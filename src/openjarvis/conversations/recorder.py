"""Fail-soft recorder that persists one interactive chat into the store.

A surface (currently only ``jarvis chat``) owns one recorder per chat. The
recorder creates the durable conversation lazily on the first user message,
appends real user turns and completed assistant turns, and never lets a
persistence problem interrupt the chat: the first failure is reported once
and the recorder disables itself for the rest of that chat. A recorder can
instead :meth:`~ConversationRecorder.resume` an existing owner conversation,
after which new turns append to it; loaded rows are never rewritten.

Nothing here logs or reports transcript content — only exception class names.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional

from openjarvis.conversations.history import RESUME_MAX_MESSAGES
from openjarvis.conversations.store import (
    OWNER_USER_ID,
    ConversationMessage,
    ConversationNotFound,
    ConversationStore,
    ConversationStoreError,
)

logger = logging.getLogger(__name__)

WarnCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class ResumedConversation:
    """An existing conversation adopted by a recorder."""

    conversation_id: str
    # The newest durable messages, oldest first, exactly as stored.
    messages: List[ConversationMessage]


class ConversationRecorder:
    """Persist one chat's turns to a :class:`ConversationStore`, fail-soft."""

    def __init__(
        self,
        store: ConversationStore,
        *,
        surface: str,
        origin: str,
        model: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        warn: Optional[WarnCallback] = None,
    ) -> None:
        self._store: Optional[ConversationStore] = store
        self._surface = surface
        self._origin = origin
        self._model = model or None
        self._agent_id = agent_id or None
        self._session_id = session_id or None
        self._metadata = dict(metadata) if metadata else None
        self._warn = warn
        self._conversation_id: Optional[str] = None
        self._disabled = False
        self._warned = False

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        surface: str,
        origin: str,
        model: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        warn: Optional[WarnCallback] = None,
    ) -> Optional[ConversationRecorder]:
        """Open the configured store, or return ``None``.

        Returns ``None`` without touching the filesystem when
        ``[conversations] enabled = false``. Returns ``None`` after one
        warning when the store cannot be opened.
        """
        conv_cfg = getattr(config, "conversations", None)
        if conv_cfg is None or not conv_cfg.enabled:
            return None
        try:
            store = ConversationStore(conv_cfg.db_path or None)
        except Exception as exc:  # noqa: BLE001 — history must never break chat
            logger.warning(
                "Conversation history unavailable (%s); continuing without it",
                type(exc).__name__,
            )
            _notify(warn, f"Conversation history disabled ({type(exc).__name__})")
            return None
        return cls(
            store,
            surface=surface,
            origin=origin,
            model=model,
            agent_id=agent_id,
            session_id=session_id,
            metadata=metadata,
            warn=warn,
        )

    @property
    def disabled(self) -> bool:
        return self._disabled

    @property
    def conversation_id(self) -> Optional[str]:
        """The current durable conversation, once the first turn is recorded."""
        return self._conversation_id

    def record_user(self, content: str) -> Optional[str]:
        """Persist a user turn; returns its ``message_id`` or ``None``."""
        if self._disabled or self._store is None:
            return None
        try:
            if self._conversation_id is None:
                user_id = self._store.ensure_owner()
                conversation = self._store.create_conversation(
                    origin=self._origin,
                    user_id=user_id,
                    metadata=self._metadata,
                )
                self._conversation_id = conversation.conversation_id
            message = self._store.append_message(
                self._conversation_id,
                "user",
                content,
                surface=self._surface,
                message_id=uuid.uuid4().hex,
                session_id=self._session_id,
            )
        except Exception as exc:  # noqa: BLE001 — history must never break chat
            self._fail(exc)
            return None
        return message.message_id

    def record_assistant(
        self, content: str, *, trace_id: Optional[str] = None
    ) -> Optional[str]:
        """Persist a completed assistant turn; returns its ``message_id``.

        Does nothing if no user turn has been recorded in the current
        conversation, so an assistant reply is never orphaned.
        """
        if self._disabled or self._store is None or self._conversation_id is None:
            return None
        try:
            message = self._store.append_message(
                self._conversation_id,
                "assistant",
                content,
                surface=self._surface,
                message_id=uuid.uuid4().hex,
                session_id=self._session_id,
                agent_id=self._agent_id,
                model=self._model,
                trace_id=trace_id,
            )
        except Exception as exc:  # noqa: BLE001 — history must never break chat
            self._fail(exc)
            return None
        return message.message_id

    def resume(
        self,
        conversation_id: Optional[str] = None,
        *,
        max_messages: int = RESUME_MAX_MESSAGES,
    ) -> Optional[ResumedConversation]:
        """Adopt an existing owner conversation so new turns append to it.

        With ``conversation_id=None``, adopts the owner's most recently
        updated conversation from this recorder's origin, or returns ``None``
        when there is none. An explicit id that is missing or not owned by
        the canonical owner raises :class:`ConversationNotFound`.

        Unlike the ``record_*`` methods this is an explicit user request, so
        failures raise instead of failing soft: store errors propagate and
        a disabled recorder raises :class:`ConversationStoreError`. Nothing is
        written; loaded messages are returned, never re-appended.
        """
        if self._disabled or self._store is None:
            raise ConversationStoreError("conversation history unavailable")
        if conversation_id is None:
            latest = self._store.list_conversations(
                user_id=OWNER_USER_ID, origin=self._origin, limit=1
            )
            if not latest:
                return None
            conversation = latest[0]
        else:
            conversation = self._store.get_conversation(conversation_id)
            if conversation is None or conversation.user_id != OWNER_USER_ID:
                raise ConversationNotFound("conversation not found")
        messages = self._store.get_messages(
            conversation.conversation_id, limit=max_messages
        )
        self._conversation_id = conversation.conversation_id
        return ResumedConversation(
            conversation_id=conversation.conversation_id, messages=messages
        )

    def transcript(self) -> Optional[List[ConversationMessage]]:
        """Return every durable message of the current conversation.

        Returns ``[]`` before the first turn is recorded, and ``None`` when
        durable history is unavailable (disabled, closed, or the read
        failed) so the caller can fall back to its own view. A failed read
        does not disable recording.
        """
        if self._disabled or self._store is None:
            return None
        if self._conversation_id is None:
            return []
        try:
            return self._store.get_messages(self._conversation_id)
        except Exception as exc:  # noqa: BLE001 — history must never break chat
            logger.debug("Conversation transcript read failed: %s", type(exc).__name__)
            return None

    def reset(self) -> None:
        """Start a new durable conversation on the next user turn."""
        self._conversation_id = None

    def close(self) -> None:
        store, self._store = self._store, None
        if store is None:
            return
        try:
            store.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Conversation store close failed: %s", type(exc).__name__)

    def _fail(self, exc: BaseException) -> None:
        self._disabled = True
        if self._warned:
            return
        self._warned = True
        logger.warning(
            "Conversation history disabled for this chat after a persistence "
            "error (%s)",
            type(exc).__name__,
        )
        _notify(
            self._warn,
            f"Conversation history disabled for this chat ({type(exc).__name__})",
        )


def _notify(warn: Optional[WarnCallback], message: str) -> None:
    if warn is None:
        return
    try:
        warn(message)
    except Exception:  # noqa: BLE001 — a broken warning sink must not break chat
        logger.debug("Conversation warning callback failed", exc_info=True)


__all__ = ["ConversationRecorder", "ResumedConversation"]
