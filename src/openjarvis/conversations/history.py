"""Rebuild and bound model-facing context from conversation messages.

A resumed chat replays only real user and assistant turns. System prompts,
persona text and memory context are rebuilt fresh on every turn by the chat
surface, so durable ``system`` and ``tool`` rows are never replayed even if
they exist. The budget below bounds what is sent to the model; older
messages stay untouched in ``conversations.db``.

The same policy (:func:`normalize_context_messages`) bounds both resumed
history and the live working context of a running chat.

Nothing here logs or reports transcript content — only counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

from openjarvis.core.types import Message, Role

# Fixed working-context budget: at most CONTEXT_MAX_MESSAGES turns, trimmed
# from the oldest side to about CONTEXT_MAX_TOKENS (estimated with
# ``estimate_prompt_tokens``). Resume also loads only the newest
# RESUME_MAX_MESSAGES durable rows.
CONTEXT_MAX_MESSAGES = 40
CONTEXT_MAX_TOKENS = 4096
RESUME_MAX_MESSAGES = CONTEXT_MAX_MESSAGES
RESUME_MAX_TOKENS = CONTEXT_MAX_TOKENS

_ELIGIBLE_ROLES = {"user": Role.USER, "assistant": Role.ASSISTANT}


@dataclass(frozen=True, slots=True)
class ResumeContext:
    """Model-facing messages rebuilt from durable history."""

    messages: List[Message]
    # Rows that were not user/assistant turns or were malformed.
    skipped: int
    # Eligible turns left out by the token budget or turn-boundary cleanup.
    trimmed: int


def normalize_context_messages(
    messages: Sequence[Message],
    *,
    max_messages: int = CONTEXT_MAX_MESSAGES,
    max_tokens: int = CONTEXT_MAX_TOKENS,
) -> List[Message]:
    """Return completed user/assistant turns (oldest first) within budget.

    1. Drop user turns that never got a reply (a failed or interrupted
       generation), wherever they occur: of consecutive user messages only
       the most recent is kept, and trailing user messages are dropped.
    2. Keep at most the newest ``max_messages`` messages.
    3. Drop the oldest messages until the estimate fits ``max_tokens``.
    4. Drop a leading assistant message so context starts on a user turn.

    Order is preserved and the input is not modified.
    """
    from openjarvis.engine._base import estimate_prompt_tokens

    items = list(messages)
    result = [
        message
        for message, following in zip(items, [*items[1:], None])
        if not (
            message.role == Role.USER
            and (following is None or following.role == Role.USER)
        )
    ]
    if len(result) > max_messages:
        result = result[len(result) - max_messages :]
    while result and estimate_prompt_tokens(result) > max_tokens:
        result.pop(0)
    if result and result[0].role == Role.ASSISTANT:
        result.pop(0)
    return result


def build_resume_context(
    rows: Sequence[Any], *, max_tokens: int = RESUME_MAX_TOKENS
) -> ResumeContext:
    """Convert durable rows (oldest first) into model context messages.

    Keeps only ``user``/``assistant`` rows whose content is a string, then
    applies :func:`normalize_context_messages`.
    """
    messages: List[Message] = []
    skipped = 0
    for row in rows:
        role = _ELIGIBLE_ROLES.get(getattr(row, "role", None))
        content = getattr(row, "content", None)
        if role is None or not isinstance(content, str):
            skipped += 1
            continue
        messages.append(Message(role=role, content=content))

    kept = normalize_context_messages(
        messages, max_messages=RESUME_MAX_MESSAGES, max_tokens=max_tokens
    )
    return ResumeContext(
        messages=kept,
        skipped=skipped,
        trimmed=len(messages) - len(kept),
    )


__all__ = [
    "CONTEXT_MAX_MESSAGES",
    "CONTEXT_MAX_TOKENS",
    "RESUME_MAX_MESSAGES",
    "RESUME_MAX_TOKENS",
    "ResumeContext",
    "build_resume_context",
    "normalize_context_messages",
]
