"""Tests for the fail-soft ConversationRecorder."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from openjarvis.conversations import (
    OWNER_USER_ID,
    ConversationRecorder,
    ConversationStore,
)
from openjarvis.core.config import JarvisConfig


@pytest.fixture
def store():
    s = ConversationStore(":memory:")
    yield s
    s.close()


def _recorder(store, **kwargs) -> ConversationRecorder:
    kwargs.setdefault("surface", "cli")
    kwargs.setdefault("origin", "cli")
    return ConversationRecorder(store, **kwargs)


def _config(db_path: Path, *, enabled: bool = True) -> JarvisConfig:
    cfg = JarvisConfig()
    cfg.conversations.enabled = enabled
    cfg.conversations.db_path = str(db_path)
    return cfg


class TestLifecycle:
    def test_no_conversation_until_first_user_turn(self, store):
        rec = _recorder(store)
        assert rec.record_assistant("orphan") is None
        assert rec.conversation_id is None
        assert store.list_conversations() == []

    def test_user_then_assistant(self, store):
        rec = _recorder(
            store,
            model="test-model",
            agent_id="simple",
            metadata={"engine": "mock"},
        )
        user_id = rec.record_user("hello")
        assistant_id = rec.record_assistant("hi there")
        assert user_id and assistant_id and user_id != assistant_id

        (conv,) = store.list_conversations()
        assert conv.conversation_id == rec.conversation_id
        assert conv.user_id == OWNER_USER_ID
        assert conv.origin == "cli"
        assert conv.metadata == {"engine": "mock"}

        user, assistant = store.get_messages(conv.conversation_id)
        assert (user.role, user.content, user.surface) == ("user", "hello", "cli")
        assert user.model is None and user.agent_id is None
        assert (assistant.role, assistant.content) == ("assistant", "hi there")
        assert assistant.model == "test-model"
        assert assistant.agent_id == "simple"
        assert assistant.session_id is None and assistant.trace_id is None

    def test_optional_references_are_passed_through(self, store):
        rec = _recorder(store, session_id="sess-9")
        rec.record_user("q")
        rec.record_assistant("a", trace_id="trace-9")
        user, assistant = store.get_messages(rec.conversation_id)
        assert user.session_id == assistant.session_id == "sess-9"
        assert assistant.trace_id == "trace-9"

    def test_reset_starts_new_conversation(self, store):
        rec = _recorder(store)
        rec.record_user("one")
        first = rec.conversation_id
        rec.reset()
        assert rec.record_assistant("dropped") is None
        rec.record_user("two")
        assert rec.conversation_id != first
        assert len(store.list_conversations()) == 2

    def test_close_is_idempotent(self, store):
        rec = _recorder(store)
        rec.close()
        rec.close()
        assert rec.record_user("after close") is None


class TestFromConfig:
    def test_disabled_config_returns_none_and_creates_no_file(self, tmp_path):
        db_path = tmp_path / "sub" / "conversations.db"
        rec = ConversationRecorder.from_config(
            _config(db_path, enabled=False), surface="cli", origin="cli"
        )
        assert rec is None
        assert not db_path.exists()
        assert not db_path.parent.exists()

    def test_enabled_config_opens_store(self, tmp_path):
        db_path = tmp_path / "conversations.db"
        rec = ConversationRecorder.from_config(
            _config(db_path), surface="cli", origin="cli"
        )
        assert rec is not None
        rec.record_user("x")
        rec.close()
        with ConversationStore(db_path) as s:
            assert len(s.list_conversations()) == 1

    def test_open_failure_warns_once_and_returns_none(self, tmp_path, caplog):
        warnings: list[str] = []
        with patch(
            "openjarvis.conversations.recorder.ConversationStore",
            side_effect=PermissionError("denied"),
        ):
            rec = ConversationRecorder.from_config(
                _config(tmp_path / "c.db"),
                surface="cli",
                origin="cli",
                warn=warnings.append,
            )
        assert rec is None
        assert warnings == ["Conversation history disabled (PermissionError)"]


class TestFailSoft:
    def test_write_failure_disables_and_warns_once(self, store, caplog):
        marker = "PRIVATE-TURN-CONTENT-91c2"
        warnings: list[str] = []
        rec = _recorder(store, warn=warnings.append)
        caplog.set_level(logging.DEBUG)
        with patch.object(
            store, "append_message", side_effect=OSError("disk full")
        ) as append:
            assert rec.record_user(marker) is None
            assert rec.disabled
            assert rec.record_user(marker) is None
            assert rec.record_assistant(marker) is None
        assert append.call_count == 1  # disabled after the first failure
        assert warnings == ["Conversation history disabled for this chat (OSError)"]
        assert caplog.text.count("Conversation history disabled") == 1
        assert marker not in caplog.text
        assert all(marker not in w for w in warnings)

    def test_invalid_metadata_fails_soft(self, store):
        warnings: list[str] = []
        rec = _recorder(store, metadata={"api_key": "x"}, warn=warnings.append)
        assert rec.record_user("hello") is None
        assert rec.disabled
        assert warnings == ["Conversation history disabled for this chat (ValueError)"]
        assert store.list_conversations() == []

    def test_broken_warn_callback_does_not_raise(self, store):
        def bad_warn(_msg: str) -> None:
            raise RuntimeError("sink broken")

        rec = _recorder(store, warn=bad_warn)
        with patch.object(store, "ensure_owner", side_effect=OSError("x")):
            assert rec.record_user("hello") is None
        assert rec.disabled
