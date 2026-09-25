"""Tests for the durable conversation store (``conversations.db``)."""

from __future__ import annotations

import logging
import multiprocessing
import sqlite3
import stat
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from openjarvis.conversations import (
    MAX_METADATA_BYTES,
    OWNER_USER_ID,
    SCHEMA_VERSION,
    ConversationStore,
    ConversationStoreError,
)

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX permission semantics"
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "conversations.db"


@pytest.fixture
def store(db_path: Path):
    s = ConversationStore(db_path)
    yield s
    s.close()


def _conv(store: ConversationStore, **kwargs):
    kwargs.setdefault("origin", "cli")
    kwargs.setdefault("user_id", store.ensure_owner())
    return store.create_conversation(**kwargs)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# ---------------------------------------------------------------------------
# Schema, pragmas, file security
# ---------------------------------------------------------------------------


class TestSchema:
    def test_creates_schema_and_sets_user_version(self, store, db_path):
        conn = sqlite3.connect(db_path)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','trigger')"
                )
            }
        finally:
            conn.close()
        assert {
            "users",
            "channel_identities",
            "conversations",
            "channel_threads",
            "messages",
            "messages_no_update",
        } <= names
        assert SCHEMA_VERSION == 1

    def test_reopen_is_idempotent_and_preserves_data(self, db_path):
        with ConversationStore(db_path) as s1:
            conv = _conv(s1)
            s1.append_message(conv.conversation_id, "user", "hi", surface="cli")
        with ConversationStore(db_path) as s2:
            assert s2.get_conversation(conv.conversation_id) is not None
            assert [m.content for m in s2.get_messages(conv.conversation_id)] == ["hi"]

    def test_pragmas(self, store):
        conn = store._conn
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_newer_schema_is_rejected(self, db_path):
        ConversationStore(db_path).close()
        conn = sqlite3.connect(db_path)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
        conn.commit()
        conn.close()
        with pytest.raises(ConversationStoreError):
            ConversationStore(db_path)
        # The rejected open must not downgrade the version.
        conn = sqlite3.connect(db_path)
        try:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        finally:
            conn.close()
        assert version == SCHEMA_VERSION + 1

    def test_in_memory_creates_no_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with ConversationStore(":memory:") as s:
            conv = _conv(s)
            s.append_message(conv.conversation_id, "user", "x", surface="cli")
            assert s.db_path is None
        assert list(tmp_path.iterdir()) == []

    @posix_only
    def test_file_and_sidecar_permissions(self, store, db_path):
        conv = _conv(store)
        store.append_message(conv.conversation_id, "user", "secret", surface="cli")
        assert _mode(db_path) == 0o600
        assert _mode(db_path.parent) == 0o700
        for suffix in ("-wal", "-shm"):
            sidecar = db_path.with_name(db_path.name + suffix)
            if sidecar.exists():
                assert _mode(sidecar) == 0o600


class TestDefaultPath:
    def test_default_path_honors_openjarvis_home(self, tmp_path, monkeypatch):
        home = tmp_path / "ojhome"
        monkeypatch.setenv("OPENJARVIS_HOME", str(home))
        with ConversationStore() as s:
            assert s.db_path == home.resolve() / "conversations.db"
        assert (home / "conversations.db").exists()

    def test_default_path_honors_xdg_data_home(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENJARVIS_HOME", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
        with ConversationStore() as s:
            expected = (tmp_path / "xdg" / "openjarvis").resolve()
            assert s.db_path == expected / "conversations.db"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_ensure_owner_is_idempotent(self, store, db_path):
        assert store.ensure_owner() == OWNER_USER_ID
        assert store.ensure_owner() == OWNER_USER_ID
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT user_id, display_name FROM users").fetchall()
        finally:
            conn.close()
        assert rows == [("owner", "")]

    def test_unknown_channel_identity_is_not_mapped(self, store, db_path):
        store.ensure_owner()
        assert store.resolve_channel_identity("telegram", "12345") is None
        conn = sqlite3.connect(db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM channel_identities").fetchone()
        finally:
            conn.close()
        assert count[0] == 0

    def test_explicit_link_then_resolve(self, store):
        owner = store.ensure_owner()
        store.link_channel_identity("telegram", "+1 555 0100", owner)
        store.link_channel_identity("telegram", "+1 555 0100", owner)  # no-op
        assert store.resolve_channel_identity("telegram", "+1 555 0100") == owner
        assert store.resolve_channel_identity("discord", "+1 555 0100") is None

    def test_link_to_unknown_user_raises(self, store):
        with pytest.raises(sqlite3.IntegrityError):
            store.link_channel_identity("telegram", "1", "nobody")

    def test_relink_to_different_user_raises(self, store):
        owner = store.ensure_owner()
        store.link_channel_identity("telegram", "1", owner)
        store._conn.execute(
            "INSERT INTO users (user_id, display_name, created_at)"
            " VALUES ('other', '', 0)"
        )
        with pytest.raises(ValueError):
            store.link_channel_identity("telegram", "1", "other")
        assert store.resolve_channel_identity("telegram", "1") == owner


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


class TestConversations:
    def test_create_and_get(self, store):
        conv = _conv(store, metadata={"engine": "ollama"})
        assert len(conv.conversation_id) == 32
        assert conv.user_id == OWNER_USER_ID
        assert conv.origin == "cli"
        assert conv.title == ""
        assert conv.metadata == {"engine": "ollama"}
        assert conv.created_at == conv.updated_at
        assert store.get_conversation(conv.conversation_id) == conv
        assert store.get_conversation("missing") is None

    def test_unowned_conversation_allowed(self, store):
        conv = store.create_conversation(origin="channel:telegram", user_id=None)
        assert conv.user_id is None

    def test_conversation_for_unknown_user_raises(self, store):
        with pytest.raises(sqlite3.IntegrityError):
            store.create_conversation(origin="cli", user_id="nobody")

    def test_caller_supplied_id(self, store):
        conv = _conv(store, conversation_id="lx3k9a2b1c")
        assert conv.conversation_id == "lx3k9a2b1c"

    @pytest.mark.parametrize("bad", ["", "has space", "x" * 129, "a/b"])
    def test_invalid_conversation_id(self, store, bad):
        with pytest.raises(ValueError):
            _conv(store, conversation_id=bad)

    def test_invalid_origin(self, store):
        with pytest.raises(ValueError):
            _conv(store, origin="")

    def test_list_orders_by_recent_activity_and_filters(self, store):
        a = _conv(store)
        b = _conv(store)
        c = store.create_conversation(origin="cli", user_id=None)
        store.append_message(
            a.conversation_id, "user", "later", surface="cli", created_at=9e9
        )
        listed = [x.conversation_id for x in store.list_conversations()]
        assert listed[0] == a.conversation_id
        assert set(listed) == {a.conversation_id, b.conversation_id, c.conversation_id}
        owned = store.list_conversations(user_id=OWNER_USER_ID)
        assert {x.conversation_id for x in owned} == {
            a.conversation_id,
            b.conversation_id,
        }
        assert len(store.list_conversations(limit=1)) == 1
        with pytest.raises(ValueError):
            store.list_conversations(limit=0)

    def test_resolve_channel_thread_is_idempotent(self, store):
        first = store.resolve_channel_thread("telegram", "chat-42", user_id=None)
        again = store.resolve_channel_thread(
            "telegram", "chat-42", user_id=store.ensure_owner()
        )
        other = store.resolve_channel_thread("telegram", "chat-43", user_id=None)
        assert first.conversation_id == again.conversation_id
        assert again.user_id is None  # owner is only applied at creation
        assert first.origin == "channel:telegram"
        assert other.conversation_id != first.conversation_id


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class TestMessages:
    def test_append_and_read_back(self, store):
        conv = _conv(store)
        msg = store.append_message(
            conv.conversation_id,
            "assistant",
            "hello",
            surface="cli",
            session_id="sess-1",
            agent_id="simple",
            model="ollama/llama3.1:8b",
            trace_id="trace-1",
            metadata={"k": 1},
        )
        assert len(msg.message_id) == 32
        (stored,) = store.get_messages(conv.conversation_id)
        assert stored == msg
        assert stored.session_id == "sess-1"
        assert stored.agent_id == "simple"
        assert stored.model == "ollama/llama3.1:8b"
        assert stored.trace_id == "trace-1"

    def test_order_is_insertion_order_even_with_equal_timestamps(self, store):
        conv = _conv(store)
        for i in range(5):
            store.append_message(
                conv.conversation_id, "user", f"m{i}", surface="cli", created_at=100.0
            )
        contents = [m.content for m in store.get_messages(conv.conversation_id)]
        assert contents == ["m0", "m1", "m2", "m3", "m4"]
        recent = [m.content for m in store.get_messages(conv.conversation_id, limit=2)]
        assert recent == ["m3", "m4"]
        with pytest.raises(ValueError):
            store.get_messages(conv.conversation_id, limit=0)

    def test_append_bumps_updated_at(self, store):
        conv = _conv(store)
        store.append_message(
            conv.conversation_id,
            "user",
            "x",
            surface="cli",
            created_at=conv.updated_at + 50,
        )
        updated = store.get_conversation(conv.conversation_id)
        assert updated.updated_at == conv.updated_at + 50
        # An older timestamp never moves updated_at backwards.
        store.append_message(
            conv.conversation_id, "user", "y", surface="cli", created_at=1.0
        )
        assert store.get_conversation(conv.conversation_id).updated_at == (
            conv.updated_at + 50
        )

    # Full field set for a message used by the idempotency tests below.
    _FULL = {
        "role": "assistant",
        "content": "private reply",
        "surface": "cli",
        "created_at": 1234.5,
        "session_id": "sess-1",
        "agent_id": "simple",
        "model": "ollama/llama3.1:8b",
        "trace_id": "trace-1",
        "metadata": {"k": "v", "n": 1},
    }

    def _append_full(self, store, conversation_id, **overrides):
        args = {**self._FULL, **overrides}
        role = args.pop("role")
        content = args.pop("content")
        return store.append_message(
            conversation_id, role, content, message_id="m-1", **args
        )

    def test_exact_retry_is_idempotent(self, store):
        conv = _conv(store)
        first = self._append_full(store, conv.conversation_id)
        # Same metadata in a different key order is the same message.
        again = self._append_full(
            store, conv.conversation_id, metadata={"n": 1, "k": "v"}
        )
        assert again == first
        assert len(store.get_messages(conv.conversation_id)) == 1

    @pytest.mark.parametrize(
        "field, changed",
        [
            ("content", "different reply"),
            ("role", "user"),
            ("surface", "web"),
            ("created_at", 9999.0),
            ("session_id", "sess-2"),
            ("session_id", None),
            ("agent_id", "other"),
            ("agent_id", None),
            ("model", "gpt-4o"),
            ("model", None),
            ("trace_id", "trace-2"),
            ("trace_id", None),
            ("metadata", {"k": "changed", "n": 1}),
            ("metadata", None),
        ],
    )
    def test_retry_with_changed_field_is_rejected(self, store, field, changed):
        conv = _conv(store)
        first = self._append_full(store, conv.conversation_id)
        with pytest.raises(ValueError) as excinfo:
            self._append_full(store, conv.conversation_id, **{field: changed})
        # The error must not echo stored or requested values.
        message = str(excinfo.value)
        assert "private reply" not in message and "different reply" not in message
        assert "changed" not in message
        assert store.get_messages(conv.conversation_id) == [first]

    def test_retry_into_other_conversation_is_rejected(self, store):
        conv = _conv(store)
        other = _conv(store)
        first = self._append_full(store, conv.conversation_id)
        with pytest.raises(ValueError):
            self._append_full(store, other.conversation_id)
        assert store.get_messages(conv.conversation_id) == [first]
        assert store.get_messages(other.conversation_id) == []

    def test_caller_message_id_with_omitted_created_at_retries_idempotently(
        self, store
    ):
        conv = _conv(store)
        args = {**self._FULL}
        args.pop("created_at")
        role, content = args.pop("role"), args.pop("content")
        first = store.append_message(
            conv.conversation_id, role, content, message_id="m-1", **args
        )
        with patch("openjarvis.conversations.store.time.time", return_value=9e9):
            # The retry omits created_at too; a later clock must not matter.
            again = store.append_message(
                conv.conversation_id, role, content, message_id="m-1", **args
            )
            assert again == first
            assert again.created_at == first.created_at
            # Supplying the original timestamp explicitly is also a match ...
            explicit = store.append_message(
                conv.conversation_id,
                role,
                content,
                message_id="m-1",
                created_at=first.created_at,
                **args,
            )
            assert explicit == first
            # ... but an explicit, different timestamp is a different message.
            with pytest.raises(ValueError):
                store.append_message(
                    conv.conversation_id,
                    role,
                    content,
                    message_id="m-1",
                    created_at=first.created_at + 1,
                    **args,
                )
        assert store.get_messages(conv.conversation_id) == [first]
        # A no-op retry never bumps the conversation's activity time.
        assert store.get_conversation(conv.conversation_id).updated_at < 9e9

    def test_append_to_unknown_conversation_raises(self, store):
        with pytest.raises(sqlite3.IntegrityError):
            store.append_message("missing", "user", "x", surface="cli")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"role": "wizard"},
            {"surface": ""},
            {"message_id": "bad id"},
            {"message_id": ""},
            {"created_at": float("nan")},
            {"created_at": True},
            {"model": "bad\nmodel"},
            {"agent_id": ""},
            {"content": None},
        ],
    )
    def test_invalid_append_arguments(self, store, kwargs):
        conv = _conv(store)
        args = {"role": "user", "content": "x", "surface": "cli"}
        args.update(kwargs)
        role = args.pop("role")
        content = args.pop("content")
        with pytest.raises(ValueError):
            store.append_message(conv.conversation_id, role, content, **args)
        assert store.get_messages(conv.conversation_id) == []

    def test_messages_are_append_only(self, store):
        conv = _conv(store)
        store.append_message(conv.conversation_id, "user", "orig", surface="cli")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store._conn.execute("UPDATE messages SET content = 'changed'")
        assert store.get_messages(conv.conversation_id)[0].content == "orig"

    def test_delete_is_not_blocked_for_future_retention(self, store):
        conv = _conv(store)
        store.append_message(conv.conversation_id, "user", "x", surface="cli")
        store._conn.execute("DELETE FROM messages")
        assert store.get_messages(conv.conversation_id) == []


# ---------------------------------------------------------------------------
# Metadata policy
# ---------------------------------------------------------------------------


class TestMetadata:
    @pytest.mark.parametrize(
        "metadata",
        [
            {"nested": {"a": 1}},
            {"items": [1, 2]},
            {1: "int key"},
            {"": "empty key"},
            {"api_key": "x"},
            {"OPENAI_API_KEY": "x"},
            {"access_token": "x"},
            {"client_secret": "x"},
            {"password": "x"},
            {"Authorization": "x"},
            {"cookie": "x"},
            {"credentials": "x"},
            {"ratio": float("inf")},
            {"blob": "x" * MAX_METADATA_BYTES},
            ["not", "a", "mapping"],
        ],
    )
    def test_rejected_metadata(self, store, metadata):
        with pytest.raises(ValueError):
            _conv(store, metadata=metadata)

    def test_accepted_scalar_metadata(self, store):
        meta = {"engine": "ollama", "n": 1, "f": 0.5, "flag": True, "none": None}
        assert _conv(store, metadata=meta).metadata == meta

    def test_message_metadata_uses_same_policy(self, store):
        conv = _conv(store)
        with pytest.raises(ValueError):
            store.append_message(
                conv.conversation_id,
                "user",
                "x",
                surface="cli",
                metadata={"bearer_token": "x"},
            )


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def _process_worker(db_path: str, conversation_id: str, count: int) -> None:
    with ConversationStore(db_path) as s:
        for i in range(count):
            s.append_message(conversation_id, "user", f"p{i}", surface="cli")


class TestConcurrency:
    def test_threads_with_separate_stores_on_one_file(self, db_path):
        with ConversationStore(db_path) as setup:
            conv_id = _conv(setup).conversation_id
        stores = [ConversationStore(db_path) for _ in range(2)]
        errors: list[BaseException] = []

        def worker(s: ConversationStore, tag: str) -> None:
            try:
                for i in range(50):
                    s.append_message(conv_id, "user", f"{tag}{i}", surface="cli")
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(s, f"t{n}-"))
            for n, s in enumerate(stores)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for s in stores:
            s.close()
        assert errors == []
        with ConversationStore(db_path) as check:
            assert len(check.get_messages(conv_id)) == 100

    def test_threads_sharing_one_store(self, store):
        conv_id = _conv(store).conversation_id
        threads = [
            threading.Thread(
                target=lambda: [
                    store.append_message(conv_id, "user", "x", surface="cli")
                    for _ in range(25)
                ]
            )
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(store.get_messages(conv_id)) == 100

    def test_multiple_processes(self, db_path):
        with ConversationStore(db_path) as setup:
            conv_id = _conv(setup).conversation_id
        ctx = multiprocessing.get_context("spawn")
        procs = [
            ctx.Process(target=_process_worker, args=(str(db_path), conv_id, 25))
            for _ in range(2)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
        assert [p.exitcode for p in procs] == [0, 0]
        with ConversationStore(db_path) as check:
            assert len(check.get_messages(conv_id)) == 50


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------


class TestPrivacy:
    def test_no_content_logged(self, store, caplog):
        marker = "TOP-SECRET-TRANSCRIPT-7f3a"
        caplog.set_level(logging.DEBUG)
        conv = _conv(store)
        store.append_message(conv.conversation_id, "user", marker, surface="cli")
        store.get_messages(conv.conversation_id)
        with pytest.raises(ValueError) as excinfo:
            store.append_message(conv.conversation_id, "wizard", marker, surface="cli")
        assert marker not in caplog.text
        assert marker not in str(excinfo.value)
