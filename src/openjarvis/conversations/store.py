"""Durable, local, append-only conversation history.

This store is the long-term record of what was actually said between the
user and the assistant. It is deliberately separate from the two runtime
``SessionStore`` implementations (``openjarvis.sessions.session`` and
``openjarvis.server.session_store``), which consolidate and decay their rows
and therefore cannot serve as durable history.

The identifiers kept here are distinct concepts and are never collapsed:

* ``user_id`` — the canonical person (``"owner"`` for the single local user).
* ``(channel, external_id)`` — an external channel identity, mapped to a
  ``user_id`` only through an explicit :meth:`link_channel_identity` call.
* ``conversation_id`` — a durable thread.
* ``session_id`` / ``agent_id`` / ``trace_id`` — optional *references* to the
  runtime session, the executing agent, and ``traces.db``. They are stored as
  plain text with no cross-database foreign keys.

Message content is append-only: an ``UPDATE`` on ``messages`` aborts. Row
deletion is intentionally not blocked so a future retention/privacy feature
can remove history, but no deletion API exists yet.

The store never logs message content and emits no events or telemetry.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Union

from openjarvis.core.paths import get_config_dir

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
OWNER_USER_ID = "owner"
ROLES = frozenset({"user", "assistant", "system", "tool"})
MAX_METADATA_BYTES = 4096
DEFAULT_BUSY_TIMEOUT_MS = 5000

# Primary-key style identifiers we generate or accept (uuid hex, frontend
# base36 ids, "owner", "channel:telegram", ...).
_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
# Free-form references (model names like "ollama/llama3:8b", external ids such
# as phone numbers): non-empty, bounded, and free of control characters.
_REF_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")
_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|passwd|authorization|cookie|credential)"
)
_MAX_TITLE_CHARS = 256
_SIDECAR_SUFFIXES = ("-wal", "-shm")


class ConversationStoreError(RuntimeError):
    """Raised when the conversation database cannot be used safely."""


@dataclass(frozen=True, slots=True)
class Conversation:
    """A durable conversation thread."""

    conversation_id: str
    user_id: Optional[str]
    title: str
    origin: str
    created_at: float
    updated_at: float
    metadata: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """A single persisted message within a conversation."""

    message_id: str
    conversation_id: str
    role: str
    content: str
    surface: str
    created_at: float
    session_id: Optional[str]
    agent_id: Optional[str]
    model: Optional[str]
    trace_id: Optional[str]
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Schema migrations — index ``i`` upgrades user_version ``i`` -> ``i + 1``.
# Statements run one by one inside the caller's transaction (``executescript``
# would implicitly COMMIT, so it is never used here).
# ---------------------------------------------------------------------------

_V1_STATEMENTS = (
    """
    CREATE TABLE users (
        user_id       TEXT PRIMARY KEY CHECK (length(user_id) > 0),
        display_name  TEXT NOT NULL DEFAULT '',
        created_at    REAL NOT NULL
    )
    """,
    """
    CREATE TABLE channel_identities (
        channel       TEXT NOT NULL CHECK (length(channel) > 0),
        external_id   TEXT NOT NULL CHECK (length(external_id) > 0),
        user_id       TEXT NOT NULL
                      REFERENCES users(user_id) ON DELETE RESTRICT,
        created_at    REAL NOT NULL,
        PRIMARY KEY (channel, external_id)
    )
    """,
    "CREATE INDEX idx_channel_identities_user ON channel_identities(user_id)",
    """
    CREATE TABLE conversations (
        conversation_id TEXT PRIMARY KEY,
        user_id         TEXT REFERENCES users(user_id) ON DELETE RESTRICT,
        title           TEXT NOT NULL DEFAULT '',
        origin          TEXT NOT NULL CHECK (length(origin) > 0),
        created_at      REAL NOT NULL,
        updated_at      REAL NOT NULL,
        metadata        TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE INDEX idx_conversations_user_updated
        ON conversations(user_id, updated_at)
    """,
    "CREATE INDEX idx_conversations_updated ON conversations(updated_at)",
    """
    CREATE TABLE channel_threads (
        channel            TEXT NOT NULL CHECK (length(channel) > 0),
        external_thread_id TEXT NOT NULL CHECK (length(external_thread_id) > 0),
        conversation_id    TEXT NOT NULL
                           REFERENCES conversations(conversation_id)
                           ON DELETE RESTRICT,
        created_at         REAL NOT NULL,
        PRIMARY KEY (channel, external_thread_id)
    )
    """,
    """
    CREATE INDEX idx_channel_threads_conversation
        ON channel_threads(conversation_id)
    """,
    """
    CREATE TABLE messages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id      TEXT NOT NULL UNIQUE,
        conversation_id TEXT NOT NULL
                        REFERENCES conversations(conversation_id)
                        ON DELETE RESTRICT,
        role            TEXT NOT NULL
                        CHECK (role IN ('user', 'assistant', 'system', 'tool')),
        content         TEXT NOT NULL,
        surface         TEXT NOT NULL CHECK (length(surface) > 0),
        created_at      REAL NOT NULL,
        session_id      TEXT,
        agent_id        TEXT,
        model           TEXT,
        trace_id        TEXT,
        metadata        TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX idx_messages_conversation ON messages(conversation_id, id)",
    """
    CREATE TRIGGER messages_no_update BEFORE UPDATE ON messages
    BEGIN
        SELECT RAISE(ABORT, 'messages are append-only');
    END
    """,
)


def _migrate_to_v1(conn: sqlite3.Connection) -> None:
    for statement in _V1_STATEMENTS:
        conn.execute(statement)


_MIGRATIONS: tuple[Callable[[sqlite3.Connection], None], ...] = (_migrate_to_v1,)
assert len(_MIGRATIONS) == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Validation helpers (error messages never echo values, only field names)
# ---------------------------------------------------------------------------


def _check_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValueError(f"invalid {field_name}")
    return value


def _check_optional_ref(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not _REF_RE.match(value):
        raise ValueError(f"invalid {field_name}")
    return value


def _check_timestamp(value: Any) -> float:
    if value is None:
        return time.time()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid created_at")
    if not math.isfinite(value):
        raise ValueError("invalid created_at")
    return float(value)


def _serialize_metadata(metadata: Optional[Mapping[str, Any]]) -> str:
    """Validate and serialize metadata.

    Metadata must be a flat mapping of string keys to scalar values, at most
    ``MAX_METADATA_BYTES`` once serialized, and must not use secret-like key
    names. It is descriptive context, not a place to stash credentials.
    """
    if metadata is None:
        return "{}"
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        if _SECRET_KEY_RE.search(key):
            raise ValueError("metadata key looks like a secret")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError("metadata values must be scalars")
    try:
        encoded = json.dumps(
            dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except ValueError:
        raise ValueError("metadata values must be finite") from None
    if len(encoded.encode("utf-8")) > MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds size limit")
    return encoded


def _locked(func):
    """Serialize access to the shared connection across threads."""

    @wraps(func)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return func(self, *args, **kwargs)

    return wrapped


_CONVERSATION_COLUMNS = (
    "conversation_id, user_id, title, origin, created_at, updated_at, metadata"
)
_MESSAGE_COLUMNS = (
    "message_id, conversation_id, role, content, surface, created_at, "
    "session_id, agent_id, model, trace_id, metadata"
)


def _row_to_conversation(row: sqlite3.Row) -> Conversation:
    return Conversation(
        conversation_id=row["conversation_id"],
        user_id=row["user_id"],
        title=row["title"],
        origin=row["origin"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        metadata=json.loads(row["metadata"]),
    )


def _row_to_message(row: sqlite3.Row) -> ConversationMessage:
    return ConversationMessage(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        surface=row["surface"],
        created_at=row["created_at"],
        session_id=row["session_id"],
        agent_id=row["agent_id"],
        model=row["model"],
        trace_id=row["trace_id"],
        metadata=json.loads(row["metadata"]),
    )


class ConversationStore:
    """SQLite-backed durable conversation store (``conversations.db``).

    Parameters
    ----------
    db_path:
        Database file. ``None`` resolves to ``get_config_dir() /
        "conversations.db"`` (honoring ``OPENJARVIS_HOME`` / ``XDG_DATA_HOME``).
        ``":memory:"`` gives a private in-memory database and creates no file.
    busy_timeout_ms:
        How long a writer waits for another process's lock before failing.
    """

    def __init__(
        self,
        db_path: Union[str, Path, None] = None,
        *,
        busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        if db_path is None:
            db_path = get_config_dir() / "conversations.db"
        self._in_memory = str(db_path) == ":memory:"
        self._path: Optional[Path] = None
        if not self._in_memory:
            from openjarvis.security.file_utils import secure_create

            self._path = Path(db_path).expanduser()
            secure_create(self._path)

        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(
            ":memory:" if self._in_memory else str(self._path),
            timeout=busy_timeout_ms / 1000,
            check_same_thread=False,
            isolation_level=None,  # explicit transactions only
        )
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
            if not self._in_memory:
                self._conn.execute("PRAGMA journal_mode = WAL")
                self._conn.execute("PRAGMA synchronous = NORMAL")
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._migrate()
            self._harden_sidecars()
        except BaseException:
            self._conn.close()
            self._closed = True
            raise

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def db_path(self) -> Optional[Path]:
        """The database file, or ``None`` for an in-memory store."""
        return self._path

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def _migrate(self) -> None:
        with self._transaction():
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise ConversationStoreError(
                    f"conversations.db schema version {version} is newer than "
                    f"supported version {SCHEMA_VERSION}; refusing to open"
                )
            for migrate in _MIGRATIONS[version:]:
                migrate(self._conn)
            if version != SCHEMA_VERSION:
                self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _harden_sidecars(self) -> None:
        """Restrict WAL/SHM sidecar files, which also contain transcript data.

        SQLite already creates them with the database file's mode; this is
        defense in depth.
        """
        if self._path is None:
            return
        for suffix in _SIDECAR_SUFFIXES:
            sidecar = self._path.with_name(self._path.name + suffix)
            if sidecar.exists():
                try:
                    os.chmod(sidecar, 0o600)
                except OSError as exc:
                    logger.warning(
                        "Could not restrict permissions on %s: %s",
                        sidecar,
                        type(exc).__name__,
                    )

    def _fetch_conversation(self, conversation_id: str) -> Optional[Conversation]:
        row = self._conn.execute(
            f"SELECT {_CONVERSATION_COLUMNS} FROM conversations"
            " WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return _row_to_conversation(row) if row else None

    def _insert_conversation(
        self,
        *,
        origin: str,
        user_id: Optional[str],
        title: str,
        metadata: Optional[Mapping[str, Any]],
        conversation_id: Optional[str],
    ) -> str:
        _check_id(origin, "origin")
        if user_id is not None:
            _check_id(user_id, "user_id")
        if not isinstance(title, str) or len(title) > _MAX_TITLE_CHARS:
            raise ValueError("invalid title")
        if conversation_id is None:
            conversation_id = uuid.uuid4().hex
        _check_id(conversation_id, "conversation_id")
        encoded = _serialize_metadata(metadata)
        now = time.time()
        self._conn.execute(
            "INSERT INTO conversations"
            " (conversation_id, user_id, title, origin,"
            " created_at, updated_at, metadata)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, user_id, title, origin, now, now, encoded),
        )
        return conversation_id

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @_locked
    def ensure_owner(self) -> str:
        """Ensure the canonical local owner exists and return its ``user_id``."""
        self._conn.execute(
            "INSERT OR IGNORE INTO users (user_id, display_name, created_at)"
            " VALUES (?, '', ?)",
            (OWNER_USER_ID, time.time()),
        )
        return OWNER_USER_ID

    @_locked
    def link_channel_identity(
        self, channel: str, external_id: str, user_id: str
    ) -> None:
        """Explicitly map an external channel identity to a canonical user.

        This is the ONLY way a channel identity becomes associated with a
        user. Re-linking to the same user is a no-op; re-linking an identity
        that already belongs to a different user raises ``ValueError``.
        """
        _check_id(channel, "channel")
        if _check_optional_ref(external_id, "external_id") is None:
            raise ValueError("invalid external_id")
        _check_id(user_id, "user_id")
        with self._transaction():
            row = self._conn.execute(
                "SELECT user_id FROM channel_identities"
                " WHERE channel = ? AND external_id = ?",
                (channel, external_id),
            ).fetchone()
            if row is not None:
                if row["user_id"] != user_id:
                    raise ValueError("channel identity is linked to another user")
                return
            self._conn.execute(
                "INSERT INTO channel_identities"
                " (channel, external_id, user_id, created_at)"
                " VALUES (?, ?, ?, ?)",
                (channel, external_id, user_id, time.time()),
            )

    @_locked
    def resolve_channel_identity(self, channel: str, external_id: str) -> Optional[str]:
        """Return the canonical user for a channel identity, or ``None``.

        Unknown identities are never mapped automatically and nothing is
        written.
        """
        row = self._conn.execute(
            "SELECT user_id FROM channel_identities"
            " WHERE channel = ? AND external_id = ?",
            (channel, external_id),
        ).fetchone()
        return row["user_id"] if row else None

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    @_locked
    def create_conversation(
        self,
        *,
        origin: str,
        user_id: Optional[str],
        title: str = "",
        metadata: Optional[Mapping[str, Any]] = None,
        conversation_id: Optional[str] = None,
    ) -> Conversation:
        """Create a new durable conversation.

        ``user_id`` may be ``None`` for conversations whose owner is unknown
        (e.g. an unmapped channel sender).
        """
        with self._transaction():
            new_id = self._insert_conversation(
                origin=origin,
                user_id=user_id,
                title=title,
                metadata=metadata,
                conversation_id=conversation_id,
            )
            conversation = self._fetch_conversation(new_id)
        assert conversation is not None
        return conversation

    @_locked
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        return self._fetch_conversation(conversation_id)

    @_locked
    def list_conversations(
        self, *, user_id: Optional[str] = None, limit: int = 50
    ) -> List[Conversation]:
        """List conversations, most recently updated first."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        sql = f"SELECT {_CONVERSATION_COLUMNS} FROM conversations"
        params: list = []
        if user_id is not None:
            sql += " WHERE user_id = ?"
            params.append(user_id)
        sql += " ORDER BY updated_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_conversation(row) for row in rows]

    @_locked
    def resolve_channel_thread(
        self,
        channel: str,
        external_thread_id: str,
        *,
        user_id: Optional[str],
    ) -> Conversation:
        """Get or create the durable conversation for an external thread.

        ``user_id`` is only applied when the conversation is created; an
        existing thread keeps its original owner.
        """
        _check_id(channel, "channel")
        if _check_optional_ref(external_thread_id, "external_thread_id") is None:
            raise ValueError("invalid external_thread_id")
        with self._transaction():
            row = self._conn.execute(
                "SELECT conversation_id FROM channel_threads"
                " WHERE channel = ? AND external_thread_id = ?",
                (channel, external_thread_id),
            ).fetchone()
            if row is not None:
                conversation_id = row["conversation_id"]
            else:
                conversation_id = self._insert_conversation(
                    origin=f"channel:{channel}",
                    user_id=user_id,
                    title="",
                    metadata=None,
                    conversation_id=None,
                )
                self._conn.execute(
                    "INSERT INTO channel_threads"
                    " (channel, external_thread_id, conversation_id, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (channel, external_thread_id, conversation_id, time.time()),
                )
            conversation = self._fetch_conversation(conversation_id)
        assert conversation is not None
        return conversation

    # ------------------------------------------------------------------
    # Messages (append-only)
    # ------------------------------------------------------------------

    @_locked
    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        surface: str,
        message_id: Optional[str] = None,
        created_at: Optional[float] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        model: Optional[str] = None,
        trace_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ConversationMessage:
        """Append a message to a conversation.

        Idempotent on ``message_id``: re-appending an identical message
        returns the stored row without writing. A retry is identical only if
        ``conversation_id``, ``role``, ``content``, ``surface``,
        ``session_id``, ``agent_id``, ``model``, ``trace_id`` and
        ``metadata`` (compared in canonical serialized form) all match the
        stored row; otherwise ``ValueError`` is raised.

        ``created_at`` retry semantics: an explicit ``created_at`` must match
        the stored timestamp exactly. When ``created_at`` is omitted, the
        store assigns the time of the *first* successful append; a retry that
        also omits it is not compared on timestamp and returns the original
        stored value.
        """
        _check_id(conversation_id, "conversation_id")
        if role not in ROLES:
            raise ValueError("invalid role")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        _check_id(surface, "surface")
        if message_id is None:
            message_id = uuid.uuid4().hex
        _check_id(message_id, "message_id")
        timestamp = _check_timestamp(created_at)
        session_id = _check_optional_ref(session_id, "session_id")
        agent_id = _check_optional_ref(agent_id, "agent_id")
        model = _check_optional_ref(model, "model")
        trace_id = _check_optional_ref(trace_id, "trace_id")
        encoded = _serialize_metadata(metadata)

        with self._transaction():
            cur = self._conn.execute(
                "INSERT INTO messages"
                " (message_id, conversation_id, role, content, surface,"
                " created_at, session_id, agent_id, model, trace_id, metadata)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(message_id) DO NOTHING",
                (
                    message_id,
                    conversation_id,
                    role,
                    content,
                    surface,
                    timestamp,
                    session_id,
                    agent_id,
                    model,
                    trace_id,
                    encoded,
                ),
            )
            inserted = cur.rowcount == 1
            if inserted:
                self._conn.execute(
                    "UPDATE conversations SET updated_at = MAX(updated_at, ?)"
                    " WHERE conversation_id = ?",
                    (timestamp, conversation_id),
                )
            row = self._conn.execute(
                f"SELECT {_MESSAGE_COLUMNS} FROM messages WHERE message_id = ?",
                (message_id,),
            ).fetchone()
        if not inserted:
            requested = {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "surface": surface,
                "session_id": session_id,
                "agent_id": agent_id,
                "model": model,
                "trace_id": trace_id,
                "metadata": encoded,
            }
            if created_at is not None:
                requested["created_at"] = timestamp
            # Compare raw persisted values; never echo them in the error.
            if any(row[name] != value for name, value in requested.items()):
                raise ValueError("message_id already used for a different message")
        return _row_to_message(row)

    @_locked
    def get_messages(
        self, conversation_id: str, *, limit: Optional[int] = None
    ) -> List[ConversationMessage]:
        """Return messages in chronological (insertion) order.

        With ``limit``, returns the most recent ``limit`` messages, still in
        chronological order.
        """
        if limit is None:
            rows = self._conn.execute(
                f"SELECT {_MESSAGE_COLUMNS} FROM messages"
                " WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        else:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError("limit must be a positive integer")
            rows = self._conn.execute(
                f"SELECT {_MESSAGE_COLUMNS} FROM messages"
                " WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
            rows.reverse()
        return [_row_to_message(row) for row in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @_locked
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._conn.close()

    def __enter__(self) -> ConversationStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = [
    "MAX_METADATA_BYTES",
    "OWNER_USER_ID",
    "ROLES",
    "SCHEMA_VERSION",
    "Conversation",
    "ConversationMessage",
    "ConversationStore",
    "ConversationStoreError",
]
