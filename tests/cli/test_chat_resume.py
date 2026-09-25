"""End-to-end tests: ``jarvis chat --resume`` / ``--conversation`` continuity."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.conversations import OWNER_USER_ID, ConversationStore
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Role
from openjarvis.engine._base import estimate_prompt_tokens
from tests.cli.test_chat_conversation_persistence import (
    MEMORY_FACT_MARKER,
    SYSTEM_PROMPT_MARKER,
    _close_spy,
    _config,
    _engine,
    _read_all,
    _run_chat,
)

TRANSCRIPT_MARKER = "TRANSCRIPT-MARKER-91ac"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "home" / "conversations.db"


def _sent(engine, call: int = -1):
    return engine.generate.call_args_list[call].args[0]


def _flat(output: str) -> str:
    """Undo Rich soft-wrapping so notices can be matched as one line."""
    return " ".join(output.split())


def _roles_contents(messages):
    return [(m.role, m.content) for m in messages]


def _seed_first_chat(db_path, *turns: tuple[str, str]) -> str:
    """Run one plain chat with the given (user, reply) turns; return its id."""
    engine = _engine(*(reply for _, reply in turns))
    text = "".join(f"{user}\n" for user, _ in turns) + "/quit\n"
    result, _ = _run_chat(_config(db_path), engine, text)
    assert result.exit_code == 0, result.output
    conversations = _read_all(db_path)
    return conversations[-1][0].conversation_id


def _seed_rows(db_path, rows, *, origin="cli", user_id=OWNER_USER_ID) -> str:
    with ConversationStore(db_path) as store:
        store.ensure_owner()
        conv = store.create_conversation(origin=origin, user_id=user_id)
        for role, content in rows:
            store.append_message(conv.conversation_id, role, content, surface="cli")
        return conv.conversation_id


# ---------------------------------------------------------------------------
# Default behavior and basic resume
# ---------------------------------------------------------------------------


def test_plain_invocations_never_resume(db_path):
    _seed_first_chat(db_path, ("hello", "Hi!"))
    engine = _engine("fresh reply")
    result, _ = _run_chat(_config(db_path), engine, "new topic\n/quit\n")
    assert result.exit_code == 0, result.output
    assert "Resumed" not in result.output
    # The second plain run saw no prior turns and made its own conversation.
    assert [m.role for m in _sent(engine) if m.role != Role.SYSTEM] == [Role.USER]
    conversations = _read_all(db_path)
    assert len(conversations) == 2
    assert [m.content for m in conversations[1][1]] == ["new topic", "fresh reply"]


def test_resume_appends_to_the_same_conversation(db_path):
    conversation_id = _seed_first_chat(db_path, ("hello", "Hi!"))
    engine = _engine("Welcome back.")
    result, publish = _run_chat(_config(db_path), engine, "again\n/quit\n", "--resume")
    assert result.exit_code == 0, result.output
    assert f"Resumed conversation {conversation_id}" in _flat(result.output)
    assert "(2 earlier messages)" in _flat(result.output)

    ((conv, messages),) = _read_all(db_path)
    assert conv.conversation_id == conversation_id
    assert [(m.role, m.content) for m in messages] == [
        ("user", "hello"),
        ("assistant", "Hi!"),
        ("user", "again"),
        ("assistant", "Welcome back."),
    ]
    # Loaded history is never republished to memory extraction.
    assert publish.call_count == 1
    assert publish.call_args.args[1:] == ("again", "Welcome back.")


def test_resume_direct_engine_context_ordering(db_path):
    _seed_first_chat(db_path, ("hello", "Hi!"))
    engine = _engine("ok")
    result, _ = _run_chat(
        _config(db_path),
        engine,
        "again\n/quit\n",
        "--resume",
        "--system",
        SYSTEM_PROMPT_MARKER,
    )
    assert result.exit_code == 0, result.output
    assert _roles_contents(_sent(engine)) == [
        (Role.SYSTEM, SYSTEM_PROMPT_MARKER),
        (Role.USER, "hello"),
        (Role.ASSISTANT, "Hi!"),
        (Role.USER, "again"),
    ]


def test_resume_across_several_turns_keeps_one_copy_of_each(db_path):
    _seed_first_chat(db_path, ("hello", "Hi!"))
    engine = _engine("r1", "r2")
    result, _ = _run_chat(
        _config(db_path),
        engine,
        "second\nthird\n/quit\n",
        "--resume",
        "--system",
        SYSTEM_PROMPT_MARKER,
    )
    assert result.exit_code == 0, result.output
    sent = _sent(engine)
    assert [m.content for m in sent] == [
        SYSTEM_PROMPT_MARKER,
        "hello",
        "Hi!",
        "second",
        "r1",
        "third",
    ]
    ((_, messages),) = _read_all(db_path)
    assert [m.content for m in messages] == [
        "hello",
        "Hi!",
        "second",
        "r1",
        "third",
        "r2",
    ]


def test_resume_by_explicit_conversation_id(db_path):
    older = _seed_first_chat(db_path, ("first chat", "A"))
    newer = _seed_first_chat(db_path, ("second chat", "B"))
    assert older != newer
    engine = _engine("back to first")
    result, _ = _run_chat(
        _config(db_path), engine, "continue\n/quit\n", "--conversation", older
    )
    assert result.exit_code == 0, result.output
    assert [m.content for m in _sent(engine) if m.role != Role.SYSTEM] == [
        "first chat",
        "A",
        "continue",
    ]
    by_id = {c.conversation_id: msgs for c, msgs in _read_all(db_path)}
    assert [m.content for m in by_id[older]] == [
        "first chat",
        "A",
        "continue",
        "back to first",
    ]
    assert [m.content for m in by_id[newer]] == ["second chat", "B"]


def test_resume_with_empty_store_starts_new_with_notice(db_path):
    engine = _engine("hello there")
    result, _ = _run_chat(_config(db_path), engine, "hi\n/quit\n", "--resume")
    assert result.exit_code == 0, result.output
    assert "No previous conversation; starting new." in result.output
    ((_, messages),) = _read_all(db_path)
    assert [m.content for m in messages] == ["hi", "hello there"]


def test_resume_selects_only_owner_cli_conversations(db_path):
    owner_cli = _seed_rows(db_path, [("user", "mine"), ("assistant", "yes")])
    # Newer, but from another origin or without an owner: never selected.
    _seed_rows(
        db_path,
        [("user", "channel msg"), ("assistant", "c")],
        origin="channel:telegram",
    )
    _seed_rows(db_path, [("user", "unowned"), ("assistant", "u")], user_id=None)
    engine = _engine("ok")
    result, _ = _run_chat(_config(db_path), engine, "next\n/quit\n", "--resume")
    assert result.exit_code == 0, result.output
    assert f"Resumed conversation {owner_cli}" in _flat(result.output)
    assert [m.content for m in _sent(engine) if m.role != Role.SYSTEM] == [
        "mine",
        "yes",
        "next",
    ]


def test_clear_after_resume_starts_new_conversation(db_path):
    original = _seed_first_chat(db_path, ("hello", "Hi!"))
    engine = _engine("r1", "r2")
    result, _ = _run_chat(
        _config(db_path), engine, "more\n/clear\nfresh start\n/quit\n", "--resume"
    )
    assert result.exit_code == 0, result.output
    # After /clear the model no longer sees any resumed turns.
    assert [m.content for m in _sent(engine) if m.role != Role.SYSTEM] == [
        "fresh start"
    ]
    by_id = {c.conversation_id: msgs for c, msgs in _read_all(db_path)}
    assert len(by_id) == 2
    assert [m.content for m in by_id.pop(original)] == ["hello", "Hi!", "more", "r1"]
    ((new_messages),) = by_id.values()
    assert [m.content for m in new_messages] == ["fresh start", "r2"]


# ---------------------------------------------------------------------------
# Agent path
# ---------------------------------------------------------------------------


def test_agent_context_receives_prior_durable_turns(db_path):
    _seed_first_chat(db_path, ("hello", "Hi!"))
    captured: list[tuple[str, AgentContext | None]] = []

    class _CapturingAgent(BaseAgent):
        agent_id = "resume_capturing_agent"

        def run(self, input, context: AgentContext | None = None, **kwargs):
            captured.append((input, context))
            return AgentResult(content=f"agent:{input}", turns=1)

    AgentRegistry.register_value("resume_capturing_agent", _CapturingAgent)
    result, _ = _run_chat(
        _config(db_path),
        _engine(),
        "again\n/quit\n",
        "--resume",
        "--agent",
        "resume_capturing_agent",
    )
    assert result.exit_code == 0, result.output
    ((user_input, context),) = captured
    assert user_input == "again"
    assert context is not None
    assert _roles_contents(context.conversation.messages) == [
        (Role.USER, "hello"),
        (Role.ASSISTANT, "Hi!"),
    ]
    ((_, messages),) = _read_all(db_path)
    assert [(m.content, m.agent_id) for m in messages[2:]] == [
        ("again", None),
        ("agent:again", "resume_capturing_agent"),
    ]


def _memory_config(db_path, tmp_path):
    from openjarvis.memory.store import LocalFactStore

    facts_path = tmp_path / "facts.jsonl"
    LocalFactStore(facts_path).add(MEMORY_FACT_MARKER, source="auto", trust="auto")
    config = _config(db_path)
    config.memory.enabled = True
    config.memory.facts_path = str(facts_path)
    config.agent.context_from_memory = True
    return config


def _assert_no_markers_on_disk(db_path):
    raw = db_path.read_bytes()
    sidecar = db_path.with_name(db_path.name + "-wal")
    if sidecar.exists():
        raw += sidecar.read_bytes()
    assert SYSTEM_PROMPT_MARKER.encode() not in raw
    assert MEMORY_FACT_MARKER.encode() not in raw


@pytest.mark.parametrize("agent", [None, "simple"], ids=["direct", "simple-agent"])
def test_resumed_turn_gets_one_system_message_and_fresh_memory(
    db_path, tmp_path, agent
):
    config = _memory_config(db_path, tmp_path)
    no_backend = patch("openjarvis.cli.ask._get_memory_backend", return_value=None)
    agent_args = ("--agent", agent) if agent else ()
    first, _ = _run_chat(
        config,
        _engine("Hi!"),
        "hello\n/quit\n",
        *agent_args,
        extra_patches=[no_backend],
    )
    assert first.exit_code == 0, first.output

    engine = _engine("ok")
    result, _ = _run_chat(
        config,
        engine,
        "again\n/quit\n",
        "--resume",
        *agent_args,
        extra_patches=[
            patch("openjarvis.cli.ask._get_memory_backend", return_value=None)
        ],
    )
    assert result.exit_code == 0, result.output
    sent = _sent(engine)
    systems = [m for m in sent if m.role == Role.SYSTEM]
    assert len(systems) == 1 and sent[0] is systems[0]
    assert sum(m.text.count(MEMORY_FACT_MARKER) for m in sent) == 1
    assert _roles_contents(sent[1:]) == [
        (Role.USER, "hello"),
        (Role.ASSISTANT, "Hi!"),
        (Role.USER, "again"),
    ]
    ((_, messages),) = _read_all(db_path)
    assert [m.role for m in messages] == ["user", "assistant"] * 2
    _assert_no_markers_on_disk(db_path)


def test_durable_system_and_tool_rows_are_never_replayed(db_path):
    _seed_rows(
        db_path,
        [
            ("system", "OLD-SYSTEM-ROW"),
            ("user", "hello"),
            ("tool", "OLD-TOOL-ROW"),
            ("assistant", "Hi!"),
        ],
    )
    engine = _engine("ok")
    result, _ = _run_chat(_config(db_path), engine, "again\n/quit\n", "--resume")
    assert result.exit_code == 0, result.output
    assert "2 skipped" in _flat(result.output)
    sent_text = " ".join(m.text for m in _sent(engine))
    assert "OLD-SYSTEM-ROW" not in sent_text and "OLD-TOOL-ROW" not in sent_text
    assert [m.content for m in _sent(engine) if m.role != Role.SYSTEM] == [
        "hello",
        "Hi!",
        "again",
    ]


# ---------------------------------------------------------------------------
# Budget and normalization
# ---------------------------------------------------------------------------


def test_resume_loads_at_most_40_messages(db_path):
    rows = []
    for i in range(30):
        rows += [("user", f"u{i}"), ("assistant", f"a{i}")]
    conversation_id = _seed_rows(db_path, rows)
    engine = _engine("ok")
    result, _ = _run_chat(_config(db_path), engine, "next\n/quit\n", "--resume")
    assert result.exit_code == 0, result.output
    prior = [m.content for m in _sent(engine) if m.role != Role.SYSTEM][:-1]
    assert len(prior) == 40
    assert prior[0] == "u10" and prior[-1] == "a29"
    # Older rows remain untouched in the database.
    with ConversationStore(db_path) as store:
        assert len(store.get_messages(conversation_id)) == 62


def test_resume_respects_token_budget(db_path):
    big = "word " * 1000  # ~1250 estimated tokens per message
    rows = []
    for i in range(6):
        rows += [("user", f"u{i} {big}"), ("assistant", f"a{i} {big}")]
    _seed_rows(db_path, rows)
    engine = _engine("ok")
    result, _ = _run_chat(_config(db_path), engine, "next\n/quit\n", "--resume")
    assert result.exit_code == 0, result.output
    prior = [m for m in _sent(engine) if m.role != Role.SYSTEM][:-1]
    assert estimate_prompt_tokens(prior) <= 4096
    assert prior[0].role == Role.USER
    assert [m.content.split()[0] for m in prior] == ["u5", "a5"]
    assert "trimmed" in _flat(result.output)


def test_trailing_dangling_user_turn_not_replayed(db_path):
    failing = MagicMock()
    failing.engine_id = "mock"
    failing.generate.side_effect = [{"content": "ok"}, RuntimeError("engine down")]
    first, _ = _run_chat(_config(db_path), failing, "one\ntwo\n/quit\n")
    assert first.exit_code == 0, first.output

    engine = _engine("fine")
    result, _ = _run_chat(_config(db_path), engine, "three\n/quit\n", "--resume")
    assert result.exit_code == 0, result.output
    assert _roles_contents([m for m in _sent(engine) if m.role != Role.SYSTEM]) == [
        (Role.USER, "one"),
        (Role.ASSISTANT, "ok"),
        (Role.USER, "three"),
    ]
    # The dangling row stays permanently in the database.
    ((_, messages),) = _read_all(db_path)
    assert [m.content for m in messages] == ["one", "ok", "two", "three", "fine"]


def test_leading_assistant_after_cap_is_dropped(db_path):
    # 41 alternating rows u0, a0, ..., a19, u20: the newest 40 start on a0 and
    # end on the unanswered u20; both edges are dropped from model context.
    rows = []
    for i in range(20):
        rows += [("user", f"u{i}"), ("assistant", f"a{i}")]
    rows.append(("user", "u20"))
    _seed_rows(db_path, rows)
    engine = _engine("ok")
    result, _ = _run_chat(_config(db_path), engine, "next\n/quit\n", "--resume")
    assert result.exit_code == 0, result.output
    prior = [m for m in _sent(engine) if m.role != Role.SYSTEM][:-1]
    assert len(prior) == 38
    assert (prior[0].role, prior[0].content) == (Role.USER, "u1")
    assert (prior[-1].role, prior[-1].content) == (Role.ASSISTANT, "a19")


# ---------------------------------------------------------------------------
# Model / engine changes
# ---------------------------------------------------------------------------


def test_resume_with_different_engine(db_path):
    conversation_id = _seed_first_chat(db_path, ("hello", "Hi!"))
    engine = _engine("from another engine")
    engine.engine_id = "other"
    result, _ = _run_chat(
        _config(db_path),
        engine,
        "again\n/quit\n",
        "--resume",
        extra_patches=[
            patch("openjarvis.engine.get_engine", return_value=("other", engine))
        ],
    )
    assert result.exit_code == 0, result.output
    ((conv, _messages),) = _read_all(db_path)
    assert conv.conversation_id == conversation_id
    # Conversation metadata is informational and is not mutated on resume.
    assert conv.metadata == {"engine": "mock"}
    assert [m.content for m in _sent(engine) if m.role != Role.SYSTEM] == [
        "hello",
        "Hi!",
        "again",
    ]


def test_resume_records_current_model(db_path):
    _seed_first_chat(db_path, ("hello", "Hi!"))
    engine = _engine("new model reply")
    result, _ = _run_chat(
        _config(db_path), engine, "again\n/quit\n", "--resume", "--model", "model-two"
    )
    assert result.exit_code == 0, result.output
    ((_, messages),) = _read_all(db_path)
    assert [m.model for m in messages] == [None, "test-model", None, "model-two"]
    assert engine.generate.call_args.kwargs["model"] == "model-two"


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def _foreign_conversation(db_path) -> str:
    with ConversationStore(db_path) as store:
        store._conn.execute(
            "INSERT INTO users (user_id, display_name, created_at)"
            " VALUES ('someone-else', '', 0)"
        )
    return _seed_rows(
        db_path,
        [("user", TRANSCRIPT_MARKER), ("assistant", "x")],
        user_id="someone-else",
    )


@pytest.mark.parametrize("kind", ["missing", "malformed-id", "unmapped", "foreign"])
def test_inaccessible_explicit_conversation_exits_1(db_path, kind):
    _seed_first_chat(db_path, ("hello", "Hi!"))
    if kind == "missing":
        conversation_id = uuid.uuid4().hex
    elif kind == "malformed-id":
        conversation_id = "../../etc/passwd[bold]"
    elif kind == "unmapped":
        conversation_id = _seed_rows(
            db_path, [("user", TRANSCRIPT_MARKER)], user_id=None
        )
    else:
        conversation_id = _foreign_conversation(db_path)
    before = [(c.conversation_id, len(m)) for c, m in _read_all(db_path)]

    engine = _engine("never")
    result, publish = _run_chat(
        _config(db_path), engine, "hi\n/quit\n", "--conversation", conversation_id
    )
    assert result.exit_code == 1
    assert "Conversation not found." in result.output
    assert "Traceback" not in result.output
    assert TRANSCRIPT_MARKER not in result.output
    engine.generate.assert_not_called()
    publish.assert_not_called()
    assert [(c.conversation_id, len(m)) for c, m in _read_all(db_path)] == before


def test_failed_resume_stops_memory_service_and_closes_store(db_path):
    memory_service = MagicMock()
    closed, spy = _close_spy()
    result, _ = _run_chat(
        _config(db_path),
        _engine(),
        "hi\n",
        "--conversation",
        "nope",
        extra_patches=[
            spy,
            patch(
                "openjarvis.memory.build_memory_service", return_value=memory_service
            ),
        ],
    )
    assert result.exit_code == 1
    memory_service.stop.assert_called_once()
    assert closed == [db_path]


@pytest.mark.parametrize("flag", [["--resume"], ["--conversation", "abc"]])
def test_resume_with_conversations_disabled_exits_1(db_path, flag):
    engine = _engine("never")
    result, _ = _run_chat(_config(db_path, enabled=False), engine, "hi\n", *flag)
    assert result.exit_code == 1
    assert "disabled or unavailable; cannot resume" in result.output
    engine.generate.assert_not_called()
    assert not db_path.exists()


def test_resume_with_store_open_failure_exits_1(db_path):
    engine = _engine("never")
    result, _ = _run_chat(
        _config(db_path),
        engine,
        "hi\n",
        "--resume",
        extra_patches=[
            patch(
                "openjarvis.conversations.recorder.ConversationStore",
                side_effect=PermissionError("denied"),
            )
        ],
    )
    assert result.exit_code == 1
    assert "Conversation history disabled (PermissionError)" in result.output
    assert "cannot resume" in result.output
    engine.generate.assert_not_called()


def test_resume_load_error_exits_1_without_content(db_path, caplog):
    _seed_first_chat(db_path, (TRANSCRIPT_MARKER, "Hi!"))
    caplog.set_level(logging.DEBUG)
    engine = _engine("never")
    result, _ = _run_chat(
        _config(db_path),
        engine,
        "hi\n",
        "--resume",
        extra_patches=[
            patch(
                "openjarvis.conversations.store.ConversationStore.get_messages",
                side_effect=sqlite3.DatabaseError("disk image is malformed"),
            )
        ],
    )
    assert result.exit_code == 1
    assert "Conversation history unavailable (DatabaseError)" in result.output
    assert TRANSCRIPT_MARKER not in result.output
    assert TRANSCRIPT_MARKER not in caplog.text
    engine.generate.assert_not_called()


def test_corrupt_message_metadata_exits_1_without_content(db_path, caplog):
    conversation_id = _seed_first_chat(db_path, (TRANSCRIPT_MARKER, "Hi!"))
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO messages (message_id, conversation_id, role, content,"
        " surface, created_at, metadata) VALUES (?, ?, 'user', ?, 'cli', 1e10, ?)",
        ("bad-row", conversation_id, TRANSCRIPT_MARKER, "{not json"),
    )
    conn.commit()
    conn.close()
    caplog.set_level(logging.DEBUG)
    engine = _engine("never")
    result, _ = _run_chat(_config(db_path), engine, "hi\n", "--resume")
    assert result.exit_code == 1
    assert "Conversation history unavailable (JSONDecodeError)" in result.output
    assert TRANSCRIPT_MARKER not in result.output
    assert TRANSCRIPT_MARKER not in caplog.text
    engine.generate.assert_not_called()


def test_write_failure_after_resume_warns_once_and_continues(db_path):
    conversation_id = _seed_first_chat(db_path, ("hello", "Hi!"))
    engine = _engine("r1", "r2")
    result, publish = _run_chat(
        _config(db_path),
        engine,
        "a\nb\n/quit\n",
        "--resume",
        extra_patches=[
            patch(
                "openjarvis.conversations.store.ConversationStore.append_message",
                side_effect=OSError("disk full"),
            )
        ],
    )
    assert result.exit_code == 0, result.output
    assert "r1" in result.output and "r2" in result.output
    assert result.output.count("Conversation history disabled") == 1
    assert publish.call_count == 2
    ((conv, messages),) = _read_all(db_path)
    assert conv.conversation_id == conversation_id
    assert len(messages) == 2


def test_resume_logs_contain_no_transcript(db_path, caplog):
    _seed_first_chat(db_path, (TRANSCRIPT_MARKER, f"reply {TRANSCRIPT_MARKER}"))
    caplog.set_level(logging.DEBUG)
    result, _ = _run_chat(_config(db_path), _engine("ok"), "next\n/quit\n", "--resume")
    assert result.exit_code == 0, result.output
    assert TRANSCRIPT_MARKER not in caplog.text
    # The terminal shows only the notice, not old transcript text.
    assert TRANSCRIPT_MARKER not in result.output


# ---------------------------------------------------------------------------
# Cleanup on Ctrl-C / EOF
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_text", ["again\n", ""], ids=["eof-after-turn", "eof"])
def test_store_closed_on_eof_after_resume(db_path, input_text):
    _seed_first_chat(db_path, ("hello", "Hi!"))
    closed, spy = _close_spy()
    result, _ = _run_chat(
        _config(db_path), _engine("ok"), input_text, "--resume", extra_patches=[spy]
    )
    assert result.exit_code == 0, result.output
    assert closed == [db_path]


def test_store_closed_after_interrupted_resumed_generation(db_path):
    conversation_id = _seed_first_chat(db_path, ("hello", "Hi!"))
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = KeyboardInterrupt
    closed, spy = _close_spy()
    result, _ = _run_chat(
        _config(db_path), engine, "again\n", "--resume", extra_patches=[spy]
    )
    assert result.exit_code == 0, result.output
    assert "Generation interrupted" in result.output
    assert closed == [db_path]
    ((conv, messages),) = _read_all(db_path)
    assert conv.conversation_id == conversation_id
    assert [m.role for m in messages] == ["user", "assistant", "user"]


# ---------------------------------------------------------------------------
# Flag validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags",
    [
        ["--resume", "--conversation", "abc"],
        ["--resume", "--voice"],
        ["--conversation", "abc", "--voice"],
    ],
    ids=["resume+conversation", "resume+voice", "conversation+voice"],
)
def test_incompatible_flags_are_usage_errors(db_path, flags):
    engine = _engine("never")
    result, _ = _run_chat(_config(db_path), engine, "hi\n", *flags)
    assert result.exit_code == 2
    assert "Usage:" in result.output
    engine.generate.assert_not_called()
    assert not db_path.exists()
