"""End-to-end tests: ``jarvis chat`` persists turns to ``conversations.db``."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.cli.chat_cmd import chat
from openjarvis.conversations import OWNER_USER_ID, ConversationStore
from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import AgentRegistry

SYSTEM_PROMPT_MARKER = "SYSTEM-PROMPT-MARKER-5d1e"
MEMORY_FACT_MARKER = "MEMORY-FACT-MARKER-b77c"


class _PersistAgent(BaseAgent):
    agent_id = "persist_chat_agent"

    def run(self, input, context: AgentContext | None = None, **kwargs):
        return AgentResult(content=f"agent:{input}", turns=1)


def _engine(*replies: str) -> MagicMock:
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = [{"content": r} for r in replies]
    return engine


def _config(db_path: Path, *, enabled: bool = True) -> JarvisConfig:
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    config.agent.default_agent = "none"
    config.agent.context_from_memory = False
    config.conversations.enabled = enabled
    config.conversations.db_path = str(db_path)
    return config


def _run_chat(config, engine, input_text: str, *args: str, extra_patches=()):
    with ExitStack() as stack:
        stack.enter_context(
            patch("openjarvis.cli.chat_cmd.load_config", return_value=config)
        )
        stack.enter_context(
            patch("openjarvis.engine.get_engine", return_value=("mock", engine))
        )
        stack.enter_context(patch("openjarvis.intelligence.register_builtin_models"))
        stack.enter_context(
            patch("openjarvis.memory.build_memory_service", return_value=None)
        )
        stack.enter_context(
            patch(
                "openjarvis.cli._model_switch.tty_wants_model_picker",
                return_value=False,
            )
        )
        stack.enter_context(
            patch(
                "openjarvis.cli._runtime_panel.tty_wants_runtime_panel",
                return_value=False,
            )
        )
        publish = stack.enter_context(
            patch("openjarvis.cli.chat_cmd.publish_completed_exchange")
        )
        for p in extra_patches:
            stack.enter_context(p)
        result = CliRunner().invoke(
            chat, ["--model", "test-model", *args], input=input_text
        )
    return result, publish


def _read_all(db_path: Path):
    with ConversationStore(db_path) as store:
        conversations = sorted(store.list_conversations(), key=lambda c: c.created_at)
        return [(c, store.get_messages(c.conversation_id)) for c in conversations]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "home" / "conversations.db"


def test_two_turn_chat_is_persisted(db_path):
    engine = _engine("Hi!", "Fine.")
    result, publish = _run_chat(
        _config(db_path), engine, "hello\nhow are you?\n/quit\n"
    )
    assert result.exit_code == 0, result.output

    ((conv, messages),) = _read_all(db_path)
    assert conv.user_id == OWNER_USER_ID
    assert conv.origin == "cli"
    assert conv.metadata == {"engine": "mock"}
    assert [(m.role, m.content) for m in messages] == [
        ("user", "hello"),
        ("assistant", "Hi!"),
        ("user", "how are you?"),
        ("assistant", "Fine."),
    ]
    timestamps = [m.created_at for m in messages]
    assert timestamps == sorted(timestamps)
    assert all(m.surface == "cli" for m in messages)
    assert [m.model for m in messages] == [None, "test-model", None, "test-model"]
    assert all(m.agent_id is None for m in messages)  # direct engine chat
    assert all(m.session_id is None and m.trace_id is None for m in messages)
    assert all(m.metadata == {} for m in messages)

    # Existing memory publishing is unchanged: once per completed turn.
    assert publish.call_count == 2
    assert publish.call_args_list[0].args[1:] == ("hello", "Hi!")
    assert publish.call_args_list[0].kwargs == {"source": "cli.chat"}


def test_agent_id_recorded_when_agent_runs(db_path):
    AgentRegistry.register_value("persist_chat_agent", _PersistAgent)
    config = _config(db_path)
    result, _ = _run_chat(
        config, _engine(), "ping\n/quit\n", "--agent", "persist_chat_agent"
    )
    assert result.exit_code == 0, result.output
    ((_, messages),) = _read_all(db_path)
    assert [(m.role, m.content, m.agent_id) for m in messages] == [
        ("user", "ping", None),
        ("assistant", "agent:ping", "persist_chat_agent"),
    ]


def test_clear_starts_new_conversation_and_slash_commands_not_stored(db_path):
    engine = _engine("A1", "A2")
    result, _ = _run_chat(
        _config(db_path),
        engine,
        "first\n/model\n/history\n/help\n/clear\nsecond\n/quit\n",
    )
    assert result.exit_code == 0, result.output
    records = _read_all(db_path)
    assert [[(m.role, m.content) for m in msgs] for _, msgs in records] == [
        [("user", "first"), ("assistant", "A1")],
        [("user", "second"), ("assistant", "A2")],
    ]


def test_generation_failure_keeps_user_turn_only(db_path):
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = [RuntimeError("engine down"), {"content": "ok"}]
    result, publish = _run_chat(_config(db_path), engine, "one\ntwo\n/quit\n")
    assert result.exit_code == 0, result.output
    ((_, messages),) = _read_all(db_path)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "one"),
        ("user", "two"),
        ("assistant", "ok"),
    ]
    assert publish.call_count == 1


def test_disabled_config_creates_no_database(db_path):
    result, _ = _run_chat(_config(db_path, enabled=False), _engine("x"), "hi\n/quit\n")
    assert result.exit_code == 0, result.output
    assert not db_path.exists()
    assert not db_path.parent.exists()


def test_immediate_quit_creates_no_conversation(db_path):
    result, _ = _run_chat(_config(db_path), _engine(), "/quit\n")
    assert result.exit_code == 0, result.output
    assert _read_all(db_path) == []


def test_store_open_failure_does_not_break_chat(db_path):
    engine = _engine("still here")
    result, publish = _run_chat(
        _config(db_path),
        engine,
        "hello\n/quit\n",
        extra_patches=[
            patch(
                "openjarvis.conversations.recorder.ConversationStore",
                side_effect=PermissionError("denied"),
            )
        ],
    )
    assert result.exit_code == 0, result.output
    assert "still here" in result.output
    assert "Conversation history disabled (PermissionError)" in result.output
    assert publish.call_count == 1


def test_write_failure_mid_chat_warns_once_and_chat_continues(db_path):
    engine = _engine("r1", "r2")
    result, publish = _run_chat(
        _config(db_path),
        engine,
        "a\nb\n/quit\n",
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
    assert "Traceback" not in result.output
    assert publish.call_count == 2


def test_voice_mode_is_not_persisted(db_path):
    config = _config(db_path)
    engine = _engine("spoken reply")
    result, _ = _run_chat(
        config,
        engine,
        "typed in voice mode\n/quit\n",
        "--voice",
        extra_patches=[patch("openjarvis.cli.chat_cmd.speak")],
    )
    assert result.exit_code == 0, result.output
    engine.generate.assert_called_once()  # a real voice-mode turn happened
    assert not db_path.exists()


def test_system_prompt_and_memory_context_are_not_stored(db_path, tmp_path):
    from openjarvis.memory.store import LocalFactStore

    facts_path = tmp_path / "facts.jsonl"
    LocalFactStore(facts_path).add(MEMORY_FACT_MARKER, source="auto", trust="auto")
    config = _config(db_path)
    config.memory.enabled = True
    config.memory.facts_path = str(facts_path)
    config.agent.context_from_memory = True
    engine = _engine("answer")

    result, _ = _run_chat(
        config,
        engine,
        f"tell me about {MEMORY_FACT_MARKER.split('-')[0]}\n/quit\n",
        "--system",
        SYSTEM_PROMPT_MARKER,
        extra_patches=[
            patch("openjarvis.cli.ask._get_memory_backend", return_value=None)
        ],
    )
    assert result.exit_code == 0, result.output
    # The engine did receive the system prompt (sanity check of the setup) ...
    sent = engine.generate.call_args.args[0]
    assert any(SYSTEM_PROMPT_MARKER in m.content for m in sent)
    assert any(MEMORY_FACT_MARKER in m.content for m in sent)
    # ... but the durable record holds only the real turns.
    ((conv, messages),) = _read_all(db_path)
    assert [m.role for m in messages] == ["user", "assistant"]
    raw = db_path.read_bytes()
    for suffix in ("-wal",):
        sidecar = db_path.with_name(db_path.name + suffix)
        if sidecar.exists():
            raw += sidecar.read_bytes()
    assert SYSTEM_PROMPT_MARKER.encode() not in raw
    assert MEMORY_FACT_MARKER.encode() not in raw


# ---------------------------------------------------------------------------
# Recorder/store cleanup is guaranteed however the REPL exits
# ---------------------------------------------------------------------------


def _close_spy():
    real_close = ConversationStore.close
    closed: list[Path] = []

    def spy(self):
        closed.append(self.db_path)
        return real_close(self)

    return closed, patch.object(ConversationStore, "close", spy)


@pytest.mark.parametrize(
    "input_text",
    ["hello\n/quit\n", "hello\n"],  # explicit /quit, and EOF on stdin
    ids=["quit", "eof"],
)
def test_store_closed_on_normal_exit(db_path, input_text):
    closed, spy = _close_spy()
    result, _ = _run_chat(
        _config(db_path), _engine("hi"), input_text, extra_patches=[spy]
    )
    assert result.exit_code == 0, result.output
    assert closed == [db_path]


def test_store_closed_after_interrupted_generation(db_path):
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = KeyboardInterrupt
    closed, spy = _close_spy()
    result, _ = _run_chat(_config(db_path), engine, "hello\n", extra_patches=[spy])
    assert result.exit_code == 0, result.output
    assert "Generation interrupted" in result.output
    assert closed == [db_path]
    ((_, messages),) = _read_all(db_path)
    assert [m.role for m in messages] == ["user"]


@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyboardInterrupt()])
def test_store_closed_when_exception_escapes_repl(db_path, error):
    # The notification check runs at the top of every loop iteration, outside
    # the per-turn error handling, so failing it on the second iteration makes
    # the error propagate out of the REPL after one completed turn.
    closed, spy = _close_spy()
    diff = patch(
        "openjarvis.cli._chat_notifications.NotificationDispatcher.diff",
        side_effect=[[], error],
    )
    result, _ = _run_chat(
        _config(db_path),
        _engine("hi"),
        "hello\nnever read\n/quit\n",
        extra_patches=[spy, diff],
    )
    assert result.exit_code != 0
    assert closed == [db_path]
    ((_, messages),) = _read_all(db_path)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "hello"),
        ("assistant", "hi"),
    ]
