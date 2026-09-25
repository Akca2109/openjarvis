"""Live ``jarvis chat`` working context stays bounded and structurally valid."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.agents._stubs import AgentContext, AgentResult, BaseAgent
from openjarvis.conversations import (
    CONTEXT_MAX_MESSAGES,
    CONTEXT_MAX_TOKENS,
    ConversationStore,
)
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Role
from openjarvis.engine._base import estimate_prompt_tokens
from tests.cli.test_chat_conversation_persistence import (
    SYSTEM_PROMPT_MARKER,
    _config,
    _read_all,
    _run_chat,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "home" / "conversations.db"


def _recording_engine(*outcomes):
    """Engine whose calls snapshot the messages sent; outcomes are replies
    (str) or exceptions to raise."""
    sent: list[list] = []
    queue = list(outcomes)

    def _generate(messages, **kwargs):
        sent.append(list(messages))
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return {"content": outcome}

    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = _generate
    return engine, sent


def _turns(messages):
    return [(m.role, m.content) for m in messages if m.role != Role.SYSTEM]


def _assert_alternating(messages):
    turns = [m for m in messages if m.role != Role.SYSTEM]
    assert turns, "expected at least the current user turn"
    assert turns[0].role == Role.USER
    for prev, cur in zip(turns, turns[1:]):
        assert prev.role != cur.role, _turns(messages)
    assert turns[-1].role == Role.USER


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_long_live_chat_is_bounded_by_message_cap(db_path):
    n = CONTEXT_MAX_MESSAGES  # twice as many messages as the cap
    engine, sent = _recording_engine(*(f"a{i}" for i in range(n)))
    text = "".join(f"u{i}\n" for i in range(n)) + "/quit\n"
    result, _ = _run_chat(
        _config(db_path), engine, text, "--system", SYSTEM_PROMPT_MARKER
    )
    assert result.exit_code == 0, result.output

    for messages in sent:
        # Prior completed turns fit the cap; plus the current user turn.
        assert len(_turns(messages)) <= CONTEXT_MAX_MESSAGES + 1
        _assert_alternating(messages)
    last = _turns(sent[-1])
    assert last[-1] == (Role.USER, f"u{n - 1}")
    assert last[0] == (Role.USER, f"u{n - 1 - CONTEXT_MAX_MESSAGES // 2}")


def test_long_live_chat_is_bounded_by_token_budget(db_path):
    big = "word " * 1200  # ~1500 estimated tokens per reply
    engine, sent = _recording_engine(*(f"a{i} {big}" for i in range(8)))
    text = "".join(f"u{i}\n" for i in range(8)) + "/quit\n"
    result, _ = _run_chat(_config(db_path), engine, text)
    assert result.exit_code == 0, result.output

    for messages in sent:
        prior = [m for m in messages if m.role != Role.SYSTEM][:-1]
        assert estimate_prompt_tokens(prior) <= CONTEXT_MAX_TOKENS
        _assert_alternating(messages)
    assert len(_turns(sent[-1])) < 2 * 8 - 1


def test_trimming_never_deletes_durable_history(db_path):
    n = CONTEXT_MAX_MESSAGES
    engine, _ = _recording_engine(*(f"a{i}" for i in range(n)))
    text = "".join(f"u{i}\n" for i in range(n)) + "/quit\n"
    result, _ = _run_chat(_config(db_path), engine, text)
    assert result.exit_code == 0, result.output

    ((_, messages),) = _read_all(db_path)
    expected = []
    for i in range(n):
        expected += [("user", f"u{i}"), ("assistant", f"a{i}")]
    assert [(m.role, m.content) for m in messages] == expected


def test_system_prompt_survives_trimming_direct_engine(db_path):
    n = CONTEXT_MAX_MESSAGES
    engine, sent = _recording_engine(*(f"a{i}" for i in range(n)))
    text = "".join(f"u{i}\n" for i in range(n)) + "/quit\n"
    result, _ = _run_chat(
        _config(db_path), engine, text, "--system", SYSTEM_PROMPT_MARKER
    )
    assert result.exit_code == 0, result.output
    for messages in sent:
        systems = [m for m in messages if m.role == Role.SYSTEM]
        assert [m.content for m in systems] == [SYSTEM_PROMPT_MARKER]
        assert messages[0].role == Role.SYSTEM


def test_system_prompt_survives_trimming_simple_agent(db_path):
    config = _config(db_path)
    config.agent.default_system_prompt = SYSTEM_PROMPT_MARKER
    n = CONTEXT_MAX_MESSAGES
    engine, sent = _recording_engine(*(f"a{i}" for i in range(n)))
    text = "".join(f"u{i}\n" for i in range(n)) + "/quit\n"
    result, _ = _run_chat(config, engine, text, "--agent", "simple")
    assert result.exit_code == 0, result.output
    assert len(sent) == n
    for messages in sent:
        assert messages[0].role == Role.SYSTEM
        assert SYSTEM_PROMPT_MARKER in messages[0].content
        assert sum(m.role == Role.SYSTEM for m in messages) == 1
        assert len(_turns(messages)) <= CONTEXT_MAX_MESSAGES + 1
        _assert_alternating(messages)


# ---------------------------------------------------------------------------
# Failure recovery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "failure", [RuntimeError("engine down"), KeyboardInterrupt()], ids=type
)
def test_failed_turn_does_not_leave_dangling_user(db_path, failure):
    engine, sent = _recording_engine("a1", failure, "a3")
    result, _ = _run_chat(
        _config(db_path),
        engine,
        "u1\nu2\nu3\n/quit\n",
        "--system",
        SYSTEM_PROMPT_MARKER,
    )
    assert result.exit_code == 0, result.output
    if isinstance(failure, RuntimeError):
        assert "Error: engine down" in result.output
    else:
        assert "Generation interrupted." in result.output
    assert [(m.role, m.content) for m in sent[-1]] == [
        (Role.SYSTEM, SYSTEM_PROMPT_MARKER),
        (Role.USER, "u1"),
        (Role.ASSISTANT, "a1"),
        (Role.USER, "u3"),
    ]
    # Durable recording semantics are unchanged: the failed user turn stays.
    ((_, messages),) = _read_all(db_path)
    assert [m.content for m in messages] == ["u1", "a1", "u2", "u3", "a3"]


def test_multiple_failures_do_not_accumulate_user_turns(db_path):
    engine, sent = _recording_engine(
        "a1", RuntimeError("x"), RuntimeError("y"), RuntimeError("z"), "a5"
    )
    result, _ = _run_chat(_config(db_path), engine, "u1\nu2\nu3\nu4\nu5\n/quit\n")
    assert result.exit_code == 0, result.output
    for messages in sent:
        _assert_alternating(messages)
    assert _turns(sent[-1]) == [
        (Role.USER, "u1"),
        (Role.ASSISTANT, "a1"),
        (Role.USER, "u5"),
    ]


def test_failure_as_first_turn_leaves_empty_valid_context(db_path):
    engine, sent = _recording_engine(RuntimeError("x"), "a2", "a3")
    result, _ = _run_chat(_config(db_path), engine, "u1\nu2\nu3\n/quit\n")
    assert result.exit_code == 0, result.output
    assert _turns(sent[1]) == [(Role.USER, "u2")]
    assert _turns(sent[2]) == [
        (Role.USER, "u2"),
        (Role.ASSISTANT, "a2"),
        (Role.USER, "u3"),
    ]


def test_agent_success_after_failure_has_valid_chronological_context(db_path):
    captured: list[tuple[str, list]] = []
    outcomes = ["a1", RuntimeError("agent boom"), RuntimeError("again"), "a4"]

    class _FlakyAgent(BaseAgent):
        agent_id = "working_context_flaky_agent"

        def run(self, input, context: AgentContext | None = None, **kwargs):
            assert context is not None
            captured.append((input, list(context.conversation.messages)))
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return AgentResult(content=outcome, turns=1)

    AgentRegistry.register_value("working_context_flaky_agent", _FlakyAgent)
    engine, _ = _recording_engine()
    result, publish = _run_chat(
        _config(db_path),
        engine,
        "u1\nu2\nu3\nu4\nu5\n/quit\n",
        "--agent",
        "working_context_flaky_agent",
    )
    # u5 has no scripted outcome: pop() from an empty list errors, which the
    # chat loop reports like any other failure.
    assert result.exit_code == 0, result.output
    assert "Error: agent boom" in result.output

    by_input = {user_input: _turns(msgs) for user_input, msgs in captured}
    assert by_input["u2"] == [(Role.USER, "u1"), (Role.ASSISTANT, "a1")]
    assert by_input["u3"] == [(Role.USER, "u1"), (Role.ASSISTANT, "a1")]
    assert by_input["u4"] == [(Role.USER, "u1"), (Role.ASSISTANT, "a1")]
    assert by_input["u5"] == [
        (Role.USER, "u1"),
        (Role.ASSISTANT, "a1"),
        (Role.USER, "u4"),
        (Role.ASSISTANT, "a4"),
    ]
    # Failures are never turned into successful assistant messages.
    assert [c.args[1:] for c in publish.call_args_list] == [
        ("u1", "a1"),
        ("u4", "a4"),
    ]
    ((_, messages),) = _read_all(db_path)
    assert [m.content for m in messages] == ["u1", "a1", "u2", "u3", "u4", "a4", "u5"]


def test_failures_then_success_then_resume_replays_only_answered_turns(db_path):
    engine, sent = _recording_engine(
        "a1", RuntimeError("x"), RuntimeError("y"), "a4", "a5"
    )
    result, _ = _run_chat(_config(db_path), engine, "u1\nu2\nu3\nu4\nu5\n/quit\n")
    assert result.exit_code == 0, result.output
    live = _turns(sent[-1])
    assert live == [
        (Role.USER, "u1"),
        (Role.ASSISTANT, "a1"),
        (Role.USER, "u4"),
        (Role.ASSISTANT, "a4"),
        (Role.USER, "u5"),
    ]

    engine, resumed = _recording_engine("a6")
    result, _ = _run_chat(_config(db_path), engine, "u6\n/quit\n", "--resume")
    assert result.exit_code == 0, result.output
    # Resume applies the same policy as the live chat did.
    assert _turns(resumed[0]) == [
        (Role.USER, "u1"),
        (Role.ASSISTANT, "a1"),
        (Role.USER, "u4"),
        (Role.ASSISTANT, "a4"),
        (Role.USER, "u5"),
        (Role.ASSISTANT, "a5"),
        (Role.USER, "u6"),
    ]
    _assert_alternating(resumed[0])
    # The failed turns remain in the durable transcript.
    ((_, messages),) = _read_all(db_path)
    assert [m.content for m in messages] == [
        "u1", "a1", "u2", "u3", "u4", "a4", "u5", "a5", "u6", "a6",
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# /history shows the durable transcript
# ---------------------------------------------------------------------------


def _history_lines(output: str) -> list[str]:
    """/history entries; prompts share a line because stdin is not echoed."""
    lines = []
    for ln in output.splitlines():
        while ln.startswith("You> "):
            ln = ln[len("You> ") :]
        if ln.startswith(("USER:", "ASSISTANT:", "SYSTEM:")):
            lines.append(ln)
    return lines


def test_history_shows_turns_trimmed_from_model_context(db_path):
    n = CONTEXT_MAX_MESSAGES
    engine, sent = _recording_engine(*(f"a{i}" for i in range(n)))
    text = "".join(f"u{i}\n" for i in range(n)) + "/history\n/quit\n"
    result, _ = _run_chat(
        _config(db_path), engine, text, "--system", SYSTEM_PROMPT_MARKER
    )
    assert result.exit_code == 0, result.output
    # u0 is out of the model context but still listed.
    assert (Role.USER, "u0") not in _turns(sent[-1])
    expected = []
    for i in range(n):
        expected += [f"USER: u{i}", f"ASSISTANT: a{i}"]
    assert _history_lines(result.output) == expected


def test_history_shows_failed_user_turns(db_path):
    engine, sent = _recording_engine("a1", RuntimeError("x"), "a3")
    result, _ = _run_chat(_config(db_path), engine, "u1\nu2\nu3\n/history\n/quit\n")
    assert result.exit_code == 0, result.output
    assert (Role.USER, "u2") not in _turns(sent[-1])
    assert _history_lines(result.output) == [
        "USER: u1",
        "ASSISTANT: a1",
        "USER: u2",
        "USER: u3",
        "ASSISTANT: a3",
    ]


def test_history_after_resume_shows_whole_durable_conversation(db_path):
    n = CONTEXT_MAX_MESSAGES  # more rows than resume loads
    first, _ = _recording_engine(*(f"a{i}" for i in range(n)))
    text = "".join(f"u{i}\n" for i in range(n)) + "/quit\n"
    assert _run_chat(_config(db_path), first, text)[0].exit_code == 0

    engine, sent = _recording_engine("b0")
    result, _ = _run_chat(_config(db_path), engine, "v0\n/history\n/quit\n", "--resume")
    assert result.exit_code == 0, result.output
    assert (Role.USER, "u0") not in _turns(sent[0])
    lines = _history_lines(result.output)
    assert lines[:2] == ["USER: u0", "ASSISTANT: a0"]
    assert lines[-2:] == ["USER: v0", "ASSISTANT: b0"]
    assert len(lines) == 2 * n + 2


def test_history_before_first_turn_and_after_clear(db_path):
    engine, _ = _recording_engine("a1")
    result, _ = _run_chat(
        _config(db_path), engine, "/history\nu1\n/clear\n/history\n/quit\n"
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("No history yet.") == 2
    assert _history_lines(result.output) == []


def test_history_falls_back_to_live_context_when_disabled(db_path):
    engine, _ = _recording_engine("a1", RuntimeError("x"))
    result, _ = _run_chat(
        _config(db_path, enabled=False),
        engine,
        "u1\nu2\n/history\n/quit\n",
        "--system",
        SYSTEM_PROMPT_MARKER,
    )
    assert result.exit_code == 0, result.output
    assert _history_lines(result.output) == [
        f"SYSTEM: {SYSTEM_PROMPT_MARKER}",
        "USER: u1",
        "ASSISTANT: a1",
    ]


def test_history_falls_back_to_live_context_when_read_fails(db_path):
    engine, _ = _recording_engine("a1")
    failing_read = patch.object(
        ConversationStore, "get_messages", side_effect=OSError("disk gone")
    )
    result, _ = _run_chat(
        _config(db_path),
        engine,
        "u1\n/history\n/quit\n",
        extra_patches=(failing_read,),
    )
    assert result.exit_code == 0, result.output
    assert "disk gone" not in result.output
    lines = _history_lines(result.output)
    assert lines[-2:] == ["USER: u1", "ASSISTANT: a1"]
    # Recording is unaffected by the failed read.
    ((_, messages),) = _read_all(db_path)
    assert [m.content for m in messages] == ["u1", "a1"]


def test_durable_history_does_not_interpret_markup_or_controls(db_path):
    engine, _ = _recording_engine(
        "[conceal]secret plan[/conceal] \x1b[2J\x1b]0;pwned\x07done"
    )
    result, _ = _run_chat(
        _config(db_path), engine, "[bold red]x[/bold red]\n/history\n/quit\n"
    )
    assert result.exit_code == 0, result.output
    lines = _history_lines(result.output)
    assert lines[0] == "USER: [bold red]x[/bold red]"
    assert "[conceal]secret plan[/conceal]" in lines[1]
    assert "\x1b" not in lines[1]
    assert "pwned" not in lines[1] and lines[1].endswith("done")


# ---------------------------------------------------------------------------
# Stage 2 compatibility
# ---------------------------------------------------------------------------


def test_resumed_chat_keeps_live_context_bounded(db_path):
    n = CONTEXT_MAX_MESSAGES // 2
    first, _ = _recording_engine(*(f"a{i}" for i in range(n)))
    text = "".join(f"u{i}\n" for i in range(n)) + "/quit\n"
    result, _ = _run_chat(_config(db_path), first, text)
    assert result.exit_code == 0, result.output

    engine, sent = _recording_engine(*(f"b{i}" for i in range(n)))
    text = "".join(f"v{i}\n" for i in range(n)) + "/quit\n"
    result, _ = _run_chat(_config(db_path), engine, text, "--resume")
    assert result.exit_code == 0, result.output

    # The first resumed turn sees the full Stage 2 window.
    assert len(_turns(sent[0])) == CONTEXT_MAX_MESSAGES + 1
    assert _turns(sent[0])[0] == (Role.USER, "u0")
    for messages in sent:
        assert len(_turns(messages)) <= CONTEXT_MAX_MESSAGES + 1
        _assert_alternating(messages)
    # The oldest resumed exchange has aged out of the live window.
    assert _turns(sent[-1])[0] == (Role.USER, f"u{n - 1}")

    ((_, messages),) = _read_all(db_path)
    assert len(messages) == 4 * n
