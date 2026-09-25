"""``jarvis chat`` records safe tool provenance on the final assistant row."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openjarvis.agents.orchestrator import OrchestratorAgent
from openjarvis.core.registry import AgentRegistry, ToolRegistry
from openjarvis.core.types import Role, ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec
from tests.cli.test_chat_conversation_persistence import _config, _read_all, _run_chat

SECRET_ARG = "SECRET-ARG-MARKER-3a8e"
RAW_OUTPUT = "RAW-TOOL-OUTPUT-MARKER-c51d"
AGENT_KEY = "prov_orchestrator"


class _ProvEchoTool(BaseTool):
    tool_id = "prov_echo"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="prov_echo", description="Echo for provenance tests.")

    def execute(self, **params) -> ToolResult:
        return ToolResult(tool_name="prov_echo", content=f"{RAW_OUTPUT}:{params}")


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "home" / "conversations.db"


@pytest.fixture(autouse=True)
def _register():
    AgentRegistry.register_value(AGENT_KEY, OrchestratorAgent)
    ToolRegistry.register_value("prov_echo", _ProvEchoTool)


def _recording_engine(*replies: dict):
    sent: list[list] = []
    queue = list(replies)

    def _generate(messages, **kwargs):
        sent.append(list(messages))
        return queue.pop(0)

    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = _generate
    return engine, sent


def _tool_turn_engine():
    return _recording_engine(
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "name": "prov_echo",
                    "arguments": json.dumps({"text": SECRET_ARG}),
                },
                {
                    "id": "call_0",  # duplicate provider ID
                    "name": "ghost_tool",
                    "arguments": json.dumps({"cmd": SECRET_ARG}),
                },
            ],
        },
        {"content": "Final answer text."},
    )


def _run_tool_turn(db_path):
    engine, sent = _tool_turn_engine()
    result, _ = _run_chat(
        _config(db_path),
        engine,
        "use the tool\n/quit\n",
        "--agent",
        AGENT_KEY,
        "--tools",
        "prov_echo",
    )
    assert result.exit_code == 0, result.output
    return sent


def test_final_assistant_row_records_safe_tool_provenance(db_path):
    _run_tool_turn(db_path)

    ((conversation, messages),) = _read_all(db_path)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "use the tool"),
        ("assistant", "Final answer text."),
    ]
    user, assistant = messages
    assert user.metadata == {}
    metadata = assistant.metadata
    assert metadata["tool_provenance"] == 1
    assert metadata["tools_used"] is True
    assert metadata["tool_call_count"] == 2
    assert metadata["tool_calls_truncated"] is False
    first, second = (entry.split("|") for entry in metadata["tool_calls"].split(";"))
    assert first == ["prov_echo", "success", "call_0"]
    assert second[:2] == ["ghost_tool", "unknown_tool"]
    assert second[2] not in ("call_0", "-", "")  # duplicate ID was replaced
    assert conversation.metadata == {"engine": "mock"}


def test_durable_rows_contain_no_raw_arguments_or_results(db_path):
    _run_tool_turn(db_path)

    ((conversation, messages),) = _read_all(db_path)
    dumped = json.dumps(
        [conversation.metadata] + [[m.role, m.content, m.metadata] for m in messages]
    )
    assert SECRET_ARG not in dumped
    assert RAW_OUTPUT not in dumped
    assert all(m.role in ("user", "assistant") for m in messages)
    for m in messages:
        assert all(isinstance(v, (str, int, float, bool)) for v in m.metadata.values())


def test_resume_after_tool_turn_replays_only_user_and_assistant_text(db_path):
    sent = _run_tool_turn(db_path)
    # The live tool loop really did carry tool messages to the model.
    assert any(m.role == Role.TOOL for m in sent[-1])

    engine, resumed = _recording_engine({"content": "Second answer."})
    result, _ = _run_chat(
        _config(db_path),
        engine,
        "next question\n/quit\n",
        "--agent",
        AGENT_KEY,
        "--tools",
        "prov_echo",
        "--resume",
    )
    assert result.exit_code == 0, result.output

    context = resumed[0]
    turns = [(m.role, m.content) for m in context if m.role != Role.SYSTEM]
    assert turns == [
        (Role.USER, "use the tool"),
        (Role.ASSISTANT, "Final answer text."),
        (Role.USER, "next question"),
    ]
    assert all(not m.tool_calls and not m.tool_call_id for m in context)
    assert all(not m.metadata for m in context if m.role != Role.SYSTEM)
    rendered = json.dumps([m.text for m in context])
    for marker in (SECRET_ARG, RAW_OUTPUT, "tool_provenance", "prov_echo|success"):
        assert marker not in rendered

    ((_, messages),) = _read_all(db_path)
    # A turn without tool calls records no provenance.
    assert messages[-1].content == "Second answer."
    assert messages[-1].metadata == {}
