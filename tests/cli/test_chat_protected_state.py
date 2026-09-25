"""``jarvis chat``: persona-aware memory tools and protected-state denials."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.agents.orchestrator import OrchestratorAgent
from openjarvis.core.registry import AgentRegistry, ToolRegistry
from openjarvis.core.types import Role
from openjarvis.security.protected_state import clear_protected_config
from openjarvis.tools.file_write import FileWriteTool
from openjarvis.tools.memory_manage import MemoryManageTool
from openjarvis.tools.user_profile_manage import UserProfileManageTool
from tests.cli.test_chat_conversation_persistence import _config, _read_all, _run_chat

AGENT_KEY = "protected_state_orchestrator"
SECRET = "SECRET-CONTENT-MARKER-77ab"


@pytest.fixture(autouse=True)
def _register():
    AgentRegistry.register_value(AGENT_KEY, OrchestratorAgent)
    for name, cls in (
        ("file_write", FileWriteTool),
        ("memory_manage", MemoryManageTool),
        ("user_profile_manage", UserProfileManageTool),
    ):
        if not ToolRegistry.contains(name):
            ToolRegistry.register_value(name, cls)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("OPENJARVIS_HOME", str(home))
    monkeypatch.delenv("OPENJARVIS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    clear_protected_config()
    yield home
    clear_protected_config()


def _engine(*replies: dict):
    sent: list[list] = []
    queue = list(replies)

    def _generate(messages, **kwargs):
        sent.append(list(messages))
        return queue.pop(0)

    engine = MagicMock()
    engine.engine_id = "mock"
    engine.generate.side_effect = _generate
    return engine, sent


def _tool_call(call_id: str, name: str, **arguments) -> dict:
    return {"id": call_id, "name": name, "arguments": json.dumps(arguments)}


# AG
def test_persona_memory_tools_edit_the_files_the_prompt_reads(home: Path):
    persona = home / "personas" / "work"
    persona.mkdir(parents=True)
    (persona / "MEMORY.md").write_text("PERSONA-MEMORY-MARKER\n")
    (persona / "USER.md").write_text("PERSONA-USER-MARKER\n")
    (home / "MEMORY.md").write_text("HOME-MEMORY-MARKER\n")
    (home / "USER.md").write_text("HOME-USER-MARKER\n")

    engine, sent = _engine(
        {
            "content": "",
            "tool_calls": [
                _tool_call("call_m", "memory_manage", action="add", entry="new fact"),
                _tool_call(
                    "call_u", "user_profile_manage", action="add", entry="new pref"
                ),
            ],
        },
        {"content": "Saved."},
    )
    confirm = MagicMock(return_value=True)
    result, _ = _run_chat(
        _config(home / "conversations.db"),
        engine,
        "remember this\n/quit\n",
        "--agent",
        AGENT_KEY,
        "--tools",
        "memory_manage,user_profile_manage",
        "--persona",
        "work",
        extra_patches=(patch("openjarvis.cli.chat_cmd.confirm_tool_call", confirm),),
    )
    assert result.exit_code == 0, result.output

    # The prompt was built from the persona files...
    system = "\n".join(m.content for m in sent[0] if m.role == Role.SYSTEM)
    assert "PERSONA-MEMORY-MARKER" in system
    assert "PERSONA-USER-MARKER" in system
    assert "HOME-MEMORY-MARKER" not in system
    # ...and the confirmed dedicated tools edited exactly those files.
    assert confirm.call_count == 2
    assert "new fact" in (persona / "MEMORY.md").read_text()
    assert "new pref" in (persona / "USER.md").read_text()
    assert (home / "MEMORY.md").read_text() == "HOME-MEMORY-MARKER\n"
    assert (home / "USER.md").read_text() == "HOME-USER-MARKER\n"


def test_default_persona_memory_tools_edit_home_files(home: Path):
    (home / "MEMORY.md").write_text("HOME-MEMORY-MARKER\n")
    engine, sent = _engine(
        {
            "content": "",
            "tool_calls": [
                _tool_call("call_m", "memory_manage", action="add", entry="home fact")
            ],
        },
        {"content": "Saved."},
    )
    result, _ = _run_chat(
        _config(home / "conversations.db"),
        engine,
        "remember\n/quit\n",
        "--agent",
        AGENT_KEY,
        "--tools",
        "memory_manage",
        extra_patches=(
            patch(
                "openjarvis.cli.chat_cmd.confirm_tool_call",
                MagicMock(return_value=True),
            ),
        ),
    )
    assert result.exit_code == 0, result.output
    assert "HOME-MEMORY-MARKER" in sent[0][0].content
    assert "home fact" in (home / "MEMORY.md").read_text()


# AH, AJ
def test_chat_protected_write_denied_and_recorded_safely(home: Path):
    (home / "MEMORY.md").write_text("# Memory\n")
    target = home / "MEMORY.md"
    engine, sent = _engine(
        {
            "content": "",
            "tool_calls": [
                _tool_call("call_w", "file_write", path=str(target), content=SECRET)
            ],
        },
        {"content": "I cannot edit that file directly."},
    )
    confirm = MagicMock(return_value=True)
    result, _ = _run_chat(
        _config(home / "conversations.db"),
        engine,
        "overwrite memory\n/quit\n",
        "--agent",
        AGENT_KEY,
        "--tools",
        "file_write",
        extra_patches=(patch("openjarvis.cli.chat_cmd.confirm_tool_call", confirm),),
    )
    assert result.exit_code == 0, result.output
    assert target.read_text() == "# Memory\n"
    confirm.assert_not_called()

    # The model saw the category-level denial, never a path echo.
    tool_messages = [m for m in sent[-1] if m.role == Role.TOOL]
    assert len(tool_messages) == 1
    assert "protected OpenJarvis memory state" in tool_messages[0].content
    assert str(home) not in tool_messages[0].content

    ((conversation, messages),) = _read_all(home / "conversations.db")
    assistant = messages[-1]
    assert assistant.metadata["tool_calls"] == (
        "file_write|protected_target_denied|call_w"
    )
    dumped = json.dumps(
        [conversation.metadata] + [[m.content, m.metadata] for m in messages]
    )
    for marker in (SECRET, str(home), "MEMORY.md", 'protected_target"'):
        assert marker not in dumped
