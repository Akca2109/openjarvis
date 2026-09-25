"""``jarvis ask``: dedicated memory tools edit the files the prompt reads."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from openjarvis.agents._stubs import AgentContext, AgentResult, ToolUsingAgent
from openjarvis.cli import cli
from openjarvis.core.config import JarvisConfig
from openjarvis.core.registry import AgentRegistry, ToolRegistry
from openjarvis.core.types import ToolCall
from openjarvis.security.protected_state import clear_protected_config
from openjarvis.security.taint import TaintLabel, TaintSet, check_taint
from openjarvis.tools.memory_manage import MemoryManageTool
from openjarvis.tools.outcomes import OUTCOME_KEY
from openjarvis.tools.user_profile_manage import UserProfileManageTool

_ask_mod = importlib.import_module("openjarvis.cli.ask")
TOOLS = "memory_manage,user_profile_manage"


class _PersonaProbeAgent(ToolUsingAgent):
    """Records its prompt builder and tools, then adds one entry via each."""

    agent_id = "persona_probe"
    seen: dict = {}

    def __init__(self, engine, model, *, prompt_builder=None, **kwargs):
        super().__init__(engine, model, **kwargs)
        type(self).seen = {
            "builder": prompt_builder,
            "tools": {t.spec.name: t for t in kwargs.get("tools") or []},
        }

    def run(self, input, context: AgentContext | None = None, **kwargs):
        results = [
            self._executor.execute(
                ToolCall(
                    id="call_m",
                    name="memory_manage",
                    arguments='{"action": "add", "entry": "ask fact"}',
                )
            ),
            self._executor.execute(
                ToolCall(
                    id="call_u",
                    name="user_profile_manage",
                    arguments='{"action": "add", "entry": "ask pref"}',
                )
            ),
        ]
        type(self).seen["results"] = results
        type(self).seen["prompt"] = self._prompt_builder_text()
        return AgentResult(content="done", tool_results=results, turns=1)

    def _prompt_builder_text(self) -> str:
        builder = type(self).seen["builder"]
        return builder.build() if builder is not None else ""


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("OPENJARVIS_HOME", str(home))
    monkeypatch.delenv("OPENJARVIS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    clear_protected_config()
    AgentRegistry.register_value("persona_probe", _PersonaProbeAgent)
    for name, cls in (
        ("memory_manage", MemoryManageTool),
        ("user_profile_manage", UserProfileManageTool),
    ):
        if not ToolRegistry.contains(name):
            ToolRegistry.register_value(name, cls)
    _PersonaProbeAgent.seen = {}
    yield home
    clear_protected_config()


def _ask(*args: str, user_input: str = "y\ny\n"):
    # Built after the fixture set OPENJARVIS_HOME so default paths use it.
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    config.traces.enabled = False
    config.agent.context_from_memory = False
    engine = MagicMock()
    engine.engine_id = "mock"
    with (
        patch.object(_ask_mod, "load_config", return_value=config),
        patch.object(_ask_mod, "get_engine", return_value=("mock", engine)),
        patch.object(_ask_mod, "discover_engines", return_value=[("mock", engine)]),
        patch.object(_ask_mod, "discover_models", return_value={"mock": ["m"]}),
        patch.object(_ask_mod, "register_builtin_models"),
        patch.object(_ask_mod, "merge_discovered_models"),
    ):
        result = CliRunner().invoke(
            cli,
            ["ask", "--agent", "persona_probe", "--tools", TOOLS, *args, "Hi"],
            input=user_input,
        )
    assert result.exit_code == 0, result.output
    return _PersonaProbeAgent.seen


def _tool_paths(seen) -> tuple[Path, Path]:
    tools = seen["tools"]
    return tools["memory_manage"]._memory_path, tools["user_profile_manage"]._user_path


def _prompt_paths(seen) -> tuple[Path, Path]:
    mf = seen["builder"]._mf_config
    return Path(mf.memory_path), Path(mf.user_path)


def _seed(directory: Path, tag: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "MEMORY.md").write_text(f"{tag}-MEMORY\n")
    (directory / "USER.md").write_text(f"{tag}-USER\n")


# A
def test_default_config_prompt_and_tools_share_home_files(home: Path):
    _seed(home, "HOME")
    seen = _ask()

    assert _tool_paths(seen) == _prompt_paths(seen)
    assert _tool_paths(seen) == (home / "MEMORY.md", home / "USER.md")
    assert "HOME-MEMORY" in seen["prompt"] and "HOME-USER" in seen["prompt"]
    assert all(r.success for r in seen["results"])
    assert "ask fact" in (home / "MEMORY.md").read_text()
    assert "ask pref" in (home / "USER.md").read_text()


# B
def test_persona_prompt_and_tools_share_persona_files(home: Path):
    persona = home / "personas" / "work"
    _seed(home, "HOME")
    _seed(persona, "PERSONA")
    seen = _ask("--persona", "work")

    assert _tool_paths(seen) == _prompt_paths(seen)
    assert _tool_paths(seen) == (persona / "MEMORY.md", persona / "USER.md")
    assert "PERSONA-MEMORY" in seen["prompt"]
    assert "HOME-MEMORY" not in seen["prompt"]
    assert "ask fact" in (persona / "MEMORY.md").read_text()
    assert "ask pref" in (persona / "USER.md").read_text()
    assert (home / "MEMORY.md").read_text() == "HOME-MEMORY\n"
    assert (home / "USER.md").read_text() == "HOME-USER\n"


# C
def test_persona_none_tools_fail_safely(home: Path):
    seen = _ask("--persona", "none")

    assert "MEMORY" not in seen["prompt"]
    for result in seen["results"]:
        assert result.success is False
        assert "disabled" in result.content
    # Nothing was written to a default (or any) memory file.
    assert not (home / "MEMORY.md").exists()
    assert not (home / "USER.md").exists()
    assert not Path("MEMORY.md").exists() and not Path("USER.md").exists()


# D
@pytest.mark.parametrize("user_input", ["n\nn\n", "\n\n", ""])
def test_dedicated_tools_still_require_confirmation(home: Path, user_input: str):
    _seed(home, "HOME")
    seen = _ask(user_input=user_input)

    for result in seen["results"]:
        assert result.success is False
        assert result.metadata[OUTCOME_KEY] == "denied"
    assert (home / "MEMORY.md").read_text() == "HOME-MEMORY\n"
    assert (home / "USER.md").read_text() == "HOME-USER\n"
    secret = TaintSet(frozenset({TaintLabel.SECRET}))
    for name, tool in seen["tools"].items():
        assert tool.spec.requires_confirmation is True
        assert check_taint(name, secret) is not None
