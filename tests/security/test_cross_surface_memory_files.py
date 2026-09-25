"""Stage 3E: every surface binds the memory-file tools to its prompt's files.

``memory_manage`` and ``user_profile_manage`` edit MEMORY.md / USER.md, which
the system prompt injects. Each production surface that exposes them must
construct them against the same effective ``MemoryFilesConfig`` its prompt is
built from, across default paths, custom paths, a config persona, and the
persona ``none`` opt-out — which must never fall back to the owner's
default-home files.
"""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.agents._stubs import AgentResult, ToolUsingAgent
from openjarvis.core.config import JarvisConfig, MemoryFilesConfig, load_config
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.prompt.builder import SystemPromptBuilder

_TOOL_NAMES = ["memory_manage", "user_profile_manage"]
_OWNER_MEMORY = "OWNER_MEMORY_SENTINEL"
_OWNER_USER = "OWNER_USER_SENTINEL"


@dataclass
class Case:
    name: str
    mf: MemoryFilesConfig
    home: Path
    memory: Path | None  # None: memory files disabled
    user: Path | None
    memory_sentinel: str = ""
    user_sentinel: str = ""

    @property
    def disabled(self) -> bool:
        return self.memory is None

    def config(self) -> JarvisConfig:
        cfg = JarvisConfig()
        cfg.memory_files = self.mf
        cfg.agent.context_from_memory = False
        cfg.skills.enabled = False
        return cfg


@pytest.fixture(params=["default", "custom", "persona", "none"])
def case(request, tmp_path, monkeypatch) -> Case:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("OPENJARVIS_HOME", str(home))
    load_config.cache_clear()
    from openjarvis.agents.tool_resolver import ensure_registries_populated
    from openjarvis.core.registry import ToolRegistry
    from openjarvis.tools.memory_manage import MemoryManageTool
    from openjarvis.tools.user_profile_manage import UserProfileManageTool

    ensure_registries_populated()  # conftest clears registries per test
    if not ToolRegistry.contains("memory_manage"):
        ToolRegistry.register_value("memory_manage", MemoryManageTool)
    if not ToolRegistry.contains("user_profile_manage"):
        ToolRegistry.register_value("user_profile_manage", UserProfileManageTool)
    (home / "MEMORY.md").write_text(_OWNER_MEMORY)
    (home / "USER.md").write_text(_OWNER_USER)

    if request.param == "default":
        return Case(
            "default",
            MemoryFilesConfig(),
            home,
            home / "MEMORY.md",
            home / "USER.md",
            _OWNER_MEMORY,
            _OWNER_USER,
        )
    if request.param == "custom":
        custom = tmp_path / "custom"
        custom.mkdir()
        (custom / "MEM.md").write_text("CUSTOM_MEMORY_SENTINEL")
        (custom / "PROFILE.md").write_text("CUSTOM_USER_SENTINEL")
        mf = MemoryFilesConfig(
            soul_path=str(custom / "SOUL.md"),
            memory_path=str(custom / "MEM.md"),
            user_path=str(custom / "PROFILE.md"),
        )
        return Case(
            "custom",
            mf,
            home,
            custom / "MEM.md",
            custom / "PROFILE.md",
            "CUSTOM_MEMORY_SENTINEL",
            "CUSTOM_USER_SENTINEL",
        )
    if request.param == "persona":
        persona = home / "personas" / "stage3e"
        persona.mkdir(parents=True)
        (persona / "MEMORY.md").write_text("PERSONA_MEMORY_SENTINEL")
        (persona / "USER.md").write_text("PERSONA_USER_SENTINEL")
        return Case(
            "persona",
            MemoryFilesConfig(persona_name="stage3e"),
            home,
            persona / "MEMORY.md",
            persona / "USER.md",
            "PERSONA_MEMORY_SENTINEL",
            "PERSONA_USER_SENTINEL",
        )
    return Case("none", MemoryFilesConfig(persona_name="none"), home, None, None)


def _by_name(tools) -> dict:
    return {tool.spec.name: tool for tool in tools}


def assert_tools_bound(tools, case: Case) -> None:
    """The memory-file tools target exactly the case's effective files."""

    by_name = _by_name(tools)
    memory = by_name["memory_manage"]
    profile = by_name["user_profile_manage"]
    # Unwrap resolver spec-override adapters, if any.
    memory = getattr(memory, "_wrapped", memory)
    profile = getattr(profile, "_wrapped", profile)

    if case.disabled:
        assert memory._disabled and profile._disabled
        added = memory.execute(action="add", entry="INJECTED")
        added_profile = profile.execute(action="add", entry="INJECTED")
        assert not added.success and not added_profile.success
        # No silent fallback to the owner's default-home files.
        assert (case.home / "MEMORY.md").read_text() == _OWNER_MEMORY
        assert (case.home / "USER.md").read_text() == _OWNER_USER
        return

    assert not memory._disabled and not profile._disabled
    assert memory._memory_path == case.memory
    assert profile._user_path == case.user
    assert case.memory_sentinel in memory.execute(action="read").content
    assert case.user_sentinel in profile.execute(action="read").content


def assert_prompt_matches(prompt: str, case: Case) -> None:
    """The prompt injects the same files the tools edit (and nothing else)."""

    if case.disabled:
        assert _OWNER_MEMORY not in prompt
        assert _OWNER_USER not in prompt
        return
    assert case.memory_sentinel in prompt
    assert case.user_sentinel in prompt
    if case.name != "default":
        assert _OWNER_MEMORY not in prompt
        assert _OWNER_USER not in prompt


class _CaptureAgent(ToolUsingAgent):
    """Real tool-using agent that records what a surface constructed."""

    agent_id = "stage3e_capture"
    instances: list["_CaptureAgent"] = []

    def run(self, input, context=None, **kwargs):
        type(self).instances.append(self)
        return AgentResult(content="captured")


@pytest.fixture
def capture_agent():
    _CaptureAgent.instances = []
    AgentRegistry.register_value("stage3e_capture", _CaptureAgent)
    yield _CaptureAgent
    _CaptureAgent.instances = []


def _engine() -> MagicMock:
    engine = MagicMock()
    engine.engine_id = "mock"
    engine.health.return_value = True
    engine.list_models.return_value = ["test-model"]
    return engine


# ---------------------------------------------------------------------------
# Shared construction point
# ---------------------------------------------------------------------------


def test_prompt_builder_reads_case_files(case):
    """Sanity: the reference prompt for each case (what surfaces must match)."""

    prompt = SystemPromptBuilder("", memory_files_config=case.mf).build()
    assert_prompt_matches(prompt, case)


def test_instantiate_registered_tool(case):
    from openjarvis.agents.tool_resolver import (
        ensure_registries_populated,
        instantiate_registered_tool,
    )
    from openjarvis.core.registry import ToolRegistry

    ensure_registries_populated()
    tools = [
        instantiate_registered_tool(
            ToolRegistry.get(name),
            name,
            engine=None,
            model="",
            memory_files_config=case.mf,
        )
        for name in _TOOL_NAMES
    ]
    assert_tools_bound(tools, case)


def test_managed_agent_resolver(case):
    from openjarvis.agents.tool_resolver import resolve_agent_tools

    custom_spec = {
        "type": "function",
        "function": {
            "name": "user_profile_manage",
            "description": "custom schema",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    with resolve_agent_tools(
        {"config": {"tools": ["memory_manage", custom_spec]}},
        engine=None,
        model="",
        memory_files_config=case.mf,
    ) as resolved:
        assert_tools_bound(resolved.instances, case)


# ---------------------------------------------------------------------------
# CLI ask / SDK / skill run (shared ``_build_tools``)
# ---------------------------------------------------------------------------


def test_cli_build_tools_defaults_to_config_memory_files(case):
    ask_module = importlib.import_module("openjarvis.cli.ask")

    assert_tools_bound(
        ask_module._build_tools(_TOOL_NAMES, case.config(), None, ""), case
    )


def test_cli_ask_run_agent(case, capture_agent):
    ask_module = importlib.import_module("openjarvis.cli.ask")

    cfg = case.config()
    ask_module._run_agent(
        "stage3e_capture",
        "hello",
        _engine(),
        "test-model",
        list(_TOOL_NAMES),
        cfg,
        EventBus(),
        0.0,
        16,
    )
    agent = capture_agent.instances[-1]
    assert_tools_bound(agent._tools, case)
    assert_prompt_matches(agent._prompt_builder.build(), case)


def test_cli_ask_persona_override_wins(case, capture_agent):
    """``--persona`` selects the effective files for prompt and tools alike."""

    ask_module = importlib.import_module("openjarvis.cli.ask")

    ask_module._run_agent(
        "stage3e_capture",
        "hello",
        _engine(),
        "test-model",
        list(_TOOL_NAMES),
        JarvisConfig(),
        EventBus(),
        0.0,
        16,
        memory_files_config=case.mf,
    )
    agent = capture_agent.instances[-1]
    assert_tools_bound(agent._tools, case)
    assert_prompt_matches(agent._prompt_builder.build(), case)


def test_sdk_agent(case, capture_agent):
    from openjarvis.sdk import Jarvis

    with patch("openjarvis.sdk.get_engine", return_value=("mock", _engine())):
        j = Jarvis(config=case.config(), model="test-model")
        try:
            j.ask("hello", agent="stage3e_capture", tools=list(_TOOL_NAMES))
        finally:
            j.close()
    agent = capture_agent.instances[-1]
    assert_tools_bound(agent._tools, case)
    assert_prompt_matches(agent._prompt_builder.build(), case)


def test_skill_run(case, tmp_path, monkeypatch):
    from click.testing import CliRunner

    from openjarvis.cli import cli

    skills = tmp_path / "skills"
    (skills / "mem").mkdir(parents=True)
    (skills / "mem" / "skill.toml").write_text(
        '[skill]\nname = "mem"\nversion = "1.0.0"\n\n'
        '[[skill.steps]]\ntool_name = "memory_manage"\n'
        'arguments_template = \'{"action": "read"}\'\n\n'
        '[[skill.steps]]\ntool_name = "user_profile_manage"\n'
        'arguments_template = \'{"action": "read"}\'\n'
    )
    ask_module = importlib.import_module("openjarvis.cli.ask")
    cfg = case.config()
    cfg.security.enabled = False
    cfg.security.capabilities.enabled = False
    cfg.learning.skills.overlay_dir = str(tmp_path / "overlays")

    built: list = []
    real_build_tools = ask_module._build_tools

    def _spy(*args, **kwargs):
        tools = real_build_tools(*args, **kwargs)
        built.extend(tools)
        return tools

    monkeypatch.setattr(ask_module, "_build_tools", _spy)
    with (
        patch("openjarvis.cli.skill_cmd._get_skill_paths", return_value=[skills]),
        patch("openjarvis.cli.skill_cmd.load_config", return_value=cfg),
        patch("openjarvis.core.config.load_config", return_value=cfg),
    ):
        CliRunner().invoke(cli, ["skill", "run", "mem"], input="y\ny\n")

    assert_tools_bound(built, case)


# ---------------------------------------------------------------------------
# jarvis serve (main agent + channel agent)
# ---------------------------------------------------------------------------


def test_serve_main_agent(case, capture_agent, monkeypatch):
    pytest.importorskip("fastapi")
    from click.testing import CliRunner

    from openjarvis.cli import cli

    serve_mod = importlib.import_module("openjarvis.cli.serve")
    config = case.config()
    config.tools.enabled = list(_TOOL_NAMES)
    config.intelligence.default_model = "test-model"
    config.telemetry.enabled = False
    config.agent_manager.enabled = False
    config.sessions.enabled = False
    config.channel.enabled = False

    engine = _engine()
    monkeypatch.setattr(serve_mod, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(serve_mod, "get_engine", lambda *a, **k: ("mock", engine))
    monkeypatch.setattr(serve_mod, "discover_engines", lambda *a, **k: {})
    monkeypatch.setattr(serve_mod, "discover_models", lambda *a, **k: {})
    sec = MagicMock(engine=engine, capability_policy=None, audit_logger=None)
    sec.rate_limiter = None
    monkeypatch.setattr("openjarvis.security.setup_security", lambda *a, **k: sec)

    captured: dict = {}

    def _capture_create_app(*args, **kwargs):
        captured["agent"] = kwargs.get("agent")
        return MagicMock(name="app")

    with (
        patch("openjarvis.server.app.create_app", side_effect=_capture_create_app),
        patch("openjarvis.server.daemon.run_server", lambda *a, **k: None),
    ):
        result = CliRunner().invoke(
            cli, ["serve", "--agent", "stage3e_capture"], catch_exceptions=False
        )

    assert result.exit_code == 0, result.output
    agent = captured["agent"]
    assert_tools_bound(agent._tools, case)
    assert_prompt_matches(agent._prompt_builder.build(), case)


def test_serve_channel_agent(case, capture_agent, monkeypatch):
    pytest.importorskip("fastapi")
    from click.testing import CliRunner

    from openjarvis.cli import cli

    serve_mod = importlib.import_module("openjarvis.cli.serve")
    config = case.config()
    config.tools.enabled = list(_TOOL_NAMES)
    config.intelligence.default_model = "test-model"
    config.telemetry.enabled = False
    config.agent_manager.enabled = False
    config.sessions.enabled = False
    config.channel.enabled = True
    config.channel.default_channel = "stage3e"
    config.channel.default_agent = "stage3e_capture"

    engine = _engine()
    monkeypatch.setattr(serve_mod, "load_config", lambda *a, **k: config)
    monkeypatch.setattr(serve_mod, "get_engine", lambda *a, **k: ("mock", engine))
    monkeypatch.setattr(serve_mod, "discover_engines", lambda *a, **k: {})
    monkeypatch.setattr(serve_mod, "discover_models", lambda *a, **k: {})
    sec = MagicMock(engine=engine, capability_policy=None, audit_logger=None)
    sec.rate_limiter = None
    monkeypatch.setattr("openjarvis.security.setup_security", lambda *a, **k: sec)

    from openjarvis.system import JarvisSystem, SystemBuilder

    monkeypatch.setattr(
        SystemBuilder, "_resolve_channel", lambda self, cfg, bus: MagicMock()
    )
    captured: dict = {}
    monkeypatch.setattr(
        JarvisSystem,
        "wire_channel",
        lambda self, bridge: captured.setdefault("tools", self.tools),
    )

    with (
        patch("openjarvis.server.app.create_app", return_value=MagicMock()),
        patch("openjarvis.server.daemon.run_server", lambda *a, **k: None),
    ):
        result = CliRunner().invoke(cli, ["serve"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert_tools_bound(captured["tools"], case)


# ---------------------------------------------------------------------------
# Managed agents: SSE, scheduled/immediate tick (also ``jarvis agents ask``)
# ---------------------------------------------------------------------------


def test_managed_sse(case):
    pytest.importorskip("fastapi")
    from openjarvis.engine._stubs import StreamChunk
    from openjarvis.server import agent_manager_routes as routes

    class _Engine:
        messages: list = []

        async def stream_full(self, messages, *, model, **kwargs):
            type(self).messages = list(messages)
            yield StreamChunk(content="done")
            yield StreamChunk(finish_reason="stop")

    resolved: list = []
    real_resolve = routes.resolve_agent_tools

    def _spy(*args, **kwargs):
        toolkit = real_resolve(*args, **kwargs)
        resolved.append(list(toolkit.instances))
        return toolkit

    manager = MagicMock()
    manager.list_messages.return_value = []
    app_state = SimpleNamespace(
        config=case.config(),
        memory_backend=None,
        channel_backend=None,
        channel_bridge=None,
    )

    async def _run() -> None:
        response = await routes._stream_managed_agent(
            manager=manager,
            agent_record={
                "id": "agent-3e",
                "name": "3e",
                "agent_type": "simple",
                "config": {
                    "model": "test-model",
                    "tools": list(_TOOL_NAMES),
                    "mcp_tools": False,
                },
            },
            user_content="hi",
            message_id="m-3e",
            engine=_Engine(),
            bus=None,
            app_state=app_state,
        )
        async for _ in response.body_iterator:
            pass

    with patch.object(routes, "resolve_agent_tools", _spy):
        asyncio.run(_run())

    assert_tools_bound(resolved[0], case)
    system_prompt = "\n".join(
        m.content for m in _Engine.messages if m.role.value == "system"
    )
    assert_prompt_matches(system_prompt, case)


def test_managed_tick(case, capture_agent, tmp_path):
    from openjarvis.agents.executor import AgentExecutor
    from openjarvis.agents.manager import AgentManager
    from tests.agents.scenario_harness import FakeSystem

    system = FakeSystem(engine=_engine())
    system.config = case.config()
    manager = AgentManager(db_path=str(tmp_path / "agents.db"))
    try:
        agent = manager.create_agent(
            "3e",
            agent_type="stage3e_capture",
            config={"model": "test-model", "tools": list(_TOOL_NAMES)},
        )
        AgentExecutor(manager, EventBus(), system=system).execute_tick(agent["id"])
    finally:
        manager.close()

    captured = capture_agent.instances[-1]
    assert_tools_bound(captured._tools, case)
    assert_prompt_matches(captured._prompt_builder.build(), case)


def test_managed_ephemeral_session_flush(case, monkeypatch):
    """Session-expiry flush runs ``simple`` through the tool-using fallback."""

    from openjarvis.agents.executor import AgentExecutor
    from openjarvis.agents.orchestrator import OrchestratorAgent
    from openjarvis.agents.simple import SimpleAgent

    if not AgentRegistry.contains("simple"):
        AgentRegistry.register_value("simple", SimpleAgent)
    captured: list = []

    def _capture_run(self, input, context=None, **kwargs):
        captured.append(self)
        return AgentResult(content="captured")

    monkeypatch.setattr(OrchestratorAgent, "run", _capture_run)

    system = SimpleNamespace(
        config=case.config(),
        engine=_engine(),
        model="test-model",
        capability_policy=None,
        rate_limiter=None,
    )
    AgentExecutor(MagicMock(), EventBus(), system=system).run_ephemeral(
        "simple", "flush", "hi", tools=list(_TOOL_NAMES)
    )
    assert_tools_bound(captured[-1]._tools, case)


# ---------------------------------------------------------------------------
# SystemBuilder / JarvisSystem
# ---------------------------------------------------------------------------


def test_system_builder_mcp_discovered_tools(case):
    from openjarvis.system import SystemBuilder

    cfg = case.config()
    builder = SystemBuilder(cfg).tools(list(_TOOL_NAMES))
    tools = builder._resolve_tools(cfg, _engine(), "test-model", None)
    assert_tools_bound(tools, case)


def test_jarvis_system_per_request_tools(case, capture_agent):
    from openjarvis.system import JarvisSystem

    system = JarvisSystem(
        config=case.config(),
        bus=EventBus(),
        engine=_engine(),
        engine_key="mock",
        model="test-model",
    )
    system.ask("hi", agent="stage3e_capture", tools=list(_TOOL_NAMES), context=False)
    assert_tools_bound(capture_agent.instances[-1]._tools, case)
