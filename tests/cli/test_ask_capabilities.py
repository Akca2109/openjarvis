"""Stage 0C: ``jarvis ask`` enforces the default personal capability policy."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from openjarvis.agents._stubs import AgentContext, AgentResult, ToolUsingAgent
from openjarvis.cli import cli
from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.security.capabilities import CapabilityPolicy
from openjarvis.tools._stubs import BaseTool, ToolSpec

_ask_mod = importlib.import_module("openjarvis.cli.ask")
_BODY_CALLS: list[str] = []


class _AdminProbeTool(BaseTool):
    tool_id = "admin_probe"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="admin_probe",
            description="Requires system:admin.",
            required_capabilities=["system:admin"],
        )

    def execute(self, **params) -> ToolResult:
        _BODY_CALLS.append("admin_probe")
        return ToolResult(tool_name="admin_probe", content="ADMIN-RAN", success=True)


class _ProbeAgent(ToolUsingAgent):
    agent_id = "cap_probe_agent"
    seen: dict = {}

    def run(self, input, context: AgentContext | None = None, **kwargs):
        type(self).seen = {
            "policy": self._executor._capability_policy,
            "agent_id": self._executor._agent_id,
        }
        results = [
            self._executor.execute(
                ToolCall(id="a", name="admin_probe", arguments="{}")
            ),
            self._executor.execute(
                ToolCall(id="b", name="calculator", arguments='{"expression": "6*7"}')
            ),
        ]
        return AgentResult(
            content=" | ".join(r.content for r in results),
            tool_results=results,
            turns=1,
        )


@pytest.fixture
def ask_setup(tmp_path):
    from openjarvis.core.config import JarvisConfig
    from openjarvis.core.registry import AgentRegistry, ToolRegistry
    from openjarvis.tools.calculator import CalculatorTool

    engine = MagicMock()
    engine.engine_id = "mock"
    engine.health.return_value = True
    engine.list_models.return_value = ["test-model"]
    config = JarvisConfig()
    config.intelligence.default_model = "test-model"
    config.traces.enabled = False
    config.security.audit_log_path = str(tmp_path / "audit.db")
    config.tools.enabled = ["admin_probe", "calculator"]

    AgentRegistry.register_value("cap_probe_agent", _ProbeAgent)
    ToolRegistry.register_value("admin_probe", _AdminProbeTool)
    if not ToolRegistry.contains("calculator"):
        ToolRegistry.register_value("calculator", CalculatorTool)
    _BODY_CALLS.clear()

    with (
        patch.object(_ask_mod, "load_config", return_value=config),
        patch.object(_ask_mod, "get_engine", return_value=("mock", engine)),
        patch.object(_ask_mod, "discover_engines", return_value=[("mock", engine)]),
        patch.object(_ask_mod, "discover_models", return_value={"mock": ["m"]}),
        patch.object(_ask_mod, "register_builtin_models"),
        patch.object(_ask_mod, "merge_discovered_models"),
    ):
        yield config


def test_ask_default_profile_denies_admin_and_allows_safe_tool(ask_setup):
    result = CliRunner().invoke(cli, ["ask", "--agent", "cap_probe_agent", "hi"])

    assert result.exit_code == 0, result.output
    assert "Capability 'system:admin' denied" in result.output
    assert "ADMIN-RAN" not in result.output
    assert _BODY_CALLS == []
    assert "42" in result.output
    seen = _ProbeAgent.seen
    assert isinstance(seen["policy"], CapabilityPolicy)
    assert seen["agent_id"] == "cap_probe_agent"


def test_ask_explicit_disable_restores_unenforced_dispatch(ask_setup):
    ask_setup.security.capabilities.enabled = False

    result = CliRunner().invoke(cli, ["ask", "--agent", "cap_probe_agent", "hi"])

    assert result.exit_code == 0, result.output
    assert "ADMIN-RAN" in result.output
    assert _ProbeAgent.seen["policy"] is None
