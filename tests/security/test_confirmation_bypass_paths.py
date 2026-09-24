"""Regression tests: requires_confirmation tools must never be auto-approved.

Covers the execution paths that previously passed
``confirm_callback=lambda _: True`` to their ToolExecutor:

- managed-agent SSE chat executor (``_stream_managed_agent``)
- deep_research managed agent (SSE and channel-binding constructions)
- ``jarvis ask`` agent / skill pipeline executors (see tests/cli/test_ask_agent.py)
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")

from openjarvis.core.registry import ToolRegistry  # noqa: E402
from openjarvis.core.types import Role, ToolCall, ToolResult  # noqa: E402
from openjarvis.engine._stubs import StreamChunk  # noqa: E402
from openjarvis.tools._stubs import BaseTool, ToolSpec  # noqa: E402

_SRC = Path(__file__).resolve().parents[2] / "src" / "openjarvis"


class _SensitiveProbe(BaseTool):
    tool_id = "sensitive_probe_confirm"
    calls = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description="Confirmation-gated probe",
            parameters={"type": "object", "properties": {}},
            requires_confirmation=True,
        )

    def execute(self, **params) -> ToolResult:
        type(self).calls += 1
        return ToolResult(tool_name=self.tool_id, content="SENSITIVE_EXECUTED")


class _SafeProbe(BaseTool):
    tool_id = "safe_probe_confirm"
    calls = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_id,
            description="Ordinary probe",
            parameters={"type": "object", "properties": {}},
        )

    def execute(self, **params) -> ToolResult:
        type(self).calls += 1
        return ToolResult(tool_name=self.tool_id, content="SAFE_EXECUTED")


@pytest.fixture(autouse=True)
def _register_probes():
    for cls in (_SensitiveProbe, _SafeProbe):
        if not ToolRegistry.contains(cls.tool_id):
            ToolRegistry.register_value(cls.tool_id, cls)
        cls.calls = 0


def _app_state(**extra):
    return SimpleNamespace(
        config=SimpleNamespace(memory_files=None, system_prompt=None),
        memory_backend=None,
        channel_backend=None,
        channel_bridge=None,
        _mcp_clients=[],
        _mcp_tools_cache=([], {}),
        **extra,
    )


class _StreamingToolEngine:
    """SSE engine: request one tool call, then record the tool result."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.turns = 0
        self.observed_tool_result = ""

    async def stream_full(self, messages, *, model, **kwargs):
        self.turns += 1
        if self.turns == 1:
            yield StreamChunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": self.tool_name, "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
            )
            return
        tool_messages = [m for m in messages if m.role is Role.TOOL]
        self.observed_tool_result = tool_messages[-1].content
        yield StreamChunk(content="complete")
        yield StreamChunk(finish_reason="stop")


class _GeneratingToolEngine:
    """Deep Research engine: request one tool call, then record the result."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.turns = 0
        self.observed_tool_result = ""

    def generate(self, messages, *, model, **kwargs):
        self.turns += 1
        if self.turns == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": json.dumps({}),
                        },
                    }
                ],
                "usage": {},
            }
        tool_messages = [m for m in messages if m.role is Role.TOOL]
        self.observed_tool_result = tool_messages[-1].content
        return {"content": "complete", "tool_calls": [], "usage": {}}


async def _run_stream(engine, agent_type: str, tool_name: str):
    from openjarvis.server.agent_manager_routes import _stream_managed_agent

    manager = MagicMock()
    manager.list_messages.return_value = []
    response = await _stream_managed_agent(
        manager=manager,
        agent_record={
            "id": f"agent-{agent_type}-{tool_name}",
            "name": "Probe Agent",
            "agent_type": agent_type,
            "config": {
                "model": "test-model",
                "max_turns": 3,
                "mcp_tools": False,
                "tools": [tool_name],
            },
        },
        user_content="Use the probe",
        message_id=f"message-{agent_type}-{tool_name}",
        engine=engine,
        bus=None,
        app_state=_app_state(),
    )
    async for _ in response.body_iterator:
        pass
    if response.background is not None:
        await response.background()


# ---------------------------------------------------------------------------
# Managed-agent SSE chat executor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_chat_refuses_requires_confirmation_tool() -> None:
    engine = _StreamingToolEngine(_SensitiveProbe.tool_id)
    await _run_stream(engine, "simple", _SensitiveProbe.tool_id)

    assert _SensitiveProbe.calls == 0
    assert "SENSITIVE_EXECUTED" not in engine.observed_tool_result
    assert "requires confirmation" in engine.observed_tool_result


@pytest.mark.asyncio
async def test_sse_chat_still_executes_ordinary_tool() -> None:
    engine = _StreamingToolEngine(_SafeProbe.tool_id)
    await _run_stream(engine, "simple", _SafeProbe.tool_id)

    assert _SafeProbe.calls == 1
    assert engine.observed_tool_result == "SAFE_EXECUTED"


# ---------------------------------------------------------------------------
# Deep Research managed agent (SSE)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deep_research_sse_refuses_requires_confirmation_tool() -> None:
    engine = _GeneratingToolEngine(_SensitiveProbe.tool_id)
    await _run_stream(engine, "deep_research", _SensitiveProbe.tool_id)

    assert engine.turns == 2
    assert _SensitiveProbe.calls == 0
    assert "requires confirmation" in engine.observed_tool_result


@pytest.mark.asyncio
async def test_deep_research_sse_still_executes_ordinary_tool() -> None:
    engine = _GeneratingToolEngine(_SafeProbe.tool_id)
    await _run_stream(engine, "deep_research", _SafeProbe.tool_id)

    assert _SafeProbe.calls == 1
    assert engine.observed_tool_result == "SAFE_EXECUTED"


# ---------------------------------------------------------------------------
# Deep Research agents built by channel bindings (iMessage / SendBlue)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channel_type", ["imessage", "sendblue"])
def test_channel_binding_deep_research_refuses_requires_confirmation_tool(
    tmp_path: Path,
    channel_type: str,
) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from openjarvis.agents import deep_research as dr_mod
    from openjarvis.agents.manager import AgentManager
    from openjarvis.server import agent_manager_routes as routes

    built: list = []

    class _RecordingDeepResearchAgent(dr_mod.DeepResearchAgent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            built.append(self)

    manager = AgentManager(db_path=str(tmp_path / "agents.db"))
    try:
        agent = manager.create_agent(name="dr", agent_type="deep_research")
        app = FastAPI()
        for router in routes.create_agent_manager_router(manager):
            app.include_router(router)
        app.state.engine = MagicMock()

        with (
            patch.object(dr_mod, "DeepResearchAgent", _RecordingDeepResearchAgent),
            patch.object(
                routes,
                "_build_deep_research_tools",
                return_value=[_SensitiveProbe(), _SafeProbe()],
            ),
            patch(
                "openjarvis.channels.imessage_daemon.is_running",
                return_value=False,
            ),
            patch("openjarvis.channels.imessage_daemon.run_daemon"),
            patch("openjarvis.channels.sendblue.SendBlueChannel"),
        ):
            config = (
                {"identifier": "+15550000000"}
                if channel_type == "imessage"
                else {
                    "api_key_id": "k",
                    "api_secret_key": "s",
                    "from_number": "+15550000000",
                }
            )
            resp = TestClient(app).post(
                f"/v1/managed-agents/{agent['id']}/channels",
                json={"channel_type": channel_type, "config": config},
            )
        assert resp.status_code == 200, resp.text
    finally:
        manager.close()

    assert len(built) == 1
    executor = built[0]._executor

    denied = executor.execute(
        ToolCall(id="c1", name=_SensitiveProbe.tool_id, arguments="{}")
    )
    assert denied.success is False
    assert "requires confirmation" in denied.content
    assert _SensitiveProbe.calls == 0

    allowed = executor.execute(
        ToolCall(id="c2", name=_SafeProbe.tool_id, arguments="{}")
    )
    assert allowed.success is True
    assert _SafeProbe.calls == 1


# ---------------------------------------------------------------------------
# Static guard: no unconditional auto-approve on the audited paths
# ---------------------------------------------------------------------------


def _always_true_lambda(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Lambda)
        and isinstance(node.body, ast.Constant)
        and node.body.value is True
    )


@pytest.mark.parametrize(
    "rel_path",
    ["server/agent_manager_routes.py", "cli/ask.py"],
)
def test_no_unconditional_auto_approve_callback(rel_path: str) -> None:
    tree = ast.parse((_SRC / rel_path).read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "confirm_callback":
            if _always_true_lambda(node.value):
                offenders.append(node.value.lineno)
        elif isinstance(node, ast.Assign) and _always_true_lambda(node.value):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "confirm_callback"
                ):
                    offenders.append(node.value.lineno)
    assert offenders == [], f"auto-approve confirm_callback at lines {offenders}"
