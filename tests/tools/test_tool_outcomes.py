"""ToolExecutor outcomes, tool-call correlation, and blocked-call events."""

from __future__ import annotations

import json
import threading

import pytest

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.security.taint import TaintLabel, TaintSet
from openjarvis.tools._stubs import BaseTool, ToolExecutor, ToolSpec
from openjarvis.tools.outcomes import (
    OUTCOME_KEY,
    TOOL_CALL_ID_KEY,
    ToolOutcome,
    tool_result_call_id,
    tool_result_outcome,
)

RAW_ARG = "RAW-ARGUMENT-MARKER-91f3"
RAW_OUTPUT = "RAW-OUTPUT-MARKER-4c2a"


class _Tool(BaseTool):
    def __init__(
        self,
        name: str = "probe",
        *,
        behavior: str = "echo",
        is_local: bool = True,
        requires_confirmation: bool = False,
        timeout_seconds: float = 30.0,
        release: threading.Event | None = None,
        required_capabilities: list[str] | None = None,
    ) -> None:
        self.tool_id = name
        self._name = name
        self._behavior = behavior
        self.is_local = is_local
        self._requires_confirmation = requires_confirmation
        self._timeout = timeout_seconds
        self._release = release
        self._capabilities = list(required_capabilities or [])
        self.calls = 0

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description="test tool",
            requires_confirmation=self._requires_confirmation,
            timeout_seconds=self._timeout,
            required_capabilities=self._capabilities,
        )

    def execute(self, **params) -> ToolResult:
        self.calls += 1
        if self._behavior == "raise":
            raise RuntimeError("boom")
        if self._behavior == "fail":
            return ToolResult(tool_name=self._name, content="nope", success=False)
        if self._behavior == "block":
            assert self._release is not None
            self._release.wait(5)
        if self._behavior == "spoof":
            return ToolResult(
                tool_name=self._name,
                content=RAW_OUTPUT,
                metadata={OUTCOME_KEY: "denied", TOOL_CALL_ID_KEY: "forged"},
            )
        return ToolResult(tool_name=self._name, content=RAW_OUTPUT)


class _DenyingRateLimiter:
    def check(self, key: str):
        return False, 2.5


class _RaisingBoundaryGuard:
    def check_outbound(self, tool_call):
        raise RuntimeError("outbound secret detected")


class _DenyAllPolicy:
    def check(self, agent_id, capability, resource=""):
        return False


def _call(name: str = "probe", call_id: str = "call_abc", args=None) -> ToolCall:
    arguments = json.dumps({"text": RAW_ARG} if args is None else args)
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _bus() -> EventBus:
    return EventBus(record_history=True)


def _types(bus: EventBus) -> list[EventType]:
    return [event.event_type for event in bus.history]


def _blocked_events(bus: EventBus):
    return [e for e in bus.history if e.event_type == EventType.TOOL_CALL_BLOCKED]


def _assert_result(result: ToolResult, outcome: ToolOutcome, call_id: str) -> None:
    assert result.metadata[OUTCOME_KEY] == outcome.value
    assert result.metadata[TOOL_CALL_ID_KEY] == call_id
    assert tool_result_outcome(result) == outcome.value
    assert tool_result_call_id(result) == call_id


def _assert_blocked(bus: EventBus, tool: str, outcome: ToolOutcome, call_id: str):
    """Exactly one safe blocked event, and no execution lifecycle events."""
    (event,) = _blocked_events(bus)
    assert event.data == {
        "tool": tool,
        "tool_call_id": call_id,
        "outcome": outcome.value,
        "agent": event.data["agent"],
    }
    serialized = json.dumps(event.data)
    assert RAW_ARG not in serialized
    assert RAW_OUTPUT not in serialized
    assert EventType.TOOL_CALL_START not in _types(bus)
    assert EventType.TOOL_CALL_END not in _types(bus)


# ---------------------------------------------------------------------------
# A / L — success
# ---------------------------------------------------------------------------


def test_success_has_outcome_and_matching_id_and_normal_lifecycle():
    bus = _bus()
    tool = _Tool()
    result = ToolExecutor([tool], bus=bus, agent_id="a1").execute(_call())

    assert result.success is True
    assert result.content == RAW_OUTPUT
    _assert_result(result, ToolOutcome.SUCCESS, "call_abc")
    assert tool.calls == 1

    assert _types(bus) == [EventType.TOOL_CALL_START, EventType.TOOL_CALL_END]
    start, end = bus.history
    assert start.data["tool_call_id"] == "call_abc"
    assert start.data["tool"] == "probe"
    assert start.data["agent"] == "a1"
    assert end.data["tool_call_id"] == "call_abc"
    assert end.data["outcome"] == "success"
    assert end.data["success"] is True
    assert end.data["result"] == RAW_OUTPUT
    assert end.data["metadata"][OUTCOME_KEY] == "success"
    assert end.data["metadata"][TOOL_CALL_ID_KEY] == "call_abc"
    assert not _blocked_events(bus)


def test_tool_cannot_spoof_its_own_outcome_or_id():
    result = ToolExecutor([_Tool(behavior="spoof")]).execute(_call())
    _assert_result(result, ToolOutcome.SUCCESS, "call_abc")


def test_empty_call_id_gets_generated_correlation_id():
    bus = _bus()
    result = ToolExecutor([_Tool()], bus=bus).execute(_call(call_id=""))
    call_id = result.metadata[TOOL_CALL_ID_KEY]
    assert call_id and call_id.startswith("call_")
    assert [e.data["tool_call_id"] for e in bus.history] == [call_id, call_id]


# ---------------------------------------------------------------------------
# B–H — calls rejected before execution
# ---------------------------------------------------------------------------


def test_unknown_tool():
    bus = _bus()
    result = ToolExecutor([_Tool()], bus=bus).execute(_call(name="missing"))
    assert result.success is False
    assert result.content == "Unknown tool: missing"
    _assert_result(result, ToolOutcome.UNKNOWN_TOOL, "call_abc")
    _assert_blocked(bus, "missing", ToolOutcome.UNKNOWN_TOOL, "call_abc")


@pytest.mark.parametrize("arguments", [f"{{not json {RAW_ARG}", json.dumps([RAW_ARG])])
def test_invalid_args(arguments):
    bus = _bus()
    tool = _Tool()
    call = ToolCall(id="call_bad", name="probe", arguments=arguments)
    result = ToolExecutor([tool], bus=bus).execute(call)
    assert result.success is False
    _assert_result(result, ToolOutcome.INVALID_ARGS, "call_bad")
    _assert_blocked(bus, "probe", ToolOutcome.INVALID_ARGS, "call_bad")
    assert tool.calls == 0


def test_rate_limited_keeps_existing_event_and_adds_blocked_event():
    bus = _bus()
    tool = _Tool()
    executor = ToolExecutor(
        [tool], bus=bus, agent_id="a1", rate_limiter=_DenyingRateLimiter()
    )
    result = executor.execute(_call())
    assert result.success is False
    assert "Rate limit exceeded" in result.content
    _assert_result(result, ToolOutcome.RATE_LIMITED, "call_abc")
    assert _types(bus) == [EventType.RATE_LIMITED, EventType.TOOL_CALL_BLOCKED]
    assert bus.history[0].data == {
        "agent_id": "a1",
        "tool": "probe",
        "tool_call_id": "call_abc",
        "wait_seconds": 2.5,
    }
    _assert_blocked(bus, "probe", ToolOutcome.RATE_LIMITED, "call_abc")
    assert tool.calls == 0


def test_boundary_blocked():
    bus = _bus()
    tool = _Tool(is_local=False)
    executor = ToolExecutor([tool], bus=bus, boundary_guard=_RaisingBoundaryGuard())
    result = executor.execute(_call())
    assert result.success is False
    assert result.content.startswith("Security block:")
    _assert_result(result, ToolOutcome.BOUNDARY_BLOCKED, "call_abc")
    _assert_blocked(bus, "probe", ToolOutcome.BOUNDARY_BLOCKED, "call_abc")
    assert tool.calls == 0


def test_capability_denied_keeps_existing_event_and_adds_blocked_event():
    bus = _bus()
    tool = _Tool(required_capabilities=["network:fetch"])
    executor = ToolExecutor(
        [tool], bus=bus, agent_id="a1", capability_policy=_DenyAllPolicy()
    )
    result = executor.execute(_call())
    assert result.success is False
    _assert_result(result, ToolOutcome.CAPABILITY_DENIED, "call_abc")
    assert _types(bus) == [EventType.CAPABILITY_DENIED, EventType.TOOL_CALL_BLOCKED]
    assert bus.history[0].data == {
        "agent_id": "a1",
        "capability": "network:fetch",
        "tool": "probe",
        "tool_call_id": "call_abc",
    }
    _assert_blocked(bus, "probe", ToolOutcome.CAPABILITY_DENIED, "call_abc")
    assert tool.calls == 0


def test_taint_blocked_keeps_existing_event_and_adds_blocked_event():
    bus = _bus()
    tool = _Tool(name="web_search")
    executor = ToolExecutor([tool], bus=bus)
    executor.begin_session()
    executor._session_taint = TaintSet(labels=frozenset({TaintLabel.SECRET}))
    result = executor.execute(_call(name="web_search"))
    assert result.success is False
    assert result.content.startswith("Taint violation:")
    _assert_result(result, ToolOutcome.TAINT_BLOCKED, "call_abc")
    assert _types(bus) == [EventType.TAINT_VIOLATION, EventType.TOOL_CALL_BLOCKED]
    assert bus.history[0].data["tool_call_id"] == "call_abc"
    assert bus.history[0].data["violation"]
    _assert_blocked(bus, "web_search", ToolOutcome.TAINT_BLOCKED, "call_abc")
    assert tool.calls == 0


def test_confirmation_denied_by_user():
    bus = _bus()
    tool = _Tool(requires_confirmation=True)
    executor = ToolExecutor(
        [tool], bus=bus, interactive=True, confirm_callback=lambda prompt: False
    )
    result = executor.execute(_call())
    assert result.success is False
    assert "denied by user" in result.content
    _assert_result(result, ToolOutcome.DENIED, "call_abc")
    _assert_blocked(bus, "probe", ToolOutcome.DENIED, "call_abc")
    assert tool.calls == 0


def test_confirmation_unavailable_defaults_to_denied():
    bus = _bus()
    tool = _Tool(requires_confirmation=True)
    result = ToolExecutor([tool], bus=bus).execute(_call())
    assert result.success is False
    _assert_result(result, ToolOutcome.DENIED, "call_abc")
    _assert_blocked(bus, "probe", ToolOutcome.DENIED, "call_abc")
    assert tool.calls == 0


def test_confirmation_approved_executes_normally():
    tool = _Tool(requires_confirmation=True)
    executor = ToolExecutor(
        [tool], interactive=True, confirm_callback=lambda prompt: True
    )
    result = executor.execute(_call())
    _assert_result(result, ToolOutcome.SUCCESS, "call_abc")
    assert tool.calls == 1


# ---------------------------------------------------------------------------
# I / J — failures after dispatch
# ---------------------------------------------------------------------------


def test_timeout():
    bus = _bus()
    release = threading.Event()
    tool = _Tool(behavior="block", timeout_seconds=0.05, release=release)
    try:
        result = ToolExecutor([tool], bus=bus).execute(_call())
    finally:
        release.set()
    assert result.success is False
    assert "timed out" in result.content
    _assert_result(result, ToolOutcome.TIMEOUT, "call_abc")
    timeout_events = [e for e in bus.history if e.event_type == EventType.TOOL_TIMEOUT]
    assert [e.data["tool_call_id"] for e in timeout_events] == ["call_abc"]
    end = bus.history[-1]
    assert end.event_type == EventType.TOOL_CALL_END
    assert end.data["outcome"] == "timeout"
    assert not _blocked_events(bus)


@pytest.mark.parametrize("behavior", ["raise", "fail"])
def test_ordinary_execution_failure_is_error(behavior):
    bus = _bus()
    result = ToolExecutor([_Tool(behavior=behavior)], bus=bus).execute(_call())
    assert result.success is False
    _assert_result(result, ToolOutcome.ERROR, "call_abc")
    assert _types(bus) == [EventType.TOOL_CALL_START, EventType.TOOL_CALL_END]
    assert bus.history[-1].data["outcome"] == "error"
    assert bus.history[-1].data["tool_call_id"] == "call_abc"


def test_capacity_exhausted(monkeypatch):
    import openjarvis.tools._stubs as stubs

    monkeypatch.setattr(stubs._TOOL_RUNNER, "submit", lambda fn, **kw: None)
    result = ToolExecutor([_Tool()]).execute(_call())
    assert result.success is False
    _assert_result(result, ToolOutcome.CAPACITY_EXHAUSTED, "call_abc")


# ---------------------------------------------------------------------------
# Legacy results without annotations
# ---------------------------------------------------------------------------


def test_outcome_helpers_infer_for_unannotated_results():
    assert tool_result_outcome(ToolResult(tool_name="x", content="")) == "success"
    failed = ToolResult(tool_name="x", content="", success=False)
    assert tool_result_outcome(failed) == "error"
    assert tool_result_call_id(failed) is None
    bogus = ToolResult(tool_name="x", content="", metadata={OUTCOME_KEY: "weird"})
    assert tool_result_outcome(bogus) == "success"
