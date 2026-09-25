"""Run-local tool-call ID hygiene across agent paths and the Gemini engine."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from openjarvis.agents.loop_guard import LoopGuard, LoopGuardConfig
from openjarvis.agents.native_react import NativeReActAgent
from openjarvis.agents.orchestrator import OrchestratorAgent
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import Message, Role, ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec
from openjarvis.tools.call_ids import (
    ToolCallIdAllocator,
    is_valid_tool_call_id,
    new_tool_call_id,
)
from openjarvis.tools.outcomes import tool_result_call_id, tool_result_outcome


class _EchoTool(BaseTool):
    tool_id = "echo"

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="echo", description="Echo text.")

    def execute(self, **params) -> ToolResult:
        return ToolResult(tool_name="echo", content=str(params.get("text", "")))


def _engine(*replies: dict) -> mock.MagicMock:
    engine = mock.MagicMock()
    engine.engine_id = "mock"
    engine._publishes_events = False
    engine.generate.side_effect = list(replies)
    return engine


def _tc(call_id, text: str, name: str = "echo") -> dict:
    tc = {"name": name, "arguments": json.dumps({"text": text})}
    if call_id is not None:
        tc["id"] = call_id
    return tc


def _agent(cls, engine, bus=None, **kwargs):
    agent = cls(
        engine,
        "test-model",
        tools=[_EchoTool()],
        bus=bus,
        max_turns=6,
        temperature=0.0,
        max_tokens=64,
        **kwargs,
    )
    agent._loop_guard = None
    return agent


def _ids(result) -> list[str]:
    return [tool_result_call_id(r) for r in result.tool_results]


def _assert_unique_valid(ids: list) -> None:
    assert all(is_valid_tool_call_id(call_id) for call_id in ids), ids
    assert len(set(ids)) == len(ids), ids


# ---------------------------------------------------------------------------
# Allocator
# ---------------------------------------------------------------------------


def test_allocator_keeps_valid_ids_and_replaces_bad_or_duplicate_ones():
    allocator = ToolCallIdAllocator()
    assert allocator.claim("toolu_01ABC") == "toolu_01ABC"
    duplicate = allocator.claim("toolu_01ABC")
    assert duplicate != "toolu_01ABC"
    claimed = [
        duplicate,
        allocator.claim(""),
        allocator.claim(None),
        allocator.claim(123),
        allocator.claim("x" * 129),
        allocator.claim("has space"),
        allocator.claim("line\nbreak"),
    ]
    _assert_unique_valid(["toolu_01ABC", *claimed])
    allocator.reset()
    assert allocator.claim("toolu_01ABC") == "toolu_01ABC"


def test_generated_ids_are_opaque_and_bounded():
    generated = {new_tool_call_id() for _ in range(200)}
    assert len(generated) == 200
    for call_id in generated:
        assert is_valid_tool_call_id(call_id)
        assert len(call_id) <= 64
        assert "echo" not in call_id


# ---------------------------------------------------------------------------
# Function-calling agents (normalized once, at the generate boundary)
# ---------------------------------------------------------------------------


def test_duplicate_and_empty_ids_become_unique_within_one_run():
    engine = _engine(
        # Duplicate within one response, an empty ID, and a missing ID.
        {
            "content": "",
            "tool_calls": [
                _tc("call_0", "a"),
                _tc("call_0", "b"),
                _tc("", "c"),
                _tc(None, "d"),
            ],
        },
        # Ollama-style index IDs repeat on the next response.
        {"content": "", "tool_calls": [_tc("call_0", "e"), _tc("call_1", "f")]},
        {"content": "done"},
    )
    bus = EventBus(record_history=True)
    agent = _agent(OrchestratorAgent, engine, bus=bus, parallel_tools=False)

    result = agent.run("go")

    assert result.content == "done"
    ids = _ids(result)
    assert len(ids) == 6
    _assert_unique_valid(ids)
    assert ids[0] == "call_0"  # a valid first-seen provider ID is kept
    assert ids[5] == "call_1"
    assert all(tool_result_outcome(r) == "success" for r in result.tool_results)

    # The model sees consistent assistant tool_calls / tool message IDs.
    final_messages = engine.generate.call_args_list[-1].args[0]
    announced = [
        tc.id for m in final_messages if m.role == Role.ASSISTANT for tc in m.tool_calls
    ]
    answered = [m.tool_call_id for m in final_messages if m.role == Role.TOOL]
    assert announced == ids
    assert answered == ids

    # Lifecycle events correlate with the same IDs.
    end_ids = [
        e.data["tool_call_id"]
        for e in bus.history
        if e.event_type == EventType.TOOL_CALL_END
    ]
    assert end_ids == ids
    # INFERENCE_END reports the normalized IDs too.
    inference_ids = [
        tc["id"]
        for e in bus.history
        if e.event_type == EventType.INFERENCE_END
        for tc in e.data["tool_calls"]
    ]
    assert inference_ids == ids


def test_ids_are_run_local_and_reset_between_runs():
    engine = _engine(
        {"content": "", "tool_calls": [_tc("call_0", "a")]},
        {"content": "one"},
        {"content": "", "tool_calls": [_tc("call_0", "b")]},
        {"content": "two"},
    )
    agent = _agent(OrchestratorAgent, engine, parallel_tools=False)
    assert _ids(agent.run("first")) == ["call_0"]
    assert _ids(agent.run("second")) == ["call_0"]


def test_normalization_does_not_mutate_engine_dicts():
    raw = {"content": "", "tool_calls": [_tc("", "a")]}
    engine = _engine(raw, {"content": "ok"})
    _agent(OrchestratorAgent, engine, parallel_tools=False).run("go")
    assert raw["tool_calls"][0]["id"] == ""


# ---------------------------------------------------------------------------
# Agent-synthesized IDs (text-protocol agents)
# ---------------------------------------------------------------------------


def test_native_react_ids_are_opaque_and_unique():
    action = 'Thought: t\nAction: echo\nAction Input: {"text": "x"}'
    engine = _engine(
        {"content": action},
        {"content": action.replace("x", "y")},
        {"content": "Final Answer: done"},
    )
    result = _agent(NativeReActAgent, engine).run("go")
    ids = _ids(result)
    assert len(ids) == 2
    _assert_unique_valid(ids)
    assert not any(call_id.startswith("react_") for call_id in ids)


def test_orchestrator_structured_ids_are_opaque_and_unique():
    step = 'THOUGHT: t\nTOOL: echo\nINPUT: {"text": "x"}'
    engine = _engine(
        {"content": step},
        {"content": step.replace("x", "y")},
        {"content": "FINAL_ANSWER: done"},
    )
    result = _agent(OrchestratorAgent, engine, mode="structured").run("go")
    ids = _ids(result)
    assert len(ids) == 2
    _assert_unique_valid(ids)
    assert not any(call_id.startswith("orch_") for call_id in ids)


# ---------------------------------------------------------------------------
# Agent-level gates carry outcome and ID too
# ---------------------------------------------------------------------------


SECRET_ARG = "SECRET-ARG-MARKER-e41b"


def _events(bus: EventBus, event_type: EventType) -> list:
    return [e for e in bus.history if e.event_type == event_type]


def _assert_blocked_contract(bus, results, outcome: str, agent_id: str) -> list:
    """Blocked results match safe TOOL_CALL_BLOCKED events and never executed."""
    blocked = [r for r in results if tool_result_outcome(r) == outcome]
    assert blocked, [tool_result_outcome(r) for r in results]
    blocked_ids = [tool_result_call_id(r) for r in blocked]
    _assert_unique_valid(blocked_ids)
    assert all(r.success is False for r in blocked)

    events = _events(bus, EventType.TOOL_CALL_BLOCKED)
    assert [e.data for e in events] == [
        {"tool": "echo", "tool_call_id": call_id, "outcome": outcome, "agent": agent_id}
        for call_id in blocked_ids
    ]
    for event in events:
        assert SECRET_ARG not in json.dumps(event.data)

    lifecycle_ids = {
        e.data["tool_call_id"]
        for e in bus.history
        if e.event_type in (EventType.TOOL_CALL_START, EventType.TOOL_CALL_END)
    }
    assert lifecycle_ids.isdisjoint(blocked_ids)
    return blocked_ids


def _provenance_outcomes(results) -> list[tuple[str, str]]:
    from openjarvis.conversations import tool_provenance_metadata

    entries = tool_provenance_metadata(results)["tool_calls"].split(";")
    return [tuple(entry.split("|")[1:]) for entry in entries]


@pytest.mark.parametrize("mode", ["function_calling", "structured"])
def test_governance_denial_is_policy_denied_and_blocked_event(mode):
    if mode == "function_calling":
        engine = _engine(
            {"content": "", "tool_calls": [_tc("call_gov", SECRET_ARG)]},
            {"content": "ok"},
        )
    else:
        engine = _engine(
            {"content": f'THOUGHT: t\nTOOL: echo\nINPUT: {{"text": "{SECRET_ARG}"}}'},
            {"content": "FINAL_ANSWER: ok"},
        )
    bus = EventBus(record_history=True)
    agent = _agent(
        OrchestratorAgent,
        engine,
        bus=bus,
        mode=mode,
        parallel_tools=False,
        agent_id="gov-agent",
        before_tool_call=lambda name, args: False,
    )
    results = agent.run("go").tool_results

    (denied,) = results
    (call_id,) = _assert_blocked_contract(bus, results, "policy_denied", "gov-agent")
    if mode == "function_calling":
        assert call_id == "call_gov"
    assert "[Governance]" in denied.content
    assert _provenance_outcomes(results) == [("policy_denied", call_id)]


@pytest.mark.parametrize("parallel", [False, True])
def test_orchestrator_loop_guard_block_is_explicit_and_blocked_event(parallel):
    same = _tc(None, SECRET_ARG)
    engine = _engine(
        {"content": "", "tool_calls": [dict(same), dict(same), dict(same)]},
        {"content": "ok"},
    )
    bus = EventBus(record_history=True)
    agent = _agent(
        OrchestratorAgent,
        engine,
        bus=bus,
        parallel_tools=parallel,
        agent_id="loop-agent",
    )
    agent._loop_guard = LoopGuard(
        LoopGuardConfig(max_identical_calls=1, warn_before_block=False)
    )
    results = agent.run("go").tool_results

    _assert_unique_valid([tool_result_call_id(r) for r in results])
    blocked_ids = _assert_blocked_contract(
        bus, results, "loop_guard_blocked", "loop-agent"
    )
    # The call that did run keeps its normal lifecycle.
    executed = [r for r in results if tool_result_outcome(r) == "success"]
    assert executed
    ended = [e.data["tool_call_id"] for e in _events(bus, EventType.TOOL_CALL_END)]
    assert ended == [tool_result_call_id(r) for r in executed]
    provenance = _provenance_outcomes(results)
    assert [p for p in provenance if p[0] == "loop_guard_blocked"] == [
        ("loop_guard_blocked", call_id) for call_id in blocked_ids
    ]


def test_native_react_loop_guard_block_is_explicit_and_blocked_event():
    action = f'Thought: t\nAction: echo\nAction Input: {{"text": "{SECRET_ARG}"}}'
    engine = _engine(
        {"content": action},
        {"content": action},
        {"content": "Final Answer: done"},
    )
    bus = EventBus(record_history=True)
    agent = _agent(NativeReActAgent, engine, bus=bus, agent_id="react-agent")
    agent._loop_guard = LoopGuard(
        LoopGuardConfig(max_identical_calls=1, warn_before_block=False)
    )
    results = agent.run("go").tool_results

    assert [tool_result_outcome(r) for r in results] == [
        "success",
        "loop_guard_blocked",
    ]
    (call_id,) = _assert_blocked_contract(
        bus, results, "loop_guard_blocked", "react-agent"
    )
    assert _provenance_outcomes(results) == [
        ("success", tool_result_call_id(results[0])),
        ("loop_guard_blocked", call_id),
    ]


# ---------------------------------------------------------------------------
# Gemini non-stream
# ---------------------------------------------------------------------------


def test_gemini_same_name_calls_no_longer_collide(monkeypatch):
    from openjarvis.engine.cloud import CloudEngine

    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    engine = CloudEngine()

    def _part(city, sig=None):
        return SimpleNamespace(
            function_call=SimpleNamespace(name="get_weather", args={"city": city}),
            text=None,
            thought_signature=sig,
        )

    def _response():
        return SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[_part("Paris", b"sig-paris"), _part("London")]
                    )
                )
            ],
            text="",
            usage_metadata=SimpleNamespace(
                prompt_token_count=1, candidates_token_count=1
            ),
        )

    client = mock.MagicMock()
    client.models.generate_content.side_effect = [_response(), _response()]
    engine._google_client = client
    modules = {
        "google": mock.MagicMock(),
        "google.genai": mock.MagicMock(),
        "google.genai.types": mock.MagicMock(),
    }
    with mock.patch.dict("sys.modules", modules):
        first = engine.generate(
            [Message(role=Role.USER, content="weather?")], model="gemini-2.5-pro"
        )
        second = engine.generate(
            [Message(role=Role.USER, content="again?")], model="gemini-2.5-pro"
        )

    ids = [tc["id"] for tc in first["tool_calls"] + second["tool_calls"]]
    assert len(ids) == 4
    _assert_unique_valid(ids)
    assert "google_get_weather" not in ids
    # Thought signatures stay keyed by the (now unique) call ID.
    assert engine._thought_sigs[first["tool_calls"][0]["id"]] == b"sig-paris"
    assert first["tool_calls"][1]["id"] not in engine._thought_sigs
