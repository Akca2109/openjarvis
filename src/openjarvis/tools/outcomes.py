"""Machine-readable tool outcomes and tool-call correlation metadata.

``ToolExecutor`` annotates every ``ToolResult`` it returns with two stable
metadata keys: :data:`OUTCOME_KEY` (a :class:`ToolOutcome` value) and
:data:`TOOL_CALL_ID_KEY` (the originating ``ToolCall.id``). Both are plain
strings — never arguments, content, or secrets — so they are safe to forward
to event subscribers and to summarize in durable provenance.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from openjarvis.core.types import ToolResult

OUTCOME_KEY = "outcome"
TOOL_CALL_ID_KEY = "tool_call_id"


class ToolOutcome(str, Enum):
    """Why a tool call finished the way it did."""

    SUCCESS = "success"
    # The tool ran (or was dispatched) and failed.
    ERROR = "error"
    TIMEOUT = "timeout"
    CAPACITY_EXHAUSTED = "capacity_exhausted"
    # The call was rejected before the tool ran.
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGS = "invalid_args"
    RATE_LIMITED = "rate_limited"
    BOUNDARY_BLOCKED = "boundary_blocked"
    CAPABILITY_DENIED = "capability_denied"
    TAINT_BLOCKED = "taint_blocked"
    # Confirmation was refused or could not be obtained (defaults to No).
    DENIED = "denied"
    # Agent-level gates outside ToolExecutor.
    LOOP_GUARD_BLOCKED = "loop_guard_blocked"
    POLICY_DENIED = "policy_denied"


def annotate_tool_result(
    result: ToolResult, *, tool_call_id: str, outcome: ToolOutcome
) -> ToolResult:
    """Record *outcome* and *tool_call_id* on *result* (in place) and return it."""
    result.metadata[OUTCOME_KEY] = outcome.value
    result.metadata[TOOL_CALL_ID_KEY] = tool_call_id
    return result


def tool_result_outcome(result: ToolResult) -> str:
    """Return *result*'s recorded outcome, inferring one for legacy results."""
    outcome = result.metadata.get(OUTCOME_KEY) if result.metadata else None
    if isinstance(outcome, str) and outcome in _OUTCOME_VALUES:
        return outcome
    return ToolOutcome.SUCCESS.value if result.success else ToolOutcome.ERROR.value


def tool_result_call_id(result: ToolResult) -> Optional[str]:
    """Return *result*'s originating tool-call id, if one was recorded."""
    call_id = result.metadata.get(TOOL_CALL_ID_KEY) if result.metadata else None
    return call_id if isinstance(call_id, str) and call_id else None


_OUTCOME_VALUES = frozenset(outcome.value for outcome in ToolOutcome)


__all__ = [
    "OUTCOME_KEY",
    "TOOL_CALL_ID_KEY",
    "ToolOutcome",
    "annotate_tool_result",
    "tool_result_call_id",
    "tool_result_outcome",
]
