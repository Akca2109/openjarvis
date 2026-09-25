"""Safe, flat tool-use provenance for a durable assistant turn.

The final assistant row of a tool-using turn records *that* tools ran, which
ones, how each call ended, and the tool-call ID that correlates each outcome
with live events. It never records tool arguments, tool output, or any other
content, and it fits the store's flat-scalar metadata schema::

    tool_provenance       1  (format version)
    tools_used            true
    tool_call_count       3
    tool_calls            "web_search|success|call_ab12;file_read|denied|call_cd34"
    tool_calls_truncated  false

``tool_calls`` lists calls in execution order as ``name|outcome|id`` entries
joined by ``;``. Names are reduced to ``[A-Za-z0-9._-]`` and IDs that are not
valid tool-call IDs are written as ``-``, so neither separator can appear in
a field. Entries beyond the size budget are dropped and flagged.

This metadata is descriptive only: it is not replayed into model context.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Union

TOOL_PROVENANCE_VERSION = 1
# Leaves ample headroom under the store's 4096-byte metadata limit.
MAX_TOOL_CALLS_CHARS = 3072
MAX_TOOL_CALL_ENTRIES = 64
_MAX_NAME_CHARS = 64
_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")

Scalar = Union[str, int, bool]


def tool_provenance_metadata(
    tool_results: Optional[Sequence[Any]],
) -> Optional[Dict[str, Scalar]]:
    """Summarize a turn's ``ToolResult`` list; ``None`` when no tools ran."""
    if not tool_results:
        return None
    from openjarvis.core.types import ToolResult
    from openjarvis.tools.call_ids import is_valid_tool_call_id
    from openjarvis.tools.outcomes import tool_result_call_id, tool_result_outcome

    results = [result for result in tool_results if isinstance(result, ToolResult)]
    if not results:
        return None

    entries = []
    used = 0
    truncated = False
    for result in results:
        call_id = tool_result_call_id(result)
        entry = "|".join(
            (
                _safe_tool_name(result.tool_name),
                tool_result_outcome(result),
                call_id if is_valid_tool_call_id(call_id) else "-",
            )
        )
        cost = len(entry) + (1 if entries else 0)
        if len(entries) >= MAX_TOOL_CALL_ENTRIES or used + cost > MAX_TOOL_CALLS_CHARS:
            truncated = True
            break
        entries.append(entry)
        used += cost

    return {
        "tool_provenance": TOOL_PROVENANCE_VERSION,
        "tools_used": True,
        "tool_call_count": len(results),
        "tool_calls": ";".join(entries),
        "tool_calls_truncated": truncated,
    }


def _safe_tool_name(name: Any) -> str:
    cleaned = _UNSAFE_NAME_RE.sub("", name if isinstance(name, str) else "")
    return cleaned[:_MAX_NAME_CHARS] or "unnamed"


__all__ = [
    "MAX_TOOL_CALLS_CHARS",
    "MAX_TOOL_CALL_ENTRIES",
    "TOOL_PROVENANCE_VERSION",
    "tool_provenance_metadata",
]
