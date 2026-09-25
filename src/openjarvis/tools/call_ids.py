"""Run-local tool-call ID hygiene.

Providers do not reliably return usable tool-call IDs: some omit them, some
derive them from the function name or the call's index in one response, so
IDs collide across responses in a multi-turn agent run. A correlation ID must
be non-empty and unique within the run; :class:`ToolCallIdAllocator` keeps
valid provider IDs and replaces empty, malformed, or duplicate ones with
opaque generated IDs.
"""

from __future__ import annotations

import re
import threading
import uuid
from typing import Any, Optional, Set

# Bounded, printable, and compatible with the conversation store's ID syntax.
_VALID_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def is_valid_tool_call_id(value: Any) -> bool:
    """Return whether *value* is a bounded, safe tool-call ID."""
    return isinstance(value, str) and bool(_VALID_ID_RE.match(value))


def new_tool_call_id() -> str:
    """Return a fresh opaque tool-call ID."""
    return f"call_{uuid.uuid4().hex[:24]}"


class ToolCallIdAllocator:
    """Hand out tool-call IDs that are unique within one agent run."""

    def __init__(self) -> None:
        self._seen: Set[str] = set()
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Forget every claimed ID (call at the start of a run)."""
        with self._lock:
            self._seen.clear()

    def claim(self, candidate: Optional[Any] = None) -> str:
        """Return *candidate* if it is valid and unused, else a fresh ID."""
        with self._lock:
            if is_valid_tool_call_id(candidate) and candidate not in self._seen:
                self._seen.add(candidate)
                return candidate
            while True:
                generated = new_tool_call_id()
                if generated not in self._seen:
                    self._seen.add(generated)
                    return generated


__all__ = ["ToolCallIdAllocator", "is_valid_tool_call_id", "new_tool_call_id"]
