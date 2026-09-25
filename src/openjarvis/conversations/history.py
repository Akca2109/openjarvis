"""Rebuild model-facing context from durable conversation messages.

A resumed chat replays only real user and assistant turns. System prompts,
persona text and memory context are rebuilt fresh on every turn by the chat
surface, so durable ``system`` and ``tool`` rows are never replayed even if
they exist. The budget below bounds what is sent to the model; older
messages stay untouched in ``conversations.db``.

Nothing here logs or reports transcript content — only counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

from openjarvis.core.types import Message, Role

# Stage 2 fixed resume budget: the newest RESUME_MAX_MESSAGES durable rows are
# loaded, then trimmed from the oldest side to about RESUME_MAX_TOKENS
# (estimated with ``estimate_prompt_tokens``).
RESUME_MAX_MESSAGES = 40
RESUME_MAX_TOKENS = 4096

_ELIGIBLE_ROLES = {"user": Role.USER, "assistant": Role.ASSISTANT}


@dataclass(frozen=True, slots=True)
class ResumeContext:
    """Model-facing messages rebuilt from durable history."""

    messages: List[Message]
    # Rows that were not user/assistant turns or were malformed.
    skipped: int
    # Eligible turns left out by the token budget or turn-boundary cleanup.
    trimmed: int


def build_resume_context(
    rows: Sequence[Any], *, max_tokens: int = RESUME_MAX_TOKENS
) -> ResumeContext:
    """Convert durable rows (oldest first) into model context messages.

    1. Keep only ``user``/``assistant`` rows whose content is a string.
    2. Drop trailing user turns that never got a reply (a failed or
       interrupted generation), so the next user turn does not follow
       another user turn.
    3. Drop the oldest messages until the estimate fits ``max_tokens``.
    4. Drop a leading assistant message so context starts on a user turn.
    """
    from openjarvis.engine._base import estimate_prompt_tokens

    messages: List[Message] = []
    skipped = 0
    for row in rows:
        role = _ELIGIBLE_ROLES.get(getattr(row, "role", None))
        content = getattr(row, "content", None)
        if role is None or not isinstance(content, str):
            skipped += 1
            continue
        messages.append(Message(role=role, content=content))

    eligible = len(messages)
    while messages and messages[-1].role == Role.USER:
        messages.pop()
    while messages and estimate_prompt_tokens(messages) > max_tokens:
        messages.pop(0)
    if messages and messages[0].role == Role.ASSISTANT:
        messages.pop(0)
    return ResumeContext(
        messages=messages,
        skipped=skipped,
        trimmed=eligible - len(messages),
    )


__all__ = [
    "RESUME_MAX_MESSAGES",
    "RESUME_MAX_TOKENS",
    "ResumeContext",
    "build_resume_context",
]
