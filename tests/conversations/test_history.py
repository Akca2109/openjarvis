"""Tests for rebuilding model context from durable conversation rows."""

from __future__ import annotations

from types import SimpleNamespace

from openjarvis.conversations import (
    RESUME_MAX_MESSAGES,
    RESUME_MAX_TOKENS,
    ConversationStore,
    build_resume_context,
)
from openjarvis.core.types import Role
from openjarvis.engine._base import estimate_prompt_tokens


def _row(role, content):
    return SimpleNamespace(role=role, content=content)


def _pairs(ctx):
    return [(m.role, m.content) for m in ctx.messages]


def test_budget_constants():
    assert RESUME_MAX_MESSAGES == 40
    assert RESUME_MAX_TOKENS == 4096


def test_user_assistant_turns_kept_in_order():
    ctx = build_resume_context(
        [
            _row("user", "u1"),
            _row("assistant", "a1"),
            _row("user", "u2"),
            _row("assistant", "a2"),
        ]
    )
    assert _pairs(ctx) == [
        (Role.USER, "u1"),
        (Role.ASSISTANT, "a1"),
        (Role.USER, "u2"),
        (Role.ASSISTANT, "a2"),
    ]
    assert ctx.skipped == 0 and ctx.trimmed == 0


def test_system_tool_and_malformed_rows_are_skipped():
    ctx = build_resume_context(
        [
            _row("system", "SYSTEM-ROW"),
            _row("user", "u1"),
            _row("tool", "TOOL-ROW"),
            _row("assistant", None),
            _row("bogus", "x"),
            SimpleNamespace(),
            _row("assistant", "a1"),
        ]
    )
    assert _pairs(ctx) == [(Role.USER, "u1"), (Role.ASSISTANT, "a1")]
    assert ctx.skipped == 5
    assert all(m.role in (Role.USER, Role.ASSISTANT) for m in ctx.messages)


def test_trailing_dangling_user_turn_is_dropped():
    ctx = build_resume_context(
        [_row("user", "u1"), _row("assistant", "a1"), _row("user", "failed")]
    )
    assert _pairs(ctx) == [(Role.USER, "u1"), (Role.ASSISTANT, "a1")]
    assert ctx.trimmed == 1


def test_all_trailing_unanswered_user_turns_are_dropped():
    ctx = build_resume_context(
        [
            _row("user", "u1"),
            _row("assistant", "a1"),
            _row("user", "failed-1"),
            _row("user", "failed-2"),
        ]
    )
    assert _pairs(ctx) == [(Role.USER, "u1"), (Role.ASSISTANT, "a1")]


def test_conversation_of_only_unanswered_user_turns_is_empty():
    ctx = build_resume_context([_row("user", "failed")])
    assert ctx.messages == []
    assert ctx.trimmed == 1


def test_answered_earlier_failure_is_preserved_mid_conversation():
    # A failed turn followed by a successful one stays faithful to history.
    ctx = build_resume_context(
        [_row("user", "failed"), _row("user", "u2"), _row("assistant", "a2")]
    )
    assert [m.content for m in ctx.messages] == ["failed", "u2", "a2"]


def test_leading_assistant_message_is_dropped():
    ctx = build_resume_context(
        [_row("assistant", "orphan"), _row("user", "u1"), _row("assistant", "a1")]
    )
    assert _pairs(ctx) == [(Role.USER, "u1"), (Role.ASSISTANT, "a1")]
    assert ctx.messages[0].role == Role.USER


def test_token_budget_trims_oldest_first_then_leading_assistant():
    big = "word " * 400  # ~500 estimated tokens each
    rows = []
    for i in range(10):
        rows += [_row("user", f"u{i} {big}"), _row("assistant", f"a{i} {big}")]
    ctx = build_resume_context(rows, max_tokens=2000)
    assert estimate_prompt_tokens(ctx.messages) <= 2000
    # Three ~505-token messages fit (a8, u9, a9); the leading a8 is then
    # dropped so context starts on a user turn.
    assert [m.content.split()[0] for m in ctx.messages] == ["u9", "a9"]
    assert ctx.trimmed == 18


def test_default_token_budget_applies():
    big = "x" * (RESUME_MAX_TOKENS * 4)  # one message alone exceeds the budget
    ctx = build_resume_context(
        [
            _row("user", "u1"),
            _row("assistant", big),
            _row("user", "u2"),
            _row("assistant", "a2"),
        ]
    )
    assert _pairs(ctx) == [(Role.USER, "u2"), (Role.ASSISTANT, "a2")]
    assert estimate_prompt_tokens(ctx.messages) <= RESUME_MAX_TOKENS


def test_works_with_real_store_rows():
    with ConversationStore(":memory:") as store:
        conv = store.create_conversation(origin="cli", user_id=store.ensure_owner())
        store.append_message(conv.conversation_id, "user", "hi", surface="cli")
        store.append_message(conv.conversation_id, "assistant", "yo", surface="cli")
        ctx = build_resume_context(store.get_messages(conv.conversation_id))
    assert _pairs(ctx) == [(Role.USER, "hi"), (Role.ASSISTANT, "yo")]
