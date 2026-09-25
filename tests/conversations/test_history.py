"""Tests for rebuilding model context from durable conversation rows."""

from __future__ import annotations

from types import SimpleNamespace

from openjarvis.conversations import (
    CONTEXT_MAX_MESSAGES,
    CONTEXT_MAX_TOKENS,
    RESUME_MAX_MESSAGES,
    RESUME_MAX_TOKENS,
    ConversationStore,
    build_resume_context,
    normalize_context_messages,
)
from openjarvis.core.types import Message, Role
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


def test_unanswered_user_turn_is_dropped_mid_conversation():
    # Only the most recent of consecutive user turns got the reply; the
    # earlier failed one is left out of model context (it stays on disk).
    ctx = build_resume_context(
        [
            _row("user", "u1"),
            _row("assistant", "a1"),
            _row("user", "failed-1"),
            _row("user", "failed-2"),
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
    assert ctx.trimmed == 2


def test_leading_unanswered_user_turn_is_dropped():
    ctx = build_resume_context(
        [_row("user", "failed"), _row("user", "u2"), _row("assistant", "a2")]
    )
    assert [m.content for m in ctx.messages] == ["u2", "a2"]


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


# ---------------------------------------------------------------------------
# Shared working-context policy (resume + live chat)
# ---------------------------------------------------------------------------


def _msg(role, content):
    return Message(role=role, content=content)


def test_context_budget_constants_are_shared_with_resume():
    assert CONTEXT_MAX_MESSAGES == RESUME_MAX_MESSAGES == 40
    assert CONTEXT_MAX_TOKENS == RESUME_MAX_TOKENS == 4096


def test_normalize_caps_message_count_and_starts_on_user():
    messages = []
    for i in range(30):
        messages += [_msg(Role.USER, f"u{i}"), _msg(Role.ASSISTANT, f"a{i}")]
    kept = normalize_context_messages(messages)
    assert len(kept) == CONTEXT_MAX_MESSAGES
    assert kept[0].content == "u10" and kept[-1].content == "a29"
    kept = normalize_context_messages(messages, max_messages=5)
    # a27 would lead after the cap, so it is dropped too.
    assert [m.content for m in kept] == ["u28", "a28", "u29", "a29"]


def test_normalize_drops_trailing_unanswered_users_and_keeps_input():
    messages = [
        _msg(Role.USER, "u1"),
        _msg(Role.ASSISTANT, "a1"),
        _msg(Role.USER, "failed-1"),
        _msg(Role.USER, "failed-2"),
    ]
    snapshot = list(messages)
    assert [m.content for m in normalize_context_messages(messages)] == ["u1", "a1"]
    assert messages == snapshot


def test_normalize_applies_token_budget():
    big = "word " * 400
    messages = []
    for i in range(10):
        messages += [_msg(Role.USER, f"u{i} {big}"), _msg(Role.ASSISTANT, f"a{i}")]
    kept = normalize_context_messages(messages, max_tokens=2000)
    assert estimate_prompt_tokens(kept) <= 2000
    assert kept[0].role == Role.USER and kept[-1].role == Role.ASSISTANT


def test_resume_context_matches_shared_policy():
    rows = []
    for i in range(30):
        rows += [_row("user", f"u{i}"), _row("assistant", f"a{i}")]
    rows.append(_row("user", "failed"))
    ctx = build_resume_context(rows)
    expected = normalize_context_messages(
        [
            _msg(Role.USER if r.role == "user" else Role.ASSISTANT, r.content)
            for r in rows
        ]
    )
    assert _pairs(ctx) == [(m.role, m.content) for m in expected]
    assert ctx.trimmed == len(rows) - len(expected)
