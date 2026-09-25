"""Flat, content-free tool provenance for durable assistant rows."""

from __future__ import annotations

import json
from pathlib import Path

from openjarvis.conversations import (
    MAX_METADATA_BYTES,
    ConversationRecorder,
    ConversationStore,
    tool_provenance_metadata,
)
from openjarvis.conversations.provenance import (
    MAX_TOOL_CALL_ENTRIES,
    MAX_TOOL_CALLS_CHARS,
)
from openjarvis.core.types import ToolResult
from openjarvis.tools.outcomes import ToolOutcome, annotate_tool_result

SECRET_ARG = "sk-SECRET-ARG-7f1e"
RAW_OUTPUT = "RAW-TOOL-OUTPUT-0d9b"


def _result(name: str, outcome: ToolOutcome, call_id: str, **kwargs) -> ToolResult:
    result = ToolResult(
        tool_name=name,
        content=RAW_OUTPUT,
        success=outcome is ToolOutcome.SUCCESS,
        metadata={"arguments": {"command": SECRET_ARG}, "nested": {"x": [1]}},
        **kwargs,
    )
    return annotate_tool_result(result, tool_call_id=call_id, outcome=outcome)


def test_no_tools_means_no_metadata():
    assert tool_provenance_metadata(None) is None
    assert tool_provenance_metadata([]) is None
    assert tool_provenance_metadata(["not a result"]) is None


def test_compact_deterministic_scalar_format():
    results = [
        _result("web_search", ToolOutcome.SUCCESS, "call_a1"),
        _result("shell_exec", ToolOutcome.DENIED, "call_b2"),
        _result("file_read", ToolOutcome.TIMEOUT, "toolu_01XYZ"),
    ]
    metadata = tool_provenance_metadata(results)
    assert metadata == {
        "tool_provenance": 1,
        "tools_used": True,
        "tool_call_count": 3,
        "tool_calls": (
            "web_search|success|call_a1;"
            "shell_exec|denied|call_b2;"
            "file_read|timeout|toolu_01XYZ"
        ),
        "tool_calls_truncated": False,
    }
    assert tool_provenance_metadata(results) == metadata
    assert all(isinstance(v, (str, int, bool)) for v in metadata.values())


def test_no_raw_arguments_or_results_leak():
    metadata = tool_provenance_metadata(
        [_result("shell_exec", ToolOutcome.SUCCESS, "call_a1")]
    )
    encoded = json.dumps(metadata)
    assert SECRET_ARG not in encoded
    assert RAW_OUTPUT not in encoded
    assert "arguments" not in encoded
    assert "nested" not in encoded


def test_hostile_names_and_ids_cannot_break_the_format():
    results = [
        _result(f"evil|name;{SECRET_ARG}\n", ToolOutcome.UNKNOWN_TOOL, "bad id;|"),
        _result("", ToolOutcome.ERROR, ""),
        _result("x" * 500, ToolOutcome.ERROR, "call_ok"),
        ToolResult(tool_name="legacy", content=RAW_OUTPUT, success=False),
    ]
    metadata = tool_provenance_metadata(results)
    entries = metadata["tool_calls"].split(";")
    assert len(entries) == 4
    for entry in entries:
        assert len(entry.split("|")) == 3
    assert entries[0].split("|")[1:] == ["unknown_tool", "-"]
    assert "\n" not in metadata["tool_calls"]
    assert entries[1] == "unnamed|error|-"
    assert entries[2] == f"{'x' * 64}|error|call_ok"
    assert entries[3] == "legacy|error|-"


def test_large_turns_are_bounded_and_flagged():
    results = [
        _result("t" * 64, ToolOutcome.SUCCESS, f"call_{i:0>100}") for i in range(200)
    ]
    metadata = tool_provenance_metadata(results)
    assert metadata["tool_call_count"] == 200
    assert metadata["tool_calls_truncated"] is True
    assert len(metadata["tool_calls"]) <= MAX_TOOL_CALLS_CHARS
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    assert len(encoded.encode()) <= MAX_METADATA_BYTES

    many = [_result("t", ToolOutcome.SUCCESS, f"c{i}") for i in range(100)]
    metadata = tool_provenance_metadata(many)
    assert len(metadata["tool_calls"].split(";")) == MAX_TOOL_CALL_ENTRIES
    assert metadata["tool_calls_truncated"] is True


def test_recorder_writes_metadata_on_assistant_row_only(tmp_path: Path):
    db_path = tmp_path / "conversations.db"
    recorder = ConversationRecorder(
        ConversationStore(db_path), surface="cli", origin="cli"
    )
    recorder.record_user("hi")
    metadata = tool_provenance_metadata(
        [_result("web_search", ToolOutcome.SUCCESS, "call_a1")]
    )
    assert recorder.record_assistant("answer", metadata=metadata) is not None
    assert recorder.disabled is False
    conversation_id = recorder.conversation_id
    recorder.close()

    with ConversationStore(db_path) as store:
        user, assistant = store.get_messages(conversation_id)
    assert user.metadata == {}
    assert assistant.metadata == metadata
