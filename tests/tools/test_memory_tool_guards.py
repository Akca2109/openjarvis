"""Stage 3B: L2 memory-file tools need user confirmation and block secrets.

``memory_manage`` / ``user_profile_manage`` write MEMORY.md / USER.md, which
are loaded into every future system prompt. A write must therefore be
approved by the user (Stage 2.5 confirmation path) and must never persist
SECRET-tainted data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openjarvis.core.types import ToolCall
from openjarvis.security.taint import SINK_POLICY, TaintLabel
from openjarvis.tools._stubs import ToolExecutor
from openjarvis.tools.memory_manage import MemoryManageTool
from openjarvis.tools.user_profile_manage import UserProfileManageTool

_SECRET_OUTPUT = "db password=hunter2hunter2"


class _SecretSourceTool:
    """Local tool whose output carries a SECRET (e.g. a file it read)."""

    is_local = True

    @property
    def spec(self):
        from openjarvis.tools._stubs import ToolSpec

        return ToolSpec(name="secret_source", description="", parameters={})

    def execute(self, **params):
        from openjarvis.core.types import ToolResult

        return ToolResult(tool_name="secret_source", content=_SECRET_OUTPUT)


@pytest.fixture(params=["memory", "profile"])
def l2_tool(request, tmp_path):
    if request.param == "memory":
        return MemoryManageTool(memory_path=tmp_path / "MEMORY.md")
    return UserProfileManageTool(user_path=tmp_path / "USER.md")


def _path(tool) -> Path:
    return getattr(tool, "_memory_path", None) or tool._user_path


def _add(tool, entry="Prefers Vim"):
    arguments = {"action": "add", "entry": entry}
    return ToolCall(id="1", name=tool.spec.name, arguments=json.dumps(arguments))


def test_i_j_spec_requires_confirmation(l2_tool):
    assert l2_tool.spec.requires_confirmation is True


def test_i_j_write_fails_closed_without_a_confirmation_callback(l2_tool):
    executor = ToolExecutor([l2_tool])

    result = executor.execute(_add(l2_tool))

    assert result.success is False
    assert "requires confirmation" in result.content
    assert not _path(l2_tool).exists()


def test_i_j_denied_confirmation_does_not_write(l2_tool):
    prompts: list[str] = []

    def deny(prompt: str) -> bool:
        prompts.append(prompt)
        return False

    executor = ToolExecutor([l2_tool], interactive=True, confirm_callback=deny)

    result = executor.execute(_add(l2_tool))

    assert result.success is False
    assert "denied by user" in result.content
    assert not _path(l2_tool).exists()
    # The user is shown what would be written.
    assert len(prompts) == 1 and "Prefers Vim" in prompts[0]


def test_i_j_approved_confirmation_writes(l2_tool):
    executor = ToolExecutor(
        [l2_tool], interactive=True, confirm_callback=lambda prompt: True
    )

    result = executor.execute(_add(l2_tool))

    assert result.success is True
    assert "Prefers Vim" in _path(l2_tool).read_text()


def test_k_sink_policy_lists_memory_file_tools():
    assert TaintLabel.SECRET in SINK_POLICY["memory_manage"]
    assert TaintLabel.SECRET in SINK_POLICY["user_profile_manage"]


def test_k_secret_tainted_session_blocks_write_before_confirmation(l2_tool):
    prompts: list[str] = []

    def approve(prompt: str) -> bool:
        prompts.append(prompt)
        return True

    executor = ToolExecutor(
        [_SecretSourceTool(), l2_tool], interactive=True, confirm_callback=approve
    )
    read = executor.execute(ToolCall(id="0", name="secret_source", arguments="{}"))
    assert read.success is True

    result = executor.execute(_add(l2_tool, "db password=hunter2hunter2"))

    assert result.success is False
    assert "Taint violation" in result.content
    assert prompts == []  # blocked before the user is even asked
    assert not _path(l2_tool).exists()


def test_k_secret_seeded_conversation_blocks_write(l2_tool):
    executor = ToolExecutor(
        [l2_tool], interactive=True, confirm_callback=lambda prompt: True
    )
    executor.begin_session(["my api token: " + "x" * 12])

    result = executor.execute(_add(l2_tool))

    assert result.success is False
    assert "Taint violation" in result.content
    assert not _path(l2_tool).exists()
