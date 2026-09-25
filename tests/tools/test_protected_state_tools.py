"""Generic tools cannot modify protected OpenJarvis state (Stage 3D)."""

from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import TTSRegistry
from openjarvis.core.types import ToolCall
from openjarvis.security.protected_state import clear_protected_config
from openjarvis.security.taint import TaintLabel, TaintSet
from openjarvis.tools._stubs import ToolExecutor
from openjarvis.tools.apply_patch import ApplyPatchTool
from openjarvis.tools.db_query import DatabaseQueryTool
from openjarvis.tools.file_write import FileWriteTool
from openjarvis.tools.image_tool import ImageGenerateTool
from openjarvis.tools.memory_manage import MemoryManageTool
from openjarvis.tools.outcomes import (
    OUTCOME_KEY,
    TOOL_CALL_ID_KEY,
    ToolOutcome,
    tool_result_outcome,
)
from openjarvis.tools.skill_manage import SkillManageTool
from openjarvis.tools.text_to_speech import TextToSpeechTool
from openjarvis.tools.user_profile_manage import UserProfileManageTool

DENIED = ToolOutcome.PROTECTED_TARGET_DENIED.value
SECRET = "SECRET-CONTENT-MARKER-5e0c"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("OPENJARVIS_HOME", str(home))
    monkeypatch.delenv("OPENJARVIS_CONFIG", raising=False)
    clear_protected_config()
    (home / "MEMORY.md").write_text("# Memory\n- original\n")
    (home / "USER.md").write_text("# User\n")
    (home / "SOUL.md").write_text("# Soul\n")
    (home / "config.toml").write_text("[engine]\n")
    yield home
    clear_protected_config()


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.chdir(ws)
    return ws


@pytest.fixture
def bus() -> EventBus:
    return EventBus(record_history=True)


def _call(executor: ToolExecutor, name: str, call_id: str = "call_p1", **args):
    return executor.execute(ToolCall(id=call_id, name=name, arguments=json.dumps(args)))


def _write(executor, path, content=SECRET, **extra):
    return _call(executor, "file_write", path=str(path), content=content, **extra)


def _snapshot(home: Path) -> dict:
    return {p.relative_to(home): p.read_bytes() for p in home.rglob("*") if p.is_file()}


def _assert_denied(result, category: str, call_id: str = "call_p1"):
    assert result.success is False
    assert result.metadata[OUTCOME_KEY] == DENIED
    assert result.metadata[TOOL_CALL_ID_KEY] == call_id
    assert result.metadata["protected_target"] == category
    assert tool_result_outcome(result) == DENIED


# ---------------------------------------------------------------------------
# file_write
# ---------------------------------------------------------------------------


# H
def test_file_write_ordinary_workspace_succeeds(home, ws, bus):
    executor = ToolExecutor([FileWriteTool()], bus)
    result = _write(executor, ws / "notes.md", "hello")
    assert result.success, result.content
    assert (ws / "notes.md").read_text() == "hello"
    assert result.metadata[OUTCOME_KEY] == "success"
    # Names that merely look like OpenJarvis files are fine outside the home.
    for name in ("MEMORY.md", "config.toml", "app.db"):
        assert _write(executor, ws / name, "x").success


# I, J, K, L, M
@pytest.mark.parametrize(
    "rel, category",
    [
        ("MEMORY.md", "memory"),
        ("USER.md", "profile"),
        ("SOUL.md", "persona"),
        ("config.toml", "config"),
        ("memory.db", "runtime_store"),
        ("conversations.db", "runtime_store"),
        ("personas/work/MEMORY.md", "memory"),
        ("tools/descriptions.toml", "instructions"),
        ("skills/evil/SKILL.md", "instructions"),
    ],
)
def test_file_write_protected_targets_denied(home, ws, bus, rel, category):
    before = _snapshot(home)
    executor = ToolExecutor([FileWriteTool()], bus)
    for mode in ("write", "append"):
        result = _write(executor, home / rel, mode=mode, create_dirs=True)
        _assert_denied(result, category)
    assert _snapshot(home) == before
    assert not (home / rel).parent.exists() or (home / rel).exists() == (
        rel in {"MEMORY.md", "USER.md", "SOUL.md", "config.toml"}
    )


def test_file_write_guides_model_to_dedicated_tools(home, ws, bus):
    executor = ToolExecutor([FileWriteTool()], bus)
    assert "memory_manage" in _write(executor, home / "MEMORY.md").content
    assert "user_profile_manage" in _write(executor, home / "USER.md").content
    assert "_manage" not in _write(executor, home / "SOUL.md").content


# N
def test_file_write_symlink_to_protected_denied(home, ws, bus):
    (ws / "harmless.md").symlink_to(home / "MEMORY.md")
    (ws / "state").symlink_to(home, target_is_directory=True)
    executor = ToolExecutor([FileWriteTool()], bus)
    _assert_denied(_write(executor, ws / "harmless.md"), "memory")
    _assert_denied(_write(executor, "state/USER.md"), "profile")
    _assert_denied(_write(executor, "state/sub/../memory.db"), "runtime_store")
    assert SECRET not in (home / "MEMORY.md").read_text()


def test_file_write_direct_call_also_denied(home, ws):
    # Defense in depth: the tool enforces the policy without ToolExecutor.
    result = FileWriteTool().execute(path=str(home / "MEMORY.md"), content=SECRET)
    assert result.success is False
    assert result.metadata == {"protected_target": "memory"}
    assert SECRET not in (home / "MEMORY.md").read_text()


# O
def test_denial_is_not_bypassed_by_confirmation_or_taint(home, ws, bus):
    confirm = MagicMock(return_value=True)
    executor = ToolExecutor(
        [FileWriteTool()], bus, interactive=True, confirm_callback=confirm
    )
    result = _write(executor, home / "SOUL.md", call_id="call_soul")
    _assert_denied(result, "persona", "call_soul")
    confirm.assert_not_called()

    # Same result whether or not the session carries taint.
    executor.begin_session()
    with executor._taint_lock:
        executor._session_taint = TaintSet(frozenset({TaintLabel.PII}))
    _assert_denied(_write(executor, home / "SOUL.md"), "persona")


def test_existing_sensitive_and_allowed_dir_checks_preserved(home, ws, bus):
    executor = ToolExecutor([FileWriteTool(allowed_dirs=[str(ws / "only")])], bus)
    result = _write(executor, ws / ".env")
    assert result.success is False and "sensitive" in result.content
    assert result.metadata[OUTCOME_KEY] == "error"
    result = _write(executor, ws / "outside.txt")
    assert result.success is False and "outside allowed" in result.content


# ---------------------------------------------------------------------------
# apply_patch
# ---------------------------------------------------------------------------

_PATCH = "--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n # Memory\n-- original\n+- injected\n"


def _patch(executor, path, **extra):
    return _call(executor, "apply_patch", patch=_PATCH, path=str(path), **extra)


# P
def test_apply_patch_ordinary_workspace_succeeds(home, ws, bus):
    target = ws / "notes.md"
    target.write_text("# Memory\n- original\n")
    result = _patch(ToolExecutor([ApplyPatchTool()], bus), target)
    assert result.success, result.content
    assert target.read_text() == "# Memory\n- injected\n"
    assert (ws / "notes.md.bak").read_text() == "# Memory\n- original\n"


# Q
def test_apply_patch_protected_target_denied(home, ws, bus):
    executor = ToolExecutor([ApplyPatchTool()], bus)
    before = _snapshot(home)
    _assert_denied(_patch(executor, home / "MEMORY.md"), "memory")
    _assert_denied(_patch(executor, home / "MEMORY.md", backup=False), "memory")
    header_only = _PATCH.replace("+++ b/f", f"+++ {home / 'MEMORY.md'}")
    result = _call(executor, "apply_patch", patch=header_only)
    _assert_denied(result, "memory")
    assert _snapshot(home) == before


# R
def test_apply_patch_symlink_protected_target_denied(home, ws, bus):
    (ws / "link.md").symlink_to(home / "MEMORY.md")
    before = _snapshot(home)
    result = _patch(ToolExecutor([ApplyPatchTool()], bus), ws / "link.md")
    _assert_denied(result, "memory")
    assert _snapshot(home) == before


# S
def test_apply_patch_backup_cannot_bypass_policy(home, ws, bus):
    target = ws / "notes.md"
    target.write_text("# Memory\n- original\n")
    (ws / "notes.md.bak").symlink_to(home / "USER.md")
    before = _snapshot(home)
    executor = ToolExecutor([ApplyPatchTool()], bus)
    _assert_denied(_patch(executor, target), "profile")
    assert _snapshot(home) == before
    assert target.read_text() == "# Memory\n- original\n"
    # Without a backup the workspace file itself is patchable.
    assert _patch(executor, target, backup=False).success
    assert _snapshot(home) == before

    # A backup that would be created inside instruction state is refused.
    other = ws / "other.md"
    other.write_text("# Memory\n- original\n")
    (home / "skills").mkdir()
    (ws / "other.md.bak").symlink_to(home / "skills" / "planted.md")
    _assert_denied(_patch(executor, other), "instructions")
    assert not (home / "skills" / "planted.md").exists()


def test_apply_patch_direct_call_also_denied(home, ws):
    # Defense in depth: the tool enforces the policy without ToolExecutor,
    # for the target itself and for a backup aliased into protected state.
    before = _snapshot(home)
    result = ApplyPatchTool().execute(patch=_PATCH, path=str(home / "MEMORY.md"))
    assert result.success is False
    assert result.metadata == {"protected_target": "memory"}

    target = ws / "notes.md"
    target.write_text("# Memory\n- original\n")
    (ws / "notes.md.bak").symlink_to(home / "SOUL.md")
    result = ApplyPatchTool().execute(patch=_PATCH, path=str(target))
    assert result.success is False
    assert result.metadata == {"protected_target": "persona"}
    assert target.read_text() == "# Memory\n- original\n"
    assert _snapshot(home) == before


# ---------------------------------------------------------------------------
# skill_manage
# ---------------------------------------------------------------------------


@pytest.fixture
def skills(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    return d


# T
def test_skill_manage_valid_operations(skills):
    import tomllib

    tool = SkillManageTool(skills_dir=skills)
    result = tool.execute(
        action="create",
        name="api-health_2",
        description="Check API health",
        steps=[
            {
                "tool_name": "http_request",
                "arguments_template": '{"url": "{endpoint}/health"}',
                "output_key": "status",
            },
            {"tool_name": "calculator"},
        ],
    )
    assert result.success, result.content
    data = tomllib.loads((skills / "api-health_2.toml").read_text())
    assert data == {
        "skill": {
            "name": "api-health_2",
            "description": "Check API health",
            "steps": [
                {
                    "tool_name": "http_request",
                    "arguments_template": '{"url": "{endpoint}/health"}',
                    "output_key": "status",
                },
                {"tool_name": "calculator"},
            ],
        }
    }
    assert "api-health_2" in tool.execute(action="list").content
    loaded = tool.execute(action="load", name="api-health_2")
    assert "Check API health" in loaded.content
    assert tool.execute(action="delete", name="api-health_2").success


# U, V
@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "..",
        "a/../../b",
        "sub/skill",
        "sub\\skill",
        "/abs/skill",
        "C:\\skill",
        ".hidden",
        "-flag",
        "name.toml",
        "x" * 65,
        "",
        "evil\n[skill]",
        42,
    ],
)
def test_skill_manage_rejects_unsafe_names(skills, tmp_path, name):
    tool = SkillManageTool(skills_dir=skills)
    for action in ("create", "load", "delete"):
        result = tool.execute(
            action=action, name=name, description="d", steps=[{"tool_name": "x"}]
        )
        assert result.success is False
    assert list(skills.iterdir()) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["skills"]


# W
def test_skill_manage_toml_payload_cannot_inject_keys(skills):
    import tomllib

    payload = 'x"\nevil = true\n[injected]\nkey = "v'
    tool = SkillManageTool(skills_dir=skills)
    result = tool.execute(
        action="create",
        name="safe",
        description=payload,
        steps=[
            {
                "tool_name": payload,
                "arguments_template": "a'\n[[skill.steps]]\ntool_name = 'shell_exec",
                "output_key": payload,
                "unexpected": "ignored",
            }
        ],
    )
    assert result.success, result.content
    data = tomllib.loads((skills / "safe.toml").read_text())
    assert set(data) == {"skill"}
    assert set(data["skill"]) == {"name", "description", "steps"}
    assert data["skill"]["description"] == payload
    (step,) = data["skill"]["steps"]
    assert set(step) == {"tool_name", "arguments_template", "output_key"}
    assert step["tool_name"] == payload


def test_skill_manage_rejects_malformed_steps(skills):
    tool = SkillManageTool(skills_dir=skills)
    assert not tool.execute(action="create", name="s", steps="nope").success
    assert not tool.execute(action="create", name="s", steps=["x"]).success
    assert list(skills.iterdir()) == []


# X
def test_skill_manage_path_cannot_escape_skills_dir(skills, home):
    (skills / "sneaky.toml").symlink_to(home / "MEMORY.md")
    tool = SkillManageTool(skills_dir=skills)
    assert not tool.execute(action="load", name="sneaky").success
    assert not tool.execute(action="delete", name="sneaky").success
    assert not tool.execute(
        action="create", name="sneaky", description=SECRET, steps=[]
    ).success
    assert (home / "MEMORY.md").exists()
    assert SECRET not in (home / "MEMORY.md").read_text()


# ---------------------------------------------------------------------------
# db_query
# ---------------------------------------------------------------------------


def _make_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE facts (v TEXT)")
    conn.execute("INSERT INTO facts VALUES ('orig')")
    conn.commit()
    conn.close()


def _rows(path: Path) -> list:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT v FROM facts").fetchall()
    finally:
        conn.close()


def _db(executor, db_path, query, **extra):
    return _call(executor, "db_query", query=query, db_path=str(db_path), **extra)


# Y
def test_db_query_read_only_protected_db_still_allowed(home, ws, bus):
    _make_db(home / "memory.db")
    executor = ToolExecutor([DatabaseQueryTool()], bus)
    result = _db(executor, home / "memory.db", "SELECT v FROM facts")
    assert result.success, result.content
    assert "orig" in result.content


# Z
def test_db_query_write_to_user_db_works(home, ws, bus):
    _make_db(ws / "app.db")
    executor = ToolExecutor([DatabaseQueryTool()], bus)
    result = _db(
        executor, ws / "app.db", "INSERT INTO facts VALUES ('new')", read_only=False
    )
    assert result.success, result.content
    assert _rows(ws / "app.db") == [("orig",), ("new",)]


# AA
@pytest.mark.parametrize(
    "name", ["memory.db", "conversations.db", "audit.db", "approvals.db"]
)
def test_db_query_write_to_protected_db_denied(home, ws, bus, name):
    _make_db(home / name)
    executor = ToolExecutor([DatabaseQueryTool()], bus)
    query = "INSERT INTO facts VALUES ('evil')"
    (ws / "alias.db").symlink_to(home / name)
    uri = f"file:{home / name}?mode=rw"
    for target in (home / name, ws / "alias.db", uri):
        result = _db(executor, target, query, read_only=False)
        _assert_denied(result, "runtime_store")
    assert _rows(home / name) == [("orig",)]


def test_db_query_writable_attach_of_protected_db_refused(home, ws, bus):
    _make_db(home / "memory.db")
    tool = DatabaseQueryTool()
    for query in (
        f"ATTACH DATABASE '{home / 'memory.db'}' AS m",
        f"VACUUM INTO '{home / 'fresh.db'}'",
    ):
        result = tool.execute(query=query, read_only=False)
        assert result.success is False, query
    assert not (home / "fresh.db").exists()
    # Attaching an ordinary database still works.
    _make_db(ws / "other.db")
    attach = "ATTACH DATABASE 'other.db' AS o"
    assert tool.execute(query=attach, read_only=False).success


# ---------------------------------------------------------------------------
# image_generate / text_to_speech
# ---------------------------------------------------------------------------


# AB
def test_image_output_into_protected_state_denied(home, ws, bus):
    executor = ToolExecutor([ImageGenerateTool()], bus)
    with patch.dict(sys.modules, {"openai": MagicMock()}):
        for rel, category in (("SOUL.md", "persona"), ("skills/x.png", "instructions")):
            result = _call(
                executor, "image_generate", prompt="cat", output_path=str(home / rel)
            )
            _assert_denied(result, category)
    assert (home / "SOUL.md").read_text() == "# Soul\n"


def _fake_openai(url: str) -> types.ModuleType:
    module = types.ModuleType("openai")
    client = MagicMock()
    client.images.generate.return_value = MagicMock(data=[MagicMock(url=url)])
    module.OpenAI = MagicMock(return_value=client)  # type: ignore[attr-defined]
    return module


# AD (image)
def test_image_output_to_workspace_still_works(home, ws, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    response = MagicMock(content=b"PNGDATA")
    with (
        patch.dict(sys.modules, {"openai": _fake_openai("https://img/x.png")}),
        patch("httpx.get", return_value=response),
    ):
        result = ImageGenerateTool().execute(prompt="cat", output_path="out.png")
    assert result.success, result.content
    assert (ws / "out.png").read_bytes() == b"PNGDATA"


class _FakeAudio:
    format = "mp3"
    voice_id = "v1"
    duration_seconds = 0.1

    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"ID3")


class _FakeTTS:
    def synthesize(self, text: str, **kwargs):
        return _FakeAudio()


@pytest.fixture
def fake_tts():
    TTSRegistry.register_value("fake_tts_3d", _FakeTTS)
    yield "fake_tts_3d"


# AC
def test_tts_output_into_protected_state_denied(home, ws, bus, fake_tts):
    executor = ToolExecutor([TextToSpeechTool()], bus)
    (ws / "voice").symlink_to(home / "skills", target_is_directory=True)
    for out_dir, category in (
        (home / "skills" / "evil", "instructions"),
        (home / "personas" / "work", "persona"),
        (ws / "voice", "instructions"),
    ):
        result = _call(
            executor,
            "text_to_speech",
            text="hi",
            backend=fake_tts,
            voice_id="v1",
            output_dir=str(out_dir),
        )
        _assert_denied(result, category)
    assert not (home / "skills").exists()
    assert not (home / "personas").exists()


# AD (tts)
def test_tts_output_elsewhere_still_works(home, ws, bus, fake_tts):
    executor = ToolExecutor([TextToSpeechTool()], bus)
    result = _call(
        executor,
        "text_to_speech",
        text="hi",
        backend=fake_tts,
        voice_id="v1",
        output_dir=str(ws / "audio"),
    )
    assert result.success, result.content
    assert Path(result.content).parent == ws / "audio"
    assert Path(result.content).read_bytes() == b"ID3"


# ---------------------------------------------------------------------------
# Dedicated memory / profile tools
# ---------------------------------------------------------------------------


# AE, AF
@pytest.mark.parametrize(
    "tool, name",
    [
        (MemoryManageTool, "memory_manage"),
        (UserProfileManageTool, "user_profile_manage"),
    ],
)
def test_dedicated_tools_still_require_confirmation(home, ws, bus, tool, name):
    from openjarvis.security.taint import check_taint

    assert tool().spec.requires_confirmation is True
    assert check_taint(name, TaintSet(frozenset({TaintLabel.SECRET}))) is not None

    before = _snapshot(home)
    # No callback -> denied; callback that says no -> denied.
    for executor in (
        ToolExecutor([tool()], bus),
        ToolExecutor([tool()], bus, interactive=True, confirm_callback=lambda _: False),
    ):
        result = _call(executor, name, action="add", entry=SECRET)
        assert result.metadata[OUTCOME_KEY] == ToolOutcome.DENIED.value
    assert _snapshot(home) == before

    approved = ToolExecutor(
        [tool()], bus, interactive=True, confirm_callback=lambda _: True
    )
    assert _call(approved, name, action="add", entry="approved fact").success


def test_dedicated_tools_disabled_for_persona_none(home):
    for tool in (MemoryManageTool(memory_path=""), UserProfileManageTool(user_path="")):
        result = tool.execute(action="add", entry="x")
        assert result.success is False
        assert "disabled" in result.content


# ---------------------------------------------------------------------------
# Events and provenance
# ---------------------------------------------------------------------------


# AH, AI
def test_denial_emits_safe_tool_call_blocked(home, ws, bus):
    executor = ToolExecutor([FileWriteTool()], bus)
    result = _write(executor, home / "MEMORY.md", call_id="call_evt")

    assert [e.event_type for e in bus.history] == [EventType.TOOL_CALL_BLOCKED]
    (event,) = bus.history
    assert event.data == {
        "tool": "file_write",
        "tool_call_id": "call_evt",
        "outcome": DENIED,
        "agent": "",
        "protected_target": "memory",
    }
    for text in (json.dumps(event.data), result.content, json.dumps(result.metadata)):
        assert str(home) not in text
        assert "MEMORY.md" not in text
        assert SECRET not in text
    assert "arguments" not in result.metadata


def test_provenance_summary_of_denial_is_safe(home, ws, bus):
    from openjarvis.conversations import tool_provenance_metadata

    executor = ToolExecutor([FileWriteTool()], bus)
    denied = _write(executor, home / "USER.md", call_id="call_den")
    metadata = tool_provenance_metadata([denied])
    assert metadata["tool_calls"] == f"file_write|{DENIED}|call_den"
    dumped = json.dumps(metadata)
    for marker in (str(home), "USER.md", SECRET, '"protected_target"'):
        assert marker not in dumped
