"""Protected persistent-state classifier (Stage 3D)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from openjarvis.core.config import JarvisConfig, MemoryFilesConfig
from openjarvis.security.protected_state import (
    ProtectedCategory,
    classify_protected_target,
    clear_protected_config,
    protected_target_message,
    register_protected_config,
)

C = ProtectedCategory


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("OPENJARVIS_HOME", str(home))
    monkeypatch.delenv("OPENJARVIS_CONFIG", raising=False)
    clear_protected_config()
    yield home
    clear_protected_config()


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    monkeypatch.chdir(ws)
    return ws


# A
def test_default_memory_files_classified(home: Path):
    assert classify_protected_target(home / "MEMORY.md") is C.MEMORY
    assert classify_protected_target(home / "memory_facts.jsonl") is C.MEMORY
    assert classify_protected_target(home / "USER.md") is C.PROFILE
    assert classify_protected_target(home / "SOUL.md") is C.PERSONA
    # Missing files are protected too: creating them is a modification.
    assert not (home / "MEMORY.md").exists()


# B
def test_persona_specific_files_classified(home: Path):
    base = home / "personas" / "work"
    assert classify_protected_target(base / "MEMORY.md") is C.MEMORY
    assert classify_protected_target(base / "USER.md") is C.PROFILE
    assert classify_protected_target(base / "SOUL.md") is C.PERSONA
    assert classify_protected_target(base / "notes.md") is C.PERSONA


def test_registered_custom_memory_files_classified(home: Path, tmp_path: Path):
    elsewhere = tmp_path / "elsewhere"
    config = JarvisConfig()
    config.memory_files = MemoryFilesConfig(
        soul_path=str(elsewhere / "soul.md"),
        memory_path=str(elsewhere / "mem.md"),
        user_path=str(elsewhere / "me.md"),
    )
    assert classify_protected_target(elsewhere / "mem.md") is None
    register_protected_config(config)
    assert classify_protected_target(elsewhere / "mem.md") is C.MEMORY
    assert classify_protected_target(elsewhere / "me.md") is C.PROFILE
    assert classify_protected_target(elsewhere / "soul.md") is C.PERSONA


def test_registered_persona_override_classified(home: Path):
    register_protected_config(
        JarvisConfig(), memory_files=MemoryFilesConfig(persona_name="work")
    )
    base = home / "personas" / "work"
    assert classify_protected_target(base / "MEMORY.md") is C.MEMORY


# C
def test_config_files_classified(home: Path, tmp_path: Path, monkeypatch):
    assert classify_protected_target(home / "config.toml") is C.CONFIG
    custom = tmp_path / "custom" / "jarvis.toml"
    assert classify_protected_target(custom) is None
    monkeypatch.setenv("OPENJARVIS_CONFIG", str(custom))
    assert classify_protected_target(custom) is C.CONFIG


# D
def test_capability_policy_classified(home: Path, tmp_path: Path):
    policy = tmp_path / "policies" / "caps.json"
    config = JarvisConfig()
    config.security.capabilities.policy_path = str(policy)
    assert classify_protected_target(policy) is None
    assert classify_protected_target(policy, config) is C.SECURITY
    register_protected_config(config)
    assert classify_protected_target(policy) is C.SECURITY


# E
@pytest.mark.parametrize(
    "name",
    [
        "memory.db",
        "conversations.db",
        "audit.db",
        "approvals.db",
        "memory.db-wal",
        "sub/dir/other.sqlite3",
    ],
)
def test_openjarvis_databases_classified(home: Path, name: str):
    assert classify_protected_target(home / name) is C.RUNTIME_STORE


def test_instruction_and_security_state_classified(home: Path):
    assert classify_protected_target(home / "tools" / "descriptions.toml") is (
        C.INSTRUCTIONS
    )
    assert classify_protected_target(home / "skills" / "x" / "skill.toml") is (
        C.INSTRUCTIONS
    )
    assert classify_protected_target(home / "prompts" / "personas" / "p.md") is (
        C.INSTRUCTIONS
    )
    assert classify_protected_target(home / ".vault_key") is C.SECURITY
    assert classify_protected_target(home / "connectors" / "gmail.json") is (C.SECURITY)


def test_workspace_skill_manifests_classified(home: Path, workspace: Path):
    assert classify_protected_target("skills/demo/SKILL.md") is C.INSTRUCTIONS
    assert classify_protected_target("skills/demo/skill.toml") is C.INSTRUCTIONS
    assert classify_protected_target("skills/demo/helper.py") is None


# F
def test_ordinary_workspace_files_not_classified(home: Path, workspace: Path):
    names = ("notes.md", "MEMORY.md", "USER.md", "SOUL.md", "config.toml")
    for name in (*names, "app.db", "src/x.py", "personas/w/MEMORY.md"):
        assert classify_protected_target(workspace / name) is None
        assert classify_protected_target(name) is None
    assert classify_protected_target(home) is None
    assert classify_protected_target(home / "logs" / "server.log") is None
    assert classify_protected_target("") is None
    assert classify_protected_target(123) is None  # type: ignore[arg-type]


# G
def test_symlink_alias_to_protected_target_classified(home: Path, workspace: Path):
    (home / "MEMORY.md").write_text("mem\n")
    link = workspace / "innocent.md"
    link.symlink_to(home / "MEMORY.md")
    assert classify_protected_target(link) is C.MEMORY

    # A chain of links, a dangling link, and a linked parent directory.
    chain = workspace / "chain.md"
    chain.symlink_to(link)
    assert classify_protected_target(chain) is C.MEMORY
    dangling = workspace / "future.md"
    dangling.symlink_to(home / "USER.md")
    assert classify_protected_target(dangling) is C.PROFILE
    dir_link = workspace / "state"
    dir_link.symlink_to(home, target_is_directory=True)
    assert classify_protected_target(dir_link / "audit.db") is C.RUNTIME_STORE
    assert classify_protected_target("state/../state/SOUL.md") is C.PERSONA


def test_hard_link_to_protected_target_classified(home: Path, workspace: Path):
    (home / "USER.md").write_text("profile\n")
    alias = workspace / "copy.md"
    try:
        os.link(home / "USER.md", alias)
    except OSError:
        pytest.skip("hard links unsupported")
    assert classify_protected_target(alias) is C.PROFILE


def test_symlinked_home_is_matched(tmp_path: Path, monkeypatch, workspace: Path):
    real = tmp_path / "real-home"
    real.mkdir()
    alias = tmp_path / "alias-home"
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("OPENJARVIS_HOME", str(alias))
    clear_protected_config()
    assert classify_protected_target(real / "MEMORY.md") is C.MEMORY
    assert classify_protected_target(alias / "memory.db") is C.RUNTIME_STORE


def test_denial_message_names_category_not_path(home: Path):
    message = protected_target_message(C.MEMORY, "file_write")
    assert "memory_manage" in message
    assert str(home) not in message
    assert "user_profile_manage" in protected_target_message(C.PROFILE, "x")
    assert "_manage" not in protected_target_message(C.CONFIG, "x")
