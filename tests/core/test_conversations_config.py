"""Tests for the ``[conversations]`` config section."""

from __future__ import annotations

from openjarvis.core.config import ConversationsConfig, JarvisConfig, load_config
from openjarvis.core.paths import get_config_dir


def test_defaults_enabled_under_config_dir():
    cfg = JarvisConfig()
    assert isinstance(cfg.conversations, ConversationsConfig)
    assert cfg.conversations.enabled is True
    assert cfg.conversations.db_path == str(get_config_dir() / "conversations.db")


def test_default_path_follows_openjarvis_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path / "ojhome"))
    expected = (tmp_path / "ojhome").resolve() / "conversations.db"
    assert ConversationsConfig().db_path == str(expected)


def test_toml_section_is_loaded(tmp_path):
    config_path = tmp_path / "config.toml"
    custom = tmp_path / "elsewhere.db"
    config_path.write_text(
        f'[conversations]\nenabled = false\ndb_path = "{custom.as_posix()}"\n',
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg.conversations.enabled is False
    assert cfg.conversations.db_path == custom.as_posix()
