"""External analytics must be opt-in; local telemetry is unaffected."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from openjarvis.analytics import is_analytics_enabled
from openjarvis.core.config import (
    AnalyticsConfig,
    JarvisConfig,
    TelemetryConfig,
    TracesConfig,
    generate_default_toml,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENJARVIS_HOME", str(tmp_path))
    monkeypatch.delenv("OPENJARVIS_CONFIG", raising=False)
    return tmp_path


def _load(home: Path, body: str | None):
    path = home / "config.toml"
    if body is not None:
        path.write_text(body, encoding="utf-8")
    return load_config(path)


def test_dataclass_default_is_disabled():
    assert AnalyticsConfig().enabled is False
    assert JarvisConfig().analytics.enabled is False
    assert is_analytics_enabled(AnalyticsConfig()) is False


@pytest.mark.parametrize(
    "body",
    [None, "", "[telemetry]\nenabled = true\n", "[analytics]\n"],
    ids=["missing-file", "empty-file", "no-analytics-section", "empty-section"],
)
def test_fresh_config_is_disabled(home, body):
    cfg = _load(home, body)
    assert is_analytics_enabled(cfg.analytics) is False


def test_generated_default_config_is_disabled(home):
    # Covers `jarvis init` / quickstart / `_bootstrap --write-config`,
    # which all write this template.
    cfg = _load(home, generate_default_toml(JarvisConfig().hardware))
    assert is_analytics_enabled(cfg.analytics) is False


def test_explicit_true_enables(home):
    cfg = _load(home, "[analytics]\nenabled = true\n")
    assert is_analytics_enabled(cfg.analytics) is True


def test_explicit_false_disables(home):
    cfg = _load(home, "[analytics]\nenabled = false\n")
    assert is_analytics_enabled(cfg.analytics) is False


@pytest.mark.parametrize("analytics", [None, "true", "false"])
def test_local_telemetry_independent_of_analytics(home, analytics):
    body = "" if analytics is None else f"[analytics]\nenabled = {analytics}\n"
    cfg = _load(home, body)
    assert cfg.telemetry.enabled is TelemetryConfig().enabled is True
    assert cfg.traces.enabled is TracesConfig().enabled is True


def test_identity_route_disabled_by_default(home):
    from openjarvis.server.analytics_routes import get_identity

    identity = get_identity()
    assert identity.enabled is False
    assert identity.key == ""
    assert identity.anon_id == ""
    assert not (home / "anon_id").exists()


def test_identity_route_enabled_on_opt_in(home):
    from openjarvis.server.analytics_routes import get_identity

    (home / "config.toml").write_text("[analytics]\nenabled = true\n")
    identity = get_identity()
    assert identity.enabled is True
    assert identity.key == AnalyticsConfig().key
    assert identity.anon_id


# ---- install.sh beacon opt-in ----------------------------------------------

_INSTALL_SH = (REPO_ROOT / "scripts/install/install.sh").read_text(encoding="utf-8")


def _shell_snippet() -> str:
    """The installer's home capture plus its analytics opt-in functions."""
    home = re.search(
        r"^OPENJARVIS_HOME_FROM_ENV=.*\n^OPENJARVIS_HOME=.*\n", _INSTALL_SH, re.M
    )
    funcs = [
        re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", _INSTALL_SH, re.S | re.M)
        for name in ("analytics_config_path", "analytics_enabled")
    ]
    assert home is not None and all(funcs)
    return home.group(0) + "".join(f.group(0) for f in funcs)


def _awk_variants() -> list[str]:
    variants = []
    if shutil.which("gawk"):
        variants += ["gawk", "gawk --posix"]
    if shutil.which("mawk"):
        variants.append("mawk")
    return variants or ["awk"]


@pytest.fixture(params=_awk_variants())
def installer(request, tmp_path):
    """Run the installer's opt-in check under a given awk implementation."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    shim = bindir / "awk"
    shim.write_text(f'#!/bin/sh\nexec {request.param} "$@"\n')
    shim.chmod(0o755)
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    def run(env: dict[str, str] | None = None) -> bool:
        full_env = {"HOME": str(fake_home), "PATH": f"{bindir}:/usr/bin:/bin"}
        full_env.update(env or {})
        script = "set -euo pipefail\n" + _shell_snippet()
        script += "if analytics_enabled; then echo on; else echo off; fi\n"
        out = subprocess.run(
            ["bash", "-c", script],
            env=full_env,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert out in ("on", "off")
        return out == "on"

    run.home = fake_home
    run.default_cfg = fake_home / ".openjarvis" / "config.toml"
    return run


def _write(path: Path, body: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(body, bytes):
        path.write_bytes(body)
    else:
        path.write_text(body, encoding="utf-8")
    return path


def test_installer_missing_config_disabled_and_creates_nothing(installer):
    assert installer() is False
    assert list(installer.home.iterdir()) == []


@pytest.mark.parametrize(
    "body",
    [
        "",
        "[telemetry]\nenabled = true\n",
        "[analytics]\n",
        "[analytics]\nenabled = false\n",
        "[analytics.sub]\nenabled = true\n",
        "[analytics]\n[server]\nenabled = true\n",
    ],
    ids=["empty", "no-section", "no-key", "false", "subtable", "other-section"],
)
def test_installer_not_opted_in(installer, body):
    _write(installer.default_cfg, body)
    assert installer() is False
    assert not (installer.home / ".openjarvis" / "anon_id").exists()


@pytest.mark.parametrize(
    "body",
    [
        "[analytics]\nenabled = true\n",
        "[analytics]\nenabled=true # opted in\n",
        "# hi\n[telemetry]\nenabled = false\n\n  [ analytics ]  \n  enabled = true\n",
        "[analytics]\r\nenabled = true\r\n",
        '[analytics]\nhost = "https://x"\nenabled = true\n[traces]\nenabled = false\n',
    ],
    ids=["plain", "comment", "whitespace", "crlf", "surrounded"],
)
def test_installer_explicit_true_enables(installer, body):
    _write(installer.default_cfg, body)
    assert installer() is True


def test_installer_generated_template(installer):
    template = generate_default_toml(JarvisConfig().hardware)
    _write(installer.default_cfg, template)
    assert installer() is False
    _write(installer.default_cfg, template + "\n[analytics]\nenabled = true\n")
    assert installer() is True


@pytest.mark.parametrize(
    "body",
    [
        '[analytics]\nenabled = "true"\n',
        "[analytics]\nenabled = True\n",
        "[analytics]\nenabled = 1\n",
        "[analytics]\nenabled =\n",
        "[analytics]\nenabled = true\nenabled = false\n",
        "[analytics]\nenabled = true\nenabled = true\n",
        "[analytics]\nenabled = true\n[analytics]\n",
        '[analytics]\n"enabled" = true\n',
        "analytics.enabled = true\n",
        "analytics = { enabled = true }\n",
        'x = """\n[analytics]\nenabled = true\n"""\n',
        "[analytics]\nx = [\n  1,\n]\nenabled = true\n",
        "[analytics\nenabled = true\n",
        "garbage\n[analytics]\nenabled = true\n",
        b"\x00\xff\xfe[analytics]\nenabled = true\n",
    ],
    ids=[
        "string",
        "capitalised",
        "integer",
        "no-value",
        "dup-key-conflict",
        "dup-key-same",
        "dup-section",
        "quoted-key",
        "top-level-dotted",
        "inline-table",
        "multiline-string",
        "multiline-array",
        "broken-header",
        "garbage-line",
        "binary",
    ],
)
def test_installer_malformed_or_ambiguous_fails_closed(installer, body):
    _write(installer.default_cfg, body)
    assert installer() is False


def test_installer_unusable_path_fails_closed(installer):
    installer.default_cfg.mkdir(parents=True)  # a directory, not a file
    assert installer() is False


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 files")
def test_installer_unreadable_config_fails_closed(installer):
    cfg = _write(installer.default_cfg, "[analytics]\nenabled = true\n")
    cfg.chmod(0)
    try:
        assert installer() is False
    finally:
        cfg.chmod(0o600)


_OPT_IN = "[analytics]\nenabled = true\n"
_OPT_OUT = "[analytics]\nenabled = false\n"


def test_installer_openjarvis_config_takes_precedence(installer, tmp_path):
    explicit = _write(tmp_path / "custom.toml", _OPT_IN)
    _write(tmp_path / "ojhome" / "config.toml", _OPT_OUT)
    _write(installer.default_cfg, _OPT_OUT)
    env = {
        "OPENJARVIS_CONFIG": str(explicit),
        "OPENJARVIS_HOME": str(tmp_path / "ojhome"),
    }
    assert installer(env) is True
    _write(explicit, _OPT_OUT)
    _write(tmp_path / "ojhome" / "config.toml", _OPT_IN)
    assert installer(env) is False


def test_installer_openjarvis_config_tilde_and_missing(installer):
    _write(installer.home / "cfg" / "c.toml", _OPT_IN)
    assert installer({"OPENJARVIS_CONFIG": "~/cfg/c.toml"}) is True
    # Explicit path that does not exist: no fallback to the default path.
    _write(installer.default_cfg, _OPT_IN)
    assert installer({"OPENJARVIS_CONFIG": "~/cfg/missing.toml"}) is False


def test_installer_openjarvis_home_beats_xdg(installer, tmp_path):
    _write(tmp_path / "ojhome" / "config.toml", _OPT_IN)
    _write(tmp_path / "xdg" / "openjarvis" / "config.toml", _OPT_OUT)
    env = {
        "OPENJARVIS_HOME": str(tmp_path / "ojhome"),
        "XDG_DATA_HOME": str(tmp_path / "xdg"),
    }
    assert installer(env) is True


def test_installer_xdg_used_when_home_unset(installer, tmp_path):
    _write(tmp_path / "xdg" / "openjarvis" / "config.toml", _OPT_IN)
    _write(installer.default_cfg, _OPT_OUT)
    assert installer({"XDG_DATA_HOME": str(tmp_path / "xdg")}) is True
    # Empty values count as unset, like the Python runtime.
    env = {"OPENJARVIS_CONFIG": "", "OPENJARVIS_HOME": "", "XDG_DATA_HOME": ""}
    assert installer(env) is False
    _write(installer.default_cfg, _OPT_IN)
    assert installer(env) is True
