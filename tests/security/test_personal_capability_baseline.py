"""Stage 0C: capability enforcement for the default personal profile."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openjarvis.core.config import JarvisConfig, load_config
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.types import ToolCall, ToolResult
from openjarvis.security import setup_security
from openjarvis.security.capabilities import (
    PERSONAL_BASELINE_GRANTS,
    RESTRICTED_BASELINE_GRANTS,
    CapabilityPolicy,
    _glob_match,
    _PythonCapabilityBackend,
)
from openjarvis.tools._stubs import BaseTool, ToolExecutor, ToolSpec


def _security(config: JarvisConfig, tmp_path):
    config.security.audit_log_path = str(tmp_path / "audit.db")
    config.security.rate_limit_enabled = False
    sec = setup_security(config, MagicMock(), EventBus())
    return sec


class _RecordingTool(BaseTool):
    """Third-party tool whose body records every invocation."""

    def __init__(
        self,
        name: str,
        capabilities: list[str],
        *,
        requires_confirmation: bool = False,
    ) -> None:
        self._name = name
        self._capabilities = capabilities
        self._requires_confirmation = requires_confirmation
        self.calls: list[dict] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description="test tool",
            required_capabilities=list(self._capabilities),
            requires_confirmation=self._requires_confirmation,
        )

    def execute(self, **params) -> ToolResult:
        self.calls.append(params)
        return ToolResult(tool_name=self._name, content="ran", success=True)


def _call(name: str) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments="{}")


# ---------------------------------------------------------------------------
# Default / profile state
# ---------------------------------------------------------------------------


class TestPersonalDefaults:
    def test_dataclass_default_enables_personal_enforcement(self):
        caps = JarvisConfig().security.capabilities
        assert caps.enabled is True
        assert caps.default_deny is True
        assert caps.baseline == "personal"
        assert caps.policy_path == ""

    def test_load_config_without_file_is_enforced(self, tmp_path):
        caps = load_config(tmp_path / "missing.toml").security.capabilities
        assert (caps.enabled, caps.default_deny, caps.baseline) == (
            True,
            True,
            "personal",
        )

    @pytest.mark.parametrize(
        ("profile", "baseline"),
        [
            ("", "personal"),
            ("personal", "personal"),
            ("shared", "restricted"),
            ("server", "restricted"),
        ],
    )
    def test_profile_baselines(self, tmp_path, profile, baseline):
        path = tmp_path / "config.toml"
        path.write_text(f'[security]\nprofile = "{profile}"\n')
        caps = load_config(path).security.capabilities
        assert caps.enabled is True
        assert caps.default_deny is True
        assert caps.baseline == baseline

    def test_setup_security_builds_personal_policy(self, tmp_path):
        sec = _security(JarvisConfig(), tmp_path)
        policy = sec.capability_policy
        assert isinstance(policy, CapabilityPolicy)
        for cap in PERSONAL_BASELINE_GRANTS:
            assert policy.check("any-agent", cap)
        assert not policy.check("any-agent", "system:admin")
        assert not policy.check("any-agent", "custom:unreviewed")
        # The anonymous identity never inherits the baseline.
        assert not policy.check("", "file:read")

    def test_baselines_never_grant_admin(self):
        assert "system:admin" not in PERSONAL_BASELINE_GRANTS
        assert "system:admin" not in RESTRICTED_BASELINE_GRANTS
        assert "*" not in PERSONAL_BASELINE_GRANTS


# ---------------------------------------------------------------------------
# Explicit overrides
# ---------------------------------------------------------------------------


class TestExplicitOverrides:
    def _load(self, tmp_path, body: str) -> JarvisConfig:
        # load_config is cached by path; give every document its own file.
        path = tmp_path / f"config-{len(list(tmp_path.glob('config-*')))}.toml"
        path.write_text(body)
        return load_config(path)

    def test_explicit_disable_turns_enforcement_off(self, tmp_path):
        config = self._load(tmp_path, "[security.capabilities]\nenabled = false\n")
        assert _security(config, tmp_path).capability_policy is None

    def test_explicit_default_allow_is_honored(self, tmp_path):
        config = self._load(tmp_path, "[security.capabilities]\ndefault_deny = false\n")
        policy = _security(config, tmp_path).capability_policy
        assert policy.check("any-agent", "system:admin")

    @pytest.mark.parametrize("profile", ["", "personal"])
    def test_legacy_explicit_default_deny_keeps_restricted_baseline(
        self, tmp_path, profile
    ):
        config = self._load(
            tmp_path,
            f'[security]\nprofile = "{profile}"\n'
            "[security.capabilities]\nenabled = true\ndefault_deny = true\n",
        )
        assert config.security.capabilities.baseline == "restricted"
        policy = _security(config, tmp_path).capability_policy
        assert policy.check("any-agent", "file:read")
        assert not policy.check("any-agent", "code:execute")
        assert not policy.check("any-agent", "file:write")

    def test_explicit_baseline_wins(self, tmp_path):
        config = self._load(
            tmp_path,
            '[security]\nprofile = "shared"\n'
            '[security.capabilities]\ndefault_deny = true\nbaseline = "personal"\n',
        )
        assert config.security.capabilities.baseline == "personal"
        config = self._load(
            tmp_path, '[security.capabilities]\nbaseline = "restricted"\n'
        )
        policy = _security(config, tmp_path).capability_policy
        assert not policy.check("any-agent", "code:execute")

    def test_policy_file_receives_no_baseline(self, tmp_path):
        policy_path = tmp_path / "policy.json"
        policy_path.write_text(
            '{"agents": [{"agent_id": "simple", "grants": '
            '[{"capability": "system:admin"}]}]}'
        )
        config = self._load(
            tmp_path, f'[security.capabilities]\npolicy_path = "{policy_path}"\n'
        )
        policy = _security(config, tmp_path).capability_policy
        assert policy.check("simple", "system:admin")
        assert not policy.check("simple", "file:read")
        assert not policy.check("other", "file:read")

    def test_unknown_baseline_fails_closed(self, tmp_path):
        config = JarvisConfig()
        config.security.capabilities.baseline = "everything"
        with pytest.raises(RuntimeError, match="Capability policy initialization"):
            _security(config, tmp_path)


# ---------------------------------------------------------------------------
# ToolExecutor behaviour under the personal default
# ---------------------------------------------------------------------------


class TestPersonalExecutor:
    @pytest.fixture
    def policy(self, tmp_path):
        return _security(JarvisConfig(), tmp_path).capability_policy

    def test_allowed_capability_executes(self, policy):
        tool = _RecordingTool("notes_read", ["memory:read", "file:read"])
        result = ToolExecutor([tool], capability_policy=policy, agent_id="simple")
        result = result.execute(_call("notes_read"))
        assert result.success, result.content
        assert tool.calls == [{}]

    def test_ordinary_safe_builtin_still_works(self, policy):
        from openjarvis.tools.calculator import CalculatorTool

        executor = ToolExecutor(
            [CalculatorTool()], capability_policy=policy, agent_id="simple"
        )
        result = executor.execute(
            ToolCall(id="c", name="calculator", arguments='{"expression": "2+3"}')
        )
        assert result.success, result.content
        assert "5" in result.content

    def test_denied_capability_refused_before_tool_body(self, policy):
        bus = EventBus(record_history=True)
        tool = _RecordingTool("admin_probe", ["system:admin"])
        executor = ToolExecutor(
            [tool], bus, capability_policy=policy, agent_id="simple"
        )
        result = executor.execute(_call("admin_probe"))
        assert not result.success
        assert "system:admin" in result.content and "denied" in result.content
        assert tool.calls == []
        denied = [e for e in bus.history if e.event_type == EventType.CAPABILITY_DENIED]
        assert denied and denied[0].data["agent_id"] == "simple"
        assert not any(e.event_type == EventType.TOOL_CALL_START for e in bus.history)

    def test_builtin_admin_floor_cannot_be_dropped(self, policy, monkeypatch):
        """A real admin built-in is denied even if its spec omits the capability."""
        from openjarvis.tools.channel_tools import ChannelListTool

        tool = ChannelListTool()
        body = MagicMock()
        monkeypatch.setattr(tool, "execute", body)
        monkeypatch.setattr(
            type(tool),
            "spec",
            property(lambda self: ToolSpec(name="channel_list", description="x")),
        )
        executor = ToolExecutor([tool], capability_policy=policy, agent_id="simple")
        result = executor.execute(_call("channel_list"))
        assert not result.success and "system:admin" in result.content
        body.assert_not_called()

    def test_mcp_tools_require_tool_invoke(self, policy, tmp_path):
        from openjarvis.tools.mcp_adapter import MCPToolAdapter

        client = MagicMock()
        client.call_tool.return_value = {"content": [{"type": "text", "text": "ok"}]}
        spec = ToolSpec(name="calculator", description="remote impersonator")
        adapter = MCPToolAdapter(client, spec)

        allowed = ToolExecutor([adapter], capability_policy=policy, agent_id="simple")
        assert allowed.execute(_call("calculator")).success

        config = JarvisConfig()
        config.security.capabilities.baseline = "restricted"
        restricted = _security(config, tmp_path).capability_policy
        denied = ToolExecutor(
            [adapter], capability_policy=restricted, agent_id="simple"
        )
        client.call_tool.reset_mock()
        result = denied.execute(_call("calculator"))
        assert not result.success and "tool:invoke" in result.content
        client.call_tool.assert_not_called()


# ---------------------------------------------------------------------------
# Capability grants and human confirmation are independent controls
# ---------------------------------------------------------------------------


class TestConfirmationIndependence:
    @pytest.fixture
    def policy(self, tmp_path):
        return _security(JarvisConfig(), tmp_path).capability_policy

    def test_granted_capability_still_requires_confirmation(self, policy):
        tool = _RecordingTool("run_code", ["code:execute"], requires_confirmation=True)
        executor = ToolExecutor([tool], capability_policy=policy, agent_id="simple")
        result = executor.execute(_call("run_code"))
        assert not result.success and "requires confirmation" in result.content
        assert tool.calls == []

        confirm = MagicMock(return_value=False)
        executor = ToolExecutor(
            [tool],
            capability_policy=policy,
            agent_id="simple",
            interactive=True,
            confirm_callback=confirm,
        )
        result = executor.execute(_call("run_code"))
        assert "denied by user" in result.content
        confirm.assert_called_once()
        assert tool.calls == []

        confirm.return_value = True
        assert executor.execute(_call("run_code")).success
        assert tool.calls == [{}]

    def test_capability_denial_precedes_and_skips_confirmation(self, policy):
        tool = _RecordingTool("spawn", ["system:admin"], requires_confirmation=True)
        confirm = MagicMock(return_value=True)
        executor = ToolExecutor(
            [tool],
            capability_policy=policy,
            agent_id="simple",
            interactive=True,
            confirm_callback=confirm,
        )
        result = executor.execute(_call("spawn"))
        assert not result.success and "system:admin" in result.content
        confirm.assert_not_called()
        assert tool.calls == []


# ---------------------------------------------------------------------------
# Pure-Python backend parity (installs without the native extension)
# ---------------------------------------------------------------------------


_GLOB_CASES = [
    ("*", "file:read"),
    ("file:read", "file:read"),
    ("file:*", "file:read"),
    ("file:*", "network:fetch"),
    ("*:read", "memory:read"),
    ("*:read", "memory:write"),
    ("f*e:r*d", "file:read"),
    ("f*e:r*d", "file:reads"),
    ("file:read", "file:readx"),
    ("[f]ile:read", "file:read"),
    ("file:?ead", "file:read"),
    ("a*a", "a"),
    ("**", ""),
    ("", ""),
]


@pytest.mark.parametrize(("pattern", "text"), _GLOB_CASES)
def test_python_glob_matches_rust(pattern, text):
    rust = pytest.importorskip("openjarvis_rust")
    native = rust.CapabilityPolicy(default_deny=True)
    native.grant("a", pattern, "*")
    python = _PythonCapabilityBackend(default_deny=True)
    python.grant("a", pattern, "*")
    assert python.check("a", text, "") == native.check("a", text, "")
    assert _glob_match(pattern, text) == native.check("a", text, "")


def test_policy_enforces_without_native_extension(monkeypatch, tmp_path):
    import openjarvis._rust_bridge as bridge

    def _missing():
        raise ImportError("no native extension")

    monkeypatch.setattr(bridge, "get_rust_module", _missing)
    sec = _security(JarvisConfig(), tmp_path)
    policy = sec.capability_policy
    assert isinstance(policy._rust_impl, _PythonCapabilityBackend)
    assert policy.check("simple", "file:write")
    assert not policy.check("simple", "system:admin")
    policy.deny("simple", "code:*")
    assert not policy.check("simple", "code:execute")
    assert policy.check("simple", "memory:read") is False  # own policy, no grant
    assert policy.check("other", "memory:read")
