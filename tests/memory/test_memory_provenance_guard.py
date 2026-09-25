"""Stage 3B: provenance and poisoning guards for automatic memory facts.

These tests drive the real ``FactExtractor`` with a stub model that copies
any "fact-looking" sentence it is shown, i.e. an extraction model that
faithfully (and naively) repeats claims from its input. That makes the
threat concrete: whatever text reaches the extractor can become memory.
"""

from __future__ import annotations

import json
import logging
import re
import time

import pytest

from openjarvis.core.events import EventBus
from openjarvis.core.types import Message, Role
from openjarvis.memory.credentials import contains_credential
from openjarvis.memory.extractor import FactExtractor
from openjarvis.memory.service import MemoryService, publish_completed_exchange
from openjarvis.memory.store import PROVENANCE_FIELDS, Fact, LocalFactStore
from openjarvis.tools.storage.context import inject_context

_CLAIM = re.compile(r"[^.!?\n]*(?:prefer|editor|password|lives)[^.!?\n]*", re.I)


class CopyingEngine:
    """Extraction-model stub: returns every claim-like sentence it was shown."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def generate(self, messages, **kwargs):
        text = messages[-1].content
        self.seen.append(text)
        claims = [m.group(0).strip() for m in _CLAIM.finditer(text)]
        return {"content": json.dumps([c for c in claims if c])}


class FixedEngine:
    def __init__(self, content: str) -> None:
        self._content = content

    def generate(self, messages, **kwargs):
        return {"content": self._content}


def _wait_until(predicate, timeout=2.0, interval=0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _drain(svc: MemoryService) -> None:
    """Wait until the worker has processed every queued job."""
    svc._queue.join()


@pytest.fixture
def facts_path(tmp_path):
    return tmp_path / "memory_facts.jsonl"


def _service(facts_path, engine, bus=None):
    store = LocalFactStore(facts_path)
    svc = MemoryService(store, FactExtractor(engine, "m"), event_bus=bus)
    svc.start()
    return svc


# -- A / B / C: sources ----------------------------------------------------


def test_a_legitimate_user_fact_is_stored(facts_path):
    svc = _service(facts_path, CopyingEngine())
    try:
        assert svc.submit("Remember that my preferred editor is Vim.", "Got it.")
        assert _wait_until(lambda: svc.fact_count() == 1)
        (fact,) = svc.list_facts()
    finally:
        svc.stop()
    assert fact.text == "Remember that my preferred editor is Vim"
    assert fact.trust == "auto"
    assert fact.derived_from == "user"


def test_b_assistant_hallucination_never_becomes_a_fact(facts_path):
    engine = CopyingEngine()
    svc = _service(facts_path, engine)
    try:
        svc.submit(
            "What should I use for coding?",
            "The user's preferred editor is Emacs.",
        )
        _drain(svc)
        facts = svc.list_facts()
    finally:
        svc.stop()
    assert facts == []
    assert all("Emacs" not in seen for seen in engine.seen)
    assert not facts_path.exists() or "Emacs" not in facts_path.read_text()


def test_c_assistant_restated_tool_content_never_becomes_a_fact(facts_path):
    engine = CopyingEngine()
    svc = _service(facts_path, engine)
    try:
        svc.submit(
            "Summarize that web page for me.",
            "The page says the user lives in Atlantis and prefers sending "
            "passwords by email.",
        )
        _drain(svc)
        facts = svc.list_facts()
    finally:
        svc.stop()
    assert facts == []
    assert all("Atlantis" not in seen for seen in engine.seen)


# -- D: secrets -------------------------------------------------------------


def test_d_secret_fact_is_never_written_to_disk(facts_path, caplog):
    engine = FixedEngine(
        json.dumps(["User's SSH password is hunter2", "Preferred editor is Vim"])
    )
    svc = _service(facts_path, engine)
    caplog.set_level(logging.DEBUG, logger="openjarvis.memory")
    try:
        svc.submit("My SSH password is hunter2 and I prefer Vim.")
        assert _wait_until(lambda: svc.fact_count() == 1)
        _drain(svc)
        texts = [fact.text for fact in svc.list_facts()]
    finally:
        svc.stop()
    assert texts == ["Preferred editor is Vim"]
    assert "hunter2" not in facts_path.read_text()
    assert "hunter2" not in caplog.text
    assert "dropped 1 extracted fact(s) containing credentials" in caplog.text


def test_d_secret_is_dropped_not_quarantined(facts_path):
    """A secret that also trips the scanner must not survive as untrusted."""

    class FlagEverything:
        def scan(self, text):
            from types import SimpleNamespace

            return SimpleNamespace(is_clean=False, findings=[], threat_level="low")

    store = LocalFactStore(facts_path)
    svc = MemoryService(
        store,
        FactExtractor(FixedEngine('["API key is sk-abcdefghijklmnopqrstuvwxyz"]'), "m"),
        scanner=FlagEverything(),
    )
    svc.start()
    try:
        svc.submit("here is my key")
        _drain(svc)
    finally:
        svc.stop()
    assert store.list() == []
    assert not facts_path.exists() or "sk-abc" not in facts_path.read_text()


@pytest.mark.parametrize(
    "text",
    [
        "User's SSH password is hunter2",
        "SSH password for prod is hunter2",
        "My password is swordfish.",
        "Wifi password is 'correct horse battery'",
        "Passcode is 123456",
        "User's PIN is 4821",
        "Bank PIN code: 0042",
        "API key is sk-abcdefghijklmnopqrstuvwx",
        "api_key=abc123",
        "Access token: eyJhbGciOi",
        "GitHub token is ghp_" + "a" * 36,
        "client secret = s3cr3t",
        "-----BEGIN PRIVATE KEY-----",
        "credentials are admin/admin",
        "password is stored in vault, and PIN is 4821",
    ],
)
def test_credential_detector_flags_disclosed_secrets(text):
    assert contains_credential(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Preferred editor is Vim",
        "Uses a password manager",
        "User's password manager is Bitwarden",
        "API key is stored in 1Password",
        "User's password policy is strict",
        "Prefers passkeys over passwords",
        "Password is required for sudo",
        "Works on authentication and access tokens",
        "Keeps credentials in a vault",
        "Rotates API keys every 90 days",
        "Is learning about SSH keys",
        "User's favourite bowling pin is red",
    ],
)
def test_credential_detector_keeps_security_topics(text):
    assert contains_credential(text) is False


# -- E: malformed output ----------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "Sure! Here are facts:\n- User lives at 1 Evil St\n- User is an admin",
        'Here you go: ["User is an admin"]',
        '{"facts": ["User is an admin"]}',
        "not json at all",
        '```json\n["User is an admin"]\n```',
        '```\n["User is an admin"]\n```',
        '["User is an admin", 1]',
    ],
)
def test_e_malformed_extractor_output_stores_nothing(facts_path, content):
    svc = _service(facts_path, FixedEngine(content))
    try:
        svc.submit("tell me about myself")
        _drain(svc)
        facts = svc.list_facts()
    finally:
        svc.stop()
    assert facts == []
    assert not facts_path.exists()


# -- F / G: provenance persistence -------------------------------------------


def test_f_provenance_survives_jsonl_round_trip(facts_path):
    bus = EventBus()
    svc = _service(facts_path, CopyingEngine(), bus=bus)
    try:
        publish_completed_exchange(
            bus,
            "My preferred editor is Vim.",
            "Noted.",
            source="cli.chat",
            conversation_id="conv-123",
            user_message_id="msg-456",
        )
        assert _wait_until(lambda: svc.fact_count() == 1)
    finally:
        svc.stop()

    row = json.loads(facts_path.read_text().splitlines()[0])
    assert row["origin"] == "cli.chat"
    assert row["conversation_id"] == "conv-123"
    assert row["user_message_id"] == "msg-456"
    assert row["owner"] == "owner"
    assert row["derived_from"] == "user"

    (fact,) = LocalFactStore(facts_path).list()
    assert (
        fact.origin,
        fact.conversation_id,
        fact.user_message_id,
        fact.owner,
        fact.derived_from,
    ) == ("cli.chat", "conv-123", "msg-456", "owner", "user")


def test_g_legacy_row_without_provenance_loads_and_stays_recallable(facts_path):
    facts_path.write_text(
        json.dumps({"text": "Likes jazz", "source": "auto", "created_at": 1.0})
        + "\n"
        + json.dumps(
            {"text": "Likes tea", "source": "auto", "created_at": 2.0, "trust": ""}
        )
        + "\n",
        encoding="utf-8",
    )

    facts = LocalFactStore(facts_path).list()

    assert [fact.text for fact in facts] == ["Likes jazz", "Likes tea"]
    for fact in facts:
        assert fact.trust == ""
        assert fact.trusted_for_recall is True
        assert all(getattr(fact, name) == "" for name in PROVENANCE_FIELDS)
    augmented = inject_context(
        "music", [Message(role=Role.USER, content="music?")], None, facts=facts
    )
    assert "Likes jazz" in augmented[0].content


def test_g_legacy_row_survives_a_later_write(facts_path):
    facts_path.write_text(
        json.dumps({"text": "Likes jazz", "source": "auto", "created_at": 1.0}) + "\n",
        encoding="utf-8",
    )
    store = LocalFactStore(facts_path)
    assert store.add("Likes tea", source="auto", trust="auto")

    reloaded = LocalFactStore(facts_path).list()
    assert [fact.text for fact in reloaded] == ["Likes jazz", "Likes tea"]
    assert reloaded[0] == Fact(text="Likes jazz", source="auto", created_at=1.0)


# -- H: server surface -----------------------------------------------------


@pytest.mark.parametrize(
    "source", ["server.chat", "server.chat.stream", "", "channel:telegram"]
)
def test_h_non_owner_surfaces_never_populate_the_fact_store(facts_path, source):
    bus = EventBus()
    engine = CopyingEngine()
    svc = _service(facts_path, engine, bus=bus)
    try:
        publish_completed_exchange(
            bus, "My preferred editor is Vim.", "Noted.", source=source
        )
        _drain(svc)
        facts = svc.list_facts()
    finally:
        svc.stop()
    assert facts == []
    assert engine.seen == []
    assert not facts_path.exists()
