"""Deterministic credential detection for automatically extracted facts.

A plaintext secret must never become durable memory: the fact store is a
readable JSONL file and recalled facts are placed into every future prompt.
Quarantining is not enough (the plaintext would still sit on disk), so the
memory service drops any fact this module flags before storage.

Detection reuses the SECRET patterns from :mod:`openjarvis.security.taint`
(API keys, GitHub tokens, private-key headers, ``password=...``) and adds a
narrow heuristic for natural-language statements such as
``"User's SSH password is hunter2"``. The heuristic requires a credential
keyword *and* an assigned value that looks like a credential, so facts that
merely discuss security concepts (``"Uses a password manager"``,
``"API key is stored in 1Password"``) are kept.
"""

from __future__ import annotations

import re

from openjarvis.security.taint import TaintLabel, auto_detect_taint

_KEYWORD = (
    r"pass(?:word|code|phrase)s?|passwd|(?P<pin>pin(?:\s*(?:code|number))?)"
    r"|api[\s_-]?keys?"
    r"|(?:access|auth(?:entication)?|bearer|refresh|session|api|oauth|github"
    r"|gitlab|slack)[\s_-]?tokens?"
    r"|(?:private|ssh|secret)[\s_-]?keys?|client[\s_-]?secrets?|secrets?"
    r"|credentials?"
)

# keyword, optional short qualifier ("for prod", "to the vpn"), then an
# assignment ("is", "was", ":", "=") and the candidate value.
_ASSIGNMENT = re.compile(
    rf"\b(?:{_KEYWORD})\b"
    r"(?:\s+(?:for|to|of|on|at)\s+[\w.@/-]+(?:\s+[\w.@/-]+)?)?"
    r"\s*(?:\bis\b|\bwas\b|\bare\b|\bequals\b|=|:)\s*"
    r"(?P<value>[\"'`][^\"'`]+[\"'`]|[^\s\"'`]+)"
    r"(?=(?P<rest>[^\n]*))",
    re.IGNORECASE,
)

# Values that describe a credential rather than disclose one.
_DESCRIPTIVE_VALUES = frozenset(
    {
        "a", "an", "the", "that", "this", "in", "on", "at", "not", "no",
        "being", "via", "from", "with", "stored", "saved", "kept", "managed",
        "set", "changed", "rotated", "expired", "required", "needed",
        "secure", "strong", "weak", "long", "short", "missing", "unknown",
        "private", "public", "encrypted", "hashed", "sensitive", "important",
        "forgotten", "reset", "valid", "invalid", "correct", "wrong", "same",
        "different", "empty", "none", "null", "redacted", "hidden",
        "confidential", "safe", "used", "shared", "generated", "updated",
        "protected",
    }
)  # fmt: skip

_TRAILING_PUNCTUATION = ".,;!?)"


def _discloses_value(match: re.Match[str]) -> bool:
    raw = match.group("value")
    rest = match.group("rest").strip()
    quoted = raw[0] in "\"'`"
    value = raw.strip("\"'`").rstrip(_TRAILING_PUNCTUATION)
    if not value or value.lower() in _DESCRIPTIVE_VALUES:
        return False
    if match.group("pin"):
        # "pin" alone is an ordinary word; only a numeric value is a PIN.
        return re.fullmatch(r"\d{3,}", value) is not None
    if quoted or any(not ch.isalpha() for ch in value):
        return True
    # A bare word counts only when it ends the clause ("password is
    # swordfish."), not when it starts a description ("is kept offline").
    ends_clause = raw[-1] in _TRAILING_PUNCTUATION or rest[:1] in {",", ";", ""}
    return ends_clause


def contains_credential(text: str) -> bool:
    """Return True when *text* appears to disclose a secret or credential."""
    if not text:
        return False
    if auto_detect_taint(text).has(TaintLabel.SECRET):
        return True
    return any(_discloses_value(m) for m in _ASSIGNMENT.finditer(text))


__all__ = ["contains_credential"]
