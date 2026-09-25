"""Protected persistent state — targets generic tools must never modify.

Protected state is OpenJarvis state whose modification changes future model
instructions, memory, security policy, or runtime stores: the persona files
(SOUL/MEMORY/USER), fact memory, skills and prompt templates, the config and
capability policy, credentials, and OpenJarvis-owned databases.

:func:`classify_protected_target` maps a path to a bounded
:class:`ProtectedCategory` (or ``None``). It is symlink-aware: the literal
path, every hop of its symlink chain, and its fully resolved form are all
checked, and an existing file is also matched by identity (device + inode) so
a hard link cannot alias a protected file. Protected locations come from the
effective OpenJarvis home (resolved on every call, so ``OPENJARVIS_HOME``
overrides apply), ``OPENJARVIS_CONFIG``, and — once :func:`setup_security`
registers it — the effective config (custom memory-file, policy, and database
paths that may live outside the home).

Categories are safe to put in results and events; paths never are.
"""

from __future__ import annotations

import os
import sys
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

PROTECTED_TARGET_KEY = "protected_target"


class ProtectedCategory(str, Enum):
    """Why a target is protected. Values are safe for events and provenance."""

    MEMORY = "memory"
    PROFILE = "profile"
    PERSONA = "persona"
    INSTRUCTIONS = "instructions"
    CONFIG = "config"
    SECURITY = "security"
    RUNTIME_STORE = "runtime_store"


_M = ProtectedCategory

# Files directly under the OpenJarvis home.
_HOME_FILES: Tuple[Tuple[str, ProtectedCategory], ...] = (
    ("MEMORY.md", _M.MEMORY),
    ("memory_facts.jsonl", _M.MEMORY),
    ("USER.md", _M.PROFILE),
    ("SOUL.md", _M.PERSONA),
    ("config.toml", _M.CONFIG),
    ("credentials.toml", _M.SECURITY),
    ("cloud-keys.env", _M.SECURITY),
    (".vault_key", _M.SECURITY),
    ("vault.enc", _M.SECURITY),
)

# Directory trees under the OpenJarvis home; everything inside is protected.
# ``personas`` is special-cased per file (MEMORY/USER/SOUL) before this.
_HOME_TREES: Tuple[Tuple[str, ProtectedCategory], ...] = (
    ("personas", _M.PERSONA),
    ("skills", _M.INSTRUCTIONS),
    ("skill-index", _M.INSTRUCTIONS),
    ("skill-cache", _M.INSTRUCTIONS),
    ("learning", _M.INSTRUCTIONS),
    ("prompts", _M.INSTRUCTIONS),
    ("templates", _M.INSTRUCTIONS),
    ("tools", _M.INSTRUCTIONS),
    ("operators", _M.INSTRUCTIONS),
    ("recipes", _M.INSTRUCTIONS),
    ("mcp", _M.CONFIG),
    ("connectors", _M.SECURITY),
)

# Files inside ``personas/<name>/``; any other persona file is PERSONA.
_PERSONA_FILES = {
    "MEMORY.md": _M.MEMORY,
    "USER.md": _M.PROFILE,
    "SOUL.md": _M.PERSONA,
}

_DB_SUFFIXES = (".db", ".sqlite", ".sqlite3")
_DB_SIDECARS = ("-wal", "-shm", "-journal")

# Skill manifests the CLI auto-discovers from ``./skills`` (``jarvis ask``
# and SkillManager both scan it), so they are part of the prompt boundary.
_WORKSPACE_SKILLS_DIR = "skills"
_SKILL_MANIFESTS = ("skill.toml", "SKILL.md")

_MAX_LINK_HOPS = 40

PathLike = Union[str, "os.PathLike[str]"]

# Sentinel category: a registered path that is a directory tree of instructions.
_TREE_INSTRUCTIONS: Any = object()

_registry_lock = threading.Lock()
_registered: Tuple[Tuple[Path, ProtectedCategory], ...] = ()


# ---------------------------------------------------------------------------
# Effective-config registration
# ---------------------------------------------------------------------------


def register_protected_config(config: Any, memory_files: Any = None) -> None:
    """Record protected paths named by the effective *config*.

    Called by :func:`openjarvis.security.setup_security`. ``memory_files``
    overrides ``config.memory_files`` (e.g. a ``--persona`` selection).
    Replaces any earlier registration.
    """
    entries = _config_entries(config, memory_files)
    with _registry_lock:
        global _registered
        _registered = tuple(entries)


def clear_protected_config() -> None:
    """Forget the registered config (tests)."""
    with _registry_lock:
        global _registered
        _registered = ()


def _config_entries(
    config: Any, memory_files: Any
) -> List[Tuple[Path, ProtectedCategory]]:
    entries: List[Tuple[Path, ProtectedCategory]] = []

    def add(value: Any, category: ProtectedCategory) -> None:
        if isinstance(value, (str, Path)) and str(value):
            entries.append((Path(value).expanduser(), category))

    mf = memory_files if memory_files is not None else _get(config, "memory_files")
    if mf is not None:
        try:
            from openjarvis.prompt.builder import resolve_memory_files

            mf = resolve_memory_files(mf)
        except Exception:
            pass
        add(_get(mf, "memory_path"), _M.MEMORY)
        add(_get(mf, "user_path"), _M.PROFILE)
        add(_get(mf, "soul_path"), _M.PERSONA)

    memory = _get(config, "memory")
    add(_get(memory, "facts_path"), _M.MEMORY)
    add(_get(memory, "db_path"), _M.RUNTIME_STORE)

    config_dir = _get(config, "_config_dir")
    if isinstance(config_dir, (str, Path)) and str(config_dir):
        add(Path(config_dir) / "config.toml", _M.CONFIG)

    security = _get(config, "security")
    add(_get(_get(security, "capabilities"), "policy_path"), _M.SECURITY)
    add(_get(security, "vault_key_path"), _M.SECURITY)
    add(_get(security, "signing_key_path"), _M.SECURITY)
    add(_get(security, "audit_log_path"), _M.RUNTIME_STORE)

    skills_dir = _get(_get(config, "skills"), "skills_dir")
    if isinstance(skills_dir, (str, Path)) and str(skills_dir):
        entries.append((Path(skills_dir).expanduser(), _TREE_INSTRUCTIONS))

    for section in ("sessions", "conversations", "telemetry", "traces", "optimize"):
        add(_get(_get(config, section), "db_path"), _M.RUNTIME_STORE)
    add(_get(_get(config, "agent_manager"), "db_path"), _M.RUNTIME_STORE)
    add(_get(_get(config, "scheduler"), "db_path"), _M.RUNTIME_STORE)
    return entries


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_protected_target(
    path: PathLike,
    config: Any = None,
) -> Optional[ProtectedCategory]:
    """Return the protected category *path* belongs to, or ``None``.

    *config* overrides the registered effective config for this call.
    """
    if not isinstance(path, (str, os.PathLike)):
        return None
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        return None

    extra = tuple(_config_entries(config, None)) if config is not None else _registered
    rules = _explicit_rules(extra)
    file_keys = {key: category for path, category in rules for key in _variants(path)}
    tree_keys = [
        (key, _M.INSTRUCTIONS)
        for path, category in extra
        if category is _TREE_INSTRUCTIONS
        for key in _variants(path)
    ]
    home_keys = _variants(_home_dir())
    skills_keys = _variants(Path.cwd() / _WORKSPACE_SKILLS_DIR)

    target = Path(raw).expanduser()
    for candidate in _candidates(target):
        key = _key(candidate)
        category = file_keys.get(key) or _classify_key(
            key, home_keys, tree_keys, skills_keys
        )
        if category is not None:
            return category
    return _classify_by_identity(target, rules)


def is_protected_target(path: PathLike, config: Any = None) -> bool:
    """Return ``True`` when :func:`classify_protected_target` matches."""
    return classify_protected_target(path, config) is not None


def first_protected_target(
    paths: Iterable[Optional[PathLike]],
) -> Optional[ProtectedCategory]:
    """Classify each non-empty path; return the first protected category."""
    for path in paths:
        if path:
            category = classify_protected_target(path)
            if category is not None:
                return category
    return None


def protected_target_message(category: ProtectedCategory, tool_name: str) -> str:
    """Model-facing denial text. Names the category, never the path."""
    base = (
        f"Access denied: the target is protected OpenJarvis {category.value}"
        f" state and cannot be modified with {tool_name}."
    )
    if category is _M.MEMORY:
        return base + " Use the memory_manage tool to change agent memory."
    if category is _M.PROFILE:
        return base + " Use the user_profile_manage tool to change the user profile."
    return base


def _home_dir() -> Path:
    from openjarvis.core.paths import get_config_dir

    return get_config_dir()


def _explicit_rules(
    extra: Iterable[Tuple[Path, Any]],
) -> List[Tuple[Path, ProtectedCategory]]:
    """Exact protected files: home files, the effective config, registrations."""
    home = _home_dir()
    rules = [(home / name, category) for name, category in _HOME_FILES]
    env_config = os.environ.get("OPENJARVIS_CONFIG")
    if env_config:
        rules.append((Path(env_config).expanduser(), _M.CONFIG))
    rules.extend(
        (path, category)
        for path, category in extra
        if isinstance(category, ProtectedCategory)
    )
    return rules


def _classify_key(
    key: str,
    home_keys: Tuple[str, ...],
    tree_keys: List[Tuple[str, ProtectedCategory]],
    skills_keys: Tuple[str, ...],
) -> Optional[ProtectedCategory]:
    for home_key in home_keys:
        rel = _relative_parts(key, home_key)
        if not rel:
            continue  # outside the home, or the home directory itself
        if rel[0] == _case("personas"):
            if len(rel) == 3:
                return _PERSONA_FILE_KEYS.get(rel[2], _M.PERSONA)
            return _M.PERSONA
        for tree, category in _HOME_TREES:
            if rel[0] == _case(tree):
                return category
        if _is_db_name(rel[-1]):
            return _M.RUNTIME_STORE

    for tree_key, category in tree_keys:
        if _relative_parts(key, tree_key) is not None:
            return category

    for skills_key in skills_keys:
        rel = _relative_parts(key, skills_key)
        if rel and rel[-1] in _SKILL_MANIFEST_KEYS:
            return _M.INSTRUCTIONS
    return None


def _classify_by_identity(
    path: Path, rules: List[Tuple[Path, ProtectedCategory]]
) -> Optional[ProtectedCategory]:
    """Match an existing file to a protected file by device and inode."""
    try:
        st = path.stat()
    except (OSError, ValueError):
        return None
    ident = (st.st_dev, st.st_ino)
    candidates: List[Tuple[Path, ProtectedCategory]] = list(rules)
    home = _home_dir()
    try:
        candidates.extend(
            (child, _M.RUNTIME_STORE)
            for child in home.iterdir()
            if _is_db_name(_case(child.name))
        )
    except OSError:
        pass
    for rule_path, category in candidates:
        try:
            rst = rule_path.stat()
        except (OSError, ValueError):
            continue
        if (rst.st_dev, rst.st_ino) == ident:
            return category
    return None


def _candidates(path: Path) -> Iterator[Path]:
    """The literal path, each symlink hop, and the fully resolved path."""
    literal = Path(os.path.abspath(path))
    yield literal
    current = literal
    for _ in range(_MAX_LINK_HOPS):
        try:
            target = current.readlink()
        except (OSError, ValueError):
            break
        current = Path(os.path.abspath(current.parent / target))
        yield current
        yield _resolve(current)
    yield _resolve(literal)


def _resolve(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return path


def _variants(path: Path) -> Tuple[str, ...]:
    """Comparison keys for a protected location: literal and resolved."""
    literal = Path(os.path.abspath(path))
    keys = {_key(literal), _key(_resolve(literal))}
    return tuple(keys)


def _key(path: Path) -> str:
    return _case(os.path.normcase(str(path)))


def _case(value: str) -> str:
    # macOS volumes are case-insensitive by default; normcase is a no-op there.
    return value.lower() if sys.platform == "darwin" else value


def _relative_parts(key: str, base_key: str) -> Optional[Tuple[str, ...]]:
    if key == base_key:
        return ()
    prefix = base_key.rstrip(os.sep) + os.sep
    if not key.startswith(prefix):
        return None
    return tuple(part for part in key[len(prefix) :].split(os.sep) if part)


def _is_db_name(name: str) -> bool:
    lowered = name.lower()
    for sidecar in _DB_SIDECARS:
        if lowered.endswith(sidecar):
            lowered = lowered[: -len(sidecar)]
            break
    return lowered.endswith(_DB_SUFFIXES)


_PERSONA_FILE_KEYS = {_case(name): cat for name, cat in _PERSONA_FILES.items()}
_SKILL_MANIFEST_KEYS = frozenset(_case(name) for name in _SKILL_MANIFESTS)


__all__ = [
    "PROTECTED_TARGET_KEY",
    "ProtectedCategory",
    "classify_protected_target",
    "clear_protected_config",
    "first_protected_target",
    "is_protected_target",
    "protected_target_message",
    "register_protected_config",
]
