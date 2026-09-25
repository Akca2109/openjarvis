"""SkillManageTool — create, list, load, or delete agent-authored skills."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional

from openjarvis.core.paths import get_config_dir
from openjarvis.core.registry import ToolRegistry
from openjarvis.core.types import ToolResult
from openjarvis.tools._stubs import BaseTool, ToolSpec

# A skill name is a bare identifier: it becomes ``<skills_dir>/<name>.toml``.
_SKILL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_OPTIONAL_STEP_FIELDS = ("arguments_template", "output_key")


@ToolRegistry.register("skill_manage")
class SkillManageTool(BaseTool):
    """Manage agent-authored procedural skills."""

    def __init__(self, skills_dir: Path | str | None = None) -> None:
        if skills_dir is None:
            skills_dir = get_config_dir() / "skills"
        self._skills_dir = Path(skills_dir).expanduser()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="skill_manage",
            description="Create, list, load, or delete agent-authored skills.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "load", "delete"],
                        "description": "Action to perform.",
                    },
                    "name": {
                        "type": "string",
                        "description": "Skill name (for create/load/delete).",
                    },
                    "description": {
                        "type": "string",
                        "description": "Skill description (for create).",
                    },
                    "steps": {
                        "type": "array",
                        "description": (
                            "List of step dicts with tool_name and optional"
                            " arguments_template (for create)."
                        ),
                    },
                },
                "required": ["action"],
            },
            category="skill",
        )

    def execute(self, **params: Any) -> ToolResult:
        action = params.get("action", "list")
        name = params.get("name", "")
        if action == "create":
            return self._create(
                name, params.get("description", ""), params.get("steps", [])
            )
        elif action == "list":
            return self._list()
        elif action == "load":
            return self._load(name)
        elif action == "delete":
            return self._delete(name)
        return ToolResult(
            tool_name=self.spec.name,
            success=False,
            content=f"Unknown action: {action}",
        )

    def _skill_path(self, name: Any) -> Optional[Path]:
        """``<skills_dir>/<name>.toml`` for a valid name that stays inside."""
        if not isinstance(name, str) or not _SKILL_NAME_RE.fullmatch(name):
            return None
        path = self._skills_dir / f"{name}.toml"
        try:
            root = self._skills_dir.resolve()
            if not path.resolve().is_relative_to(root):
                return None
        except (OSError, RuntimeError):
            return None
        return path

    def _invalid_name(self) -> ToolResult:
        return ToolResult(
            tool_name=self.spec.name,
            success=False,
            content=(
                "Invalid skill name: use 1-64 letters, digits, '_' or '-',"
                " starting with a letter or digit."
            ),
        )

    def _create(self, name: str, description: str, steps: List[dict]) -> ToolResult:
        if not name:
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content="Skill name is required.",
            )
        if self._skill_path(name) is None:
            return self._invalid_name()
        if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content="Skill steps must be a list of objects.",
            )
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        # Re-check now that the directory exists and resolves.
        path = self._skill_path(name)
        if path is None:
            return self._invalid_name()
        # Serialize with a TOML writer so no field can inject tables or keys.
        import tomlkit

        skill: dict[str, Any] = {"name": name, "description": str(description)}
        if steps:
            skill["steps"] = [
                {"tool_name": str(step.get("tool_name", ""))}
                | {key: str(step[key]) for key in _OPTIONAL_STEP_FIELDS if key in step}
                for step in steps
            ]
        path.write_text(tomlkit.dumps({"skill": skill}))
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content=f"Created skill: {name}",
        )

    def _list(self) -> ToolResult:
        if not self._skills_dir.exists():
            return ToolResult(
                tool_name=self.spec.name,
                success=True,
                content="No skills directory found.",
            )
        skills = []
        for f in sorted(self._skills_dir.glob("*.toml")):
            skills.append(f.stem)
        if not skills:
            return ToolResult(
                tool_name=self.spec.name,
                success=True,
                content="No skills found.",
            )
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content="Available skills:\n" + "\n".join(f"- {s}" for s in skills),
        )

    def _load(self, name: str) -> ToolResult:
        path = self._skill_path(name)
        if path is None:
            return self._invalid_name()
        if not path.exists():
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content=f"Skill not found: {name}",
            )
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content=path.read_text(),
        )

    def _delete(self, name: str) -> ToolResult:
        path = self._skill_path(name)
        if path is None:
            return self._invalid_name()
        if not path.exists():
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                content=f"Skill not found: {name}",
            )
        path.unlink()
        return ToolResult(
            tool_name=self.spec.name,
            success=True,
            content=f"Deleted skill: {name}",
        )
