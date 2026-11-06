"""How a HarnessSpec becomes a pi agent.

Every decision about what a harness file means is made here, on Honing's side,
so it can be checked before an episode starts and tested without pi or Node.js.
The worker (``integrations/pi/worker.mjs``) receives the resulting
:class:`AgentPlan` and only applies it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

from honing.core import CampaignRuleError, ContractError, HarnessFile, HarnessFileKind, HarnessSpec, require

AGENT = "pi"
SETTINGS = "pi/settings.json"
SYSTEM_PROMPT = "SYSTEM.md"
DEFAULT_TOOLS = ("read", "bash", "edit", "write")
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max")

_SKILL_FILE = "skill.md"
_SETTINGS_KEYS = frozenset({"defaultTools", "defaultThinkingLevel"})
# pi lists skills to the model only when it has a tool that can read a skill's file.
_SKILL_READERS = frozenset({"read", "bash"})
_FRONTMATTER = re.compile(r"\A---\r?\n(?P<body>.*?)\r?\n---(?:\r?\n|\Z)", re.DOTALL)
_DESCRIPTION = re.compile(r"^description:[ \t]*\S", re.MULTILINE)
_MAX_DESCRIPTION = 1024


@dataclass(frozen=True)
class ToolPolicy:
    """What the operator lets a harness give the agent.

    A harness chooses among these in ``pi/settings.json``; it never adds to them.

    Attributes:
        allowed: pi tools a harness may enable.
        default: Tools the agent gets when the harness names none.
    """

    allowed: frozenset[str] = frozenset(DEFAULT_TOOLS)
    default: tuple[str, ...] = DEFAULT_TOOLS

    def __post_init__(self) -> None:
        require(set(self.default) <= self.allowed, f"default tools {list(self.default)} must be allowed")


@dataclass(frozen=True)
class SkillPlan:
    """One skill: its name, its defining file, the files that come with it, and its description.

    ``description`` is None when the file's own frontmatter gives one, which pi
    reads; otherwise it is the description the worker gives pi for the file.
    """

    name: str
    path: str
    resources: tuple[str, ...] = ()
    description: str | None = None


@dataclass(frozen=True)
class AgentPlan:
    """A harness resolved against a :class:`ToolPolicy`: exactly what the worker builds.

    Attributes:
        system_prompt: The file that replaces pi's system prompt, or None to keep pi's.
        append_system_prompt: Instruction files appended, in this order, to the system prompt.
        skills: Skills to load, each with its resource files.
        files: Other settings files, copied into the agent's working directory.
        tools: Tool names the agent gets.
        settings: pi settings from ``pi/settings.json``.
        routes: How each harness path reaches the agent, for the delivery report.
    """

    system_prompt: str | None
    append_system_prompt: tuple[str, ...]
    skills: tuple[SkillPlan, ...]
    files: tuple[str, ...]
    tools: tuple[str, ...]
    settings: Mapping[str, object]
    routes: Mapping[str, str]

    def to_json(self) -> dict[str, object]:
        """Returns the plan as JSON-native values, the form the worker reads."""
        return {
            "system_prompt": self.system_prompt,
            "append_system_prompt": list(self.append_system_prompt),
            "skills": [
                {"name": s.name, "path": s.path, "resources": list(s.resources), "description": s.description}
                for s in self.skills
            ],
            "files": list(self.files),
            "tools": list(self.tools),
            "settings": dict(self.settings),
            "routes": dict(self.routes),
        }


def plan_agent(spec: HarnessSpec, policy: ToolPolicy) -> AgentPlan:
    """Resolves a harness into the agent the worker builds.

    ``SYSTEM.md`` replaces pi's system prompt; every other instruction file is
    appended to it, in path order. A skill file named ``SKILL.md`` defines a
    skill named after its directory, and every other skill file under that
    directory is its resource. A skill file under no ``SKILL.md`` is a skill of
    its own, named after its path. pi loads a skill only with a description: a
    skill file with no frontmatter is described by its first line, and one
    whose frontmatter has no description is refused. ``pi/settings.json`` sets
    tools from ``policy`` and the thinking level; other settings files are
    passed to the agent as files.

    Args:
        spec: The harness.
        policy: What the operator allows.

    Returns:
        The resolved plan.

    Raises:
        ContractError: The harness is for another agent.
        CampaignRuleError: ``harness_tools_allowed`` when the harness asks for
            a tool ``policy`` does not allow, ``harness_well_formed`` when a
            settings file or skill is invalid.
    """
    if spec.agent != AGENT:
        raise ContractError(f"the pi runner runs agent {AGENT!r}, not {spec.agent!r}")
    by_kind: dict[HarnessFileKind, list[str]] = {}
    for path, file in sorted(spec.files, key=lambda item: item[0]):
        by_kind.setdefault(file.kind, []).append(path)
    unsupported = sorted(set(by_kind) - {"instructions", "skill", "settings"})
    if unsupported:
        raise CampaignRuleError("harness_well_formed", f"the pi runner cannot run {unsupported} files")

    instructions = by_kind.get("instructions", [])
    system_prompt = SYSTEM_PROMPT if SYSTEM_PROMPT in instructions else None
    appended = tuple(path for path in instructions if path != SYSTEM_PROMPT)
    skills = _skills(spec, by_kind.get("skill", []))
    settings, tools = _settings(spec.get(SETTINGS), policy)
    if skills and not _SKILL_READERS & set(tools):
        raise CampaignRuleError(
            "harness_well_formed",
            f"the harness has skills, but pi lists them only with the read or bash tool: {tools}",
        )
    files = tuple(path for path in by_kind.get("settings", []) if path != SETTINGS)

    routes = dict.fromkeys(appended, "append_system_prompt")
    if system_prompt is not None:
        routes[system_prompt] = "system_prompt"
    for skill in skills:
        routes[skill.path] = f"skill:{skill.name}"
        routes.update(dict.fromkeys(skill.resources, f"skill_resource:{skill.name}"))
    routes.update(dict.fromkeys(files, "workspace_file"))
    if spec.get(SETTINGS) is not None:
        routes[SETTINGS] = "settings"
    return AgentPlan(
        system_prompt=system_prompt,
        append_system_prompt=appended,
        skills=skills,
        files=files,
        tools=tools,
        settings=settings,
        routes=routes,
    )


def settings_file(tools: Iterable[str] | None = None, thinking_level: str | None = None) -> HarnessFile:
    """Builds ``pi/settings.json``.

    Args:
        tools: Tool names, or None for the operator's default.
        thinking_level: pi's thinking level, or None for pi's default.

    Returns:
        The settings file.
    """
    settings: dict[str, object] = {}
    if tools is not None:
        settings["defaultTools"] = list(tools)
    if thinking_level is not None:
        settings["defaultThinkingLevel"] = thinking_level
    return HarnessFile("settings", json.dumps(settings, indent=2) + "\n")


def _skills(spec: HarnessSpec, paths: list[str]) -> tuple[SkillPlan, ...]:
    roots = {str(PurePosixPath(p).parent): p for p in paths if PurePosixPath(p).name.lower() == _SKILL_FILE}
    resources: dict[str, list[str]] = {root: [] for root in roots}
    loose: list[str] = []
    for path in paths:
        if path in roots.values():
            continue
        # The nearest enclosing SKILL.md owns a file; a skill inside another skill keeps its own files.
        owners = [root for root in roots if root == "." or path.startswith(root + "/")]
        if owners:
            resources[max(owners, key=len)].append(path)
        else:
            loose.append(path)
    named = [
        (PurePosixPath(root).name or "skill", skill, tuple(resources[root])) for root, skill in sorted(roots.items())
    ]
    named += [(str(PurePosixPath(path).with_suffix("")).replace("/", "-"), path, ()) for path in loose]
    names = [name for name, _, _ in named]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise CampaignRuleError("harness_well_formed", f"two skills are named {duplicates}")
    return tuple(
        SkillPlan(name, path, extra, _description(path, spec.get(path).content))  # type: ignore[union-attr]
        for name, path, extra in named
    )


def _description(path: str, content: str) -> str | None:
    frontmatter = _FRONTMATTER.match(content)
    if frontmatter is not None:
        if _DESCRIPTION.search(frontmatter["body"]) is None:
            raise CampaignRuleError(
                "harness_well_formed", f"{path} has frontmatter with no description, so pi would not load the skill"
            )
        return None
    first = next((line.lstrip("#").strip() for line in content.splitlines() if line.lstrip("#").strip()), "")
    return (first or f"Instructions from {path}")[:_MAX_DESCRIPTION]


def _settings(file: HarnessFile | None, policy: ToolPolicy) -> tuple[dict[str, object], tuple[str, ...]]:
    if file is None:
        return {}, policy.default
    try:
        settings = json.loads(file.content)
    except json.JSONDecodeError as error:
        raise CampaignRuleError("harness_well_formed", f"{SETTINGS} is not JSON: {error}") from None
    if not isinstance(settings, dict):
        raise CampaignRuleError("harness_well_formed", f"{SETTINGS} must be a JSON object")
    unknown = sorted(set(settings) - _SETTINGS_KEYS)
    if unknown:
        raise CampaignRuleError(
            "harness_well_formed", f"{SETTINGS} has keys {unknown}; a harness may set only {sorted(_SETTINGS_KEYS)}"
        )
    tools = settings.get("defaultTools", list(policy.default))
    if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
        raise CampaignRuleError("harness_well_formed", f"{SETTINGS}: defaultTools must be a list of tool names")
    refused = sorted(set(tools) - policy.allowed)
    if refused:
        raise CampaignRuleError(
            "harness_tools_allowed", f"{SETTINGS} enables tools {refused}; allowed are {sorted(policy.allowed)}"
        )
    level = settings.get("defaultThinkingLevel", "off")
    if level not in THINKING_LEVELS:
        raise CampaignRuleError(
            "harness_well_formed", f"{SETTINGS}: defaultThinkingLevel must be one of {THINKING_LEVELS}"
        )
    return settings, tuple(tools)
