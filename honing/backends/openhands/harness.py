"""How a HarnessSpec becomes an OpenHands agent.

Every decision about what a harness file means is made here, on Honing's side,
so it can be checked before an episode starts and tested without OpenHands.
The worker receives the resulting :class:`AgentPlan` and only applies it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from honing.core import CampaignRuleError, ContractError, HarnessFile, HarnessFileKind, HarnessSpec, require

AGENT = "openhands"
AGENT_SETTINGS = "openhands/agent.json"
MCP_SETTINGS = "openhands/mcp.json"
DEFAULT_TOOLS = ("terminal", "file_editor")
SKILLS_DIR = "skills"

_SKILL_FILE = "skill.md"
_AGENT_KEYS = frozenset({"tools", "max_iterations"})


@dataclass(frozen=True)
class ToolPolicy:
    """What the operator lets a harness give the agent.

    A harness chooses among these; it never adds to them. That lets an
    evolution method change the agent's toolset without being able to make it
    run an arbitrary command.

    Attributes:
        allowed: Tools a harness may enable in ``openhands/agent.json``.
        default: Tools the agent gets when the harness names none.
        mcp_registry: MCP server definitions by name, in OpenHands' format. A
            harness enables servers by name in ``openhands/mcp.json``.
        tool_modules: Modules the worker imports before resolving tools, so a
            custom tool registered there with ``openhands.sdk.register_tool``
            can be allowed by name.
    """

    allowed: frozenset[str] = frozenset(DEFAULT_TOOLS)
    default: tuple[str, ...] = DEFAULT_TOOLS
    mcp_registry: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    tool_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require(set(self.default) <= self.allowed, f"default tools {list(self.default)} must be allowed")


@dataclass(frozen=True)
class SkillPlan:
    """One skill: its name, its defining file, and the files that come with it."""

    name: str
    path: str
    resources: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentPlan:
    """A harness resolved against a :class:`ToolPolicy`: exactly what the worker builds.

    Attributes:
        instructions: Instruction files, appended in this order to the system message.
        skills: Skills to load, each with its resource files.
        files: Other settings files, copied into the agent's working directory.
        tools: Tool names the agent gets.
        tool_modules: Modules to import before resolving ``tools``.
        mcp_servers: MCP server definitions to attach, by name.
        max_iterations: The conversation's step limit.
        routes: How each harness path reaches the agent, for the delivery report.
    """

    instructions: tuple[str, ...]
    skills: tuple[SkillPlan, ...]
    files: tuple[str, ...]
    tools: tuple[str, ...]
    tool_modules: tuple[str, ...]
    mcp_servers: Mapping[str, Mapping[str, object]]
    max_iterations: int
    routes: Mapping[str, str]

    def to_json(self) -> dict[str, object]:
        """Returns the plan as JSON-native values, the form the worker reads."""
        return {
            "instructions": list(self.instructions),
            "skills": [{"name": s.name, "path": s.path, "resources": list(s.resources)} for s in self.skills],
            "files": list(self.files),
            "tools": list(self.tools),
            "tool_modules": list(self.tool_modules),
            "mcp_servers": {name: dict(server) for name, server in self.mcp_servers.items()},
            "max_iterations": self.max_iterations,
            "routes": dict(self.routes),
        }


def plan_agent(spec: HarnessSpec, policy: ToolPolicy, max_iterations: int) -> AgentPlan:
    """Resolves a harness into the agent the worker builds.

    Instructions are read in path order. A skill file named ``SKILL.md``
    defines a skill named after its directory, and every other skill file under
    that directory is its resource. A skill file under no ``SKILL.md`` is a
    skill of its own, named after its path. ``openhands/agent.json`` and
    ``openhands/mcp.json`` choose tools and MCP servers from ``policy``; other
    settings files are passed to the agent as files.

    Args:
        spec: The harness.
        policy: What the operator allows.
        max_iterations: The runner's step limit; the harness may lower it.

    Returns:
        The resolved plan.

    Raises:
        ContractError: The harness is for another agent.
        CampaignRuleError: ``harness_tools_allowed`` when the harness asks for
            a tool or MCP server ``policy`` does not allow, ``harness_well_formed``
            when a settings file or skill layout is invalid.
    """
    if spec.agent != AGENT:
        raise ContractError(f"the OpenHands runner runs agent {AGENT!r}, not {spec.agent!r}")
    by_kind: dict[HarnessFileKind, list[str]] = {}
    for path, file in sorted(spec.files, key=lambda item: item[0]):
        by_kind.setdefault(file.kind, []).append(path)
    unsupported = sorted(set(by_kind) - {"instructions", "skill", "settings"})
    if unsupported:
        raise CampaignRuleError("harness_well_formed", f"the OpenHands runner cannot run {unsupported} files")

    skills = _skills(by_kind.get("skill", []))
    settings = by_kind.get("settings", [])
    tools, iterations = _agent_settings(spec.get(AGENT_SETTINGS), policy, max_iterations)
    servers = _mcp_settings(spec.get(MCP_SETTINGS), policy)
    files = tuple(path for path in settings if path not in (AGENT_SETTINGS, MCP_SETTINGS))

    routes = {path: "system_message" for path in by_kind.get("instructions", [])}
    for skill in skills:
        routes[skill.path] = f"skill:{skill.name}"
        routes.update({resource: f"skill_resource:{skill.name}" for resource in skill.resources})
    routes.update({path: "workspace_file" for path in files})
    if AGENT_SETTINGS in settings:
        routes[AGENT_SETTINGS] = "tools"
    if MCP_SETTINGS in settings:
        routes[MCP_SETTINGS] = "mcp"
    return AgentPlan(
        instructions=tuple(by_kind.get("instructions", [])),
        skills=skills,
        files=files,
        tools=tools,
        tool_modules=policy.tool_modules,
        mcp_servers=servers,
        max_iterations=iterations,
        routes=routes,
    )


def load_mcp_registry(path: Path) -> dict[str, dict[str, object]]:
    """Reads MCP server definitions from a ``{"mcpServers": {...}}`` file.

    That is the format Claude Code, Cursor, and OpenHands use, so an existing
    configuration can serve as the operator's registry.

    Args:
        path: The JSON file.

    Returns:
        The server definitions by name.

    Raises:
        ContractError: The file has no ``mcpServers`` object.
    """
    servers = json.loads(path.read_text(encoding="utf-8")).get("mcpServers")
    if not isinstance(servers, dict) or not all(isinstance(server, dict) for server in servers.values()):
        raise ContractError(f"{path} has no mcpServers object of server definitions")
    return servers


def skill_files(directory: Path, prefix: str = SKILLS_DIR) -> tuple[tuple[str, HarnessFile], ...]:
    """Reads a local skills directory as harness files.

    Each skill keeps its layout under ``prefix``: ``<directory>/<name>/SKILL.md``
    becomes ``<prefix>/<name>/SKILL.md``, with its scripts and references beside it.

    Args:
        directory: A directory of skills, such as ``~/.claude/skills``.
        prefix: Where the skills go in the harness.

    Returns:
        ``(path, file)`` pairs to add to a :class:`~honing.core.HarnessSpec`.

    Raises:
        ContractError: A file is not UTF-8 text, which a harness cannot hold.
    """
    directory = directory.expanduser()
    return tuple(
        (str(PurePosixPath(prefix) / path.relative_to(directory).as_posix()), HarnessFile("skill", _text(path)))
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )


def harness_from_directory(directory: Path) -> HarnessSpec:
    """Reads an OpenHands harness laid out as files on disk.

    Top-level ``*.md`` files are instructions, everything under ``skills/`` is
    a skill file, and every file under ``openhands/`` is settings.

    Args:
        directory: The harness directory.

    Returns:
        The harness.

    Raises:
        ContractError: A file is outside that layout, or is not UTF-8 text.
    """
    directory = directory.expanduser()
    files: list[tuple[str, HarnessFile]] = []
    stray: list[str] = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        relative = path.relative_to(directory)
        if any(part.startswith(".") for part in relative.parts):
            continue
        top = relative.parts[0]
        if len(relative.parts) == 1 and path.suffix == ".md":
            files.append((relative.as_posix(), HarnessFile("instructions", _text(path))))
        elif top == "openhands":
            files.append((relative.as_posix(), HarnessFile("settings", _text(path))))
        elif top != SKILLS_DIR:
            stray.append(relative.as_posix())
    if stray:
        raise ContractError(f"{directory} has files outside the harness layout: {stray}")
    if (directory / SKILLS_DIR).is_dir():
        files.extend(skill_files(directory / SKILLS_DIR))
    return HarnessSpec(agent=AGENT, files=tuple(files))


def agent_settings(tools: Iterable[str] | None = None, max_iterations: int | None = None) -> HarnessFile:
    """Builds ``openhands/agent.json``.

    Args:
        tools: Tool names, or None for the operator's default.
        max_iterations: A step limit, or None for the runner's.

    Returns:
        The settings file.
    """
    settings: dict[str, object] = {}
    if tools is not None:
        settings["tools"] = list(tools)
    if max_iterations is not None:
        settings["max_iterations"] = max_iterations
    return HarnessFile("settings", json.dumps(settings, indent=2) + "\n")


def mcp_settings(servers: Iterable[str]) -> HarnessFile:
    """Builds ``openhands/mcp.json``, enabling registry servers by name.

    Args:
        servers: Names from the operator's MCP registry.

    Returns:
        The settings file.
    """
    return HarnessFile("settings", json.dumps({"enabled": list(servers)}, indent=2) + "\n")


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ContractError(f"{path} is not UTF-8 text; a harness holds text files only") from None


def _skills(paths: list[str]) -> tuple[SkillPlan, ...]:
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
    plans = [
        SkillPlan(PurePosixPath(root).name or "skill", skill, tuple(resources[root]))
        for root, skill in sorted(roots.items())
    ]
    plans += [SkillPlan(str(PurePosixPath(path).with_suffix("")).replace("/", "-"), path) for path in loose]
    names = [plan.name for plan in plans]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise CampaignRuleError("harness_well_formed", f"two skills are named {duplicates}")
    return tuple(plans)


def _settings_object(file: HarnessFile | None, path: str) -> dict[str, object] | None:
    if file is None:
        return None
    try:
        settings = json.loads(file.content)
    except json.JSONDecodeError as error:
        raise CampaignRuleError("harness_well_formed", f"{path} is not JSON: {error}") from None
    if not isinstance(settings, dict):
        raise CampaignRuleError("harness_well_formed", f"{path} must be a JSON object")
    return settings


def _agent_settings(file: HarnessFile | None, policy: ToolPolicy, limit: int) -> tuple[tuple[str, ...], int]:
    settings = _settings_object(file, AGENT_SETTINGS)
    if settings is None:
        return policy.default, limit
    unknown = sorted(set(settings) - _AGENT_KEYS)
    if unknown:
        raise CampaignRuleError("harness_well_formed", f"{AGENT_SETTINGS} has unknown keys {unknown}")
    tools = settings.get("tools", list(policy.default))
    if not isinstance(tools, list) or not all(isinstance(tool, str) for tool in tools):
        raise CampaignRuleError("harness_well_formed", f"{AGENT_SETTINGS}: tools must be a list of tool names")
    refused = sorted(set(tools) - policy.allowed)
    if refused:
        raise CampaignRuleError(
            "harness_tools_allowed", f"{AGENT_SETTINGS} enables tools {refused}; allowed are {sorted(policy.allowed)}"
        )
    iterations = settings.get("max_iterations", limit)
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations <= 0:
        raise CampaignRuleError("harness_well_formed", f"{AGENT_SETTINGS}: max_iterations must be a positive integer")
    return tuple(tools), min(limit, iterations)


def _mcp_settings(file: HarnessFile | None, policy: ToolPolicy) -> dict[str, Mapping[str, object]]:
    settings = _settings_object(file, MCP_SETTINGS)
    if settings is None:
        return {}
    if set(settings) != {"enabled"}:
        raise CampaignRuleError(
            "harness_well_formed", f"{MCP_SETTINGS} must hold only 'enabled'; a harness enables servers by name"
        )
    enabled = settings["enabled"]
    if not isinstance(enabled, list) or not all(isinstance(name, str) for name in enabled):
        raise CampaignRuleError("harness_well_formed", f"{MCP_SETTINGS}: enabled must be a list of server names")
    refused = sorted(set(enabled) - set(policy.mcp_registry))
    if refused:
        raise CampaignRuleError(
            "harness_tools_allowed",
            f"{MCP_SETTINGS} enables MCP servers {refused}; the registry has {sorted(policy.mcp_registry)}",
        )
    return {name: policy.mcp_registry[name] for name in enabled}
