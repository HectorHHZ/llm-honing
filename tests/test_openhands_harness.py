"""How an OpenHands harness resolves into an agent plan, and how local skills and MCP configs come in."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from honing.backends.openhands import (
    AGENT_SETTINGS,
    MCP_SETTINGS,
    ToolPolicy,
    agent_settings,
    harness_from_directory,
    load_mcp_registry,
    mcp_settings,
    plan_agent,
    skill_files,
)
from honing.core import CampaignRuleError, ContractError, HarnessFile, HarnessSpec

FETCH = {"command": "uvx", "args": ["mcp-server-fetch"]}
POLICY = ToolPolicy(
    allowed=frozenset({"terminal", "file_editor", "grep", "my_search"}),
    mcp_registry={"fetch": FETCH},
    tool_modules=("my_tools",),
)


def _spec(*files: tuple[str, HarnessFile]) -> HarnessSpec:
    return HarnessSpec(agent="openhands", files=files)


def _skill(text: str = "---\nname: x\ndescription: d\n---\nbody\n") -> HarnessFile:
    return HarnessFile("skill", text)


@pytest.mark.unit
def test_each_kind_takes_its_route() -> None:
    spec = _spec(
        ("b.md", HarnessFile("instructions", "second")),
        ("AGENTS.md", HarnessFile("instructions", "first")),
        ("skills/search/SKILL.md", _skill()),
        ("skills/search/scripts/find.sh", _skill("grep -rn \"$1\" .\n")),
        ("skills/search/references/tables.md", _skill("units are in headers\n")),
        ("notes/style.md", _skill("be brief\n")),
        ("config/extra.yaml", HarnessFile("settings", "a: 1\n")),
        (AGENT_SETTINGS, agent_settings(["grep", "my_search"], 7)),
        (MCP_SETTINGS, mcp_settings(["fetch"])),
    )

    plan = plan_agent(spec, POLICY, max_iterations=30)

    assert plan.instructions == ("AGENTS.md", "b.md")
    assert [(s.name, s.path, s.resources) for s in plan.skills] == [
        ("search", "skills/search/SKILL.md", ("skills/search/references/tables.md", "skills/search/scripts/find.sh")),
        ("notes-style", "notes/style.md", ()),
    ]
    assert plan.files == ("config/extra.yaml",)
    assert (plan.tools, plan.tool_modules, plan.max_iterations) == (("grep", "my_search"), ("my_tools",), 7)
    assert plan.mcp_servers == {"fetch": FETCH}
    assert plan.routes == {
        "AGENTS.md": "system_message",
        "b.md": "system_message",
        "skills/search/SKILL.md": "skill:search",
        "skills/search/references/tables.md": "skill_resource:search",
        "skills/search/scripts/find.sh": "skill_resource:search",
        "notes/style.md": "skill:notes-style",
        "config/extra.yaml": "workspace_file",
        AGENT_SETTINGS: "tools",
        MCP_SETTINGS: "mcp",
    }
    assert set(plan.routes) == {path for path, _ in spec.files}
    assert json.loads(json.dumps(plan.to_json()))["skills"][0]["name"] == "search"


@pytest.mark.unit
def test_a_skill_inside_a_skill_keeps_its_own_resources() -> None:
    spec = _spec(
        ("skills/a/SKILL.md", _skill()),
        ("skills/a/helper.md", _skill("a's\n")),
        ("skills/a/b/SKILL.md", _skill()),
        ("skills/a/b/helper.md", _skill("b's\n")),
    )

    skills = {s.name: s.resources for s in plan_agent(spec, POLICY, 30).skills}

    assert skills == {"a": ("skills/a/helper.md",), "b": ("skills/a/b/helper.md",)}


@pytest.mark.unit
def test_without_settings_the_agent_gets_the_operator_s_defaults() -> None:
    plan = plan_agent(_spec(("AGENTS.md", HarnessFile("instructions", "x"))), ToolPolicy(), 30)

    assert (plan.tools, plan.mcp_servers, plan.max_iterations) == (("terminal", "file_editor"), {}, 30)


@pytest.mark.unit
def test_a_harness_may_lower_the_step_limit_but_not_raise_it() -> None:
    plan = plan_agent(_spec((AGENT_SETTINGS, agent_settings(max_iterations=99))), POLICY, 30)

    assert plan.max_iterations == 30


@pytest.mark.unit
@pytest.mark.parametrize(
    ("files", "rule", "message"),
    [
        (((AGENT_SETTINGS, agent_settings(["browser_use"])),), "harness_tools_allowed", "enables tools"),
        (((MCP_SETTINGS, mcp_settings(["shell"])),), "harness_tools_allowed", "enables MCP servers \\['shell'\\]"),
        (
            ((MCP_SETTINGS, HarnessFile("settings", json.dumps({"mcpServers": {"x": {"command": "rm"}}}))),),
            "harness_well_formed",
            "enables servers by name",
        ),
        (((AGENT_SETTINGS, HarnessFile("settings", '{"shell": "bash"}')),), "harness_well_formed", "unknown keys"),
        (((AGENT_SETTINGS, HarnessFile("settings", "{nope")),), "harness_well_formed", "is not JSON"),
        (((AGENT_SETTINGS, agent_settings(max_iterations=0)),), "harness_well_formed", "positive integer"),
        ((("a/x/SKILL.md", _skill()), ("b/x/SKILL.md", _skill())), "harness_well_formed", "named \\['x'\\]"),
        ((("hooks/pre.sh", HarnessFile("hook", "echo")),), "harness_well_formed", "cannot run \\['hook'\\]"),
    ],
)
def test_a_harness_asking_for_more_than_the_operator_allows_is_refused(
    files: tuple[tuple[str, HarnessFile], ...], rule: str, message: str
) -> None:
    with pytest.raises(CampaignRuleError, match=message) as refused:
        plan_agent(_spec(*files), POLICY, 30)

    assert refused.value.rule == rule


@pytest.mark.unit
def test_another_agent_is_refused() -> None:
    with pytest.raises(ContractError, match="not 'claude_code'"):
        plan_agent(HarnessSpec(agent="claude_code"), POLICY, 30)


@pytest.mark.unit
def test_a_local_skills_directory_keeps_its_layout(tmp_path: Path) -> None:
    (tmp_path / "search" / "scripts").mkdir(parents=True)
    (tmp_path / "search" / "SKILL.md").write_text("---\nname: search\ndescription: d\n---\nUse grep.\n")
    (tmp_path / "search" / "scripts" / "find.sh").write_text("grep -rn x .\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref\n")

    files = dict(skill_files(tmp_path))

    assert sorted(files) == ["skills/search/SKILL.md", "skills/search/scripts/find.sh"]
    assert {file.kind for file in files.values()} == {"skill"}
    plan = plan_agent(_spec(*files.items()), POLICY, 30)
    assert plan.skills[0].resources == ("skills/search/scripts/find.sh",)


@pytest.mark.unit
def test_a_binary_skill_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")

    with pytest.raises(ContractError, match="not UTF-8 text"):
        skill_files(tmp_path)


@pytest.mark.unit
def test_a_harness_directory_maps_onto_kinds(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Answer from the documents.\n")
    (tmp_path / "openhands").mkdir()
    (tmp_path / "openhands" / "agent.json").write_text('{"tools": ["grep"]}\n')
    (tmp_path / "skills" / "search").mkdir(parents=True)
    (tmp_path / "skills" / "search" / "SKILL.md").write_text("---\nname: search\ndescription: d\n---\nx\n")

    spec = harness_from_directory(tmp_path)

    assert {path: file.kind for path, file in spec.files} == {
        "AGENTS.md": "instructions",
        "openhands/agent.json": "settings",
        "skills/search/SKILL.md": "skill",
    }
    (tmp_path / "notes.txt").write_text("stray\n")
    with pytest.raises(ContractError, match="outside the harness layout: \\['notes.txt'\\]"):
        harness_from_directory(tmp_path)


@pytest.mark.unit
def test_an_existing_mcp_config_serves_as_the_registry(tmp_path: Path) -> None:
    config = tmp_path / ".mcp.json"
    config.write_text(json.dumps({"mcpServers": {"fetch": FETCH}}))

    assert load_mcp_registry(config) == {"fetch": FETCH}
    config.write_text(json.dumps({"servers": {}}))
    with pytest.raises(ContractError, match="no mcpServers"):
        load_mcp_registry(config)


@pytest.mark.unit
def test_a_default_tool_must_be_allowed() -> None:
    with pytest.raises(ContractError, match="must be allowed"):
        ToolPolicy(allowed=frozenset({"terminal"}), default=("terminal", "file_editor"))
