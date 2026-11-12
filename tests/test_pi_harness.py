"""How a pi harness resolves into an agent plan: each file kind's route, and what is refused before an episode."""

from __future__ import annotations

import json

import pytest

from honing.backends.pi import SETTINGS, ToolPolicy, plan_agent, settings_file
from honing.core import CampaignRuleError, ContractError, HarnessFile, HarnessSpec

POLICY = ToolPolicy(allowed=frozenset({"read", "bash", "edit", "write", "grep"}))


def _spec(*files: tuple[str, HarnessFile]) -> HarnessSpec:
    return HarnessSpec(agent="pi", files=files)


def _skill(text: str = "---\nname: x\ndescription: d\n---\nbody\n") -> HarnessFile:
    return HarnessFile("skill", text)


@pytest.mark.unit
def test_each_kind_takes_its_route() -> None:
    spec = _spec(
        ("b.md", HarnessFile("instructions", "second")),
        ("AGENTS.md", HarnessFile("instructions", "first")),
        ("SYSTEM.md", HarnessFile("instructions", "You are a careful analyst.")),
        ("skills/search/SKILL.md", _skill()),
        ("skills/search/scripts/find.sh", _skill('grep -rn "$1" .\n')),
        ("notes/style.md", _skill("# Be brief\n\nAnswer in one line.\n")),
        ("config/extra.yaml", HarnessFile("settings", "a: 1\n")),
        (SETTINGS, settings_file(["read", "grep"], "low")),
    )

    plan = plan_agent(spec, POLICY, reasoning=True)

    assert (plan.system_prompt, plan.append_system_prompt) == ("SYSTEM.md", ("AGENTS.md", "b.md"))
    assert [(s.name, s.path, s.resources, s.description) for s in plan.skills] == [
        ("search", "skills/search/SKILL.md", ("skills/search/scripts/find.sh",), None),
        ("notes-style", "notes/style.md", (), "Be brief"),
    ]
    assert plan.files == ("config/extra.yaml",)
    assert plan.tools == ("read", "grep")
    assert plan.settings == {"defaultTools": ["read", "grep"], "defaultThinkingLevel": "low"}
    assert plan.routes == {
        "SYSTEM.md": "system_prompt",
        "AGENTS.md": "append_system_prompt",
        "b.md": "append_system_prompt",
        "skills/search/SKILL.md": "skill:search",
        "skills/search/scripts/find.sh": "skill_resource:search",
        "notes/style.md": "skill:notes-style",
        "config/extra.yaml": "workspace_file",
        SETTINGS: "settings",
    }
    assert set(plan.routes) == {path for path, _ in spec.files}
    assert json.loads(json.dumps(plan.to_json()))["skills"][1]["description"] == "Be brief"


@pytest.mark.unit
def test_without_settings_the_agent_gets_the_default_tools_and_keeps_pis_prompt() -> None:
    plan = plan_agent(_spec(("AGENTS.md", HarnessFile("instructions", "Solve it."))), POLICY)

    assert plan.system_prompt is None
    assert plan.tools == POLICY.default
    assert plan.settings == {}


@pytest.mark.unit
def test_a_skill_file_with_no_frontmatter_is_described_by_its_first_line_or_its_path() -> None:
    plan = plan_agent(
        _spec(("a.md", _skill("\n\n  Check units first.\nThen answer.\n")), ("b.md", _skill(""))), POLICY
    )

    assert [s.description for s in plan.skills] == ["Check units first.", "Instructions from b.md"]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("files", "rule", "message"),
    [
        ((("hook.sh", HarnessFile("hook", "exit 0\n")),), "harness_well_formed", "cannot run"),
        (((SETTINGS, settings_file(["read", "browser"])),), "harness_tools_allowed", "browser"),
        (((SETTINGS, HarnessFile("settings", "{")),), "harness_well_formed", "not JSON"),
        (((SETTINGS, HarnessFile("settings", "[]")),), "harness_well_formed", "JSON object"),
        (((SETTINGS, HarnessFile("settings", '{"shellPath": "/bin/evil"}')),), "harness_well_formed", "shellPath"),
        (((SETTINGS, HarnessFile("settings", '{"defaultTools": "read"}')),), "harness_well_formed", "list of tool"),
        (((SETTINGS, settings_file(thinking_level="huge")),), "harness_well_formed", "defaultThinkingLevel"),
        ((("s/SKILL.md", _skill("---\nname: s\n---\nbody\n")),), "harness_well_formed", "no description"),
        (
            (("s/SKILL.md", _skill('---\nname: s\ndescription: ""\n---\nbody\n')),),
            "harness_well_formed",
            "no description",
        ),
        (((SETTINGS, settings_file(thinking_level="high")),), "harness_well_formed", "not a reasoning model"),
        ((("a/x/SKILL.md", _skill()), ("b/x/SKILL.md", _skill())), "harness_well_formed", "two skills"),
        ((("s/SKILL.md", _skill()), (SETTINGS, settings_file(["edit"]))), "harness_well_formed", "read or bash"),
    ],
)
def test_a_harness_pi_could_not_run_as_intended_is_refused(
    files: tuple[tuple[str, HarnessFile], ...], rule: str, message: str
) -> None:
    with pytest.raises(CampaignRuleError, match=message) as refusal:
        plan_agent(_spec(*files), POLICY)

    assert refusal.value.rule == rule


@pytest.mark.unit
def test_a_harness_for_another_agent_is_a_contract_error() -> None:
    with pytest.raises(ContractError, match="runs agent 'pi'"):
        plan_agent(HarnessSpec(agent="openhands"), POLICY)


@pytest.mark.unit
def test_default_tools_must_be_allowed() -> None:
    with pytest.raises(ContractError, match="must be allowed"):
        ToolPolicy(allowed=frozenset({"read"}), default=("read", "bash"))


@pytest.mark.unit
def test_a_thinking_level_of_off_needs_no_reasoning_model() -> None:
    plan = plan_agent(_spec((SETTINGS, settings_file(thinking_level="off"))), POLICY)

    assert plan.settings == {"defaultThinkingLevel": "off"}
