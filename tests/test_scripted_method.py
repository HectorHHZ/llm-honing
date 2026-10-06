"""The scripted harness method: fixed edits, one per move, each judged by the gate."""

from __future__ import annotations

from pathlib import Path

import pytest
from integrations.officeqa.scripted import ScriptedEdit, ScriptedMethod, method

from honing.campaign import ScriptedScheduler
from honing.core import ContractError, Move, protocols
from honing.e2e.components import PipelineContext
from tests.fakes.builder import build_campaign
from tests.fakes.harness import INSTRUCTIONS_PATH

VERIFY = ScriptedEdit(INSTRUCTIONS_PATH, "instructions", "Solve the task.\nVerify the result.\n", "ask to verify")
SKILL = ScriptedEdit("skills/sums/SKILL.md", "skill", "---\nname: sums\ndescription: d\n---\nAdd carefully.\n")


@pytest.mark.unit
def test_each_move_proposes_the_next_edit_and_the_gate_decides(tmp_path: Path) -> None:
    scripted = ScriptedMethod([VERIFY, SKILL])
    fixture = build_campaign(tmp_path, method=scripted)
    seed = fixture.campaign.chain.head().harness

    first, second, third = fixture.campaign.run(ScriptedScheduler([Move(kind="harness")] * 3))

    # The fake policy rewards "verify" in the instructions and ignores skills.
    assert first.release is not None
    assert first.release.harness is not None and seed is not None
    assert first.release.harness.get(INSTRUCTIONS_PATH).content == VERIFY.content  # type: ignore[union-attr]
    assert second.decision is not None and not second.decision.selected
    assert third.skipped_reason == "scripted proposed nothing"
    assert fixture.campaign.chain.head().method_state == {"schema_version": 1, "next": 1, "verdicts": ["select"]}


@pytest.mark.unit
def test_the_factory_writes_json_settings_and_declares_the_kinds_it_edits() -> None:
    built = method(
        None,
        edits=[
            {"path": "openhands/agent.json", "kind": "settings", "json": {"tools": ["grep"]}},
            {"path": "AGENTS.md", "kind": "instructions", "content": "x"},
        ],
    )

    assert isinstance(built, protocols.EvolveMethod)
    assert built.edits[0].content == '{\n  "tools": [\n    "grep"\n  ]\n}\n'
    assert built.requires.file_kinds == frozenset({"settings", "instructions"})


@pytest.mark.unit
def test_an_edit_can_come_from_a_local_file(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\nx\n")

    built = method(
        PipelineContext(root=tmp_path, work_dir=tmp_path),
        edits=[{"path": "skills/s/SKILL.md", "kind": "skill", "file": "{root}/SKILL.md"}],
    )

    assert built.edits[0].content.startswith("---\nname: s")
    with pytest.raises(ContractError, match="exactly one of content, json, and file"):
        method(None, edits=[{"path": "a", "kind": "skill", "content": "x", "file": "y"}])


@pytest.mark.unit
def test_a_method_needs_an_edit() -> None:
    with pytest.raises(Exception, match="at least one edit"):
        ScriptedMethod([])
