"""The three REA-Issac fault builders change only their named component."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from honing.backends.rea_issac.catalog import OPTIONS, action_for
from honing.backends.rea_issac.harness import parse_harness, seed_harness
from honing.backends.rea_issac.model import make_request
from honing.backends.rea_issac.scenarios import ReaIssacScenario
from honing.backends.rea_issac_agent import fault_probe, faults
from honing.backends.rea_issac_agent.faults import (
    OPPOSITE,
    WEIGHTS_REVERSE_MOVE,
    build_reversed_move_dataset,
    reverse_movement_harness,
    swap_move_axes,
)
from honing.backends.rea_issac_agent.make_agent_demos import build_dataset
from honing.backends.rea_issac_agent.train_agent import load_batch
from integrations.rea_issac.engine import Engine

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "honing/backends/rea_issac_agent"


@pytest.mark.unit
def test_fault_freeze_rejects_changed_clean_agent_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = json.loads((AGENT / "fault_freeze.json").read_text())
    for name in frozen["files"]:
        copied = tmp_path / name
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes((ROOT / name).read_bytes())
    package = tmp_path / "honing/backends/rea_issac_agent"
    (package / "fault_freeze.json").write_text(json.dumps(frozen))
    monkeypatch.setattr(faults, "PACKAGE", package)
    monkeypatch.setattr(faults, "ROOT", tmp_path)
    assert faults.verify_clean_baseline()["baseline_commit"] == frozen["baseline_commit"]
    (package / "config/questions.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen REA-Issac agent file changed"):
        faults.verify_clean_baseline()


@pytest.mark.unit
def test_harness_fault_changes_only_model_facing_movement_instruction() -> None:
    clean = seed_harness(AGENT)
    faulty = reverse_movement_harness(clean)
    observation = Engine(1002).observe()
    scenario = ReaIssacScenario(1002, 900, 3)
    before = make_request(observation, scenario, parse_harness(clean), [])
    after = make_request(observation, scenario, parse_harness(faulty), [])
    assert clean.spec_id != faulty.spec_id
    assert before["parts"] == after["parts"]
    assert before["questions"]["move"]["criteria"] == after["questions"]["move"]["criteria"]
    assert before["questions"]["move"]["instructions"] != after["questions"]["move"]["instructions"]
    assert before["questions"]["shoot"] == after["questions"]["shoot"]
    assert before["questions"]["bomb"] == after["questions"]["bomb"]


@pytest.mark.unit
def test_controller_fault_changes_only_executed_movement() -> None:
    before = {"move_x": 1, "move_y": 0, "shoot_x": -1, "shoot_y": 1, "bomb": False}
    after = swap_move_axes(before)
    assert after == {**before, "move_x": 0, "move_y": 1}
    assert before["move_x"] == 1


@pytest.mark.unit
def test_weights_fault_dataset_has_real_wrong_move_labels_and_separate_hash(tmp_path: Path) -> None:
    source, faulty = tmp_path / "source", tmp_path / "faulty"
    original = build_dataset(source, per_seed=60, max_state_chars=12000, selection="balanced_rooms")
    summary = build_reversed_move_dataset(source, faulty)
    batch, loaded = load_batch(faulty)
    original_rows = [json.loads(line) for line in (source / "decisions.jsonl").read_text().splitlines()]
    fault_rows = [json.loads(line) for line in (faulty / "decisions.jsonl").read_text().splitlines()]
    by_identity = {(row["seed"], row["tick"]): row for row in original_rows}
    assert summary == loaded
    assert summary["fault_family"] == WEIGHTS_REVERSE_MOVE
    assert summary["source_decisions_sha256"] == original["decisions_sha256"]
    assert summary["decisions_sha256"] != original["decisions_sha256"]
    assert len(batch.decisions) == len(fault_rows) < len(original_rows)
    assert sum(row["split"] == "validation" for row in fault_rows) >= 30
    for row in fault_rows:
        clean = by_identity[row["seed"], row["tick"]]
        assert row["request"]["parts"] == clean["request"]["parts"]
        assert row["request"]["questions"] == {"move": clean["request"]["questions"]["move"]}
        assert row["chosen"] == {"move": OPPOSITE[clean["chosen"]["move"]]}
        assert row["chosen"]["move"] != clean["chosen"]["move"]
        assert row["source_release"] == WEIGHTS_REVERSE_MOVE


@pytest.mark.unit
def test_same_state_probe_checks_component_boundary_and_excludes_teacher_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "demos"
    build_dataset(dataset, per_seed=60, max_state_chars=12000, selection="balanced_rooms")
    selected = json.loads((AGENT / "fault_freeze.json").read_text())["checkpoint_sha256"]

    class FakeClient:
        def __init__(self, url: str) -> None:
            self.url = url

        def health(self) -> dict[str, Any]:
            revision = "b" * 64 if self.url == "weights" else selected
            return {"metadata": {"revision": revision, "weightsSha256": revision}}

        def decide(self, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            move = (
                "N"
                if self.url == "weights"
                else "W" if "Movement controls are inverted" in request["questions"]["move"]["instructions"] else "E"
            )
            choices = {"move": move, "shoot": "NONE", "bomb": "NO"}
            reply = {
                "id": request["id"],
                "stateText": request["parts"]["essential"][0],
                "trimmed": [],
                "tokenAudit": {},
                "answers": {
                    name: {
                        "choice": choices[name],
                        "probabilities": {option: float(option == choices[name]) for option in options},
                    }
                    for name, options in OPTIONS.items()
                },
            }
            return action_for(choices), reply

    monkeypatch.setattr(fault_probe, "DecisionClient", FakeClient)
    clean, harness, weights = (tmp_path / name for name in ("clean", "harness", "weights"))
    fault_probe.record_probes(dataset, clean, decision_server="clean", expected_revision=selected)
    fault_probe.record_probes(
        dataset,
        harness,
        decision_server="clean",
        expected_revision=selected,
        harness_fault="harness-reverse-move-v1",
    )
    fault_probe.record_probes(dataset, weights, decision_server="weights", expected_revision="b" * 64)
    assert fault_probe.compare_probes(clean, harness, component="harness")["selected_move_changes"] > 0
    assert fault_probe.compare_probes(clean, weights, component="weights")["move_probability_effects_over_0_01"] > 0
    assert "chosen" not in json.loads((clean / "probes.jsonl").read_text().splitlines()[0])
    with pytest.raises(ValueError, match="clean probe has a harness fault"):
        fault_probe.compare_probes(harness, weights, component="weights")
    with (weights / "probes.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        fault_probe.compare_probes(clean, weights, component="weights")
