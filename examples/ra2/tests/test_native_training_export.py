import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_training_export", ROOT / "league/export_native_training.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(seed=110, choice="B"):
    return {
        "split": "train",
        "map": "mp03t4.map",
        "seed": seed,
        "seat": "red",
        "tick": 9,
        "stage": 2,
        "parts": {"essential": ["Normal own state"], "details": ["detail"], "memory": [], "logs": ["old log"]},
        "questions": {
            "next_action": {
                "type": "choice",
                "instructions": "Choose an owned unit",
                "criteria": {"A": "Cancel selection", "B": "Our tank"},
            }
        },
        "answers": {"next_action": {"choice": choice}},
        "privileged": "DO NOT COPY",
    }


def measure(state, questions):
    return {
        key: {
            "inputTokens": len(state.split()) + 10,
            "stateTokens": len(state.split()),
            "candidatePathSha256": [
                module.hash_value([state, q["instructions"], value]) for value in q["criteria"].values()
            ],
            "promptSha256": module.hash_value([state, q]),
        }
        for key, q in questions.items()
    }


def packed(value, number=1, family="nanojev", limit=2048):
    source = {"manifestSha256": "1" * 64, "samplesSha256": "2" * 64, "path": "/fixture/normal-source"}
    return module.pack_row(value, source, number, family, measure, limit, 105)


@pytest.mark.parametrize("family", ["nanojev", "jev-omni"])
def test_native_export_preserves_all_choices_and_expert_label(family):
    original = row()
    result = packed(original, family=family)
    assert result["questions"] == original["questions"]
    assert result["gold"] == {"next_action": "B"}
    assert result["metadata"]["goldIndex"] == 1
    assert result["gold_label_kind"] == "hard_gold_unspecified"
    assert result["split"] == "train"
    assert result["state"] == "Normal own state\ndetail\nold log"
    assert "DO NOT COPY" not in json.dumps(result)
    assert "privileged" not in result
    assert packed(row(105), family=family)["split"] == "dev"


def test_native_optional_trimming_is_recorded_but_essential_overflow_rejected():
    result = packed(row(), limit=14)
    assert result["state"] == "Normal own state\ndetail"
    assert result["metadata"]["trimmed"] == [{"section": "logs", "text": "old log"}]
    with pytest.raises(ValueError, match="Essential"):
        packed(row(), limit=5)


def test_deduplication_uses_native_encoding_and_removes_validation_overlap():
    train = packed(row(), 1)
    duplicate = packed(row(), 2)
    validation = packed(row(105), 3)
    novel_row = row(105)
    novel_row["parts"]["essential"] = ["A different normal state"]
    novel = packed(novel_row, 4)
    groups, removed = module.separate_encoded([train, duplicate, validation, novel])
    assert groups["train"] == [train] and groups["dev"] == [novel]
    assert {item["reason"] for item in removed} == {"duplicate_native_input", "validation_input_seen_in_training"}
    assert not {x["state_id"] for x in groups["train"]} & {x["state_id"] for x in groups["dev"]}


@pytest.mark.parametrize("seed", [105, 110])
def test_conflicting_labels_for_one_actual_input_are_rejected(seed):
    with pytest.raises(ValueError, match="Conflicting labels"):
        module.separate_encoded([packed(row(), 1), packed(row(seed, "A"), 2)])


def test_reordering_options_changes_native_identity_and_label_index():
    original = packed(row())
    changed = row()
    q = changed["questions"]["next_action"]
    q["criteria"] = {"B": "Our tank", "A": "Cancel selection"}
    result = packed(changed)
    assert result["state_id"] != original["state_id"]
    assert result["metadata"]["goldIndex"] == 0


def approved(tmp_path, seed=110):
    directory = tmp_path / "episode"
    (directory / "source").mkdir(parents=True)
    (directory / "source/policy.py").write_text("fixture normal policy\n")
    manifest = {
        "kind": "teacher-demonstration",
        "notModelEvaluation": True,
        "split": "train",
        "map": "mp03t4.map",
        "seed": seed,
        "seat": "red",
        "sourceSha256": {"policy.py": module.digest(directory / "source/policy.py")},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "result.json").write_text(
        json.dumps(manifest | {"finalPlayers": "PRIVATE OBSERVER MUST NOT BECOME MODEL INPUT"})
    )
    (directory / "samples.jsonl").write_text(json.dumps(row(seed)) + "\n")
    source = {
        "path": str(directory),
        "map": "mp03t4.map",
        "seed": seed,
        "seat": "red",
        "manifestSha256": module.digest(directory / "manifest.json"),
        "samplesSha256": module.digest(directory / "samples.jsonl"),
    }
    check = tmp_path / "approved.json"
    check.write_text(json.dumps({"passed": True, "developmentOrHeldoutUsed": False, "trainingSources": [source]}))
    return check, directory, manifest


def test_registered_samples_source_and_episode_split_are_rechecked(tmp_path):
    check, directory, manifest = approved(tmp_path)
    protocol = {"train": {"maps": ["mp03t4.map"], "seeds": [105, 110]}}
    assert len(module.verified_sources(check, protocol)) == 1
    module.validate_row(row(), manifest)
    with pytest.raises(ValueError, match="does not belong"):
        module.validate_row(row(1000), manifest)
    (directory / "samples.jsonl").write_text("changed input")
    with pytest.raises(ValueError, match="data changed"):
        module.verified_sources(check, protocol)


def test_approved_registry_does_not_override_engine_heldout_boundary(tmp_path):
    check, _, _ = approved(tmp_path, seed=1000)
    with pytest.raises(ValueError, match="non-training"):
        module.verified_sources(check, {"train": {"maps": ["mp03t4.map"], "seeds": [105, 110]}})
