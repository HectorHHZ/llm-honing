"""Prospective Tank War sampling excludes seen resets and balances new instances."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts/prepare-multifamily-blind.py"
SPEC = importlib.util.spec_from_file_location("tank_multifamily_blind", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


@pytest.mark.unit
def test_fault_selection_excludes_seen_seeds_and_rotates_instances() -> None:
    labels = []
    for layer in module.LAYERS:
        for instance in (f"{layer}-new-a", f"{layer}-new-b"):
            labels.extend(
                {
                    "layer": layer,
                    "fault_instance": instance,
                    "seed": seed,
                    "seat": seat,
                    "fault_loss": True,
                    "cause_label": layer,
                    "verified_local_effect": True,
                }
                for seed in (1, 2, 3)
                for seat in ("blue", "red")
            )
        labels.append(
            {
                "layer": layer,
                "fault_instance": f"{layer}-new-a",
                "seed": 4,
                "seat": "blue",
                "fault_loss": True,
                "cause_label": layer,
                "verified_local_effect": True,
            }
        )
    chosen = module.select_faults(labels, {1, 2, 3}, per_layer=4)
    assert len(chosen) == 12
    for layer in module.LAYERS:
        group = [row for row in chosen if row["layer"] == layer]
        assert {row["fault_instance"] for row in group} == {f"{layer}-new-a", f"{layer}-new-b"}
        assert all(row["seed"] != 4 for row in group)
        assert len({(row["fault_instance"], row["seed"]) for row in group}) == len(group)


@pytest.mark.unit
def test_controls_require_unmodified_losses_on_distinct_seeds() -> None:
    controls = [
        {"seed": 1, "seat": "blue", "outcome": -1, "cause_label": None},
        {"seed": 1, "seat": "red", "outcome": -1, "cause_label": None},
        {"seed": 2, "seat": "blue", "outcome": 1, "cause_label": None},
        {"seed": 3, "seat": "blue", "outcome": -1, "cause_label": None},
        {"seed": 4, "seat": "blue", "outcome": -1, "cause_label": None},
    ]
    chosen = module.select_controls(controls, {1, 2, 3}, count=3)
    assert {row["seed"] for row in chosen} == {1, 3}
    assert len(chosen) == 2


@pytest.mark.unit
def test_legacy_source_is_excluded_before_blind_sampling(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "integrations/tank-war/scripts/match.mjs"
    source.parent.mkdir(parents=True)
    source.write_text("current")
    game = tmp_path / "game"
    game.mkdir()
    (game / "manifest.json").write_text(json.dumps({"sources": {"scripts/match.mjs": module.sha(source)}}))
    assert module.replay_source_compatible({"fault_run": str(game)}, repo)
    source.write_text("changed")
    assert not module.replay_source_compatible({"fault_run": str(game)}, repo)
