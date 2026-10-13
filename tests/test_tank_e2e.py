"""The tank campaign script: the dry-run config, its moves, startup checks, and the full dry run (with Node)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from honing.backends.tank.e2e import _move, build, main
from honing.core import ContractError

REPO = Path(__file__).resolve().parents[1]
DRY_RUN = REPO / "configs/e2e/tank_dry_run.json"
NODE = shutil.which("node")

sys.path.insert(0, str(REPO / "scripts/e2e"))
from check_campaign import problems  # noqa: E402


def _config(**changes: object) -> dict[str, object]:
    return {**json.loads(DRY_RUN.read_text()), **changes}


@pytest.mark.unit
def test_grpo_plays_rollouts_under_the_heads_harness_and_other_moves_are_refused() -> None:
    assert _move({"kind": "train", "algorithm": "grpo"}).train.rollout == "harness"  # type: ignore[union-attr]
    assert _move({"kind": "train", "algorithm": "sft"}).train.rollout == "none"  # type: ignore[union-attr]
    harness = _move({"kind": "harness", "label": "gepa", "episode_budget": 6}).harness
    assert (harness.label, harness.episode_budget) == ("gepa", 6)  # type: ignore[union-attr]
    with pytest.raises(ContractError, match="not 'prune'"):
        _move({"kind": "prune"})


@pytest.mark.unit
def test_a_config_is_refused_before_any_round_when_the_machine_or_the_config_lacks_something(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ContractError, match="command node"):
        build(_config(), REPO, tmp_path / "a")
    monkeypatch.undo()
    components = _config()["components"]
    without_method = {k: v for k, v in components.items() if k != "method"}  # type: ignore[union-attr]
    with pytest.raises(ContractError, match="names no method"):
        build(_config(requires={}, components=without_method), REPO, tmp_path / "b")
    with pytest.raises(ContractError, match="unknown task split 'dev'"):
        build(_config(requires={}, gate_tasks={"split": "dev"}), REPO, tmp_path / "c")


@pytest.mark.unit
def test_main_reports_a_refused_config_without_a_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_config(requires={"paths": ["/nonexistent/tank"]})))
    assert main(["--config", str(config), "--root", str(REPO), "--work-dir", str(tmp_path / "out")]) == 2
    assert "cannot start: this config needs path /nonexistent/tank" in capsys.readouterr().err


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(NODE is None, reason="needs Node 22")
def test_the_dry_run_publishes_the_gepa_harness_and_the_sft_step_and_rolls_back_a_tied_grpo_step(
    tmp_path: Path,
) -> None:
    work = tmp_path / "out"
    assert main(["--config", str(DRY_RUN), "--root", str(REPO), "--work-dir", str(work)]) == 0

    summary = json.loads((work / "summary.json").read_text())
    assert [(s["move"], s["result"]) for s in summary["steps"]] == [
        ("harness", "published"),
        ("train", "published"),
        ("train", "rolled back"),
    ]
    assert summary["head"] == "r0002"
    assert problems(work, []) == []
    # The published harness is GEPA's rewrite of the weapon instructions.
    head = json.loads((work / "releases/commits.jsonl").read_text().splitlines()[1])
    assert "whenever" in json.dumps(head)
    kinds = {
        json.loads((p / "manifest.json").read_text())["kind"]
        for p in (work / "runs").iterdir()
        if (p / "manifest.json").exists()
    }
    assert kinds == {"evaluation", "rollout", "teacher-demonstration"}
