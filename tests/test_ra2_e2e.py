"""The RA2 end-to-end script (#70): the dry-run config, swapping components by config alone, and startup checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import ClassVar

import pytest

from honing.backends.ra2.e2e import E2EContext, build, main, missing_requirements
from honing.backends.ra2.runner import RA2EpisodeRunner
from honing.campaign import ScriptedScheduler
from honing.core import (
    ContractError,
    EpisodeOutcome,
    HarnessSpec,
    ModelSnapshot,
    PolicyRef,
    Task,
    TrainingBatch,
    TrainJobSpec,
    TrainResult,
)
from tests.fakes.model import FakeTrainer
from tests.fakes.ra2.e2e import trainer as fake_trainer

REPO = Path(__file__).resolve().parents[1]
DRY_RUN = REPO / "configs/e2e/ra2_dry_run.json"


def _config(**changes: object) -> dict[str, object]:
    config = json.loads(DRY_RUN.read_text())
    return {**config, **changes}


@pytest.mark.unit
def test_the_dry_run_rolls_back_a_tied_harness_move_and_publishes_a_winning_train_move(tmp_path: Path) -> None:
    code = main(["--config", str(DRY_RUN), "--root", str(REPO), "--work-dir", str(tmp_path / "out")])

    assert code == 0
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["head"] == "r0001"
    harness, train = summary["steps"]
    assert (harness["move"], harness["result"], harness["reason"]) == (
        "harness",
        "rolled back",
        "0 wins, 0 losses, 4 ties",
    )
    assert (train["move"], train["result"], train["release"]) == ("train", "published", "r0001")
    assert train["candidate_mean"] == 1.0 and train["incumbent_mean"] == 0.5 and train["episodes"] == 8
    commits = (tmp_path / "out/releases/commits.jsonl").read_text().splitlines()
    assert len(commits) == 3  # release 0, the rolled-back harness candidate, and release 1
    matches = [p for p in (tmp_path / "out/runs").iterdir() if (p / "result.json").exists()]
    assert len(matches) == 16


# ---------------------------------------------------------------- components are swapped by config alone


class RecordingRunner(RA2EpisodeRunner):
    """The RA2 runner, counting the episodes it plays."""

    played: ClassVar[list[str]] = []

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        RecordingRunner.played.append(task.task_id)
        return await super().run(spec, task, policy)


def recording_runner(context: E2EContext, command: list[str]) -> RecordingRunner:
    return RecordingRunner(
        suite=context.suite,
        endpoints=context.component("endpoints"),
        integration_dir=context.integration_dir,
        runs_dir=context.work_dir / "runs",
        command=[part.format(python=sys.executable, root=str(context.root)) for part in command],
    )


class RecordingTrainer:
    """A trainer from outside the RA2 backend: delegates to the fake and records each job and batch."""

    def __init__(self, inner: FakeTrainer) -> None:
        self._inner = inner
        self.support = inner.support
        self.calls: list[tuple[TrainJobSpec, TrainingBatch | None]] = []

    def train(self, job: TrainJobSpec, base: ModelSnapshot, batch: TrainingBatch | None) -> TrainResult:
        self.calls.append((job, batch))
        return self._inner.train(job, base, batch)


def recording_trainer(context: E2EContext) -> RecordingTrainer:
    trainer = RecordingTrainer(fake_trainer(context))
    context.shared["recording_trainer"] = trainer
    return trainer


@pytest.mark.unit
def test_a_new_trainer_and_runner_are_tried_by_changing_the_config(tmp_path: Path) -> None:
    config = _config()
    components = dict(config["components"])  # type: ignore[call-overload]
    components["trainer"] = {"factory": "tests.test_ra2_e2e:recording_trainer"}
    components["runner"] = {
        "factory": "tests.test_ra2_e2e:recording_runner",
        "args": {"command": ["{python}", "{root}/tests/fakes/ra2/fake_match.py"]},
    }
    RecordingRunner.played = []

    run = build({**config, "components": components}, REPO, tmp_path / "out")
    harness, train = run.campaign.run(ScriptedScheduler(run.moves))

    trainer = run.context.shared["recording_trainer"]
    assert isinstance(trainer, RecordingTrainer)
    ((job, batch),) = trainer.calls
    assert job.contract == "typed_decisions" and batch is not None and len(batch.decisions) == 9
    assert train.release is not None and train.decision is not None and train.decision.selected
    assert harness.release is None
    assert len(RecordingRunner.played) == 16  # two gates, 4 tasks, candidate and incumbent


# ---------------------------------------------------------------- startup checks


@pytest.mark.unit
def test_missing_requirements_are_named_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("RA2_GAME_DIR", raising=False)
    config = _config(
        requires={"env": ["RA2_GAME_DIR"], "commands": ["no-such-command-xyz"], "paths": ["/no/such/path"]}
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    code = main(["--config", str(path), "--root", str(REPO), "--work-dir", str(tmp_path / "out")])

    assert code == 2
    error = capsys.readouterr().err
    assert "environment variable RA2_GAME_DIR" in error and "command no-such-command-xyz" in error
    assert "path /no/such/path" in error
    assert not (tmp_path / "out").exists()
    assert missing_requirements({}) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"trainer": None}, "names no trainer component"),
        ({"trainer": {"factory": "tests.fakes.ra2.e2e"}}, "package.module:function"),
        ({"trainer": {"factory": "tests.fakes.ra2.e2e:no_such_factory"}}, "cannot load the trainer factory"),
        ({"trainer": {"factory": "tests.fakes.ra2.e2e:endpoints"}}, "returned QualityEndpoints, not a Trainer"),
        ({"compressor": {"factory": "x:y"}}, "unknown components"),
    ],
)
def test_a_bad_component_entry_is_refused(tmp_path: Path, change: dict[str, object], message: str) -> None:
    components = {**_config()["components"], **change}  # type: ignore[dict-item]
    components = {k: v for k, v in components.items() if v is not None}
    with pytest.raises(ContractError, match=message):
        build(_config(components=components), REPO, tmp_path / "out")


@pytest.mark.unit
def test_only_harness_and_train_moves_can_be_scheduled(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="not 'prune'"):
        build(_config(schedule=[{"kind": "prune"}]), REPO, tmp_path / "out")
