"""The trainer contract a training backend (Slime, ...) must pass: it passes on the stand-in, and fails, by name, on
trainers that break one part of it each."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from honing.core import TeacherRef, TrainingBatch, TrainJobSpec, TrainResult
from honing.e2e.check import trainer_main
from honing.e2e.trainer_conformance import check_trainer
from tests.fakes.builder import TASKS
from tests.fakes.data import FakeRolloutProvider, ListDatasetProvider, token_batch
from tests.fakes.harness import SEED_HARNESS
from tests.fakes.model import SLIME_LIKE, FakeModelSource, FakeTrainer, FakeWorld
from tests.fakes.pipeline import SlimeLikeTrainer

REPO = Path(__file__).resolve().parents[1]
ALL_PLANES = REPO / "configs/e2e/all_planes.json"
JOBS = (
    TrainJobSpec("sft", steps=10),
    TrainJobSpec("grpo", steps=10, rollout="harness"),
    TrainJobSpec("distill", steps=10, teacher=TeacherRef(kind="lineage", ref="parent")),
)
ALL_CASES = ("support", "train:sft", "train:grpo", "train:distill", "fresh", "refuses")


def _check(trainer: Any, world: FakeWorld, jobs: tuple[TrainJobSpec, ...] = JOBS, rollout: Any = "default") -> dict:  # type: ignore[type-arg]
    checks = check_trainer(
        trainer,
        base=FakeModelSource(world).materialize(Path("unused")),
        jobs=jobs,
        data=ListDatasetProvider([token_batch()]),
        rollout=FakeRolloutProvider() if rollout == "default" else rollout,
        tasks=TASKS,
        harness=SEED_HARNESS,
    )
    return {check.name: (check.passed, check.detail) for check in checks}


def _failing(results: dict) -> set[str]:  # type: ignore[type-arg]
    return {name for name, (passed, _) in results.items() if not passed}


@pytest.mark.unit
def test_the_slime_like_stand_in_passes_every_check() -> None:
    world = FakeWorld()
    results = _check(SlimeLikeTrainer(world, gain=0.1), world)
    assert tuple(results) == ALL_CASES and _failing(results) == set()
    assert "rollouts batch" in results["train:grpo"][1]
    assert "undeclared dpo" in results["refuses"][1]


class NoChange:
    """Returns the base weights unchanged."""

    support = SLIME_LIKE

    def train(self, job: TrainJobSpec, base: Any, batch: TrainingBatch | None) -> TrainResult:
        return TrainResult(snapshot=base, metrics={})


class WrongFormat(SlimeLikeTrainer):
    """Returns weights in Megatron's format instead of the release's Hugging Face safetensors."""

    def train(self, job: TrainJobSpec, base: Any, batch: TrainingBatch | None) -> TrainResult:
        result = super().train(job, base, batch)
        checkpoint = replace(result.snapshot.checkpoint, weight_format="megatron_torch_dist")
        return replace(result, snapshot=replace(result.snapshot, checkpoint=checkpoint))


class Overwrites(SlimeLikeTrainer):
    """Writes every run's new weights to the same place."""

    def train(self, job: TrainJobSpec, base: Any, batch: TrainingBatch | None) -> TrainResult:
        result = super().train(job, base, batch)
        checkpoint = replace(result.snapshot.checkpoint, location="/runs/latest")
        return replace(result, snapshot=replace(result.snapshot, checkpoint=checkpoint))


class NanLoss(SlimeLikeTrainer):
    """Reports a loss that is not a number."""

    def train(self, job: TrainJobSpec, base: Any, batch: TrainingBatch | None) -> TrainResult:
        return replace(super().train(job, base, batch), metrics={"loss": math.nan})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("make", "failing", "says"),
    [
        (lambda w: FakeTrainer(w, support=SLIME_LIKE), {"refuses"}, "trained with no batch"),
        (lambda w: NoChange(), {"train:sft", "train:grpo", "train:distill", "fresh", "refuses"}, "did not change"),
        (
            lambda w: WrongFormat(w, gain=0.1),
            {"train:sft", "train:grpo", "train:distill", "fresh", "refuses"},
            "megatron_torch_dist, not hf_safetensors",
        ),
        (lambda w: Overwrites(w, gain=0.1), {"fresh"}, "to the first run's location"),
        (lambda w: NanLoss(w, gain=0.1), {"train:sft", "train:grpo", "train:distill", "fresh", "refuses"}, "loss"),
    ],
    ids=["accepts-anything", "changes-nothing", "wrong-weight-format", "overwrites-a-checkpoint", "nan-loss"],
)
def test_a_trainer_that_breaks_the_contract_fails_the_check_it_breaks(make: Any, failing: set[str], says: str) -> None:
    world = FakeWorld()
    results = _check(make(world), world)
    assert _failing(results) == failing
    assert any(says in detail for name, (_, detail) in results.items() if name in failing)


@pytest.mark.unit
def test_a_job_the_trainer_does_not_declare_stops_at_support() -> None:
    world = FakeWorld()
    results = _check(SlimeLikeTrainer(world, gain=0.1), world, jobs=(TrainJobSpec("dpo"),))
    assert list(results) == ["support"] and not results["support"][0]
    assert "cannot run dpo" in results["support"][1]


@pytest.mark.unit
def test_an_on_policy_job_without_a_rollout_component_says_so() -> None:
    world = FakeWorld()
    results = _check(SlimeLikeTrainer(world, gain=0.1), world, rollout=None)
    assert _failing(results) == {"train:grpo"}
    assert "names no rollout component" in results["train:grpo"][1]


@pytest.mark.unit
def test_a_rollout_batch_from_another_release_is_refused_before_training() -> None:
    world = FakeWorld()
    results = _check(SlimeLikeTrainer(world, gain=0.1), world, rollout=FakeRolloutProvider(stamp_release="r0007"))
    assert _failing(results) == {"train:grpo"}
    assert "batch_from_one_release" in results["train:grpo"][1]


@pytest.mark.unit
def test_the_check_command_runs_on_a_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = trainer_main(["--config", str(ALL_PLANES), "--root", str(REPO), "--work-dir", str(tmp_path / "check")])
    assert code == 0
    assert "trainer check: ok" in capsys.readouterr().out


@pytest.mark.unit
def test_the_config_names_the_jobs_to_try(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = json.loads(ALL_PLANES.read_text())
    config["trainer_check"] = {"jobs": [{"algorithm": "sft", "steps": 1}]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert trainer_main(["--config", str(path), "--root", str(REPO), "--work-dir", str(tmp_path / "check")]) == 0
    out = capsys.readouterr().out
    assert "train:sft" in out and "train:grpo" not in out
