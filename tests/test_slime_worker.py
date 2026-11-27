"""The conversions ``honing.backends.slime.worker`` runs in Slime's image, with that image's packages faked.

The worker imports torch, Slime's ``Sample`` and TensorBoard's event reader by
name when a command runs, so each test puts a stand-in under that name: torch's
``save``/``load`` as pickle, ``Sample`` as a record built from its dict, and an
event reader over scalars the test logged. What is checked is the worker's own
part: the file layouts on both sides, which fields survive the round trip, and
which value of each scalar the metrics file keeps.
"""

from __future__ import annotations

import json
import pickle
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pytest

from honing.backends.slime import worker
from honing.backends.slime.rollout_data import read_samples, slime_samples, write_samples
from honing.core import DataContract, PolicySample, TrainingBatch


@dataclass
class Sample:
    """The fields of Slime's ``Sample`` the worker reads back; ``from_dict`` ignores the rest, as Slime's does."""

    index: int | None = None
    group_index: int | None = None
    tokens: Sequence[int] = ()
    response_length: int = 0
    loss_mask: Sequence[int] | None = None
    reward: float | None = None
    rollout_log_probs: Sequence[float] | None = None
    teacher_log_probs: Sequence[float] | None = None
    metadata: Mapping[str, Any] | None = None
    status: str = "pending"

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Sample:
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in row.items() if key in names})


@pytest.fixture
def slime_image(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """torch (pickle) and ``slime.utils.types.Sample``; returns every ``torch.load`` call's keyword arguments."""
    loads: list[dict[str, Any]] = []

    def save(value: object, path: Path) -> None:
        Path(path).write_bytes(pickle.dumps(value))

    def load(path: Path, **kwargs: Any) -> object:
        loads.append(kwargs)
        return pickle.loads(Path(path).read_bytes())

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(save=save, load=load))
    monkeypatch.setitem(sys.modules, "slime.utils.types", types.SimpleNamespace(Sample=Sample))
    return loads


class Scalar:
    def __init__(self, step: int, value: float) -> None:
        self.step, self.value = step, value


@pytest.fixture
def tensorboard(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, list[float]]]:
    """An event reader over ``logged``: a run directory's name maps each tag to the values logged, in order."""
    logged: dict[str, dict[str, list[float]]] = {}

    class EventAccumulator:
        def __init__(self, path: str, size_guidance: Mapping[str, int]) -> None:
            # 0 keeps every event; TensorBoard's default samples 10,000 of them.
            assert size_guidance == {"scalars": 0}
            self.run = logged[Path(path).name]
            self.loaded = False

        def Reload(self) -> None:  # noqa: N802 - TensorBoard's name
            self.loaded = True

        def Tags(self) -> dict[str, list[str]]:  # noqa: N802
            assert self.loaded
            return {"scalars": list(self.run)}

        def Scalars(self, tag: str) -> list[Scalar]:  # noqa: N802
            return [Scalar(step, value) for step, value in enumerate(self.run[tag])]

    module = "tensorboard.backend.event_processing.event_accumulator"
    monkeypatch.setitem(sys.modules, module, types.SimpleNamespace(EventAccumulator=EventAccumulator))
    return logged


def log_run(root: Path, name: str) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / f"events.out.tfevents.1700000000.{name}").write_bytes(b"")
    return run


def batch() -> TrainingBatch:
    def sample(group: str, reward: float) -> PolicySample:
        return PolicySample(
            tokens=(5, 6, 7, 8),
            loss_mask=(0, 1, 0, 1),
            reward=reward,
            logprobs=(0.0, -0.5, -0.25, -1.0),
            group=group,
            source_release="r0001",
        )

    return TrainingBatch(
        batch_id="b1",
        contract=DataContract("grouped_on_policy_samples", requires_reward=True, requires_logprobs=True),
        samples=(sample("g0", 1.0), sample("g0", 0.0), sample("g1", 0.5), sample("g1", 0.25)),
    )


@pytest.mark.functional
def test_a_rollout_saved_for_slime_reads_back_as_the_samples_honing_wrote(
    tmp_path: Path, slime_image: list[dict[str, Any]]
) -> None:
    rows, _ = slime_samples(batch())
    source = write_samples(tmp_path / "rollout" / "0.json", rows, rollout_id=4)
    saved = tmp_path / "nested" / "rollout" / "0.pt"

    assert worker.main(["save-rollout", str(source), str(saved)]) == 0
    assert worker.main(["load-rollout", str(saved), str(tmp_path / "back.json")]) == 0

    assert pickle.loads(saved.read_bytes()) == {"rollout_id": 4, "samples": rows}
    back = json.loads((tmp_path / "back.json").read_text())
    assert back["rollout_id"] == 4
    assert read_samples(tmp_path / "back.json") == batch().samples
    # The full Sample objects need pickle, which torch refuses under weights_only=True.
    assert slime_image == [{"weights_only": False}]


@pytest.mark.unit
def test_loading_keeps_only_the_fields_honing_reads_and_absent_optional_lists_stay_none(
    tmp_path: Path, slime_image: list[dict[str, Any]]
) -> None:
    row = {
        "index": 0,
        "group_index": 0,
        "tokens": (1, 2, 3),
        "response_length": 2,
        "loss_mask": None,
        "reward": 1.0,
        "rollout_log_probs": None,
        "teacher_log_probs": (-0.1, -0.2),
        "metadata": {"batch_id": "b"},
        "status": "completed",
    }
    (tmp_path / "in.pt").write_bytes(pickle.dumps({"rollout_id": 1, "samples": [row]}))

    worker.load_rollout(tmp_path / "in.pt", tmp_path / "out.json")

    text = (tmp_path / "out.json").read_text()
    assert text.endswith("}\n")
    assert json.loads(text)["samples"] == [
        {
            "index": 0,
            "group_index": 0,
            "tokens": [1, 2, 3],
            "response_length": 2,
            "loss_mask": None,
            "reward": 1.0,
            "rollout_log_probs": None,
            "teacher_log_probs": [-0.1, -0.2],
            "metadata": {"batch_id": "b"},
        }
    ]


@pytest.mark.unit
def test_saving_writes_only_the_rollout_id_and_samples(tmp_path: Path, slime_image: list[dict[str, Any]]) -> None:
    (tmp_path / "in.json").write_text(json.dumps({"rollout_id": 2, "samples": [{"index": 0}], "extra": "dropped"}))

    worker.save_rollout(tmp_path / "in.json", tmp_path / "out.pt")

    assert pickle.loads((tmp_path / "out.pt").read_bytes()) == {"rollout_id": 2, "samples": [{"index": 0}]}


@pytest.mark.functional
def test_metrics_keep_each_scalar_s_last_and_first_value_and_count_training_steps(
    tmp_path: Path, tensorboard: dict[str, dict[str, list[float]]]
) -> None:
    root = tmp_path / "tensorboard"
    log_run(root, "a")
    log_run(root / "deeper", "b")
    (root / "no-events").mkdir()
    tensorboard["a"] = {"train/loss": [3.0, 2.0, 1.5], "rollout/reward": [0.1, 0.2, 0.3, 0.4, 0.5]}
    tensorboard["b"] = {"train/grad_norm": [9.0, 8.0], "perf/step_time": [1.0]}

    assert worker.main(["metrics", str(root), str(tmp_path / "metrics.json")]) == 0

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    # rollout/* has more events than any train/* tag, but only training steps are counted.
    assert metrics["steps"] == 3
    assert metrics["scalars"] == {
        "train/loss": 1.5,
        "train/loss@first": 3.0,
        "rollout/reward": 0.5,
        "rollout/reward@first": 0.1,
        "train/grad_norm": 8.0,
        "train/grad_norm@first": 9.0,
        "perf/step_time": 1.0,
        "perf/step_time@first": 1.0,
    }


@pytest.mark.unit
def test_a_tag_logged_by_two_runs_keeps_the_value_of_the_run_that_sorts_last(
    tmp_path: Path, tensorboard: dict[str, dict[str, list[float]]]
) -> None:
    log_run(tmp_path, "run-2")
    log_run(tmp_path, "run-1")
    tensorboard["run-1"] = {"train/loss": [5.0, 4.0]}
    tensorboard["run-2"] = {"train/loss": [3.0]}

    worker.read_metrics(tmp_path, tmp_path / "metrics.json")

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics == {"steps": 2, "scalars": {"train/loss": 3.0, "train/loss@first": 3.0}}


@pytest.mark.unit
def test_a_directory_without_event_files_has_no_steps_and_no_scalars(
    tmp_path: Path, tensorboard: dict[str, dict[str, list[float]]]
) -> None:
    (tmp_path / "empty").mkdir()

    worker.read_metrics(tmp_path, tmp_path / "metrics.json")

    assert json.loads((tmp_path / "metrics.json").read_text()) == {"steps": 0, "scalars": {}}


@pytest.mark.functional
@pytest.mark.parametrize(
    "argv", [[], ["convert", "a", "b"], ["metrics", "only-one"]], ids=["no-command", "unknown", "missing-target"]
)
def test_the_command_line_refuses_a_command_it_does_not_have(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        worker.main(argv)

    assert raised.value.code == 2
    assert "usage: honing.backends.slime.worker" in capsys.readouterr().err
