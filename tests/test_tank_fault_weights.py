"""The fault library's weights faults through the real direction-edit trainer, on a tiny NanoJev-layout checkpoint.

Needs torch, so it runs in CI's weight-edit job; the test job, which has no torch, skips it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
safetensors = pytest.importorskip("safetensors.numpy")
pytest.importorskip("torch")

from honing.backends.jev.direction_edit import (  # noqa: E402
    HEAD_WEIGHT,
    DirectionEditSettings,
    NanoJevDirectionEditTrainer,
    direction,
)
from honing.backends.jev.nanojev import nanojev_snapshot  # noqa: E402
from honing.backends.tank.faults import FAULTS, FIRE_EDIT, build_library, checksums  # noqa: E402
from honing.core import TrainJobSpec  # noqa: E402
from tests.test_direction_edit import _base_checkpoint, _batch, _inputs, _pins  # noqa: E402

INTEGRATION = Path(__file__).resolve().parents[1] / "integrations/tank-war"


class NoSft:
    """The sft trainer slot: no fault built here runs sft."""

    def train(self, *args: object) -> None:
        raise AssertionError("no fault built here runs sft")


def _editor(root: Path) -> NanoJevDirectionEditTrainer:
    root.mkdir()
    settings = DirectionEditSettings(
        pins_path=_pins(root), source_dir=root, work_dir=root / "edits", question="weapon", target="fire"
    )
    return NanoJevDirectionEditTrainer(settings, reader=lambda decisions, parent: _inputs(len(decisions)))


def _weights(location: str) -> dict[str, np.ndarray]:
    return safetensors.load_file(str(Path(location) / "best.safetensors"))  # type: ignore[no-any-return]


@pytest.mark.unit
def test_the_reverse_fire_edit_builds_twice_to_one_checkpoint_that_changes_only_the_head(tmp_path: Path) -> None:
    base = nanojev_snapshot(_base_checkpoint(tmp_path))
    job = TrainJobSpec("direction_edit", contract="typed_decisions", direction_edit=FIRE_EDIT)
    clean = _editor(tmp_path / "clean-edit").train(job, base, _batch()).snapshot
    # The stand-in reader gives the head inputs of weapon questions only.
    faults = tuple(
        fault
        for fault in FAULTS
        if (fault.edit and fault.edit.question == "weapon") or fault.earlier or fault.cause == "chance"
    )

    first, second = (
        build_library(
            tmp_path / name,
            integration_dir=INTEGRATION,
            clean=clean,
            earlier=base,
            batch=_batch(),
            trainer=NoSft(),  # type: ignore[arg-type]
            edit_trainer=_editor(tmp_path / f"{name}-edits"),
            faults=faults,
        )
        for name in ("first", "second")
    )

    assert checksums(first.manifest) == checksums(second.manifest)
    entries = {entry["id"]: entry for entry in first.manifest["faults"]}
    assert list(entries) == ["weights-earlier-checkpoint", "weights-reverse-fire-edit", "chance-clean"]
    assert entries["weights-earlier-checkpoint"]["artifacts"]["weights"] == base.checkpoint.checksum
    reverse = entries["weights-reverse-fire-edit"]
    assert reverse["artifacts"]["weights"] not in {base.checkpoint.checksum, clean.checkpoint.checksum}
    # Three units of the fire direction come off the clean head, and nothing else changes.
    before, after = _weights(clean.checkpoint.location), _weights(reverse["weights_location"])
    assert set(before) == set(after)
    for name in before:
        if name != HEAD_WEIGHT:
            assert np.array_equal(before[name], after[name]), name
    unit, _ = direction(_inputs(), "weapon", "fire")
    assert np.allclose(after[HEAD_WEIGHT], before[HEAD_WEIGHT] - 3.0 * unit, atol=1e-5)
    assert np.allclose(after[HEAD_WEIGHT], _weights(base.checkpoint.location)[HEAD_WEIGHT] - 2.0 * unit, atol=1e-5)
