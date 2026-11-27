"""The config factories for the decision-model trainers build their settings without a GPU or a checkout."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from honing.backends.jev import e2e
from honing.backends.jev.kev_trainer import KevTrainer
from honing.backends.jev.nanojev_trainer import NanoJevTrainer
from honing.backends.jev.trainer import LocalJevTrainer
from honing.core import ContractError, Trainer


def _context(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(root=tmp_path / "repo", work_dir=tmp_path / "run")


@pytest.mark.unit
def test_the_nanojev_trainer_reads_its_checkout_and_device_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HONING_NANOJEV_SOURCE", str(tmp_path / "NanoJev"))
    monkeypatch.setenv("HONING_NANOJEV_TRAIN_GPU", "3")

    trainer = e2e.nanojev_trainer(_context(tmp_path), max_length=4096, head_lr=1e-3, validation_groups="^val:")

    assert isinstance(trainer, NanoJevTrainer) and isinstance(trainer, Trainer)
    settings = trainer._settings
    assert (settings.source_dir, settings.gpu) == (tmp_path / "NanoJev", "3")
    assert settings.pins_path == (tmp_path / "repo/integrations/ra2/pins.json").resolve()
    assert settings.work_dir == tmp_path / "run/training/nanojev"
    assert (settings.max_length, settings.head_lr, settings.validation_groups) == (4096, 1e-3, "^val:")
    assert settings.backbone_lr == 1e-5  # the released recipe's default


@pytest.mark.unit
def test_the_kev_and_laya_trainers_are_placed_the_same_way(tmp_path: Path) -> None:
    context = _context(tmp_path)

    kev = e2e.kev_trainer(context, source="/kev", gpu="1", batch_size=2)
    laya = e2e.laya_trainer(context, source="/laya", gpu="0", model_record="records/base.json", seed=11)

    assert isinstance(kev, KevTrainer) and isinstance(laya, LocalJevTrainer)
    assert (kev._settings.source_dir, kev._settings.gpu, kev._settings.batch_size) == (Path("/kev"), "1", 2)
    assert kev._settings.work_dir == tmp_path / "run/training/kev" and kev._settings.model == "kev-0.8b"
    assert (laya._settings.laya_source, laya._settings.seed) == (Path("/laya"), 11)
    assert laya._settings.model_record == (tmp_path / "repo/records/base.json").resolve()
    assert laya._settings.work_dir == tmp_path / "run/training/laya"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("factory", "tuning", "message"),
    [
        (e2e.nanojev_trainer, {"learning_rate": 1e-4}, r"nanojev_trainer factory has no setting \['learning_rate'\]"),
        (e2e.nanojev_trainer, {"work_dir": "/elsewhere"}, r"no setting \['work_dir'\]; it takes \['backbone_lr'"),
        (e2e.kev_trainer, {"batch_questions": 4}, r"kev_trainer factory has no setting \['batch_questions'\]"),
        (e2e.laya_trainer, {"laya_source": "/x"}, r"laya_trainer factory has no setting \['laya_source'\]"),
    ],
)
def test_a_setting_the_trainer_does_not_have_or_the_factory_places_is_refused_by_name(
    tmp_path: Path, factory: object, tuning: dict[str, object], message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        factory(_context(tmp_path), source="/src", gpu="0", **tuning)  # type: ignore[operator]
