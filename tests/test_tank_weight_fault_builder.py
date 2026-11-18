"""The v0 weights candidate must be a real, isolated checkpoint edit."""

import importlib.util
import json
import math
import struct
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "integrations/tank-war/scripts/build-weight-faults.py"
SPEC = importlib.util.spec_from_file_location("tank_weight_fault_builder", SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def make_checkpoint(root: Path, scalar: float, other: float = 7.0, linked: bool = False) -> Path:
    root.mkdir(parents=True)
    folder = root / "checkpoint"
    folder.mkdir()
    header = {
        "scalar.weight": {"dtype": "F32", "shape": [1, 1024], "data_offsets": [0, 4096]},
        "other.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [4096, 4100]},
    }
    raw = json.dumps(header).encode()
    weights = folder / "best.safetensors"
    target = root / "source.safetensors" if linked else weights
    target.write_bytes(
        struct.pack("<Q", len(raw)) + raw + struct.pack("<1024f", *([scalar] * 1024)) + struct.pack("<f", other)
    )
    if linked:
        weights.symlink_to(target)
    for name in builder.INFERENCE_AUX_FILES:
        path = folder / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(name, encoding="utf-8")
    manifest = root / "checkpoint.json"
    manifest.write_text(json.dumps({"path": "checkpoint", "weightsSha256": builder.sha256(weights)}))
    return manifest


def freeze(clean: Path, prior: Path, reverse: Path) -> dict:
    return {
        "model": {"clean_weights_sha256": json.loads(clean.read_text())["weightsSha256"]},
        "selected_faults": {
            "weights_builder_inputs": {
                "prior_edit_weights_sha256": json.loads(prior.read_text())["weightsSha256"],
                "reverse_edit_weights_sha256": json.loads(reverse.read_text())["weightsSha256"],
                "checkpoint_aux_sha256": builder.auxiliary_hashes(reverse.parent / "checkpoint"),
            }
        },
    }


def test_builds_real_scalar_edit_and_rejects_other_tensor_change(tmp_path: Path) -> None:
    clean = make_checkpoint(tmp_path / "clean", 0.0)
    prior = make_checkpoint(tmp_path / "prior", 1 / math.sqrt(1024))
    reverse = make_checkpoint(tmp_path / "reverse", -1 / math.sqrt(1024))
    output = tmp_path / "faults"

    records = builder.build(prior, reverse, output, (-3.0, -4.0), freeze(clean, prior, reverse), clean)
    assert [record["name"] for record in records] == ["reverse_strength_3p0", "reverse_strength_4p0"]
    edited = output / "reverse_strength_3p0/checkpoint/best.safetensors"
    start, end, _, size = builder.tensor_span(edited)
    assert builder.same_except_scalar(edited, tmp_path / "clean/checkpoint/best.safetensors", start, end, size)
    assert builder.scalar_values(edited, start)[0] == pytest.approx(-3 / math.sqrt(1024))
    fault = json.loads((output / "reverse_strength_3p0/fault.json").read_text())
    assert fault["layer"] == "weights"
    assert fault["weights_sha256"] == builder.sha256(edited)

    dirty = make_checkpoint(tmp_path / "dirty", -1 / math.sqrt(1024), other=8.0)
    with pytest.raises(ValueError, match=r"outside scalar\.weight"):
        builder.build(prior, dirty, tmp_path / "rejected", (-3.0,), freeze(clean, prior, dirty), clean)


def test_symlinked_source_weight_is_not_modified(tmp_path: Path) -> None:
    clean = make_checkpoint(tmp_path / "clean", 0.0)
    prior = make_checkpoint(tmp_path / "prior", 1 / math.sqrt(1024))
    reverse = make_checkpoint(tmp_path / "reverse", -1 / math.sqrt(1024), linked=True)
    source = reverse.parent / "source.safetensors"
    before = builder.sha256(source)

    builder.build(prior, reverse, tmp_path / "faults", (-3.0,), freeze(clean, prior, reverse), clean)

    assert builder.sha256(source) == before
    assert not (tmp_path / "faults/reverse_strength_3p0/checkpoint/best.safetensors").is_symlink()


def test_rejects_changed_non_weight_file(tmp_path: Path) -> None:
    clean = make_checkpoint(tmp_path / "clean", 0.0)
    prior = make_checkpoint(tmp_path / "prior", 1 / math.sqrt(1024))
    reverse = make_checkpoint(tmp_path / "reverse", -1 / math.sqrt(1024))
    frozen = freeze(clean, prior, reverse)
    (reverse.parent / "checkpoint/tokenizer/tokenizer.json").write_text("different", encoding="utf-8")

    with pytest.raises(ValueError, match="non-weight files differ from freeze"):
        builder.build(prior, reverse, tmp_path / "rejected", (-3.0,), frozen, clean)
