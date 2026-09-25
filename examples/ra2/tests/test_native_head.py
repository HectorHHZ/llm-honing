# ruff: noqa: E402
# Standalone example imports follow its explicit sys.path bootstrap.
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
    from safetensors.torch import save_file
except ModuleNotFoundError as error:
    raise unittest.SkipTest("Native head tensor tests require optional Torch and safetensors") from error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "league"))
from integration.models.head_checkpoint import (
    apply_head_state,
    effective_weight_digest,
    freeze_body_and_select_head,
    head_state,
    load_head_artifact,
)
from integration.models.worker import digest
from train_native_head import select_rows, tensor_set_digest


class NanoFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.scalar = torch.nn.Linear(4, 1)


class OmniFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 256)
        self.register_buffer("mu", torch.zeros(1, 4))
        self.register_buffer("sd", torch.ones(1, 4))


def adapter(family):
    if family == "nanojev":
        return SimpleNamespace(predictor=SimpleNamespace(model=NanoFixture()))
    return SimpleNamespace(model=torch.nn.Linear(4, 4), head=OmniFixture())


class NativeHeadTests(unittest.TestCase):
    def test_only_native_head_gets_gradients(self):
        for family in ["nanojev", "jev-omni"]:
            with self.subTest(family=family):
                value = adapter(family)
                selected = freeze_body_and_select_head(value, family)
                body = value.predictor.model.backbone if family == "nanojev" else value.model
                self.assertTrue(selected)
                self.assertFalse(any(p.requires_grad for p in body.parameters()))
                self.assertTrue(all(p.requires_grad for p in selected.values()))

    def test_rejects_backbone_replacement_and_partial_heads(self):
        value = adapter("nanojev")
        state = {k: v.clone() for k, v in head_state(value, "nanojev").items()}
        with self.assertRaisesRegex(ValueError, "exactly"):
            apply_head_state(value, "nanojev", {**state, "backbone.weight": torch.zeros(4, 4)})
        del state["scalar.bias"]
        with self.assertRaisesRegex(ValueError, "exactly"):
            apply_head_state(value, "nanojev", state)

    def test_rejects_changed_normalization_or_nonfinite_values(self):
        value = adapter("jev-omni")
        state = {k: v.clone() for k, v in head_state(value, "jev-omni").items()}
        state["sd"].fill_(2)
        with self.assertRaisesRegex(ValueError, "normalization"):
            apply_head_state(value, "jev-omni", state)
        state["sd"].fill_(1)
        state["linear.weight"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "values"):
            apply_head_state(value, "jev-omni", state)

    def test_rejects_wrong_dtype_or_shape(self):
        value = adapter("nanojev")
        state = {k: v.clone() for k, v in head_state(value, "nanojev").items()}
        state["scalar.weight"] = state["scalar.weight"].double()
        with self.assertRaisesRegex(ValueError, "dtype"):
            apply_head_state(value, "nanojev", state)
        state["scalar.weight"] = torch.zeros(1, 5)
        with self.assertRaisesRegex(ValueError, "shape"):
            apply_head_state(value, "nanojev", state)

    def test_artifact_roundtrip_checks_parent_and_file_bytes(self):
        value = adapter("nanojev")
        parent = {
            "weightSetSha256": "a" * 64,
            "preparedManifestSha256": "b" * 64,
            "preparedManifestPath": "/fixture/parent.json",
        }
        state = {k: v.clone() + 0.5 for k, v in head_state(value, "nanojev").items()}
        body_before = tensor_set_digest(dict(value.predictor.model.backbone.named_parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "head.safetensors"
            save_file(state, str(weights))
            sha = digest(weights)
            manifest = {
                "kind": "trained-native-decision-head",
                "family": "nanojev",
                "parent": parent,
                "weightFilename": "head.safetensors",
                "headSha256": sha,
                "effectiveWeightSetSha256": effective_weight_digest(parent["weightSetSha256"], sha),
                "trainingScope": "native-head-only",
                "backboneUpdated": False,
            }
            path = root / "checkpoint.json"
            path.write_text(json.dumps(manifest))
            load_head_artifact(value, "nanojev", path, parent)
            self.assertEqual(tensor_set_digest(head_state(value, "nanojev")), tensor_set_digest(state))
            self.assertEqual(tensor_set_digest(dict(value.predictor.model.backbone.named_parameters())), body_before)
            with self.assertRaisesRegex(ValueError, "another native parent"):
                load_head_artifact(value, "nanojev", path, {**parent, "weightSetSha256": "c" * 64})
            with weights.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "modified"):
                load_head_artifact(value, "nanojev", path, parent)

    def test_selection_is_reproducible_and_covers_stages_without_duplicates(self):
        rows = [{"id": str(i), "metadata": {"stage": i % 6 + 1}} for i in range(60)]
        a = select_rows(rows, 32, 7)
        self.assertEqual(a, select_rows(rows, 32, 7))
        self.assertEqual({row["metadata"]["stage"] for row in a[:6]}, set(range(1, 7)))
        self.assertEqual(len({row["id"] for row in a}), 32)
        self.assertNotEqual(a, select_rows(rows, 32, 8))
        with self.assertRaises(ValueError):
            select_rows(rows, 61, 7)


if __name__ == "__main__":
    unittest.main()
