# ruff: noqa: E402
# Standalone example imports follow its explicit sys.path bootstrap.
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integration.models.head_checkpoint import effective_weight_digest, trained_runtime
from integration.models.provenance import verify_runtime
from integration.models.worker import digest


class HeadRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.parent = {
            "family": "jev-omni",
            "repo": "fixture/omni",
            "revision": "pinned",
            "path": "/fixture/weights",
            "weightSetSha256": "a" * 64,
            "preparedManifestPath": "/fixture/prepared.json",
            "preparedManifestSha256": "b" * 64,
        }
        weights = self.root / "head.safetensors"
        weights.write_bytes(b"CPU provenance fixture, not weights")
        head_sha = digest(weights)
        self.manifest = {
            "kind": "trained-native-decision-head",
            "family": "jev-omni",
            "parent": copy.deepcopy(self.parent),
            "weightFilename": "head.safetensors",
            "headSha256": head_sha,
            "effectiveWeightSetSha256": effective_weight_digest(self.parent["weightSetSha256"], head_sha),
            "trainingScope": "native-head-only",
            "backboneUpdated": False,
            "actualOptimizerSteps": 8,
        }
        self.path = self.root / "checkpoint.json"
        self.path.write_text(json.dumps(self.manifest))
        self.proof = {
            "verified": True,
            "family": "jev-omni",
            "checkpointSha256": digest(self.path),
            "effectiveWeightSetSha256": self.manifest["effectiveWeightSetSha256"],
            "freshProcessReload": True,
            "savedLogitsBitwiseEqual": True,
            "officialNativePredictionEquivalent": True,
            "frozenBodyAndBuffersEqualToPinnedParent": True,
            "learnedHeadTensorDigestVerified": True,
            "actualLearningVerified": True,
        }
        self.proof_path = self.root / "independent-verification.json"
        self.proof_path.write_text(json.dumps(self.proof))

    def tearDown(self):
        self.tmp.cleanup()

    def runtime(self):
        return {
            **trained_runtime(self.parent, self.path),
            "protocol": "native-choice-jsonl-v1",
            "modelWeightsLoaded": True,
            "parameterStorage": "float32",
            "forwardAutocast": "bfloat16",
            "maxLen": 2048,
        }

    def test_accepts_explicit_independently_verified_head_and_preserves_parent(self):
        before = copy.deepcopy(self.parent)
        runtime = self.runtime()
        self.assertEqual(self.parent, before)
        self.assertNotEqual(runtime["weightSetSha256"], self.parent["weightSetSha256"])
        self.assertEqual(runtime["baseModel"]["weightSetSha256"], self.parent["weightSetSha256"])
        self.assertEqual(runtime["learnedHead"]["manifestSha256"], digest(self.path))
        with patch("integration.models.provenance.verify_prepared", return_value=({}, self.parent)):
            verify_runtime(runtime, {})

    def test_rejects_missing_or_inconsistent_independent_proof(self):
        for key in [
            "verified",
            "freshProcessReload",
            "savedLogitsBitwiseEqual",
            "actualLearningVerified",
            "learnedHeadTensorDigestVerified",
        ]:
            proof = {**self.proof, key: False}
            self.proof_path.write_text(json.dumps(proof))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "independent"):
                self.runtime()
        self.proof_path.write_text(json.dumps({**self.proof, "checkpointSha256": "f" * 64}))
        with self.assertRaisesRegex(ValueError, "independent"):
            self.runtime()
        self.proof_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.runtime()

    def test_runtime_reconstruction_rejects_tampering_or_unmarked_training(self):
        runtime = self.runtime()
        mutations = [
            lambda x: x["baseModel"].update(weightSetSha256="c" * 64),
            lambda x: x["learnedHead"].update(headSha256="d" * 64),
            lambda x: x.update(weightSetSha256=self.parent["weightSetSha256"]),
            lambda x: x.pop("learnedHead"),
        ]
        with patch("integration.models.provenance.verify_prepared", return_value=({}, self.parent)):
            for mutate in mutations:
                changed = copy.deepcopy(runtime)
                mutate(changed)
                with self.assertRaises(ValueError):
                    verify_runtime(changed, {})

    def test_rejects_changed_artifact_or_failed_run(self):
        (self.root / "error.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "failed training"):
            self.runtime()
        (self.root / "error.json").unlink()
        (self.root / "head.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "modified"):
            self.runtime()

    def test_old_upstream_runtime_still_verifies(self):
        runtime = {
            **self.parent,
            "protocol": "native-choice-jsonl-v1",
            "modelWeightsLoaded": True,
            "parameterStorage": "float32",
            "forwardAutocast": "bfloat16",
            "maxLen": 2048,
        }
        with patch("integration.models.provenance.verify_prepared", return_value=({}, self.parent)):
            verify_runtime(runtime, {})


if __name__ == "__main__":
    unittest.main()
