"""Regression checks for the Exp 0 episode accounting and offline evaluator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from honing.backends.rea_issac_agent.collect_exp0 import (
    FAULTS,
    FORMAT,
    HASH_FILE,
    SEEDS,
    _files,
    _index,
    _write,
    verify,
)
from honing.backends.rea_issac_agent.score_exp0 import MODES, score


def _attempts() -> list[dict[str, Any]]:
    return [
        {
            "episode_id": f"rea-issac-s{seed}-{arm}",
            "seed": seed,
            "arm": arm,
            "path": f"attempts/s{seed}-{arm}",
            "weights_sha256": "test-weights",
            "harness_spec_id": "test-spec",
            "outcome": {"success": arm == "no_injection" or seed == SEEDS[0]},
            "status": "eligible",
        }
        for seed in SEEDS
        for arm in FAULTS
    ]


class CollectionTests(unittest.TestCase):
    def test_fixed_groups_keep_the_four_arms_and_prefer_injected_failures(self) -> None:
        index = _index(_attempts())
        self.assertEqual(len(index), 40)
        self.assertEqual({row["seed"] for row in index}, set(SEEDS[1:11]))
        self.assertEqual({row["annotation_status"] for row in index}, {"control", "verified_component"})
        self.assertEqual({row["whole_episode_effect_status"] for row in index}, {"untested", "not_applicable"})
        self.assertTrue(all(row["paired_clean_outcome"]["success"] for row in index))

    def test_incomplete_group_cannot_fill_the_quota(self) -> None:
        attempts = _attempts()
        for row in attempts:
            if row["seed"] in SEEDS[:3] and row["arm"] == "controller":
                row["status"] = "excluded"
        with self.assertRaisesRegex(ValueError, "only 9 of 10"):
            _index(attempts)

    def test_verify_detects_changed_evidence_bytes(self) -> None:
        attempts = _attempts()
        index = _index(attempts)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _write(output / "protocol.json", {"format": FORMAT})
            _write(output / "preflight_receipt.json", {"passed": True})
            _write(output / "attempts.json", attempts)
            _write(output / "episode_index.json", index)
            _write(
                output / "split_manifest.json",
                {"v0-development-evaluation": sorted(row["episode_id"] for row in index)},
            )
            for row in index:
                run = output / row["path"]
                run.mkdir(parents=True)
                _write(
                    run / "manifest.json",
                    {
                        "scenario": {"seed": row["seed"]},
                        "spec_id": row["harness_spec_id"],
                        "policy_checksum": row["weights_sha256"],
                    },
                )
                _write(run / "result.json", row["outcome"])
                _write(
                    run / "component_evidence.json",
                    {"audit": {"verified": True}, "identity": {"passed": True}, "component_effect": True},
                )
            _write(output / HASH_FILE, _files(output))
            receipt = verify(output)
            self.assertEqual(receipt["components"], dict.fromkeys(FAULTS, 10))
            _write(output / index[0]["path"] / "result.json", {"success": True, "changed": True})
            with self.assertRaisesRegex(ValueError, "hashes differ"):
                verify(output)

    def test_scores_abstentions_and_keeps_false_positives_separate(self) -> None:
        index = _index(_attempts())
        replies = [
            {
                "episode_id": row["episode_id"],
                "mode": mode,
                "label": "abstain" if mode == "read" else row["verified_component"] or "weights",
            }
            for row in index
            for mode in MODES
        ]
        metrics = score(index, replies)
        self.assertEqual(metrics["read"]["component_accuracy"], 0)
        self.assertEqual(metrics["probe"]["component_accuracy"], 1)
        self.assertEqual(metrics["read"]["control_false_positive_rate"], 0)
        self.assertEqual(metrics["probe"]["control_false_positive_rate"], 1)
        self.assertIsNone(metrics["probe"]["total_cost_usd"])
        self.assertEqual(metrics["paired_probe_minus_read"]["reset_group_bootstrap_95"], [1, 1])
        with self.assertRaisesRegex(ValueError, "one saved diagnosis"):
            score(index, replies[:-1])
