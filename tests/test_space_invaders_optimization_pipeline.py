"""State, evidence and Gate checks for the integration-only game pipeline."""

from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
INTEGRATION = HERE / "integrations" / "space-invaders"
SPEC = importlib.util.spec_from_file_location("optimization_pipeline", INTEGRATION / "optimization_pipeline.py")
assert SPEC is not None and SPEC.loader is not None
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)
ADAPTER_SPEC = importlib.util.spec_from_file_location("pipeline_sft_adapter", INTEGRATION / "pipeline_sft_adapter.py")
assert ADAPTER_SPEC is not None and ADAPTER_SPEC.loader is not None
adapter = importlib.util.module_from_spec(ADAPTER_SPEC)
ADAPTER_SPEC.loader.exec_module(adapter)
RUNNER_SPEC = importlib.util.spec_from_file_location("pipeline_game_runner", INTEGRATION / "pipeline_game_runner.py")
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)
BUNDLE_SPEC = importlib.util.spec_from_file_location("pipeline_bundle", INTEGRATION / "pipeline_bundle.py")
assert BUNDLE_SPEC is not None and BUNDLE_SPEC.loader is not None
bundle = importlib.util.module_from_spec(BUNDLE_SPEC)
BUNDLE_SPEC.loader.exec_module(bundle)


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        code = self.root / "code"
        code.mkdir()
        for name in pipeline.CODE_NAMES:
            (code / name).write_text(f"# frozen {name}\n")
        health = self.root / "health.json"
        health.write_text(json.dumps({"ready": True, "metadata": {"weightsSha256": "old"}}))
        trace = self.root / "current.jsonl"
        trace.write_text("{}\n")
        video = self.root / "current.mp4"
        video.write_bytes(b"video")
        self.config = {
            "schema": 1,
            "game": "ALE/SpaceInvaders-v5",
            "environment": {"ale": "ALE/SpaceInvaders-v5"},
            "seeds": {"development": [1, 2], "validation": [3, 4], "gate": [5, 6]},
            "gate_seed_ledger": str(self.root / "ledger"),
            "bootstrap_seed": 31,
            "incumbent": {"health": str(health), "weights_sha256": "old", "checkpoint": "fake/checkpoint"},
            "base_model": {"weights_sha256": "base"},
            "remote": {"host": "ds-serv2.ucsd.edu", "gpu_uuid": "GPU-4-test"},
            "code": {name: str(code / name) for name in pipeline.CODE_NAMES},
            "review_inputs": [
                {"kind": "code", "path": str(code / "lane_pilot.py")},
                {"kind": "trace", "path": str(trace)},
                {"kind": "video", "path": str(video)},
            ],
            "runner_command": [
                sys.executable,
                str(HERE / "tests/fakes/space_invaders_pipeline_runner.py"),
                "--run",
                "{run}",
                "--split",
                "{split}",
                "--arm",
                "{arm}",
                "--seeds",
                "{seeds}",
                "--code-dir",
                "{code_dir}",
                "--weights-sha256",
                "{weights_sha256}",
                "--summary",
                "{summary}",
                "--trace",
                "{trace}",
            ],
        }
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config))
        self.run = self.root / "run"

    def test_sft_stages_optional_replay_once(self) -> None:
        train = self.root / "train.jsonl"
        validation = self.root / "validation.jsonl"
        replay = self.root / "replay.jsonl.gz"
        bundle_path = self.root / "bundle.tar.gz"
        without_replay = adapter.staging_sources(train, validation, replay, bundle_path)
        self.assertNotIn((replay, "replay.jsonl.gz"), without_replay)
        replay.write_bytes(b"replay")
        with_replay = adapter.staging_sources(train, validation, replay, bundle_path)
        self.assertEqual(with_replay.count((replay, "replay.jsonl.gz")), 1)

    def freeze_and_review(self) -> None:
        pipeline.freeze(self.config_path, self.run)
        reviewer = [
            sys.executable,
            "-c",
            "import json,pathlib; "
            "(pathlib.Path.cwd().parent/'independent-review.json').write_text("
            "json.dumps({'findings':[{'failure':'missed shots',"
            "'evidence':'current.jsonl','confidence':0.7}]}))",
        ]
        pipeline.review(self.run, reviewer)

    def proposal(self, method: str = "manual_code") -> Path:
        data = {
            "bottleneck": "model_ranking" if method == "sft" else "execution_code",
            "method": method,
            "primary_target": "weights" if method == "sft" else "lane_controller.py",
            "hypothesis": "A single change improves score",
            "metrics": ["mean_score", "model_calls"],
            "controls": ["incumbent", "random", "base", "code"],
            "budget": {"development_games": 20, "validation_games": 4, "gate_games": 10},
            "validation_min_mean_gain": 1,
            "gate": {"selector": "no_regression", "min_mean_gain": 1, "require_positive_ci_low": True},
        }
        path = self.root / "proposal.json"
        path.write_text(json.dumps(data))
        return path

    def test_manual_candidate_runs_validation_and_honing_gate(self) -> None:
        self.freeze_and_review()
        pipeline.register(self.run, self.proposal())
        candidate = self.root / "candidate"
        candidate.mkdir()
        for name in pipeline.CODE_NAMES:
            candidate.joinpath(name).write_bytes((self.run / "incumbent-code" / name).read_bytes())
        (candidate / "lane_controller.py").write_text("# changed execution code\n")
        pipeline.develop(self.run, candidate)
        pipeline.validation(self.run)
        pipeline.gate(self.run)
        result = pipeline.read(self.run / "gate-result.json")
        self.assertTrue(result["passed"])
        self.assertEqual(result["mean_scores"]["candidate"], 120)
        self.assertEqual(result["wins"], 2)
        self.assertEqual(result["honing_gate"]["policy"], "no_regression")
        self.assertEqual(pipeline.read(self.run / "manifest.json")["stage"], "gated")
        self.assertEqual(len(pipeline.read(self.run / "allowed-inputs.json")["files"]), 3)
        next_config = {
            **self.config,
            "seeds": {"development": [7], "validation": [8, 9], "gate": [10, 11]},
            "code": {name: str(self.run / "candidate-code" / name) for name in pipeline.CODE_NAMES},
        }
        next_path = self.root / "next-config.json"
        next_path.write_text(json.dumps(next_config))
        pipeline.next_round(self.run, next_path, self.root / "next-run")
        self.assertEqual(pipeline.read(self.root / "next-run" / "manifest.json")["stage"], "frozen")

    def test_gate_seeds_cannot_be_reused(self) -> None:
        pipeline.freeze(self.config_path, self.run)
        with self.assertRaisesRegex(ValueError, "already reserved"):
            pipeline.freeze(self.config_path, self.root / "another")

    def test_failed_validation_keeps_incumbent_for_next_round(self) -> None:
        self.config["fake_candidate_score"] = 80
        self.config_path.write_text(json.dumps(self.config))
        self.freeze_and_review()
        pipeline.register(self.run, self.proposal())
        candidate = self.root / "candidate"
        candidate.mkdir()
        for name in pipeline.CODE_NAMES:
            (candidate / name).write_bytes((self.run / "incumbent-code" / name).read_bytes())
        (candidate / "lane_controller.py").write_text("# failed candidate\n")
        pipeline.develop(self.run, candidate)
        pipeline.validation(self.run)
        outcome = pipeline.read(self.run / "validation-result.json")
        self.assertFalse(outcome["passed"])
        self.assertIn("mean score gain below preregistered validation threshold", outcome["reasons"])
        with self.assertRaisesRegex(ValueError, "Validation failed"):
            pipeline.gate(self.run)
        next_config = {**self.config, "seeds": {"development": [7], "validation": [8, 9], "gate": [10, 11]}}
        next_path = self.root / "next-config.json"
        next_path.write_text(json.dumps(next_config))
        pipeline.next_round(self.run, next_path, self.root / "next-run")
        self.assertEqual(pipeline.read(self.root / "next-run" / "manifest.json")["weights_sha256"], "old")

    def test_unadapted_meta_harness_is_not_mislabeled(self) -> None:
        self.freeze_and_review()
        with self.assertRaisesRegex(ValueError, "no Space Invaders method adapter"):
            pipeline.register(self.run, self.proposal("meta_harness"))

    def test_valid_model_choice_override_is_detected(self) -> None:
        summary = self.root / "summary.json"
        trace = self.root / "trace.jsonl"
        health = self.root / "server-health.json"
        health.write_text(json.dumps({"ready": True, "metadata": {"weightsSha256": "old"}}))
        (self.root / "model-snapshot.json").write_text(
            json.dumps(
                {
                    "weights_sha256": "old",
                    "checkpoint": "fake/checkpoint",
                    "parameters": 1000,
                    "architecture": {"num_layers": 2, "hidden_size": 16, "attention_heads": 2, "ffn_size": 32},
                }
            )
        )
        summary.write_text(
            json.dumps(
                {
                    "pipeline_metadata": {
                        "code_sha256": {"a": "b"},
                        "weights_sha256": "old",
                        "environment_sha256": "env",
                        "model_health_sha256": pipeline.digest(health),
                    },
                    "results": [{"seed": 1, "reward": 1, "steps": 1, "finished": True, "lives": 0, "model_calls": 1}],
                }
            )
        )
        trace.write_text(
            json.dumps(
                {
                    "seed": 1,
                    "step": 0,
                    "reward": 1,
                    "wave": 1,
                    "lives_before": 1,
                    "lives_after": 0,
                    "player_x": 50,
                    "action": "FIRE",
                    "model_choice": "A",
                    "target_type": "alien",
                    "locked_bullet_x": 70,
                    "lane_options": [{"bullet_x": 50}, {"bullet_x": 70}],
                    "selected_lane": {"bullet_x": 70},
                }
            )
            + "\n"
        )
        audit = pipeline.audit_arm(summary, trace, [1], {"a": "b"}, "old", "env")
        self.assertEqual(audit["choice_violations"], [{"seed": 1, "step": 0, "raw_choice": "A"}])

    def test_sft_gate_requires_all_four_model_controls(self) -> None:
        self.freeze_and_review()
        pipeline.register(self.run, self.proposal("sft"))
        log = self.run / "training.log"
        log.write_text("validation accuracy=0.7\n")

        def training_command(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            (self.run / "method-result.json").write_text(
                json.dumps(
                    {
                        "method": "sft",
                        "weights_sha256": "new",
                        "checkpoint": "fake/new-checkpoint",
                        "checkpoint_manifest": "fake/new-manifest.json",
                        "training_metrics": {"validation_accuracy": 0.7},
                        "training_log": str(log),
                        "equivalent_metrics_explanation": "Loss and held-out accuracy are in the log.",
                        "development_games_spent": 8,
                        "training_host": "ds-serv2.ucsd.edu",
                        "gpu_uuid": "GPU-4-test",
                        "data_seeds": [1, 2],
                    }
                )
            )
            return subprocess.CompletedProcess(args, 0)

        with patch.object(pipeline.subprocess, "run", side_effect=training_command):
            pipeline.develop(
                self.run,
                method_command=[
                    sys.executable,
                    str(INTEGRATION / "pipeline_sft_adapter.py"),
                    "--data-dir",
                    "fake-data",
                ],
            )
        pipeline.validation(self.run)
        pipeline.gate(self.run)
        result = pipeline.read(self.run / "gate-result.json")
        self.assertEqual(set(result["mean_scores"]), {"incumbent", "candidate", "random", "base", "code"})
        self.assertTrue(result["passed"])

    def test_sft_rejects_replay_that_contains_final_gate_seed(self) -> None:
        self.freeze_and_review()
        pipeline.register(self.run, self.proposal("sft"))
        data = self.root / "sft-data"
        data.mkdir()
        (data / "train.jsonl").write_text((json.dumps({"seed": 1, "label": "A"}) + "\n") * 200)
        (data / "validation.jsonl").write_text((json.dumps({"seed": 2, "label": "B"}) + "\n") * 60)
        with gzip.open(data / "replay.jsonl.gz", "wt") as output:
            output.write(json.dumps({"seed": 5, "label": "A"}) + "\n")
        with (
            patch.object(sys, "argv", ["pipeline_sft_adapter.py", "--run", str(self.run), "--data-dir", str(data)]),
            self.assertRaisesRegex(ValueError, "Gate seeds"),
        ):
            adapter.main()

    def test_gpu_hold_requires_matching_command_and_uuid(self) -> None:
        settings = {
            "hold_unit": "gpu4-new-task-hold.service",
            "runtime": "/model/python",
            "hold_script": "/hold.py",
            "gpu_uuid": "GPU-4-test",
        }
        outputs = iter(["123\n", "/other/python /hold.py\n"])

        def process(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args, 0, stdout=next(outputs))

        with (
            patch.object(runner, "run", side_effect=process),
            patch.object(runner, "gpu_processes", return_value=[("GPU-4-test", 123)]),
            self.assertRaisesRegex(RuntimeError, "command"),
        ):
            runner.verify_hold(settings)

    def test_frozen_honing_bundle_is_verified_before_extraction(self) -> None:
        source = HERE / "honing" / "__init__.py"
        archive = self.root / "honing.tar.gz"
        version = {"honing/__init__.py": bundle.sha(source)}
        checksum = bundle.make_bundle(HERE, version, archive)
        destination = self.root / "remote"
        destination.mkdir()
        bundle.extract_bundle(archive, destination, checksum)
        self.assertEqual((destination / "honing" / "__init__.py").read_bytes(), source.read_bytes())
        with self.assertRaisesRegex(ValueError, "wrong hash"):
            bundle.extract_bundle(archive, self.root, "not-the-sha")


if __name__ == "__main__":
    unittest.main()
