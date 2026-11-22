"""The #184 preflight keeps model identity, paired seeds and replay evidence."""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from honing.backends.rea_issac.catalog import OPTIONS
from honing.backends.rea_issac_agent.controller import NAVIGATE_EMPTY_ROOMS
from honing.backends.rea_issac_agent.faults import CONTROLLER_SWAP_MOVE_AXES, HARNESS_REVERSE_MOVE
from honing.backends.rea_issac_agent.match import audit_agent_run
from honing.backends.rea_issac_agent.model_preflight import PINS, run_preflight


class ModelPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.revision = json.loads(PINS.read_text())["nanojevModels"]["nanojev"]["revision"]
        self.requests: list[dict[str, Any]] = []
        self.metadata = {"revision": self.revision, "weightsSha256": "a" * 64, "family": "nanojev"}
        requests, metadata = self.requests, self.metadata

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                assert self.path == "/v1/health"
                self._send({"ready": True, "metadata": metadata})

            def do_POST(self) -> None:
                assert self.path == "/v1/decide"
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(request)
                choices = {"move": "E", "shoot": "N", "bomb": "NO"}
                self._send(
                    {
                        "id": request["id"],
                        "stateText": request["parts"]["essential"][0],
                        "trimmed": [],
                        "tokenAudit": {},
                        "answers": {
                            name: {
                                "choice": choices[name],
                                "probabilities": {
                                    option: (
                                        (0.7 if option == "E" else 0.3 if option == "W" else 0.0)
                                        if name == "move"
                                        else float(option == choices[name])
                                    )
                                    for option in options
                                },
                            }
                            for name, options in OPTIONS.items()
                        },
                    }
                )

            def _send(self, value: dict[str, Any]) -> None:
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_paired_runs_audit_and_record_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preflight"
            summary = run_preflight(self.url, output, seeds=(7,), max_ticks=6, decision_frames=3)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual([(r["seed"], r["arm"]) for r in summary["games"]], [(7, "baseline"), (7, "model")])
            self.assertEqual(len(self.requests), 2)
            self.assertEqual(summary["weights_sha256"], "a" * 64)
            self.assertEqual(summary, json.loads((output / "summary.json").read_text()))
            self.assertTrue((output / "s7-model" / "replay.json").exists())
            self.assertNotIn(self.url, (output / "summary.json").read_text())
            with self.assertRaises(FileExistsError):
                run_preflight(self.url, output, seeds=(7,), max_ticks=6)

    def test_wrong_checkpoint_is_rejected_before_collection(self) -> None:
        self.metadata["revision"] = "another-checkpoint"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preflight"
            with self.assertRaisesRegex(ValueError, "checkpoint revision"):
                run_preflight(self.url, output, seeds=(7,), max_ticks=6)
            self.assertFalse(output.exists())
            self.assertFalse(self.requests)

    def test_wrong_weight_checksum_is_rejected_before_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preflight"
            with self.assertRaisesRegex(ValueError, "weights differ"):
                run_preflight(
                    self.url,
                    output,
                    seeds=(7,),
                    max_ticks=6,
                    expected_weights_sha256="b" * 64,
                )
            self.assertFalse(output.exists())
            self.assertFalse(self.requests)

    def test_model_controller_pair_is_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "agent-preflight"
            summary = run_preflight(
                self.url,
                output,
                seeds=(1002,),
                max_ticks=72,
                decision_frames=3,
                agent_controller=NAVIGATE_EMPTY_ROOMS,
            )
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["agent_controller"], NAVIGATE_EMPTY_ROOMS)
            run = output / "s1002-model"
            decisions = [json.loads(line) for line in (run / "agent_decisions.jsonl").read_text().splitlines()]
            self.assertTrue(any(row["controller_reason"] == "model_combat" for row in decisions))
            self.assertTrue(any(row["selected_action"] != row["executed_action"] for row in decisions))
            self.assertEqual(decisions[0]["model_response"]["answers"]["move"]["probabilities"]["E"], 0.7)
            recorded = [json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines()]
            self.assertEqual(recorded[0]["response"]["answers"]["move"]["probabilities"]["E"], 0.0)
            self.assertEqual(audit_agent_run(run)["decisions"], len(decisions))
            decisions[0]["selected_action"]["move_x"] = -1
            (run / "agent_decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in decisions))
            with self.assertRaisesRegex(ValueError, "agent metadata changed"):
                audit_agent_run(run)
            evidence = json.loads((run / "agent.json").read_text())
            evidence["agent_decisions_sha256"] = hashlib.sha256(
                (run / "agent_decisions.jsonl").read_bytes()
            ).hexdigest()
            (run / "agent.json").write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError, "agent selected/executed record"):
                audit_agent_run(run)

    def test_controller_fault_records_clean_and_swapped_actions(self) -> None:
        selected = json.loads((PINS.parents[2] / "honing/backends/rea_issac_agent/fault_freeze.json").read_text())[
            "checkpoint_sha256"
        ]
        self.metadata.update(revision=selected, weightsSha256=selected)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "controller-fault"
            summary = run_preflight(
                self.url,
                run,
                seeds=(1002,),
                max_ticks=72,
                agent_controller=NAVIGATE_EMPTY_ROOMS,
                expected_revision=selected,
                controller_fault=CONTROLLER_SWAP_MOVE_AXES,
            )
            assert summary["controller_fault"] == CONTROLLER_SWAP_MOVE_AXES
            records = [
                json.loads(line) for line in (run / "s1002-model/agent_decisions.jsonl").read_text().splitlines()
            ]
            assert any(record["fault_triggered"] for record in records)
            assert all(
                record["controller_before_fault"] != record["executed_action"]
                for record in records
                if record["fault_triggered"]
            )
            assert audit_agent_run(run / "s1002-model")["fault_triggers"] > 0

    def test_harness_fault_changes_rendered_request(self) -> None:
        selected = json.loads((PINS.parents[2] / "honing/backends/rea_issac_agent/fault_freeze.json").read_text())[
            "checkpoint_sha256"
        ]
        self.metadata.update(revision=selected, weightsSha256=selected)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "harness-fault"
            summary = run_preflight(
                self.url,
                run,
                seeds=(1002,),
                max_ticks=6,
                expected_revision=selected,
                harness_fault=HARNESS_REVERSE_MOVE,
            )
            assert summary["harness_fault"] == HARNESS_REVERSE_MOVE
            assert "Movement controls are inverted" in self.requests[0]["questions"]["move"]["instructions"]

    def test_harness_fault_rejects_a_changed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "wrong-model"
            with self.assertRaisesRegex(ValueError, "frozen clean checkpoint"):
                run_preflight(self.url, output, seeds=(1002,), harness_fault=HARNESS_REVERSE_MOVE)
            self.assertFalse(output.exists())
            self.assertFalse(self.requests)

    def test_mixed_faults_are_rejected_before_a_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "mixed"
            with self.assertRaisesRegex(ValueError, "only one component fault"):
                run_preflight(
                    self.url,
                    output,
                    seeds=(1002,),
                    agent_controller=NAVIGATE_EMPTY_ROOMS,
                    harness_fault=HARNESS_REVERSE_MOVE,
                    controller_fault=CONTROLLER_SWAP_MOVE_AXES,
                )
            self.assertFalse(output.exists())
            self.assertFalse(self.requests)


if __name__ == "__main__":
    unittest.main()
