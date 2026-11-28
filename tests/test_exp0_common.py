"""End-to-end and unit tests of the Exp 0 v0 common tools, on synthetic episodes of both decision shapes.

  python -m pytest tests/test_exp0_common.py

Writes everything under ``EXP0_TEST_OUT`` (default: a temporary directory, removed at exit).
Starts only in-process fake decision services on free 127.0.0.1 ports and the
stub CLI; no real model, no game, no network beyond localhost. Pipeline per shape: collect (with a failed
attempt, a rejected-response attempt and reserve groups) -> validate -> probes selfcheck -> re-ask -> audit
-> read inputs -> probe capture x2 + compare -> stub diagnosis read/probe (with a retry) -> score ->
fill cards -> bundle release -> verify_release.py in a fresh venv without packages -> tampering checks.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parents[1] / "integrations" / "_shared" / "exp0"
TESTS = COMMON / "tests"
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(COMMON))

import audit_set
import bundle
import diagnosis_inputs
import probes
import score
import synthetic_game as sg
import validate_episode
from adapter import load_adapter
from contract import (
    ResponseRejected,
    guard_decision,
    guard_response,
    read_json,
    read_rows,
    same_request,
    write_json,
    write_rows,
)
from fake_laya import FakeLaya, answer
from schema_lite import SchemaError, errors, load_schema

_TEMPORARY = None if os.environ.get("EXP0_TEST_OUT") else tempfile.TemporaryDirectory(prefix="exp0-common-")
OUT = Path(os.environ.get("EXP0_TEST_OUT") or _TEMPORARY.name)
PYTHON = sys.executable


def run(args: list[str], env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    process = subprocess.run(
        args, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})}
    )
    if check and process.returncode:
        raise AssertionError(f"{' '.join(args)} failed ({process.returncode}):\n{process.stdout}\n{process.stderr}")
    return process


@pytest.mark.unit
class GuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = sg.render("option", {"hp": 3, "enemies_dx": [1], "pits_dx": [], "exit_dx": None}, "d1-t2")
        self.reply = answer(self.request, "clean")

    def test_accepts_a_complete_reply(self) -> None:
        parsed = guard_response(self.request, json.dumps(self.reply))
        self.assertEqual(parsed["server_choice"], self.reply["answers"]["action"]["choice"])
        self.assertAlmostEqual(sum(parsed["normalized"].values()), 1.0)

    def test_rejections(self) -> None:
        cases = {
            "id": lambda r: r.update(id="other"),
            "sum": lambda r: r["answers"]["action"]["probabilities"].update(wait=0.9),
            "argmax": lambda r: r["answers"]["action"].update(
                choice=min(r["answers"]["action"]["probabilities"], key=r["answers"]["action"]["probabilities"].get)
            ),
            "trimmed": lambda r: r.update(trimmed=[{"section": "logs", "text": "x"}]),
            "truncated": lambda r: r["tokenAudit"]["action"].update(stateTruncated=True),
            "state": lambda r: r.update(stateText="something else"),
            "options": lambda r: r["answers"]["action"]["probabilities"].pop("wait"),
        }
        for name, change in cases.items():
            reply = copy.deepcopy(self.reply)
            change(reply)
            with self.subTest(name), self.assertRaises(ResponseRejected):
                guard_response(self.request, json.dumps(reply))
        with self.assertRaises(ResponseRejected):
            guard_response(self.request, b"not json")
        unknown = copy.deepcopy(self.request)
        unknown["parts"]["situation"] = ["ignored by Laya"]
        with self.assertRaises(ResponseRejected):
            guard_response(unknown, json.dumps(answer(unknown, "clean")))


@pytest.mark.functional
class OrderTests(unittest.TestCase):
    """The order of a request's options is data: rows keep it, comparisons see it, the validator checks it."""

    def test_fake_model_is_order_sensitive_and_same_request_sees_order(self) -> None:
        request = sg.render("native", {"hp": 3, "enemies_dx": [2], "pits_dx": [], "exit_dx": None}, "d1-t2")
        reordered = copy.deepcopy(request)
        criteria = reordered["questions"]["action"]["criteria"]
        reordered["questions"]["action"]["criteria"] = dict(sorted(criteria.items()))
        self.assertEqual(request, reordered)  # dict equality ignores the order
        self.assertFalse(same_request(request, reordered))
        self.assertNotEqual(
            answer(request, "clean")["answers"]["action"]["probabilities"],
            answer(reordered, "clean")["answers"]["action"]["probabilities"],
        )

    def test_validator_catches_sorted_rows(self) -> None:
        root = OUT / "order" / "collection"
        shutil.rmtree(root.parent, ignore_errors=True)
        root.mkdir(parents=True)
        with FakeLaya("clean") as clean, FakeLaya("alternate") as alternate:
            plan = sg.plan("native", clean.sha, alternate.sha)
            write_json(root / "collection.json", plan)
            run_dir = sg.collect(root, plan, plan["groups"][1], "no_injection", 1, clean.url)
        self.assertEqual(validate_episode.validate(root)[0]["status"], "valid")
        rows = read_rows(run_dir / "decisions.jsonl")
        (run_dir / "decisions.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        report = validate_episode.validate(root)[0]
        self.assertEqual(report["status"], "invalid")
        self.assertIn("never write them with sort_keys", " ".join(report["errors"]))


@pytest.mark.functional
class SelectionAndWindowsTests(unittest.TestCase):
    """Contract 0.1.2: per-group policy seeds for sampled selection; option rows that send no native input."""

    def test_per_group_seed_sampling_and_empty_windows(self) -> None:
        root = OUT / "sampling" / "collection"
        shutil.rmtree(root.parent, ignore_errors=True)
        root.mkdir(parents=True)
        with FakeLaya("clean") as clean, FakeLaya("alternate") as alternate:
            plan = sg.plan("option", clean.sha, alternate.sha, groups=6)
            plan["selection"] = {"rule": "sample", "seed": None, "per_group_seed": "scenario.policy_seed"}
            plan["extra"] = {"wait_sends_nothing": True}  # an option that ends without native input (HK KP case)
            for number, group in enumerate(plan["groups"]):
                group["scenario"]["policy_seed"] = 7000 + number
            write_json(root / "collection.json", plan)
            for group in plan["groups"]:
                sg.collect(root, plan, group, "no_injection", 1, clean.url)
                sg.collect(root, plan, group, "controller", 1, clean.url)
        reports = validate_episode.validate(root)
        self.assertEqual(Counter(r["status"] for r in reports), Counter(valid=len(reports)), reports[:1])
        rows = [row for path in validate_episode.runs_of(root) for row in read_rows(path / "decisions.jsonl")]
        self.assertTrue(
            any(row["executed"] and not row["executed"]["windows"] for row in rows),
            "the synthetic game should produce an option that sends no input",
        )
        self.assertTrue(
            any(row["model_choice"] != row["server_choice"] for row in rows if row["kind"] == "model"),
            "sampling should sometimes pick another option than the argmax",
        )
        run_dir = validate_episode.runs_of(root)[0]
        manifest = read_json(run_dir / "manifest.json")
        manifest["selection"]["seed"] += 1
        write_json(run_dir / "manifest.json", manifest)
        report = validate_episode.validate(root)[0]
        self.assertEqual(report["status"], "invalid")


@pytest.mark.functional
class TwoQuestionTests(unittest.TestCase):
    """Contract 0.2.0: a decision asks several questions in one request (Tank War v0; full HK raw keys)."""

    def setUp(self) -> None:
        self.request = sg.render_keys({"hp": 3, "enemies_dx": [1], "pits_dx": [], "exit_dx": None}, "d1-t2")
        self.reply = answer(self.request, "clean")

    def test_guard_accepts_and_rejects(self) -> None:
        parsed = guard_decision(self.request, json.dumps(self.reply), ["move", "attack"])
        self.assertEqual(set(parsed["questions"]), {"move", "attack"})
        self.assertEqual(parsed["questions"]["attack"]["server_choice"], "strike")  # "enemy is 1 step" raises it
        cases = {
            "answer missing": lambda r: r["answers"].pop("attack"),
            "audit missing": lambda r: r["tokenAudit"].pop("move"),
            "sum": lambda r: r["answers"]["attack"]["probabilities"].update(none=0.9),
            "argmax": lambda r: r["answers"]["move"].update(
                choice=min(r["answers"]["move"]["probabilities"], key=r["answers"]["move"]["probabilities"].get)
            ),
            "truncated": lambda r: r["tokenAudit"]["attack"].update(stateTruncated=True),
            "option count": lambda r: r["tokenAudit"]["move"].update(options=3),
        }
        for name, change in cases.items():
            reply = copy.deepcopy(self.reply)
            change(reply)
            with self.subTest(name), self.assertRaises(ResponseRejected):
                guard_decision(self.request, json.dumps(reply), ["move", "attack"])
        with self.assertRaises(ResponseRejected):  # the plan's question order is part of the request
            guard_decision(self.request, json.dumps(self.reply), ["attack", "move"])
        with self.assertRaises(ResponseRejected):  # the single-question guard refuses a two-question request
            guard_response(self.request, json.dumps(self.reply), "move")

    def test_validator_on_two_question_rows(self) -> None:
        root = OUT / "keys-validator" / "collection"
        shutil.rmtree(root.parent, ignore_errors=True)
        root.mkdir(parents=True)
        with FakeLaya("clean") as clean, FakeLaya("alternate") as alternate:
            plan = sg.plan_keys(clean.sha, alternate.sha, groups=2)
            write_json(root / "collection.json", plan)
            for condition in ("no_injection", "harness", "controller"):
                sg.collect_keys(root, plan, plan["groups"][0], condition, 1, clean.url)
            sg.collect_keys(
                root, plan, plan["groups"][1], "controller", 1, clean.url, controller_instance="extend-first-move-v1"
            )
            sg.collect_keys(root, plan, plan["groups"][0], "weights", 1, alternate.url)
        reports = {r["run"]: r for r in validate_episode.validate(root)}
        self.assertTrue(
            all(r["status"] == "valid" for r in reports.values()), [r for r in reports.values() if r["errors"]][:1]
        )
        self.assertTrue(
            all(r["summary"]["questions"] == 2 * r["summary"]["model_decisions"] for r in reports.values())
        )
        self.assertTrue(reports["s00-controller-a1"]["summary"]["triggered"])
        self.assertTrue(reports["s01-controller-a1"]["summary"]["triggered"])
        extended = [
            w
            for row in read_rows(root / "s01-controller-a1" / "decisions.jsonl")
            for w in ((row["executed"] or {}).get("windows") or [])
            if "intended_ticks" in w
        ]
        self.assertEqual(len(extended), 1)

        def tampered(run: str, change) -> dict:
            source = root / run
            target = root.parent / "tampered" / run
            shutil.rmtree(target.parent, ignore_errors=True)
            shutil.copytree(source, target)
            rows = read_rows(target / "decisions.jsonl")
            change(rows)
            (target / "decisions.jsonl").unlink()
            write_rows(target / "decisions.jsonl", rows)
            return validate_episode.validate(target, root / "collection.json")[0]

        def first_model(rows):
            return next(row for row in rows if row["kind"] == "model")

        cases = {
            "flat probabilities": (
                "s00-no_injection-a1",
                lambda rows: first_model(rows).update(probabilities=first_model(rows)["probabilities"]["move"]),
            ),
            "choice of one question": (
                "s00-no_injection-a1",
                lambda rows: first_model(rows).update(
                    model_choice={"move": first_model(rows)["model_choice"]["move"]}
                ),
            ),
            "executed option not offered": (
                "s00-no_injection-a1",
                lambda rows: first_model(rows)["executed"].update(option={"move": "fly", "attack": "none"}),
            ),
            "question order": (
                "s00-no_injection-a1",
                lambda rows: first_model(rows).update(
                    model_choice=dict(reversed(list(first_model(rows)["model_choice"].items())))
                ),
            ),
            "change without fault record": (
                "s00-controller-a1",
                lambda rows: [row.update(fault=None) for row in rows if row["fault"]],
            ),
            "intended length equal": (
                "s01-controller-a1",
                lambda rows: [
                    w.update(intended_ticks=w["ticks"])
                    for row in rows
                    for w in ((row["executed"] or {}).get("windows") or [])
                    if "intended_ticks" in w
                ],
            ),
        }
        for name, (run, change) in cases.items():
            with self.subTest(name):
                self.assertEqual(tampered(run, change)["status"], "invalid")


@pytest.mark.unit
class SchemaLiteTests(unittest.TestCase):
    def test_refuses_unsupported_keywords(self) -> None:
        path = OUT / "bad.schema.json"
        OUT.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"type": "object", "dependentRequired": {"a": ["b"]}}))
        with self.assertRaises(SchemaError):
            load_schema(path)

    def test_keywords(self) -> None:
        schema = {
            "type": "object",
            "required": ["a"],
            "properties": {"a": {"enum": [1, 2]}, "b": {"type": "integer"}},
            "if": {"properties": {"a": {"const": 2}}},
            "then": {"required": ["b"]},
        }
        self.assertEqual(errors({"a": 1}, schema), [])
        self.assertTrue(errors({"a": 2}, schema))
        self.assertTrue(errors({"a": True}, schema))
        self.assertTrue(errors({"a": 1, "b": 1.5}, schema))

    def test_all_shipped_schemas_load(self) -> None:
        for path in sorted((COMMON / "schemas").glob("*.schema.json")):
            load_schema(path)


@pytest.mark.unit
class ScoreTests(unittest.TestCase):
    def test_abstention_and_invalid_are_incorrect(self) -> None:
        index = []
        for number in range(40):
            condition = ("harness", "weights", "controller", "no_injection")[number % 4]
            row = {
                "episode_id": f"e{number:03d}",
                "condition": condition,
                "scenario_group": f"g{number % 10}",
                "annotation_status": "control" if condition == "no_injection" else "verified",
                "fault_instance": None if condition == "no_injection" else f"{condition}-v1",
            }
            if condition != "no_injection":
                row["verified_component"] = condition
            index.append(row)
        replies = []
        for row in index:
            for mode in ("read", "probe"):
                truth = row.get("verified_component")
                if mode == "read":
                    reply = {"cause": "uncertain", "confidence": 0.2}
                elif truth == "weights":
                    reply = {"status": "invalid"}
                else:
                    reply = {"cause": truth or "weights", "confidence": 0.9}
                replies.append(
                    {
                        "episode_id": row["episode_id"],
                        "mode": mode,
                        "model": "m",
                        "usage": None,
                        "cost_usd": None,
                        **reply,
                    }
                )
        result = score.score(index, replies)
        self.assertEqual(result["read"]["component_accuracy"], 0.0)
        self.assertEqual(result["read"]["abstention_rate"], 1.0)
        self.assertAlmostEqual(result["probe"]["component_accuracy"], 20 / 30)
        self.assertEqual(result["probe"]["confusion_matrix"]["weights"]["invalid"], 10)
        self.assertEqual(result["probe"]["control_false_positive_rate"], 1.0)
        self.assertIsNone(result["probe"]["cost_usd"])
        self.assertEqual(result, score.score(index, replies))


@pytest.mark.integration
@pytest.mark.slow
class PipelineTests(unittest.TestCase):
    """One full pipeline per decision shape; the option shape uses the re-ask path for weights evidence."""

    @classmethod
    def setUpClass(cls) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        cls.venv = OUT / "fresh-venv"
        if not cls.venv.exists():
            run([PYTHON, "-m", "venv", "--without-pip", str(cls.venv)])
        cls.fresh_python = str(cls.venv / "bin" / "python")
        listing = run(
            [
                cls.fresh_python,
                "-I",
                "-c",
                "import sys, pkgutil; print(sorted(m.name for m in pkgutil.iter_modules() "
                "if 'site-packages' in str(getattr(m.module_finder, 'path', ''))))",
            ]
        ).stdout.strip()
        cls.fresh_packages = listing

    def test_native_shape(self) -> None:
        self._pipeline("native", salt_weights=False)

    def test_option_shape(self) -> None:
        self._pipeline("option", salt_weights=True)

    def _pipeline(self, shape: str, *, salt_weights: bool) -> None:
        base = OUT / shape
        shutil.rmtree(base, ignore_errors=True)
        root = base / "collection"
        root.mkdir(parents=True)
        adapter_spec = f"{TESTS / 'synthetic_game.py'}:{'NativeAdapter' if shape == 'native' else 'OptionAdapter'}"
        with (
            FakeLaya("clean") as clean,
            FakeLaya("alternate") as alternate,
            FakeLaya("clean", corrupt_every=1) as broken,
        ):
            plan = sg.plan(shape, clean.sha, alternate.sha)
            write_json(root / "collection.json", plan)
            for group in plan["groups"]:
                for condition in ("no_injection", "harness", "weights", "controller"):
                    url = alternate.url if condition == "weights" else clean.url
                    salt = f"-w{group['id']}" if salt_weights and condition == "weights" else ""
                    attempt = 1
                    if group["id"] == "s00" and condition == "harness":
                        sg.collect(root, plan, group, condition, attempt, url, fail_at=0)
                        attempt += 1
                    if group["id"] == "s01" and condition == "controller":
                        sg.collect(root, plan, group, condition, attempt, broken.url)
                        attempt += 1
                    sg.collect(root, plan, group, condition, attempt, url, request_salt=salt)

            for path in validate_episode.runs_of(root)[:3]:  # raw extras: one text log and one binary frame each
                (path / "extra" / "frames").mkdir(parents=True)
                (path / "extra" / "frames" / "0001.jpg").write_bytes(b"\xff\xd8\xff\xe0" + bytes(64))
                (path / "extra" / "bridge.log").write_text("bridge ok\n")
            reports = validate_episode.validate(root)
            statuses = Counter(report["status"] for report in reports)
            self.assertEqual(statuses["invalid"], 0, [r for r in reports if r["status"] == "invalid"][:2])
            self.assertEqual(statuses["incomplete"], 2)
            cli = run([PYTHON, "-B", str(COMMON / "validate_episode.py"), str(root)])
            self.assertIn('"incomplete":2', cli.stdout)

            check = probes.selfcheck(root, load_adapter(adapter_spec), None)
            self.assertTrue(check["ok"], check)

            audit_set.reask(root, root / "weights-reask.jsonl", clean.url, None)
            self.assertEqual(audit_set.audit(root, root / "audit", root / "weights-reask.jsonl", None), 0)
            index = read_json(root / "audit" / "episodes.json")
            attempts = read_rows(root / "audit" / "attempts.jsonl")
            self.assertEqual(
                Counter(row["condition"] for row in index),
                Counter(dict.fromkeys(("no_injection", "harness", "weights", "controller"), 10)),
            )
            excluded = [row for row in attempts if row["annotation_status"] == "excluded"]
            self.assertEqual(len(excluded), 2, excluded)
            methods = Counter(row["evidence"]["method"] for row in index if row.get("evidence"))
            expected_weights = "same_request_reask" if salt_weights else "matched_clean_identical_request"
            self.assertEqual(methods[expected_weights], 10, methods)
            for row in index:
                groups = {r["split"] for r in index if r["scenario_group"] == row["scenario_group"]}
                self.assertEqual(len(groups), 1)

            diagnosis_inputs.build(root, root / "audit" / "episodes.json", root / "read-inputs.jsonl")
            texts = read_rows(root / "read-inputs.jsonl")
            runs = {row["run"] for row in index}
            for item in texts:
                self.assertEqual(set(item), {"episode_id", "text"})
                self.assertFalse(any(name in item["text"] for name in runs))
                self.assertNotIn("omit-first", item["text"])

            adapter = load_adapter(adapter_spec)
            probes.capture(root, index, clean.url, "clean", adapter, root / "probe-reference.jsonl")
            probes.capture(root, index, alternate.url, "alternate", adapter, root / "probe-alternate.jsonl")
            self.assertTrue(
                probes.compare(
                    root,
                    index,
                    root / "probe-reference.jsonl",
                    root / "probe-alternate.jsonl",
                    adapter,
                    root / "probes",
                )
            )
            with self.assertRaises(ValueError):
                probes.capture(root, index, clean.url, "alternate", adapter, base / "wrong-role.jsonl")

        state = base / "stub-state"
        state.mkdir()
        run(
            [
                PYTHON,
                "-B",
                str(COMMON / "diagnose.py"),
                str(root / "read-inputs.jsonl"),
                str(root / "diagnosis-read"),
                "--mode",
                "read",
                "--backend",
                "stub",
            ],
            env={"STUB_CODEX_MODE": "invalid_once", "STUB_CODEX_STATE": str(state)},
        )
        run(
            [
                PYTHON,
                "-B",
                str(COMMON / "diagnose.py"),
                str(root / "read-inputs.jsonl"),
                str(root / "diagnosis-probe"),
                "--mode",
                "probe",
                "--probes",
                str(root / "probes" / "probe-inputs.jsonl"),
                "--backend",
                "stub",
            ]
        )
        read_replies = read_rows(root / "diagnosis-read" / "diagnoses.jsonl")
        self.assertTrue(all(reply["attempts"] == 2 for reply in read_replies))
        refused = run(
            [
                PYTHON,
                "-B",
                str(COMMON / "diagnose.py"),
                str(root / "read-inputs.jsonl"),
                str(base / "real"),
                "--mode",
                "read",
                "--backend",
                "codex",
            ],
            env={"EXP0_DIAGNOSIS_APPROVED": ""},
            check=False,
        )
        self.assertNotEqual(refused.returncode, 0)
        self.assertFalse((base / "real").exists())
        tool = run(
            [
                PYTHON,
                "-B",
                str(COMMON / "diagnose.py"),
                str(root / "read-inputs.jsonl"),
                str(base / "tool-run"),
                "--mode",
                "read",
                "--backend",
                "stub",
            ],
            env={"STUB_CODEX_MODE": "tool"},
            check=False,
        )
        self.assertNotEqual(tool.returncode, 0)
        self.assertIn("non-diagnostic item", tool.stderr)

        scores_text = run(
            [
                PYTHON,
                "-I",
                "-B",
                str(COMMON / "score.py"),
                str(root / "audit" / "episodes.json"),
                str(root / "diagnosis-read" / "diagnoses.jsonl"),
                str(root / "diagnosis-probe" / "diagnoses.jsonl"),
            ]
        ).stdout
        scores = json.loads(scores_text)
        (base / "scores.json").write_text(scores_text)
        self.assertEqual(scores["read"]["verified_episodes"], 30)

        values = {
            name: f"[test value for {name}]"
            for name in (
                "GAME_NAME",
                "ISSUE",
                "ISSUE_URL",
                "GAME_BUILD_DESCRIPTION",
                "OBSERVATION_CONDITION",
                "OBSERVATION_SUMMARY",
                "EPISODE_DEFINITION",
                "WIN_RULE",
                "LOSS_RULE",
                "CAP_RULE",
                "OTHER_RULE",
                "HARNESS_FAULT_SUMMARY",
                "WEIGHTS_EVIDENCE_SUMMARY",
                "CONTROLLER_FAULT_SUMMARY",
                "KNOWN_LIMITS",
                "WEIGHTS_EVIDENCE_RULE",
                "CONTROLLER_SELECTED",
                "CONTROLLER_EXECUTED",
                "REVISIONS_AND_DISAGREEMENTS",
                "PILOTS_OUTSIDE_SCORED_SET",
                "HELDOUT_NOTE",
                "HARNESS_ID",
                "CONTROLLER_ID",
                "SHAPE",
                "SHAPE_SUMMARY",
                "CLEAN_CHECKPOINT",
                "ALTERNATE_CHECKPOINT",
                "START_BREAKPOINT",
                "END_CONDITIONS",
                "CAPS",
                "SELECTION_RULE",
                "ATTEMPT_CAP",
                "HARNESS_FAULT_RULE",
                "CONTROLLER_FAULT_RULE",
                "REPLAY_OR_TIMING",
                "DIAGNOSIS_MODEL",
                "CLI_OR_API_VERSION",
                "REVISION_AND_COST_AVAILABILITY",
                "ATTEMPT_SUMMARY",
                "PER_CLASS_NARRATIVE",
                "REPO_ROOT",
                "ENVIRONMENT",
                "ADAPTER_SPEC",
                "SERVICES",
                "COLLECT_COMMANDS",
                "CLEAN_URL",
                "ALTERNATE_URL",
                "RELEASE_ROOT",
                "GAME_ID",
                "CARDS_DIR",
                "CODE_REVISION",
                "SOURCE_FLAGS",
                "PINS_EVIDENCE",
                "EPISODE_EVIDENCE",
                "PILOT_EVIDENCE",
                "HARNESS_SMOKE",
                "WEIGHTS_SMOKE",
                "CONTROLLER_SMOKE",
                "USAGE_SUMMARY",
            )
        }
        write_json(base / "card-values.json", values)
        filled = run(
            [
                PYTHON,
                "-B",
                str(COMMON / "fill_cards.py"),
                str(root),
                str(root / "audit"),
                str(base / "cards"),
                "--values",
                str(base / "card-values.json"),
                "--scores",
                str(base / "scores.json"),
            ],
            check=False,
        )
        self.assertEqual(filled.returncode, 0, filled.stdout)

        release_root = base / "release"
        out = release_root / f"synthetic-{shape}" / "v0"
        with self.assertRaises(ValueError):  # unfilled cards are refused
            bundle.release_bundle(
                root,
                root / "audit",
                release_root / "x" / f"synthetic-{shape}" / "v0",
                COMMON / "templates",
                {},
                False,
                "0" * 40,
                {},
                {},
            )
        run(
            [
                PYTHON,
                "-B",
                str(COMMON / "bundle.py"),
                "release",
                str(root),
                str(root / "audit"),
                str(out),
                "--cards",
                str(base / "cards"),
                "--code-revision",
                "1" * 40,
                "--source",
                f"synthetic={TESTS / 'synthetic_game.py'}",
            ]
        )
        verified = json.loads(
            run([self.fresh_python, "-I", "-B", str(out / "tools" / "verify_release.py"), str(out)]).stdout
        )
        self.assertTrue(verified["offline_scores_match"])
        self.assertEqual(verified["episodes"], 40)
        (base / "verify-receipt.json").write_text(
            json.dumps({**verified, "fresh_venv_packages": self.fresh_packages}, indent=2, sort_keys=True) + "\n"
        )

        report = read_json(out / "dataset.json")
        self.assertEqual(sum(len(v) for v in report["extra_files_skipped"].values()), 3)
        self.assertFalse(list(out.rglob("*.jpg")))
        self.assertTrue(list(out.rglob("bridge.log")))
        self._upload_tool(out, base)

        self._tamper(out, "evaluation/read/diagnoses.jsonl", self._flip_cause, rehash=False, expect="checksums")
        self._tamper(
            out, "evaluation/read/diagnoses.jsonl", self._flip_cause, rehash=True, expect="offline scores differ"
        )
        self._tamper(out, "evaluation/read-inputs.jsonl", self._leak_run, rehash=True, expect="evaluator metadata")
        self._tamper(out, "evaluator/episodes.json", self._relabel, rehash=True, expect="")
        if shape == "native":
            self._variant_release(root, base, out)

    def _variant_release(self, root: Path, base: Path, primary: Path) -> None:
        """2026-11-26: a pre-registered probe variant (another frozen prompt) is bundled and verified separately."""
        variant = root / "diagnosis-probe-variant"
        run(
            [
                PYTHON,
                "-B",
                str(COMMON / "diagnose.py"),
                str(root / "read-inputs.jsonl"),
                str(variant),
                "--mode",
                "probe",
                "--probes",
                str(root / "probes" / "probe-inputs.jsonl"),
                "--backend",
                "stub",
                "--prompt",
                str(COMMON / "diagnosis_prompt_probe_variant.txt"),
            ]
        )
        settings = read_json(variant / "settings.json")
        self.assertEqual(settings["prompt_file"], "diagnosis_prompt_probe_variant.txt")
        self.assertNotIn("prompt_file", read_json(root / "diagnosis-probe" / "settings.json"))
        out = base / "release-variant" / primary.parent.name / "v0"
        run(
            [
                PYTHON,
                "-B",
                str(COMMON / "bundle.py"),
                "release",
                str(root),
                str(root / "audit"),
                str(out),
                "--cards",
                str(base / "cards"),
                "--code-revision",
                "1" * 40,
                "--source",
                f"synthetic={TESTS / 'synthetic_game.py'}",
                "--variant",
                f"probe-variant={variant}",
            ]
        )
        verified = json.loads(
            run([self.fresh_python, "-I", "-B", str(out / "tools" / "verify_release.py"), str(out)]).stdout
        )
        self.assertEqual(verified["variant_diagnoses"], {"probe-variant": 40})
        self.assertEqual(
            read_json(out / "evaluation" / "scores.json"), read_json(primary / "evaluation" / "scores.json")
        )
        self.assertTrue((out / "diagnosis_prompt_probe_variant.txt").is_file())
        self._tamper(
            out,
            "evaluation/probe-variant/diagnoses.jsonl",
            self._flip_cause,
            rehash=True,
            expect="variant probe-variant differ",
        )
        wrong = root / "diagnosis-wrong-variant"
        shutil.copytree(variant, wrong)
        changed = read_json(wrong / "settings.json")
        changed["model"] = "another-model"
        (wrong / "settings.json").write_text(json.dumps(changed) + "\n")
        with self.assertRaises(ValueError):  # a variant must keep the primary runs' frozen settings
            bundle.release_bundle(
                root,
                root / "audit",
                base / "release-wrong" / primary.parent.name / "v0",
                base / "cards",
                {},
                False,
                "1" * 40,
                {},
                {},
                variants={"probe-variant": wrong},
            )

    def test_two_question_shape(self) -> None:
        """Contract 0.2.0: every decision asks move + attack in one request; the whole chain to a verified bundle."""
        self._pipeline_keys(OUT / "keys", per_condition=10, groups=14, dry_run=False)

    def test_dry_run_bundle(self) -> None:
        """A dry-run plan (fewer episodes per condition) runs the whole chain, says so, and is never uploadable."""
        out = self._pipeline_keys(OUT / "keys-dry", per_condition=2, groups=3, dry_run=True)
        refused = subprocess.run(
            [PYTHON, "-B", str(COMMON / "upload_release.py"), "check", str(out)],
            capture_output=True,
            text=True,
            env={k: v for k, v in os.environ.items() if k != "HF_TOKEN_FILE"},
        )
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("dry-run bundle", refused.stderr)

    def _pipeline_keys(self, base: Path, *, per_condition: int, groups: int, dry_run: bool) -> Path:
        shutil.rmtree(base, ignore_errors=True)
        root = base / "collection"
        root.mkdir(parents=True)
        adapter_spec = f"{TESTS / 'synthetic_game.py'}:KeysAdapter"
        keys_count = 2
        with FakeLaya("clean") as clean, FakeLaya("alternate") as alternate:
            plan = sg.plan_keys(clean.sha, alternate.sha, groups=groups)
            plan["per_condition"] = per_condition
            plan["initial_groups"] = per_condition
            if dry_run:
                plan["dry_run"] = True
                plan["game_id"], plan["dataset_dir"] = "synthetic-keys-dryrun", "synthetic-keys-dryrun/v0/"
            write_json(root / "collection.json", plan)
            for number, group in enumerate(plan["groups"]):
                for condition in ("no_injection", "harness", "weights", "controller"):
                    url = alternate.url if condition == "weights" else clean.url
                    salt = f"-w{group['id']}" if condition == "weights" and number % 2 == 0 else ""
                    instance = "extend-first-move-v1" if number >= groups - 4 else "drop-first-strike-v1"
                    sg.collect_keys(
                        root, plan, group, condition, 1, url, controller_instance=instance, request_salt=salt
                    )
            reports = validate_episode.validate(root)
            self.assertEqual(
                Counter(r["status"] for r in reports),
                Counter(valid=len(reports)),
                [r for r in reports if r["status"] != "valid"][:1],
            )
            self.assertTrue(probes.selfcheck(root, load_adapter(adapter_spec), None)["ok"])
            audit_set.reask(root, root / "weights-reask.jsonl", clean.url, None)
            self.assertEqual(audit_set.audit(root, root / "audit", root / "weights-reask.jsonl", None), 0)
            index = read_json(root / "audit" / "episodes.json")
            self.assertEqual(len(index), 4 * per_condition)
            attempts = read_rows(root / "audit" / "attempts.jsonl")
            verified = [row for row in attempts if row["annotation_status"] == "verified"]
            self.assertEqual(
                len(verified), 3 * groups, [row for row in attempts if row["annotation_status"] == "excluded"][:2]
            )
            for row in verified:
                if row["condition"] == "weights":
                    self.assertEqual(set(row["evidence"]["clean_probabilities"]), {"move", "attack"})
                if row["condition"] == "harness":
                    self.assertEqual(set(row["evidence"]["change_summary"]["questions"]), {"move", "attack"})
            if not dry_run:
                methods = Counter(row["evidence"]["method"] for row in verified if row["condition"] == "weights")
                self.assertEqual(set(methods), {"matched_clean_identical_request", "same_request_reask"}, methods)
                lengths = [
                    row
                    for row in verified
                    if row["condition"] == "controller"
                    and any("requested_ticks" in item for item in row["evidence"]["first_mismatches"])
                ]
                self.assertEqual(len(lengths), 4)
            diagnosis_inputs.build(root, root / "audit" / "episodes.json", root / "read-inputs.jsonl")
            for item in read_rows(root / "read-inputs.jsonl"):
                self.assertIn("  move: ", item["text"])
                self.assertIn("  attack: ", item["text"])
                self.assertNotIn("first-strike", item["text"])
            adapter = load_adapter(adapter_spec)
            probes.capture(root, index, clean.url, "clean", adapter, root / "probe-reference.jsonl")
            probes.capture(root, index, alternate.url, "alternate", adapter, root / "probe-alternate.jsonl")
            self.assertTrue(
                probes.compare(
                    root,
                    index,
                    root / "probe-reference.jsonl",
                    root / "probe-alternate.jsonl",
                    adapter,
                    root / "probes",
                )
            )
            identity = read_rows(root / "probes" / "identity.jsonl")
            self.assertTrue(
                all(item["questions"] == keys_count * item["decisions"] for item in identity), identity[:1]
            )
        for mode, extra in (("read", []), ("probe", ["--probes", str(root / "probes" / "probe-inputs.jsonl")])):
            run(
                [
                    PYTHON,
                    "-B",
                    str(COMMON / "diagnose.py"),
                    str(root / "read-inputs.jsonl"),
                    str(root / f"diagnosis-{mode}"),
                    "--mode",
                    mode,
                    *extra,
                    "--backend",
                    "stub",
                ]
            )
        scores_text = run(
            [
                PYTHON,
                "-I",
                "-B",
                str(COMMON / "score.py"),
                str(root / "audit" / "episodes.json"),
                str(root / "diagnosis-read" / "diagnoses.jsonl"),
                str(root / "diagnosis-probe" / "diagnoses.jsonl"),
            ]
        ).stdout
        (base / "scores.json").write_text(scores_text)
        self.assertEqual(json.loads(scores_text)["read"]["verified_episodes"], 3 * per_condition)
        values = {
            name: f"[test value for {name}]"
            for name in (
                "GAME_NAME",
                "ISSUE",
                "ISSUE_URL",
                "GAME_BUILD_DESCRIPTION",
                "OBSERVATION_CONDITION",
                "OBSERVATION_SUMMARY",
                "EPISODE_DEFINITION",
                "WIN_RULE",
                "LOSS_RULE",
                "CAP_RULE",
                "OTHER_RULE",
                "HARNESS_FAULT_SUMMARY",
                "WEIGHTS_EVIDENCE_SUMMARY",
                "CONTROLLER_FAULT_SUMMARY",
                "KNOWN_LIMITS",
                "WEIGHTS_EVIDENCE_RULE",
                "CONTROLLER_SELECTED",
                "CONTROLLER_EXECUTED",
                "REVISIONS_AND_DISAGREEMENTS",
                "PILOTS_OUTSIDE_SCORED_SET",
                "HELDOUT_NOTE",
                "HARNESS_ID",
                "CONTROLLER_ID",
                "SHAPE",
                "SHAPE_SUMMARY",
                "CLEAN_CHECKPOINT",
                "ALTERNATE_CHECKPOINT",
                "START_BREAKPOINT",
                "END_CONDITIONS",
                "CAPS",
                "SELECTION_RULE",
                "ATTEMPT_CAP",
                "HARNESS_FAULT_RULE",
                "CONTROLLER_FAULT_RULE",
                "REPLAY_OR_TIMING",
                "DIAGNOSIS_MODEL",
                "CLI_OR_API_VERSION",
                "REVISION_AND_COST_AVAILABILITY",
                "ATTEMPT_SUMMARY",
                "PER_CLASS_NARRATIVE",
                "REPO_ROOT",
                "ENVIRONMENT",
                "ADAPTER_SPEC",
                "SERVICES",
                "COLLECT_COMMANDS",
                "CLEAN_URL",
                "ALTERNATE_URL",
                "RELEASE_ROOT",
                "GAME_ID",
                "CARDS_DIR",
                "CODE_REVISION",
                "SOURCE_FLAGS",
                "PINS_EVIDENCE",
                "EPISODE_EVIDENCE",
                "PILOT_EVIDENCE",
                "HARNESS_SMOKE",
                "WEIGHTS_SMOKE",
                "CONTROLLER_SMOKE",
                "USAGE_SUMMARY",
            )
        }
        write_json(base / "card-values.json", values)
        filled = run(
            [
                PYTHON,
                "-B",
                str(COMMON / "fill_cards.py"),
                str(root),
                str(root / "audit"),
                str(base / "cards"),
                "--values",
                str(base / "card-values.json"),
                "--scores",
                str(base / "scores.json"),
            ],
            check=False,
        )
        self.assertEqual(filled.returncode, 0, filled.stdout)
        out = base / "release" / plan["dataset_dir"].rstrip("/")
        run(
            [
                PYTHON,
                "-B",
                str(COMMON / "bundle.py"),
                "release",
                str(root),
                str(root / "audit"),
                str(out),
                "--cards",
                str(base / "cards"),
                "--code-revision",
                "2" * 40,
                "--source",
                f"synthetic={TESTS / 'synthetic_game.py'}",
            ]
        )
        verified_release = json.loads(
            run([self.fresh_python, "-I", "-B", str(out / "tools" / "verify_release.py"), str(out)]).stdout
        )
        self.assertTrue(verified_release["offline_scores_match"])
        self.assertEqual(verified_release["episodes"], 4 * per_condition)
        self.assertEqual(verified_release["dry_run"], dry_run)
        self.assertEqual(verified_release["questions_per_decision"], 2)
        (base / "verify-receipt.json").write_text(json.dumps(verified_release, indent=2, sort_keys=True) + "\n")
        if dry_run:
            undeclared = base / "undeclared"
            shutil.copytree(root, undeclared / "collection")
            plan_copy = read_json(undeclared / "collection" / "collection.json")
            plan_copy.pop("dry_run")
            write_json(undeclared / "collection" / "collection.json", plan_copy)
            with self.assertRaises(ValueError):  # fewer than 10 per condition without the dry-run mark: refused
                bundle.bundle(
                    undeclared / "collection", root / "audit", undeclared / plan["dataset_dir"].rstrip("/"), {}, False
                )
        else:
            self._tamper(
                out, "evaluation/read/diagnoses.jsonl", self._flip_cause, rehash=True, expect="offline scores differ"
            )
            self._tamper(out, "evaluator/episodes.json", self._relabel, rehash=True, expect="")
        return out

    def _upload_tool(self, out: Path, base: Path) -> None:
        """check / plan / dry-run upload need no token and no network; executing commands are guarded."""
        environment = {key: value for key, value in os.environ.items() if key != "HF_TOKEN_FILE"}
        tool = str(COMMON / "upload_release.py")
        probe = (
            "import sys, json; sys.path.insert(0, %r); import upload_release as u; "
            "from pathlib import Path; r = u.plan(Path(%r), 'honinglab/exp0-data', 'abc', Path(%r)); "
            "print(json.dumps({'hf_imported': 'huggingface_hub' in sys.modules, 'files': r['files']}))"
        )
        result = subprocess.run(
            [PYTHON, "-B", "-c", probe % (str(COMMON), str(out), str(base / "plan-work"))],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        self.assertFalse(json.loads(result.stdout)["hf_imported"])
        dry = subprocess.run(
            [PYTHON, "-B", tool, "upload", str(out), "--parent-revision", "abc"],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        self.assertTrue(json.loads(dry.stdout)["dry_run"])
        guarded = subprocess.run(
            [
                PYTHON,
                "-B",
                tool,
                "download-verify",
                "--game-dir",
                "x/v0/",
                "--revision",
                "abc",
                "--out",
                str(base / "dl"),
            ],
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertNotEqual(guarded.returncode, 0)
        self.assertFalse((base / "dl").exists())
        import upload_release

        self.assertEqual(upload_release._sanitize("401 for token SECRETVALUE", "SECRETVALUE"), "401 for token <token>")
        polluted = base / "polluted" / out.parent.name / "v0"
        shutil.copytree(out, polluted)
        (polluted / "evaluation" / "screenshot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(32))
        refused = subprocess.run(
            [PYTHON, "-B", tool, "check", str(polluted)], capture_output=True, text=True, env=environment
        )
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("content policy", refused.stderr)

    @staticmethod
    def _flip_cause(path: Path) -> None:
        rows = read_rows(path)
        rows[0]["cause"] = "harness" if rows[0].get("cause") != "harness" else "weights"
        rows[0].pop("status", None)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    @staticmethod
    def _leak_run(path: Path) -> None:
        rows = read_rows(path)
        index = json.loads((path.parents[1] / "evaluator" / "episodes.json").read_text())
        rows[0]["text"] += "\n" + index[0]["run"]
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    @staticmethod
    def _relabel(path: Path) -> None:
        index = json.loads(path.read_text())
        harness = next(row for row in index if row["condition"] == "harness")
        harness["verified_component"] = "controller"
        path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

    def _tamper(self, out: Path, relative: str, change, *, rehash: bool, expect: str) -> None:
        copy_root = out.parents[2] / "tampered" / relative.replace("/", "_") / ("rehashed" if rehash else "raw")
        target = copy_root / out.parent.name / "v0"
        shutil.rmtree(copy_root, ignore_errors=True)
        shutil.copytree(out, target)
        change(target / relative)
        if rehash:
            write_json(target / "sha256.json", bundle.file_hashes(target))
        result = run(
            [self.fresh_python, "-I", "-B", str(target / "tools" / "verify_release.py"), str(target)], check=False
        )
        self.assertNotEqual(result.returncode, 0, f"tampered {relative} passed verification")
        self.assertIn(expect, result.stderr)


def teardown_module() -> None:
    """Leave no bare-name modules of the tool directory behind for later test modules."""
    for name, module in list(sys.modules.items()):
        if str(COMMON) in str(getattr(module, "__file__", "") or ""):
            del sys.modules[name]
    for entry in (str(TESTS), str(COMMON)):
        while entry in sys.path:
            sys.path.remove(entry)


if __name__ == "__main__":
    unittest.main()
