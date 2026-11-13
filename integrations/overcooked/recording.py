"""Exact native state/action recordings and re-execution audits."""

from __future__ import annotations

import json
import random
import shutil
import uuid
from pathlib import Path
from typing import Any

from integrations.overcooked.decisions import make_request, validate_reply
from integrations.overcooked.engine import ROOT, Engine, Scenario, canonical, digest, write_json
from integrations.overcooked.policy import baseline

FORMAT = "overcooked-match-v2"
GRADING = "native-score-softsign-v1"


def source_identity() -> str:
    return digest(
        {
            name: (ROOT / name).read_text(encoding="utf-8")
            for name in (
                "engine.py",
                "policy.py",
                "decisions.py",
                "recording.py",
                "harness.py",
                "pins.json",
                "match.py",
                "adapter.py",
                "scenarios.py",
            )
        }
    )


def new_run(prefix: str = "match") -> Path:
    return ROOT / "runs" / f"{prefix}-{uuid.uuid4().hex[:12]}"


def result_for(frame: dict[str, Any], scenario: Scenario) -> dict[str, Any]:
    return {
        "steps": frame["step"],
        "score": frame["score"],
        "deliveries": frame["deliveries"],
        "success": frame["deliveries"] >= scenario.target_deliveries,
        "grading": GRADING,
        "grade": frame["score"] / (frame["score"] + scenario.score_scale),
        "complete": frame["status"] != "running",
        "final_frame_hash": frame["frame_hash"],
    }


class Recorder:
    def __init__(self, engine: Engine, out: Path, metadata: dict[str, Any]) -> None:
        self.out, self.engine = out.resolve(), engine
        self.out.mkdir(parents=True, exist_ok=False)
        self.manifest = {
            "format": FORMAT,
            "scenario": engine.scenario.record(),
            "grading": GRADING,
            "runtime": engine.runtime,
            "source_identity": source_identity(),
            "clock": "one tick per joint action; no simulation during inference",
            **metadata,
        }
        write_json(self.out / "manifest.json", self.manifest)
        self.steps = (self.out / "steps.jsonl").open("x", encoding="utf-8")
        self.decisions = (self.out / "decisions.jsonl").open("x", encoding="utf-8")
        self._write(self.steps, {"actions": None, "frame": engine.observe()})
        self.closed = False

    @staticmethod
    def _write(stream: Any, value: object) -> None:
        stream.write(canonical(value) + "\n")
        stream.flush()

    def append(
        self, before: dict[str, Any], actions: list[str], frame: dict[str, Any], decision: dict[str, Any]
    ) -> None:
        if self.closed:
            raise RuntimeError("recording is closed")
        self._write(self.decisions, {"step": before["step"], "before_frame_hash": before["frame_hash"], **decision})
        self._write(self.steps, {"actions": actions, "frame": frame})

    def finish(self) -> dict[str, Any]:
        result = result_for(self.engine.observe(), self.engine.scenario)
        if not self.closed:
            self.steps.close()
            self.decisions.close()
            self.closed = True
        write_json(self.out / "result.json", result)
        return result

    def snapshot(self) -> Path:
        destination = new_run("export")
        destination.mkdir(parents=True)
        for name in ("manifest.json", "steps.jsonl", "decisions.jsonl"):
            shutil.copyfile(self.out / name, destination / name)
        if (self.out / "harness").exists():
            shutil.copytree(self.out / "harness", destination / "harness")
        write_json(destination / "result.json", result_for(self.engine.observe(), self.engine.scenario))
        return destination

    def abort(self, error: Exception) -> None:
        self.steps.close()
        self.decisions.close()
        self.closed = True
        write_json(self.out / "error.json", {"error": str(error), "kind": type(error).__name__})


def replay(run: Path) -> dict[str, Any]:
    """Re-run every recorded joint action, audit model inputs and reproduce every frame."""
    run = run.resolve()
    if (run / "error.json").exists():
        raise ValueError("failed runs cannot pass replay verification")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["format"] != FORMAT
        or manifest.get("grading") != GRADING
        or manifest["source_identity"] != source_identity()
    ):
        raise ValueError("recording format or integration source differs")
    scenario = Scenario(**manifest["scenario"])
    rows = [json.loads(line) for line in (run / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    decisions = [json.loads(line) for line in (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(decisions) + 1 or not rows or rows[0]["actions"] is not None:
        raise ValueError("recording has missing or extra steps/decisions")
    if len(decisions) > scenario.max_steps:
        raise ValueError("recording exceeds task budget")
    controller = manifest["controller"]
    if controller not in {"model", "teacher", "baseline", "human", "actions"}:
        raise ValueError("unknown recorded controller")
    history: list[dict[str, Any]] = []
    rng = random.Random(manifest.get("policy_seed", 0))
    if controller in {"model", "teacher"}:
        from integrations.overcooked.harness import load_harness, parse_harness

        harness = load_harness(run / "harness")
        if harness.spec_id != manifest["spec_id"]:
            raise ValueError("recorded harness identity differs")
        settings = parse_harness(harness)
    frames = []
    with Engine(scenario) as engine:
        if engine.runtime != manifest["runtime"]:
            raise ValueError("native engine or dependency versions differ")
        frame = engine.observe()

        def check(actual: dict[str, Any], recorded: dict[str, Any]) -> None:
            if actual != recorded:
                raise ValueError(f"native replay differs at step {actual['step']}")
            frames.append(actual)

        check(frame, rows[0]["frame"])
        for index, (row, decision) in enumerate(zip(rows[1:], decisions, strict=True)):
            if decision["step"] != index or decision["before_frame_hash"] != frame["frame_hash"]:
                raise ValueError("decision does not match its pre-action state")
            actions = row["actions"]
            p = scenario.controlled_player
            if controller != "actions" and actions[1 - p] != baseline(frame, 1 - p, scenario.partner):
                raise ValueError("recorded partner action differs from its public-state policy")
            if controller in {"baseline", "teacher"} and actions[p] != baseline(frame, p):
                raise ValueError("recorded baseline action differs")
            if controller in {"model", "teacher"}:
                recent = history[-settings["history_steps"] :] if settings["history_steps"] else []
                request = make_request(frame, p, settings["instructions"], recent, settings["notes"])
                if request != decision["request"]:
                    raise ValueError("recorded model input differs from the actual state/rules/harness")
                if controller == "teacher":
                    if decision["response"] is not None:
                        raise ValueError("rule-teacher runs cannot contain model replies")
                    chosen = baseline(frame, p)
                else:
                    probs = validate_reply(request, decision["response"])
                    chosen = (
                        rng.choices(list(probs), weights=list(probs.values()))[0]
                        if manifest["sample"]
                        else decision["response"]["answers"]["action"]["choice"]
                    )
                if chosen != actions[p] or decision["chosen"] != chosen:
                    raise ValueError("executed action does not match the acting model distribution")
            elif decision.get("request") is not None or decision.get("response") is not None:
                raise ValueError("scripted/human runs cannot contain model decisions")
            frame = engine.step(actions)
            check(frame, row["frame"])
            history.append({"actions": actions, "reward": frame["reward"]})
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        if result != result_for(frame, scenario):
            raise ValueError("result does not match native replay")
    return {"verified": True, "manifest": manifest, "result": result, "frames": frames}
