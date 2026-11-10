"""Re-simulate a recorded game and bind its model decisions to public states."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from honing.backends.rea_issac.catalog import OPTIONS, PROTOCOL
from honing.backends.rea_issac.harness import load_harness, parse_harness
from honing.backends.rea_issac.match import FORMAT, history_entry, replay_record, runtime_fingerprint
from honing.backends.rea_issac.model import make_request, validate_reply
from honing.backends.rea_issac.scenarios import ReaIssacScenario
from integrations.rea_issac.engine import VERSION, Engine
from integrations.rea_issac.policy import Baseline


def audit_run(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Verify the real engine, saved harness, complete model context and executed actions."""
    manifest = json.loads((run / "manifest.json").read_text())
    if (
        manifest.get("format") != FORMAT
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("engine_version") != VERSION
        or manifest.get("runtime") != runtime_fingerprint()
        or manifest.get("actions") != {k: list(v) for k, v in OPTIONS.items()}
        or manifest.get("controller") not in ("model", "baseline")
    ):
        raise ValueError("unsupported REA-Issac recording or changed engine source")
    spec = load_harness(run / "harness")
    harness = parse_harness(spec)
    if spec.spec_id != manifest["spec_id"] or list(harness.files) != manifest["harness"]:
        raise ValueError("saved harness differs from the manifest")
    scenario = ReaIssacScenario(**manifest["scenario"])
    rows = [json.loads(s) for s in (run / "steps.jsonl").read_text().splitlines() if s.strip()]
    decisions = [json.loads(s) for s in (run / "decisions.jsonl").read_text().splitlines() if s.strip()]
    if not rows or len(rows) != len(decisions) + 1 or len(decisions) > scenario.max_ticks:
        raise ValueError("incomplete trajectory")
    engine = Engine(scenario.seed, scenario.max_ticks)
    observation = engine.observe()
    if rows[0] != {"event": "reset", "observation": observation, "state_hash": engine.digest()}:
        raise ValueError("initial state differs from the engine")
    baseline, inference_ms = Baseline(), 0.0
    history: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for row, decision in zip(rows[1:], decisions, strict=True):
        frames = min(scenario.decision_frames, scenario.max_ticks - engine.tick)
        if (
            engine.status != "running"
            or decision["tick"] != engine.tick
            or decision["before_state_hash"] != engine.digest()
            or decision["frames"] != frames
        ):
            raise ValueError("decision does not match the game clock or state")
        latency = decision["decision_ms"]
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError("invalid decision latency")
        if manifest["controller"] == "model":
            request = make_request(observation, scenario, harness, history)
            if request != decision["request"]:
                raise ValueError("model context differs from the public game state and harness")
            action = validate_reply(request, decision["response"])
            inference_ms += latency
        else:
            if decision["request"] is not None or decision["response"] is not None:
                raise ValueError("baseline recording contains model calls")
            action = baseline.act(observation)
        if action != decision["action"]:
            raise ValueError("executed action differs from the controller")
        entry = {"action": action, "frames": frames}
        actions.append(entry)
        observation = engine.step(action, frames)
        if row != {"event": "step", **entry, "observation": observation, "state_hash": engine.digest()}:
            raise ValueError(f"replayed state differs at tick {engine.tick}")
        history.append(history_entry(action, observation))
    result = json.loads((run / "result.json").read_text())
    expected = {
        **engine.result(),
        "decisions": len(actions),
        "inference_calls": len(actions) if manifest["controller"] == "model" else 0,
        "inference_ms": inference_ms,
    }
    if engine.status == "running" or any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("result differs from the completed game")
    if json.loads((run / "replay.json").read_text()) != replay_record(engine, actions):
        raise ValueError("UI replay differs from the recorded game")
    return manifest, rows, result


def verify_replay(run: Path) -> dict[str, Any]:
    _manifest, rows, result = audit_run(run)
    return {"verified": True, "decisions": len(rows) - 1, "tick": result["tick"], "final_hash": result["final_hash"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_replay(args.run), sort_keys=True))


if __name__ == "__main__":
    main()
