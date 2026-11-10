"""Run the real CPU simulation with a rule policy or typed model decisions."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, TextIO

from honing.backends.rea_issac.catalog import OPTIONS, PROTOCOL
from honing.backends.rea_issac.harness import load_harness, parse_harness, seed_harness
from honing.backends.rea_issac.model import DecisionClient, make_request
from honing.backends.rea_issac.scenarios import ReaIssacScenario
from honing.backends.rea_issac.source import runtime_fingerprint as runtime_fingerprint
from honing.core import CampaignRuleError, HarnessSpec
from integrations.rea_issac import engine as game_engine
from integrations.rea_issac.engine import VERSION, Engine
from integrations.rea_issac.policy import Baseline

FORMAT = "rea-issac-match-v1"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def record(stream: TextIO, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    stream.flush()


def history_entry(action: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": action,
        "tick": observation["tick"],
        "health": observation["player"]["health"],
        "room": observation["room"]["id"],
        "status": observation["status"],
        "metrics": observation["metrics"],
    }


def replay_record(engine: Engine, actions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "engine_version": VERSION,
        "seed": engine.seed,
        "max_ticks": engine.max_ticks,
        "actions": actions,
        "final_hash": engine.digest(),
        "result": engine.result(),
    }


def save_harness(out: Path, spec: HarnessSpec) -> None:
    directory = out / "harness"
    directory.mkdir()
    for path, file in spec.files:
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file.content, encoding="utf-8")
    write_json(directory / "index.json", [{"path": p, "kind": f.kind} for p, f in spec.files])


def run_match(
    scenario: ReaIssacScenario,
    out: Path,
    spec: HarnessSpec,
    *,
    policy: str = "baseline",
    decision_server: str | None = None,
    policy_checksum: str | None = None,
) -> dict[str, Any]:
    """Inference pauses the game clock; model errors stop without a scripted substitute."""
    if policy not in ("baseline", "model") or (policy == "model") != (decision_server is not None):
        raise ValueError("policy must be baseline, or model with a decision server")
    out.mkdir(parents=True, exist_ok=False)
    engine: Engine | None = None
    actions: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        harness = parse_harness(spec)
        save_harness(out, spec)
        client = DecisionClient(decision_server) if decision_server else None
        health = client.health() if client else None
        manifest = {
            "format": FORMAT,
            "protocol": PROTOCOL,
            "engine_version": VERSION,
            "runtime": runtime_fingerprint(),
            "scenario": scenario.record(),
            "kind": "evaluation" if client else "scripted",
            "controller": policy,
            "actions": {k: list(v) for k, v in OPTIONS.items()},
            "harness": list(harness.files),
            "spec_id": spec.spec_id,
            "policy_checksum": policy_checksum,
            "model": health,
            "clock": "30 ticks/second; explicit step only; no advancement during inference",
        }
        write_json(out / "manifest.json", manifest)
        engine = Engine(scenario.seed, scenario.max_ticks)
        observation = engine.observe()
        baseline, inference_ms = Baseline(), 0.0
        history: list[dict[str, Any]] = []
        with (out / "steps.jsonl").open("x") as steps, (out / "decisions.jsonl").open("x") as decisions:
            record(steps, {"event": "reset", "observation": observation, "state_hash": engine.digest()})
            while engine.status == "running":
                before_hash, before_tick = engine.digest(), engine.tick
                request, response = None, None
                frames = min(scenario.decision_frames, scenario.max_ticks - engine.tick)
                tick = time.perf_counter()
                if client:
                    request = make_request(observation, scenario, harness, history)
                    action, response = client.decide(request)
                else:
                    action = baseline.act(observation)
                latency_ms = (time.perf_counter() - tick) * 1000
                inference_ms += latency_ms if client else 0
                observation = engine.step(action, frames)
                entry = {"action": action, "frames": frames}
                actions.append(entry)
                record(
                    decisions,
                    {
                        "tick": before_tick,
                        "before_state_hash": before_hash,
                        **entry,
                        "request": request,
                        "response": response,
                        "decision_ms": latency_ms,
                    },
                )
                record(steps, {"event": "step", **entry, "observation": observation, "state_hash": engine.digest()})
                history.append(history_entry(action, observation))
                if engine.tick // 300 != before_tick // 300 or engine.status != "running":
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "tick": engine.tick,
                                "status": engine.status,
                                "decisions": len(actions),
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
            result = {
                **engine.result(),
                "decisions": len(actions),
                "inference_calls": len(actions) if client else 0,
                "inference_ms": inference_ms,
                "elapsed_seconds": time.perf_counter() - started,
            }
            write_json(out / "result.json", result)
            return result
    except (Exception, KeyboardInterrupt) as error:
        if engine is not None:
            # A signal can arrive inside a step. Preserve the last completed
            # action sequence rather than export an unreplayable partial tick.
            engine = Engine(scenario.seed, scenario.max_ticks)
            for completed_action in actions:
                engine.step(completed_action["action"], completed_action["frames"])
        write_json(
            out / "error.json",
            {
                "kind": "harness" if isinstance(error, CampaignRuleError) else "match",
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    finally:
        if engine is not None:
            write_json(out / "replay.json", replay_record(engine, actions))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-ticks", type=int, default=6000)
    parser.add_argument("--decision-frames", type=int, default=3)
    parser.add_argument("--policy", choices=("baseline", "model"), default="baseline")
    parser.add_argument("--decision-server")
    parser.add_argument("--policy-checksum")
    parser.add_argument("--harness-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    integration = Path(game_engine.__file__).resolve().parent
    spec = load_harness(args.harness_dir) if args.harness_dir else seed_harness(integration)
    scenario = ReaIssacScenario(args.seed, args.max_ticks, args.decision_frames)
    out = args.out or Path.cwd() / "runs" / "rea_issac" / f"s{args.seed}-{args.policy}-{uuid.uuid4().hex[:12]}"
    result = run_match(
        scenario,
        out,
        spec,
        policy=args.policy,
        decision_server=args.decision_server,
        policy_checksum=args.policy_checksum,
    )
    print(json.dumps({"event": "saved", "out": str(out), **result}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
