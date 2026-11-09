"""Lockstep native Spiders episodes with exact model inputs and executed actions."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from honing.backends.griddly.catalog import FORMAT
from honing.backends.griddly.harness import load_harness, parse_harness, seed_harness, write_harness
from honing.backends.griddly.model import DecisionClient, baseline, make_request, validate_reply
from honing.backends.griddly.scenarios import GriddlyScenario
from honing.backends.griddly.source import adapter_sha256, source_files
from honing.core import CampaignRuleError, HarnessSpec
from integrations.griddly.catalog import ACTIONS, GAME, GAME_SHA256, PROTOCOL
from integrations.griddly.engine import GriddlyEngine, canonical, runtime_versions


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def history_entry(action: str, frame: dict[str, Any]) -> dict[str, Any]:
    return {"action": action, "reward": frame["reward"], "status": frame["status"]}


def summary(frame: dict[str, Any], reward: float) -> dict[str, Any]:
    return {
        "status": frame["status"],
        "success": frame["status"] == "won",
        "steps": frame["step"],
        "reward": reward,
        "final_frame_hash": frame["frame_hash"],
    }


def run_match(
    scenario: GriddlyScenario,
    out: Path,
    spec: HarnessSpec,
    *,
    policy: str = "random",
    policy_seed: int = 1,
    decision_server: str | None = None,
    policy_checksum: str | None = None,
    actions: tuple[str, ...] = (),
    sample: bool = False,
    record_video: bool = False,
) -> dict[str, Any]:
    if policy not in ("random", "baseline", "teacher", "actions", "model"):
        raise ValueError("unknown policy")
    if (policy == "model") != (decision_server is not None) or (sample and policy != "model"):
        raise ValueError("only model play takes a server and can sample")
    if type(policy_seed) is not int or not 0 <= policy_seed < 2**32:
        raise ValueError("policy_seed must be uint32")
    if policy == "actions" and (len(actions) < scenario.max_steps or any(a not in ACTIONS for a in actions)):
        raise ValueError("prescribed actions must cover the budget and use native names")
    if policy != "actions" and actions:
        raise ValueError("prescribed actions require policy=actions")
    out.mkdir(parents=True, exist_ok=False)
    writer: Any = None
    started = time.perf_counter()
    try:
        harness = parse_harness(spec)
        write_harness(spec, out / "harness")
        source = out / "source"
        source.mkdir()
        for name, path in source_files().items():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        client = DecisionClient(decision_server) if decision_server is not None else None
        manifest = {
            "format": FORMAT,
            "adapter_sha256": adapter_sha256(),
            "protocol": PROTOCOL,
            "game": GAME,
            "game_sha256": GAME_SHA256,
            "scenario": scenario.record(),
            "runtime": runtime_versions(),
            "controller": policy,
            "kind": (
                "rollout"
                if sample
                else "evaluation" if client else "teacher-demonstration" if policy == "teacher" else "scripted"
            ),
            "sample": sample,
            "policy_seed": policy_seed,
            "policy_checksum": policy_checksum,
            "model": client.health() if client else None,
            "spec_id": spec.spec_id,
            "harness": list(harness.files),
            "actions": ACTIONS,
            "record_video": record_video,
            "clock": "one native step per action; inference does not advance game time",
        }
        write_json(out / "manifest.json", manifest)
        with (
            GriddlyEngine(scenario, render=record_video) as engine,
            (out / "steps.jsonl").open("x", encoding="utf-8") as steps,
            (out / "decisions.jsonl").open("x", encoding="utf-8") as decisions,
        ):

            def record(stream: Any, row: dict[str, Any]) -> None:
                stream.write(canonical(row) + "\n")
                stream.flush()

            frame = engine.reset()
            record(steps, {"event": "reset", "frame": frame})
            if record_video:
                imageio = __import__("imageio.v2", fromlist=["get_writer"])
                writer = imageio.get_writer(out / "replay.mp4", fps=10, codec="libx264", macro_block_size=1)
                writer.append_data(engine.pixels())
            rng, total_reward = random.Random(policy_seed), 0.0
            history: list[dict[str, Any]] = []
            while frame["status"] == "running":
                before, request, reply = frame, None, None
                forced = len(frame["legal_actions"]) == 1
                tick = time.perf_counter()
                if client and not forced:
                    request = make_request(frame, scenario, harness, history)
                    reply = client.decide(request)
                    probabilities = validate_reply(request, reply)
                    action = (
                        rng.choices(list(probabilities), weights=list(probabilities.values()))[0]
                        if sample
                        else reply["answers"]["action"]["choice"]
                    )
                elif policy == "actions":
                    action = actions[frame["step"]]
                elif forced:
                    action = frame["legal_actions"][0]
                elif policy in ("baseline", "teacher"):
                    action = baseline(frame)
                    if policy == "teacher":
                        request = make_request(frame, scenario, harness, history)
                else:
                    action = rng.choice(frame["legal_actions"])
                latency_ms = (time.perf_counter() - tick) * 1000
                frame = engine.step(action)
                total_reward += frame["reward"]
                record(
                    decisions,
                    {
                        "step": before["step"],
                        "before_frame_hash": before["frame_hash"],
                        "request": request,
                        "response": reply,
                        "chosen": {"action": action},
                        "applied": action,
                        "forced_action": forced,
                        "decision_ms": latency_ms,
                    },
                )
                record(steps, {"event": "step", "action": action, "frame": frame})
                history.append(history_entry(action, frame))
                if writer is not None:
                    writer.append_data(engine.pixels())
            result = {**summary(frame, total_reward), "elapsed_seconds": time.perf_counter() - started}
            if writer is not None:
                writer.close()
                writer = None
            write_json(out / "result.json", result)
            return result
    except (Exception, KeyboardInterrupt) as error:
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
        if writer is not None:
            writer.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--policy", choices=("random", "baseline", "teacher", "actions", "model"), default="random")
    parser.add_argument("--policy-seed", type=int, default=1)
    parser.add_argument("--actions", type=Path)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--decision-server")
    parser.add_argument("--policy-checksum")
    parser.add_argument("--harness-dir", type=Path)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    integration = Path(__file__).resolve().parents[3] / "integrations/griddly"
    actions = json.loads(args.actions.read_text(encoding="utf-8")) if args.actions else []
    if not isinstance(actions, list) or any(not isinstance(a, str) for a in actions):
        parser.error("actions must be a JSON list of native names")
    spec = load_harness(args.harness_dir) if args.harness_dir else seed_harness(integration)
    out = args.out or integration / "runs" / f"spiders-{args.policy}-{uuid.uuid4().hex[:12]}"
    result = run_match(
        GriddlyScenario(args.seed, args.level, args.max_steps),
        out,
        spec,
        policy=args.policy,
        policy_seed=args.policy_seed,
        actions=tuple(actions),
        sample=args.sample,
        decision_server=args.decision_server,
        policy_checksum=args.policy_checksum,
        record_video=args.record_video,
    )
    print(json.dumps({"out": str(out), **result}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
