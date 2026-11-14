"""Record native baseline or model episodes using the same engine as the GUI."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from integrations.overcooked.decisions import DecisionClient, make_request, validate_reply
from integrations.overcooked.engine import Engine, Scenario
from integrations.overcooked.harness import dump_harness, load_harness, parse_harness, seed_harness
from integrations.overcooked.policy import joint_baseline
from integrations.overcooked.recording import Recorder, new_run, replay


def run_match(
    scenario: Scenario,
    out: Path,
    *,
    policy: str = "baseline",
    decision_server: str | None = None,
    policy_checksum: str | None = None,
    spec: Any = None,
    sample: bool = False,
    policy_seed: int = 0,
    record_images: bool = True,
) -> dict[str, Any]:
    if policy not in {"baseline", "teacher", "model"} or (policy == "model") != (decision_server is not None):
        raise ValueError("choose baseline, or model with a decision endpoint")
    if sample and policy != "model":
        raise ValueError("only model runs can sample")
    if type(policy_seed) is not int or not 0 <= policy_seed < 2**32:
        raise ValueError("policy_seed must be uint32")
    spec = seed_harness() if spec is None else spec
    settings = parse_harness(spec)
    client = DecisionClient(decision_server) if decision_server else None
    health = client.health() if client else None
    with Engine(scenario) as engine:
        recorder = Recorder(
            engine,
            out,
            {
                "controller": policy,
                "kind": (
                    "rollout"
                    if sample
                    else "evaluation" if client else "teacher-demonstration" if policy == "teacher" else "scripted"
                ),
                "model": health,
                "policy_checksum": policy_checksum,
                "spec_id": spec.spec_id,
                "sample": sample,
                "policy_seed": policy_seed,
            },
        )
        try:
            dump_harness(spec, out / "harness")
            frame = engine.observe()
            if record_images:
                (out / "first-frame.png").write_bytes(engine.png())
            history: list[dict[str, Any]] = []
            rng = random.Random(policy_seed)
            while frame["status"] == "running":
                before = frame
                actions = joint_baseline(frame, scenario.controlled_player, scenario.partner)
                request, response = None, None
                start = time.perf_counter()
                if client:
                    recent = history[-settings["history_steps"] :] if settings["history_steps"] else []
                    request = make_request(
                        frame, scenario.controlled_player, settings["instructions"], recent, settings["notes"]
                    )
                    response = client.decide(request)
                    probs = validate_reply(request, response)
                    actions[scenario.controlled_player] = (
                        rng.choices(list(probs), weights=list(probs.values()))[0]
                        if sample
                        else response["answers"]["action"]["choice"]
                    )
                elif policy == "teacher":
                    recent = history[-settings["history_steps"] :] if settings["history_steps"] else []
                    request = make_request(
                        frame, scenario.controlled_player, settings["instructions"], recent, settings["notes"]
                    )
                latency = (time.perf_counter() - start) * 1000
                frame = engine.step(actions)
                recorder.append(
                    before,
                    actions,
                    frame,
                    {
                        "request": request,
                        "response": response,
                        "chosen": actions[scenario.controlled_player],
                        "latency_ms": latency,
                    },
                )
                history.append({"actions": actions, "reward": frame["reward"]})
            if record_images:
                (out / "last-frame.png").write_bytes(engine.png())
            return {"out": str(out.resolve()), **recorder.finish()}
        except Exception as error:
            recorder.abort(error)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("baseline", "teacher", "model"), default="baseline")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--old-dynamics", action="store_true")
    parser.add_argument("--controlled-player", type=int, default=0)
    parser.add_argument("--partner", choices=("role", "solo", "stay", "solo_slow"), default="solo")
    parser.add_argument("--target-deliveries", type=int, default=2)
    parser.add_argument("--score-scale", type=float, default=40.0)
    parser.add_argument("--decision-server")
    parser.add_argument("--policy-checksum")
    parser.add_argument("--harness-dir", type=Path)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    scenario = Scenario(
        args.layout,
        args.max_steps,
        args.old_dynamics,
        args.controlled_player,
        args.partner,
        args.target_deliveries,
        args.score_scale,
    )
    out = (args.out or new_run()).resolve()
    result = run_match(
        scenario,
        out,
        policy=args.policy,
        decision_server=args.decision_server,
        policy_checksum=args.policy_checksum,
        spec=load_harness(args.harness_dir) if args.harness_dir else None,
        sample=args.sample,
        policy_seed=args.policy_seed,
        record_images=not args.no_images,
    )
    if args.verify:
        result["verified"] = replay(out)["verified"]
    print(json.dumps(result))


if __name__ == "__main__":
    main()
