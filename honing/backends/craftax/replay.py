"""Audit saved state/action records, or verify them by replaying the native game."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from honing.backends.craftax.catalog import ACHIEVEMENTS, ACTIONS, PROTOCOL
from honing.backends.craftax.engine import CraftaxEngine, frame_hash, runtime_versions
from honing.backends.craftax.match import FORMAT, summary
from honing.backends.craftax.model import validate_reply
from honing.backends.craftax.scenarios import CraftaxScenario


def audit_run(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Check artifact consistency without JAX; this does not simulate the game."""
    manifest = json.loads((run / "manifest.json").read_text())
    if (
        manifest.get("format") != FORMAT
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("actions") != list(ACTIONS)
    ):
        raise ValueError("unsupported Craftax recording")
    config = manifest["scenario"]
    scenario = CraftaxScenario(**{**config, "goals": tuple(config["goals"])})
    rows = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines() if line.strip()]
    decisions = [json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines() if line.strip()]
    if not rows or rows[0]["event"] != "reset" or len(decisions) != len(rows) - 1:
        raise ValueError("incomplete trajectory or decisions")
    total_reward = 0.0
    for index, row in enumerate(rows):
        frame = row["frame"]
        achievements = frame["achievements"]
        if (
            frame["step"] != index
            or frame["frame_hash"] != frame_hash(frame)
            or len(set(achievements)) != len(achievements)
            or not set(achievements) <= ACHIEVEMENTS
            or type(frame["reward"]) not in (int, float)
            or not math.isfinite(frame["reward"])
        ):
            raise ValueError(f"invalid recorded frame {index}")
        complete = set(scenario.goals) <= set(achievements)
        if (
            (frame["status"] == "goal_complete") != complete
            or frame["status"] not in ("running", "goal_complete", "death", "native_terminal", "step_limit")
            or type(frame["terminated"]) is not bool
            or type(frame["truncated"]) is not bool
            or frame["truncated"] != (frame["status"] != "running" and not frame["terminated"])
        ):
            raise ValueError(f"inconsistent terminal status at {index}")
        if frame["status"] == "step_limit" and index != scenario.max_steps:
            raise ValueError("step-limit frame does not match the scenario")
        if index == 0:
            if frame["reward"] != 0:
                raise ValueError("reset must have zero reward")
            continue
        previous, decision = rows[index - 1]["frame"], decisions[index - 1]
        if (
            row["event"] != "step"
            or row["action"] not in ACTIONS
            or previous["status"] != "running"
            or decision["step"] != index - 1
            or decision["before_frame_hash"] != previous["frame_hash"]
            or decision["chosen"] != {"action": row["action"]}
            or not set(previous["achievements"]) <= set(achievements)
        ):
            raise ValueError(f"invalid action sequence at {index}")
        if manifest["controller"] == "model":
            request = decision["request"]
            if set(request["questions"]["action"]["criteria"]) != set(ACTIONS):
                raise ValueError("a model decision omitted native options")
            if validate_reply(request, decision["response"]) != row["action"]:
                raise ValueError("executed action differs from the model's answer")
        elif decision["request"] is not None or decision["response"] is not None:
            raise ValueError("a scripted run contains model decisions")
        total_reward += frame["reward"]
    result = json.loads((run / "result.json").read_text())
    if rows[-1]["frame"]["status"] == "running" or any(
        result.get(k) != v for k, v in summary(rows[-1]["frame"], total_reward).items()
    ):
        raise ValueError("result differs from the completed trajectory")
    if not 0 <= result["steps"] <= scenario.max_steps:
        raise ValueError("trajectory exceeded the task budget")
    return manifest, rows, result


def verify_replay(run: Path) -> dict[str, Any]:
    manifest, rows, _result = audit_run(run)
    if manifest["runtime"] != runtime_versions():
        raise ValueError("native replay needs the recorded game/JAX/NumPy versions and platform")
    config = manifest["scenario"]
    with CraftaxEngine(CraftaxScenario(**{**config, "goals": tuple(config["goals"])})) as engine:
        if engine.reset() != rows[0]["frame"]:
            raise ValueError("native reset differs from the recording")
        for row in rows[1:]:
            if engine.step(row["action"]) != row["frame"]:
                raise ValueError(f"native replay differs at step {row['frame']['step']}")
    return {"verified": True, "steps": len(rows) - 1, "final_frame_hash": rows[-1]["frame"]["frame_hash"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--audit-only", action="store_true", help="Check logs without starting the native engine")
    args = parser.parse_args()
    if args.audit_only:
        _manifest, rows, _result = audit_run(args.run)
        report = {"audited": True, "steps": len(rows) - 1}
    else:
        report = verify_replay(args.run)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
