"""Rebuild decisions and re-execute actions in a fresh native Spiders instance."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from honing.backends.griddly.harness import load_harness, parse_harness
from honing.backends.griddly.match import history_entry, summary
from honing.backends.griddly.model import baseline, make_request, validate_reply
from honing.backends.griddly.scenarios import GriddlyScenario
from honing.backends.griddly.source import adapter_sha256, source_files
from integrations.griddly.catalog import ACTIONS, FORMAT, GAME, GAME_SHA256, PROTOCOL
from integrations.griddly.engine import GriddlyEngine, digest, runtime_versions


def audit_run(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Check logs and reconstruct exact model requests without loading native dependencies."""
    if (run / "error.json").exists():
        raise ValueError("an incomplete or failed run cannot be audited as a completed episode")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("adapter_sha256") != adapter_sha256():
        raise ValueError("recording requires the original Griddly adapter source")
    for name, path in source_files().items():
        if (run / "source" / name).read_bytes() != path.read_bytes():
            raise ValueError("recorded source snapshot differs from the original adapter")
    if (manifest.get("format"), manifest.get("protocol"), manifest.get("game"), manifest.get("game_sha256")) != (
        FORMAT,
        PROTOCOL,
        GAME,
        GAME_SHA256,
    ) or manifest.get("actions") != ACTIONS:
        raise ValueError("unsupported Griddly recording or native rules")
    scenario = GriddlyScenario(**manifest["scenario"])
    spec = load_harness(run / "harness")
    harness = parse_harness(spec)
    if manifest.get("spec_id") != spec.spec_id or manifest.get("harness") != list(harness.files):
        raise ValueError("recorded harness differs from delivered files")
    policy, sample = manifest.get("controller"), manifest.get("sample")
    kind = (
        "rollout"
        if sample
        else "evaluation" if policy == "model" else "teacher-demonstration" if policy == "teacher" else "scripted"
    )
    if (
        policy not in ("random", "baseline", "teacher", "actions", "model")
        or type(sample) is not bool
        or (sample and policy != "model")
        or manifest.get("kind") != kind
    ):
        raise ValueError("inconsistent controller or sampling mode")
    if type(manifest.get("policy_seed")) is not int or not 0 <= manifest["policy_seed"] < 2**32:
        raise ValueError("invalid policy seed")
    rng = random.Random(manifest["policy_seed"])
    rows = [
        json.loads(line) for line in (run / "steps.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    decisions = [
        json.loads(line) for line in (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if (
        not rows
        or rows[0]["event"] != "reset"
        or len(decisions) != len(rows) - 1
        or len(decisions) > scenario.max_steps
    ):
        raise ValueError("incomplete or over-budget trajectory")
    reward = 0.0
    history: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        frame = row["frame"]
        if frame["step"] != index or frame["frame_hash"] != digest(
            {k: v for k, v in frame.items() if k != "frame_hash"}
        ):
            raise ValueError(f"invalid frame hash or step at {index}")
        if type(frame["reward"]) not in (int, float) or not math.isfinite(frame["reward"]):
            raise ValueError("nonfinite frame reward")
        if index == 0:
            if frame["reward"] != 0 or frame["status"] != "running":
                raise ValueError("invalid reset frame")
            continue
        previous, decision = rows[index - 1]["frame"], decisions[index - 1]
        action, forced = row["action"], len(previous["legal_actions"]) == 1
        if (
            row["event"] != "step"
            or previous["status"] != "running"
            or action not in previous["legal_actions"]
            or decision["step"] != index - 1
            or decision["before_frame_hash"] != previous["frame_hash"]
            or decision["chosen"] != {"action": action}
            or decision["applied"] != action
            or decision["forced_action"] is not forced
        ):
            raise ValueError(f"invalid action or decision at step {index}")
        latency = decision["decision_ms"]
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError("invalid decision latency")
        request = (
            make_request(previous, scenario, harness, history)
            if not forced and policy in ("model", "teacher")
            else None
        )
        if decision["request"] != request:
            raise ValueError("recorded request differs from public state, legal actions or harness")
        if policy == "model" and not forced:
            assert request is not None
            probabilities = validate_reply(request, decision["response"])
            expected = (
                rng.choices(list(probabilities), weights=list(probabilities.values()))[0]
                if sample
                else decision["response"]["answers"]["action"]["choice"]
            )
        else:
            if decision["response"] is not None:
                raise ValueError("a non-model or forced decision contains a model reply")
            expected = (
                previous["legal_actions"][0]
                if forced
                else (
                    baseline(previous)
                    if policy in ("baseline", "teacher")
                    else rng.choice(previous["legal_actions"]) if policy == "random" else action
                )
            )
        if expected != action:
            raise ValueError("executed action differs from the recorded policy distribution or controller")
        reward += frame["reward"]
        history.append(history_entry(action, frame))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    if rows[-1]["frame"]["status"] == "running" or any(
        result.get(k) != v for k, v in summary(rows[-1]["frame"], reward).items()
    ):
        raise ValueError("result differs from completed trajectory")
    return manifest, rows, result


def verify_replay(run: Path) -> dict[str, Any]:
    manifest, rows, _result = audit_run(run)
    if manifest["runtime"] != runtime_versions():
        raise ValueError("native replay requires the recorded runtime versions and platform")
    with GriddlyEngine(GriddlyScenario(**manifest["scenario"])) as engine:
        if engine.reset() != rows[0]["frame"]:
            raise ValueError("native reset differs from recording")
        for row in rows[1:]:
            if engine.step(row["action"]) != row["frame"]:
                raise ValueError(f"native replay differs at step {row['frame']['step']}")
    return {"verified": True, "steps": len(rows) - 1, "final_frame_hash": rows[-1]["frame"]["frame_hash"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        _manifest, rows, _result = audit_run(args.run)
        result = {"audited": True, "steps": len(rows) - 1}
    else:
        result = verify_replay(args.run)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
