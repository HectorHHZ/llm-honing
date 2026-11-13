"""Frozen small experiments through Honing's real runner/grader, with raw metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from honing.core import ArchitectureSpec, CheckpointRef, LayerShape, PolicyEndpoint, PolicyRef, run_graded
from integrations.overcooked.adapter import OvercookedEpisodeRunner, OvercookedGrader
from integrations.overcooked.decisions import DecisionClient
from integrations.overcooked.engine import ROOT, digest, write_json
from integrations.overcooked.harness import load_harness, seed_harness
from integrations.overcooked.match import run_match
from integrations.overcooked.recording import replay
from integrations.overcooked.scenarios import load_suite


def metrics(run: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in (run / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    decisions = [json.loads(line) for line in (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    p = manifest["scenario"]["controlled_player"]
    latencies = [row["latency_ms"] for row in decisions]
    first = next((row["frame"]["step"] for row in rows if row["frame"]["deliveries"] > 0), None)
    tokens, truncations, longest, streak = [], 0, 0, 0
    for before, after, decision in zip(rows, rows[1:], decisions, strict=False):
        same = before["frame"]["state"]["players"][p]["position"] == after["frame"]["state"]["players"][p]["position"]
        streak = streak + 1 if same else 0
        longest = max(longest, streak)
        if decision.get("response"):
            audit = decision["response"]["tokenAudit"]["action"]
            tokens.append(audit.get("inputTokens") or 0)
            truncations += bool(decision["response"]["trimmed"]) or audit.get("stateTruncated") is not False
    return {
        **result,
        "first_delivery_step": first,
        "actions": dict(Counter(row["actions"][p] for row in rows[1:])),
        "longest_stationary_streak": longest,
        "mean_decision_ms": statistics.fmean(latencies) if latencies else None,
        "max_input_tokens": max(tokens, default=0),
        "context_truncations": truncations,
    }


async def experiment(args: Any) -> dict[str, Any]:
    config = json.loads(args.protocol.read_text(encoding="utf-8"))
    suite_path = ROOT / config["suite"]
    suite = load_suite(suite_path)
    selected = config["pilot_tasks"][args.split]
    tasks = [t for t in suite.tasks(args.split) if t.task_id in selected]
    if len(tasks) != len(selected):
        raise ValueError("protocol selects tasks outside its named split")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = load_harness(args.harness_dir) if args.harness_dir else seed_harness()
    identity = digest({"protocol": config, "suite": suite_path.read_text(encoding="utf-8")})
    summary: dict[str, Any] = {
        "protocol": config,
        "protocol_sha256": identity,
        "split": args.split,
        "controller": args.policy,
        "spec_id": spec.spec_id,
        "episodes": [],
        "complete": False,
    }
    if args.policy == "model":
        health = DecisionClient(args.endpoint).health()
        model = health["metadata"]
        if model.get("family") != "nanojev":
            raise ValueError("this pilot requires the verified NanoJev model family")
        if args.expected_checksum and model.get("weightsSha256") != args.expected_checksum:
            raise ValueError("the server is not serving the requested checkpoint")
        if model.get("inference") != config["inference"]:
            raise ValueError("server inference mode differs from the frozen protocol")
        discovery = json.loads((ROOT / ".cache/model-discovery.json").read_text())
        shape = discovery["architecture"]
        architecture = ArchitectureSpec(shape["hidden_size"], tuple(LayerShape(**v) for v in shape["layers"]))
        policy = PolicyRef(
            CheckpointRef(model["path"], "hf_safetensors", model["weightsSha256"]), architecture, "master"
        )

        class Endpoints:
            def resolve(self, requested: PolicyRef) -> PolicyEndpoint:
                if requested.checkpoint.checksum != model["weightsSha256"]:
                    raise ValueError("checkpoint identity differs")
                return PolicyEndpoint(args.endpoint, "served-native-tokenizer", "nanojev", "jev_decision")

        runner = OvercookedEpisodeRunner(suite=suite, endpoints=Endpoints(), runs_dir=out / "runs")
        grader = OvercookedGrader(suite=suite)
        summary["model"] = model
    write_json(out / "protocol.json", {"protocol_sha256": identity, **config})
    write_json(out / "summary.json", summary)
    for task in tasks:
        started = time.perf_counter()
        if args.policy == "model":
            record = await run_graded(runner, grader, spec, task, policy)
            run = Path(record.outcome.workspace_location)
            item = metrics(run)
            if item["grade"] != record.grade.score:
                raise ValueError("grader and recording disagree")
        else:
            run = out / task.task_id
            run_match(suite.scenario(task.task_id), run, policy="baseline", record_images=False, spec=spec)
            replay(run)
            item = metrics(run)
        summary["episodes"].append(
            {
                "task_id": task.task_id,
                "scenario": suite.scenario(task.task_id).record(),
                "run": str(run),
                "verified": True,
                "elapsed_seconds": time.perf_counter() - started,
                **item,
            }
        )
        write_json(out / "summary.json", summary)
        print(
            json.dumps(
                {
                    "completed": task.task_id,
                    "score": item["score"],
                    "deliveries": item["deliveries"],
                    "first_delivery_step": item["first_delivery_step"],
                    "out": str(run),
                }
            ),
            flush=True,
        )
    summary["complete"] = True
    summary["mean_native_score"] = statistics.fmean(row["score"] for row in summary["episodes"])
    summary["mean_grade"] = statistics.fmean(row["grade"] for row in summary["episodes"])
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments/pilot-v1.json")
    parser.add_argument("--split", choices=("train", "val", "heldout"), default="train")
    parser.add_argument("--policy", choices=("baseline", "model"), default="model")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8803")
    parser.add_argument("--expected-checksum")
    parser.add_argument("--harness-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(experiment(args))
    print(
        json.dumps(
            {
                "out": str(args.out.resolve()),
                "complete": True,
                "mean_native_score": result["mean_native_score"],
                "mean_grade": result["mean_grade"],
            }
        )
    )


if __name__ == "__main__":
    main()
