"""Fixed-weight base-Laya harness experiments using Honing's real runner and grader."""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import shutil
import statistics
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from honing.core import (
    ArchitectureSpec,
    CheckpointRef,
    HoneRelease,
    LayerShape,
    ModelSnapshot,
    PolicyEndpoint,
    PolicyRef,
    UpdateCandidate,
    run_graded,
)
from honing.release import EpisodeEvaluator, Gate, WinLossSelector
from integrations.overcooked.adapter import OvercookedEpisodeRunner, OvercookedGrader
from integrations.overcooked.decisions import DecisionClient
from integrations.overcooked.engine import ROOT, Engine, Scenario, digest, write_json
from integrations.overcooked.harness import dump_harness, load_harness, parse_harness, seed_harness
from integrations.overcooked.match import run_match
from integrations.overcooked.policy import baseline
from integrations.overcooked.recording import Recorder, replay, source_identity
from integrations.overcooked.scenarios import Suite

DIRECTORY = ROOT / "experiments/laya-harness-v1"
ENDPOINT = "http://127.0.0.1:8805"
INSTRUCTIONS = (
    "Choose chef 0's next physical action to maximize delivered soup. "
    "If holding soup, serve it. With a dish, collect ready soup. With an onion, fill an idle pot. "
    "With empty hands, start a full idle pot or fetch a needed onion or dish. "
    "Move toward the needed station, face it, then interact. Let the teammate pass. "
    "Wait only when useful work is unavailable."
)


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hash(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def suite() -> Suite:
    return Suite(tuple(json.loads(line) for line in (DIRECTORY / "tasks.jsonl").read_text().splitlines()))


def candidate(name: str) -> Any:
    return load_harness(DIRECTORY / "candidates" / name)


def snapshot() -> ModelSnapshot:
    raw = read(ROOT / ".cache/laya-model-discovery.json")["snapshot"]
    return ModelSnapshot(
        CheckpointRef(**raw["checkpoint"]),
        ArchitectureSpec(
            raw["architecture"]["hidden_size"], tuple(LayerShape(**layer) for layer in raw["architecture"]["layers"])
        ),
        raw["parameters"],
    )


class Endpoints:
    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        metadata = DecisionClient(ENDPOINT).health()["metadata"]
        if metadata["weightsSha256"] != policy.checkpoint.checksum or metadata["maxLen"] != 2048:
            raise ValueError("The server changed weights or the frozen inference budget")
        return PolicyEndpoint(ENDPOINT, "laya-pinned-tokenizer", "laya", "jev_decision")


def runner() -> OvercookedEpisodeRunner:
    return OvercookedEpisodeRunner(suite=suite(), endpoints=Endpoints(), runs_dir=ROOT / ".cache/laya-episodes")


def index() -> list[dict[str, Any]]:
    path = DIRECTORY / "episodes.json"
    return read(path) if path.exists() else []


def archive(run: Path, phase: str, name: str, task: str, grade: float | None = None) -> dict[str, Any]:
    manifest, result = read(run / "manifest.json"), read(run / "result.json")
    if manifest["source_identity"] != source_identity() or not result["complete"] or result["steps"] != 400:
        raise ValueError("Only complete 400-tick episodes from this frozen source can be archived")
    key = f"{phase}--{name}--{task}"
    out = DIRECTORY / "data/episodes" / key
    out.mkdir(parents=True, exist_ok=False)
    for filename in ("manifest.json", "result.json"):
        shutil.copyfile(run / filename, out / filename)
    if (run / "harness/files.json").exists():
        (out / "harness").mkdir()
        shutil.copyfile(run / "harness/files.json", out / "harness/files.json")
    for filename in ("steps.jsonl", "decisions.jsonl"):
        (out / (filename + ".gz")).write_bytes(gzip.compress((run / filename).read_bytes(), mtime=0))
    decisions = [json.loads(line) for line in (run / "decisions.jsonl").read_text().splitlines()]
    frames = [json.loads(line)["frame"] for line in (run / "steps.jsonl").read_text().splitlines()]
    audits = [row["response"]["tokenAudit"]["action"] for row in decisions if row["response"]]
    entry = {
        "key": key,
        "phase": phase,
        "candidate": name,
        "task_id": task,
        "layout": manifest["scenario"]["layout"],
        "score": result["score"],
        "grade": result["grade"],
        "honing_grade": grade,
        "deliveries": result["deliveries"],
        "steps": result["steps"],
        "source_identity": manifest["source_identity"],
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "spec_id": manifest.get("spec_id"),
        "weights_sha256": manifest.get("policy_checksum"),
        "model_calls": len(audits),
        "actions": dict(Counter(row["chosen"] for row in decisions)),
        "native_trajectory_sha256": digest([frame["frame_hash"] for frame in frames]),
        "mean_latency_ms": statistics.fmean(row.get("latency_ms", 0) for row in decisions),
        "max_input_tokens": max((audit["inputTokens"] for audit in audits), default=0),
        "max_state_tokens": max((audit["stateTokens"] for audit in audits), default=0),
        "truncated_replies": sum(audit["stateTruncated"] is not False for audit in audits),
        "files": {p.relative_to(DIRECTORY).as_posix(): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()},
    }
    rows = index()
    rows.append(entry)
    write_json(DIRECTORY / "episodes.json", rows)
    print(
        json.dumps({k: entry[k] for k in ("phase", "candidate", "task_id", "score", "deliveries", "actions")}),
        flush=True,
    )
    return entry


def run_one(name: str, task_id: str, phase: str) -> dict[str, Any]:
    for old in index():
        if (old["phase"], old["candidate"], old["task_id"]) == (phase, name, task_id):
            if old["source_identity"] != source_identity() or old["spec_id"] != candidate(name).spec_id:
                raise ValueError("Refusing to reuse a changed source or candidate")
            return old
    spec = candidate(name)
    task = next(
        task for split in ("train", "val", "heldout") for task in suite().tasks(split) if task.task_id == task_id
    )
    policy = PolicyRef(snapshot().checkpoint, snapshot().architecture, "master")
    record = asyncio.run(run_graded(runner(), OvercookedGrader(suite=suite()), spec, task, policy))
    if record.outcome.exit_code != 0:
        raise RuntimeError("Model episode failed: " + str(record.outcome.workspace_location))
    return archive(Path(record.outcome.workspace_location), phase, name, task_id, record.score)


def initialize() -> None:
    from honing.core import HarnessFile

    DIRECTORY.mkdir(parents=True, exist_ok=False)
    protocol = {
        "version": "laya-harness-v1",
        "feature_base": "6f252b2c23cfe37f4346189fa70ef9bd1c200a35",
        "model": "convaiinnovations/laya",
        "revision": "1c5edc17a7acd8701df6fc341c0d179f1c62c982",
        "weights_sha256": snapshot().checkpoint.checksum,
        "weights_fixed": True,
        "source_identity": source_identity(),
        "controlled_chef": 0,
        "partner": "solo",
        "ticks": 400,
        "experiment_code_sha256": {name: code_hash(ROOT / name) for name in ("laya_runtime.py", "laya_experiment.py")},
        "manual_cooking": True,
        "choice_keys": ["NORTH", "SOUTH", "EAST", "WEST", "STAY", "INTERACT"],
        "inference": {"max_len": 2048, "head_max_len": 512, "argmax": True, "sampling": False},
        "initial_candidates": ["seed", "readable", "readable_objective", "readable_local_effects"],
        "max_candidates_including_seed": 6,
        "max_adaptive_candidates": 2,
        "search": (
            "Two train layouts; greatest mean raw team score, strict improvement over seed; "
            "ties keep earlier candidate."
        ),
        "gate": (
            "Only the search winner; real Honing Gate with WinLossSelector on one unseen validation layout; "
            "ties reject."
        ),
        "heldout": (
            "After selection is frozen, seed and selected unique specs on two heldout layouts; "
            "never used to select."
        ),
        "controls": ["chef0_stay", "rule_chef0"],
        "limits": "Five deterministic layouts; no independent seed repetitions or statistical significance claim.",
        "wandb": "SDK absent and no API key configured; local per-episode and per-decision records are retained.",
    }
    write_json(DIRECTORY / "protocol.json", protocol)
    rows = []
    for split, layouts in (
        ("train", ("cramped_room", "asymmetric_advantages")),
        ("val", ("coordination_ring",)),
        ("heldout", ("forced_coordination", "counter_circuit_o_1order")),
    ):
        rows.extend(
            {
                "task_id": layout,
                "split": split,
                "prompt": "Maximize team delivery score in 400 ticks.",
                "overcooked": Scenario(layout=layout, partner="solo").record(),
            }
            for layout in layouts
        )
    (DIRECTORY / "tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (DIRECTORY / "candidates").mkdir()
    for name in protocol["initial_candidates"]:
        spec = seed_harness()
        if name != "seed":
            spec = spec.with_file(
                "overcooked/presentation.json",
                HarnessFile(
                    "settings",
                    json.dumps(
                        {
                            "context": "readable",
                            "choices": "local_effects" if name == "readable_local_effects" else "static",
                        }
                    ),
                ),
            )
        if name in {"readable_objective", "readable_local_effects"}:
            spec = spec.with_file(
                "overcooked/questions.json", HarnessFile("instructions", json.dumps({"action": INSTRUCTIONS}))
            )
        dump_harness(spec, DIRECTORY / "candidates" / name)
    health = DecisionClient(ENDPOINT).health()["metadata"]
    write_json(DIRECTORY / "model.json", {k: v for k, v in health.items() if k not in {"path", "gpuUuid"}})
    (DIRECTORY / ".gitattributes").write_text("data/** -text\n")
    print(json.dumps(protocol), flush=True)


def controls() -> None:
    for row in suite().records:
        scenario = suite().scenario(row["task_id"])
        for name in ("chef0_stay", "rule_chef0"):
            if any(
                old["phase"] == "control" and old["candidate"] == name and old["task_id"] == row["task_id"]
                for old in index()
            ):
                continue
            out = ROOT / ".cache/laya-controls" / (name + "--" + row["task_id"])
            if name == "rule_chef0":
                run_match(scenario, out, record_images=False)
            else:
                with Engine(scenario) as game:
                    recorder = Recorder(game, out, {"controller": "actions", "kind": "control", "control": name})
                    frame = game.observe()
                    while frame["status"] == "running":
                        before = frame
                        actions = ["STAY", baseline(frame, 1, "solo")]
                        frame = game.step(actions)
                        recorder.append(before, actions, frame, {"request": None, "response": None, "chosen": "STAY"})
                    recorder.finish()
            replay(out)
            archive(out, "control", name, row["task_id"])


def gate() -> None:
    if (DIRECTORY / "selection.json").exists():
        raise ValueError("Selection is already frozen")
    means = {}
    order = read(DIRECTORY / "protocol.json")["initial_candidates"]
    order += sorted(p.name for p in (DIRECTORY / "candidates").iterdir() if p.is_dir() and p.name not in order)
    for name in order:
        rows = [r for r in index() if r["phase"] == "search" and r["candidate"] == name]
        if len(rows) != 2:
            raise ValueError("Every candidate must complete both search layouts before selection")
        means[name] = statistics.fmean(row["score"] for row in rows)
    best = max(means, key=means.get)
    selection = {"search_means": means, "search_winner": best, "selected": "seed"}
    if means[best] > means["seed"]:
        snap = snapshot()
        head = HoneRelease("laya-seed", UpdateCandidate("init", None, snap, harness=candidate("seed")), None)
        proposal = UpdateCandidate("harness", head.release_id, snap, harness=candidate(best))
        evaluator = EpisodeEvaluator(runner(), OvercookedGrader(suite=suite()), suite().tasks("val"))
        evaluation, decision = Gate(evaluator, WinLossSelector()).judge(proposal, head)
        for record in evaluation.records:
            name = best if record.spec_id == candidate(best).spec_id else "seed"
            archive(Path(record.outcome.workspace_location), "val", name, record.task_id, record.score)
        selection["gate"] = asdict(decision)
        if decision.selected:
            selection["selected"] = best
    else:
        run_one("seed", "coordination_ring", "val")
        selection["gate"] = {
            "outcome": "reject",
            "reason": "No strict search improvement; no candidate validation call.",
        }
    write_json(DIRECTORY / "selection.json", selection)
    print(json.dumps(selection), flush=True)


def verify_archives() -> None:
    for row in index():
        out = ROOT / ".cache/laya-archive-audit" / row["key"]
        out.mkdir(parents=True, exist_ok=False)
        for relative, expected in row["files"].items():
            source = DIRECTORY / relative
            if file_hash(source) != expected:
                raise ValueError("Archive file checksum mismatch: " + relative)
            dest = out / source.relative_to(DIRECTORY / "data/episodes" / row["key"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix == ".gz":
                dest.with_suffix("").write_bytes(gzip.decompress(source.read_bytes()))
            else:
                shutil.copyfile(source, dest)
        if not replay(out)["verified"]:
            raise ValueError("Extracted native replay failed")
    write_json(
        DIRECTORY / "archive-verification.json",
        {
            "verified": True,
            "episodes": len(index()),
            "files": sum(len(row["files"]) for row in index()),
            "source_identity": source_identity(),
        },
    )
    print("All archived episodes passed checksum and native replay verification.")


def report() -> None:
    rows = index()
    with (DIRECTORY / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        keys = ["phase", "candidate", "layout", "score", "deliveries", "steps", "model_calls", "max_input_tokens"]
        writer = csv.DictWriter(stream, keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        DIRECTORY / "summary.json",
        {
            "selection": read(DIRECTORY / "selection.json"),
            "episodes": len(rows),
            "real_model_decisions": sum(row["model_calls"] for row in rows),
            "model_actions": dict(sum((Counter(row["actions"]) for row in rows if row["model_calls"]), Counter())),
            "archive_verification": read(DIRECTORY / "archive-verification.json"),
            "weights_sha256": snapshot().checkpoint.checksum,
            "source_identity": source_identity(),
            "wandb": (
                "Not configured; full per-episode metrics and raw per-decision probabilities " "are archived locally."
            ),
        },
    )
    print(json.dumps(read(DIRECTORY / "summary.json")), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("init", "search", "controls", "gate", "heldout", "verify", "report"))
    parser.add_argument("--candidate")
    args = parser.parse_args()
    if args.mode == "init":
        initialize()
        return
    if read(DIRECTORY / "protocol.json")["source_identity"] != source_identity():
        raise ValueError("Play/audit source changed after protocol freeze")
    for name, expected in read(DIRECTORY / "protocol.json")["experiment_code_sha256"].items():
        if code_hash(ROOT / name) != expected:
            raise ValueError("Experiment driver changed after protocol freeze")
    if args.mode == "search":
        if (DIRECTORY / "selection.json").exists():
            raise ValueError("Search cannot continue after validation/selection")
        if len(list((DIRECTORY / "candidates").iterdir())) > 6:
            raise ValueError("Candidate budget exceeded")
        parse_harness(candidate(args.candidate))
        for task in suite().tasks("train"):
            run_one(args.candidate, task.task_id, "search")
    elif args.mode == "controls":
        controls()
    elif args.mode == "gate":
        gate()
    elif args.mode == "heldout":
        chosen = read(DIRECTORY / "selection.json")["selected"]
        for name in dict.fromkeys(("seed", chosen)):
            for task in suite().tasks("heldout"):
                run_one(name, task.task_id, "heldout")
    elif args.mode == "verify":
        verify_archives()
    else:
        report()


if __name__ == "__main__":
    main()
