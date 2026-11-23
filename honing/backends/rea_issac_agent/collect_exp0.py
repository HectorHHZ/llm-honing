"""Collect and validate the frozen REA-Issac Exp 0 v0 episode pool."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from honing.backends.rea_issac.harness import parse_harness, seed_harness
from honing.backends.rea_issac.model import DecisionClient
from honing.backends.rea_issac.scenarios import ReaIssacScenario
from honing.backends.rea_issac_agent.faults import (
    CONTROLLER_SWAP_MOVE_AXES,
    HARNESS_REVERSE_MOVE,
    WEIGHTS_REVERSE_MOVE,
    reverse_movement_harness,
    verify_clean_baseline,
)
from honing.backends.rea_issac_agent.match import audit_agent_run, run_agent_match

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
FORMAT = "rea-issac-exp0-v0-collection-v1"
FAULTS = {
    "no_injection": None,
    "harness": HARNESS_REVERSE_MOVE,
    "weights": WEIGHTS_REVERSE_MOVE,
    "controller": CONTROLLER_SWAP_MOVE_AXES,
}
FAULT_WEIGHTS = "feea1950fecde6e702a2fd95eaffa83f5917544d88b62ac4ae05727f256287d6"
SEEDS = tuple(range(9101, 9113))
GROUPS = 10
MAX_TICKS = 1800
PROBE_STATES = 16
IDENTITY_TOLERANCE = 0.01
HASH_FILE = "SHA256SUMS.json"


def _write(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _files(root: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"unexpected symlink in collection: {path.relative_to(root)}")
        if not path.is_file() or path == root / HASH_FILE:
            continue
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        hashes[path.relative_to(root).as_posix()] = digest.hexdigest()
    return hashes


def _attest(client: DecisionClient, sha: str) -> None:
    metadata = client.health()["metadata"]
    if metadata.get("revision") != sha or metadata.get("weightsSha256") != sha:
        raise ValueError("NanoJev service did not attest the fixed checkpoint")


def _probability_delta(before: dict[str, Any], after: dict[str, Any]) -> float:
    return max(
        abs(value - after["answers"][name]["probabilities"][option])
        for name, question in before["answers"].items()
        for option, value in question["probabilities"].items()
    )


def _evidence(
    run: Path, arm: str, clean: DecisionClient, original: DecisionClient, clean_move: str
) -> dict[str, Any]:
    # The match writer audits before it returns. Recheck while building independent label evidence.
    audit = audit_agent_run(run)
    decisions = _rows(run / "decisions.jsonl")
    originals = _rows(run / "agent_decisions.jsonl")
    request = decisions[0]["request"]
    selected, reply = original.decide(request)
    delta = _probability_delta(originals[0]["model_response"], reply)
    identity = {
        "request_id": request["id"],
        "reply": reply,
        "max_probability_delta": delta,
        "passed": delta <= IDENTITY_TOLERANCE and selected == originals[0]["selected_action"],
    }
    comparisons = []
    if arm in ("harness", "weights"):
        count = min(PROBE_STATES, len(decisions))
        for index in (step * len(decisions) // count for step in range(count)):
            candidate = decisions[index]["request"]
            baseline_request = copy.deepcopy(candidate)
            if arm == "harness":
                baseline_request["questions"]["move"]["instructions"] = clean_move
            baseline_selected, baseline_reply = clean.decide(baseline_request)
            observed = originals[index]
            comparisons.append(
                {
                    "request_id": candidate["id"],
                    "clean_request": baseline_request,
                    "clean_reply": baseline_reply,
                    "clean_selected_action": baseline_selected,
                    "max_probability_delta": _probability_delta(baseline_reply, observed["model_response"]),
                    "selected_action_changed": baseline_selected != observed["selected_action"],
                    "request_changed": baseline_request != candidate,
                }
            )
    component_effect = {
        "no_injection": True,
        "harness": all(row["request_changed"] for row in comparisons),
        "weights": any(
            row["max_probability_delta"] > IDENTITY_TOLERANCE or row["selected_action_changed"] for row in comparisons
        ),
        "controller": audit["fault_triggers"] > 0,
    }[arm]
    return {"audit": audit, "identity": identity, "component_effect": component_effect, "comparisons": comparisons}


def _index(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[int, dict[str, dict[str, Any]]] = {}
    seen = set()
    for attempt in attempts:
        key = (attempt["seed"], attempt["arm"])
        if key in seen or key[0] not in SEEDS or key[1] not in FAULTS:
            raise ValueError("unexpected or duplicate Exp 0 attempt")
        seed, arm = key
        if attempt["path"] != f"attempts/s{seed}-{arm}" or attempt["episode_id"] != f"rea-issac-s{seed}-{arm}":
            raise ValueError("attempt path or episode ID does not match the protocol")
        seen.add(key)
        if attempt["status"] == "eligible":
            by_group.setdefault(key[0], {})[key[1]] = attempt
    complete = [seed for seed, arms in by_group.items() if set(arms) == set(FAULTS)]
    ranked = sorted(
        complete,
        key=lambda seed: (-sum(not by_group[seed][arm]["outcome"]["success"] for arm in FAULTS if FAULTS[arm]), seed),
    )
    if len(ranked) < GROUPS:
        raise ValueError(f"only {len(ranked)} of {GROUPS} required reset groups have four verified episodes")
    index = []
    for seed in sorted(ranked[:GROUPS]):
        clean = by_group[seed]["no_injection"]["outcome"]
        for arm in FAULTS:
            row = by_group[seed][arm]
            index.append(
                {
                    **row,
                    "split": "v0-development-evaluation",
                    "reset_group": f"rea-issac-s{seed}",
                    "injection_layer": arm,
                    "injection_instance": FAULTS[arm],
                    "verified_component": arm if FAULTS[arm] else None,
                    "annotation_status": "verified_component" if FAULTS[arm] else "control",
                    "annotation_provenance": "replay+agent-audit+matched-request-v1",
                    "paired_clean_outcome": clean,
                    "whole_episode_effect_status": "untested" if FAULTS[arm] else "not_applicable",
                }
            )
    return index


def _source() -> str:
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "honing", "integrations/rea_issac"], cwd=ROOT, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def collect(output: Path, clean_url: str, fault_url: str, *, resume: bool = False) -> dict[str, Any]:
    """Record a bounded attempt pool, retaining rejected groups and raw audit evidence."""
    source = _source()
    frozen = verify_clean_baseline()
    clean_sha = frozen["checkpoint_sha256"]
    clients = {"clean": DecisionClient(clean_url), "fault": DecisionClient(fault_url)}
    _attest(clients["clean"], clean_sha)
    _attest(clients["fault"], FAULT_WEIGHTS)
    clean_spec = seed_harness(PACKAGE)
    fault_spec = reverse_movement_harness(clean_spec)
    clean_move = parse_harness(clean_spec).instructions["move"]
    mode = frozen["controller_mode"]
    protocol = {
        "format": FORMAT,
        "game": "rea-issac",
        "code_commit": source,
        "baseline_commit": frozen["baseline_commit"],
        "runtime": platform.platform(),
        "python": platform.python_version(),
        "seeds": list(SEEDS),
        "decision_frames": frozen["decision_frames"],
        "max_ticks": MAX_TICKS,
        "clean_harness": clean_spec.spec_id,
        "fault_harness": fault_spec.spec_id,
        "controller_mode": mode,
        "clean_weights": clean_sha,
        "fault_weights": FAULT_WEIGHTS,
        "fault_instances": FAULTS,
        "identity_tolerance": IDENTITY_TOLERANCE,
        "matched_probe_states": PROBE_STATES,
        "selection": "top ten eligible reset groups by injected non-success count, then ascending seed",
    }
    if resume:
        if (output / HASH_FILE).exists():
            return verify(output)
        if _read(output / "protocol.json") != protocol:
            raise ValueError("resumed collection differs from the frozen protocol")
        attempts = _read(output / "attempts.json")
        for attempt in attempts:
            if attempt["status"] == "running":
                attempt.update(status="excluded", exclusion_reason="interrupted before verification")
        _write(output / "attempts.json", attempts)
    else:
        output.mkdir(parents=True, exist_ok=False)
        attempts = []
        _write(output / "protocol.json", protocol)
        _write(output / "attempts.json", attempts)
    smoke_receipt = output / "preflight_receipt.json"
    if not smoke_receipt.exists():
        smoke_attempts = output / "preflight_attempts"
        smoke_attempts.mkdir(exist_ok=True)
        smoke = smoke_attempts / str(len(list(smoke_attempts.iterdir())) + 1)
        result = run_agent_match(
            ReaIssacScenario(8001, MAX_TICKS, frozen["decision_frames"]),
            smoke,
            clean_spec,
            decision_server=clients["clean"].base_url,
            policy_checksum=clean_sha,
            controller_mode=mode,
        )
        _attest(clients["clean"], clean_sha)
        _write(
            smoke_receipt,
            {"passed": result["rooms_cleared"] > 0, "outcome": result, "path": smoke.relative_to(output).as_posix()},
        )
    if not _read(smoke_receipt)["passed"]:
        raise ValueError("clean model did not clear a room in the documented seed 8001 preflight")
    started = {(attempt["seed"], attempt["arm"]) for attempt in attempts}
    for seed in SEEDS:
        for arm in FAULTS:
            if (seed, arm) in started:
                continue
            run = Path("attempts") / f"s{seed}-{arm}"
            spec = fault_spec if arm == "harness" else clean_spec
            weights_fault = arm == "weights"
            sha = FAULT_WEIGHTS if weights_fault else clean_sha
            client = clients["fault" if weights_fault else "clean"]
            scenario = ReaIssacScenario(seed, MAX_TICKS, frozen["decision_frames"])
            attempt = {
                "episode_id": f"rea-issac-s{seed}-{arm}",
                "seed": seed,
                "arm": arm,
                "path": run.as_posix(),
                "harness_spec_id": spec.spec_id,
                "weights_sha256": sha,
                "status": "running",
            }
            attempts.append(attempt)
            _write(output / "attempts.json", attempts)
            _attest(client, sha)
            result = run_agent_match(
                scenario,
                output / run,
                spec,
                decision_server=client.base_url,
                policy_checksum=sha,
                controller_mode=mode,
                controller_fault=CONTROLLER_SWAP_MOVE_AXES if arm == "controller" else None,
            )
            evidence = _evidence(output / run, arm, clients["clean"], client, clean_move)
            _attest(client, sha)
            _attest(clients["clean"], clean_sha)
            _write(output / run / "component_evidence.json", evidence)
            reason = None
            if not evidence["identity"]["passed"]:
                reason = "same-request identity exceeded tolerance or changed the selected action"
            elif not evidence["component_effect"]:
                reason = "no verified component effect in this episode"
            attempt.update(status="excluded" if reason else "eligible", exclusion_reason=reason, outcome=result)
            _write(output / "attempts.json", attempts)
    index = _index(attempts)
    included = {row["episode_id"] for row in index}
    _write(
        output / "exclusions.json",
        [
            {"episode_id": row["episode_id"], "reason": row["exclusion_reason"] or "reset group not selected"}
            for row in attempts
            if row["episode_id"] not in included
        ],
    )
    _write(output / "episode_index.json", index)
    _write(output / "split_manifest.json", {"v0-development-evaluation": sorted(included)})
    _write(output / HASH_FILE, _files(output))
    return verify(output)


def verify(output: Path) -> dict[str, Any]:
    """Verify exact file coverage, hashes, labels and grouped counts with no model credentials."""
    if _read(output / HASH_FILE) != _files(output):
        raise ValueError("collection file list or hashes differ")
    if _read(output / "protocol.json")["format"] != FORMAT:
        raise ValueError("unsupported collection format")
    if not _read(output / "preflight_receipt.json")["passed"]:
        raise ValueError("clean model preflight was not verified")
    attempts = _read(output / "attempts.json")
    terminal = {"eligible", "excluded"}
    if len(attempts) != len(SEEDS) * len(FAULTS) or any(row["status"] not in terminal for row in attempts):
        raise ValueError("attempt ledger is incomplete")
    index = _index(attempts)
    if _read(output / "episode_index.json") != index:
        raise ValueError("episode index differs from the recorded attempt selection")
    split = {"v0-development-evaluation": sorted(row["episode_id"] for row in index)}
    if _read(output / "split_manifest.json") != split:
        raise ValueError("split does not match the selected episodes")
    for row in index:
        run = output / row["path"]
        manifest = _read(run / "manifest.json")
        if (
            manifest["scenario"]["seed"] != row["seed"]
            or manifest["spec_id"] != row["harness_spec_id"]
            or manifest["policy_checksum"] != row["weights_sha256"]
            or _read(run / "result.json") != row["outcome"]
        ):
            raise ValueError(f"recorded version or outcome differs for {row['episode_id']}")
        evidence = _read(run / "component_evidence.json")
        if not (evidence["audit"]["verified"] and evidence["identity"]["passed"] and evidence["component_effect"]):
            raise ValueError(f"invalid evidence for {row['episode_id']}")
    return {
        "episodes": len(index),
        "components": dict(Counter(row["injection_layer"] for row in index)),
        "reset_groups": len({row["reset_group"] for row in index}),
        "attempts": len(attempts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    runner = subcommands.add_parser("collect")
    runner.add_argument("--out", type=Path, required=True)
    runner.add_argument("--clean-server", required=True)
    runner.add_argument("--fault-server", required=True)
    runner.add_argument("--resume", action="store_true")
    checker = subcommands.add_parser("verify")
    checker.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "collect":
        result = collect(args.out, args.clean_server, args.fault_server, resume=args.resume)
    else:
        result = verify(args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
