"""Record and compare fixed same-state REA-Issac fault probes without gold labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from honing.backends.rea_issac.model import DecisionClient, validate_reply
from honing.backends.rea_issac_agent.faults import (
    HARNESS_REVERSE_MOVE,
    REVERSED_MOVE_INSTRUCTION,
    verify_clean_baseline,
)
from honing.backends.rea_issac_agent.train_agent import load_batch

FORMAT = "rea-issac-agent-fault-probe-v1"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_requests(dataset: Path, *, per_group: int = 2, limit: int = 32) -> list[dict[str, Any]]:
    """Predeclare first two requests per seed/room, in dataset order, before seeing model replies."""
    if per_group < 1 or limit < 1:
        raise ValueError("probe selection limits must be positive")
    load_batch(dataset)
    rows = [json.loads(line) for line in (dataset / "decisions.jsonl").read_text().splitlines() if line]
    selected: list[dict[str, Any]] = []
    group_counts: dict[tuple[int, str], int] = {}
    for row in rows:
        key = (row["seed"], row["room"])
        if group_counts.get(key, 0) < per_group and len(selected) < limit:
            selected.append(row["request"])
            group_counts[key] = group_counts.get(key, 0) + 1
    return selected


def record_probes(
    dataset: Path,
    out: Path,
    *,
    decision_server: str,
    expected_revision: str,
    harness_fault: str | None = None,
    per_group: int = 2,
    limit: int = 32,
) -> dict[str, Any]:
    """Query one identified checkpoint on preselected saved requests; never pass teacher labels."""
    if out.exists():
        raise FileExistsError(out)
    if harness_fault not in (None, HARNESS_REVERSE_MOVE):
        raise ValueError("unsupported harness fault")
    verify_clean_baseline()
    client = DecisionClient(decision_server)
    metadata = client.health()["metadata"]
    if metadata.get("revision") != expected_revision or metadata.get("weightsSha256") != expected_revision:
        raise ValueError("probe server does not attest the expected checkpoint")
    requests = select_requests(dataset, per_group=per_group, limit=limit)
    out.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "status": "running",
        "source_decisions_sha256": json.loads((dataset / "summary.json").read_text())["decisions_sha256"],
        "weights_sha256": expected_revision,
        "harness_fault": harness_fault,
        "selection": {"per_group": per_group, "limit": limit},
        "requests": len(requests),
        "responses": 0,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    try:
        with (out / "probes.jsonl").open("x") as stream:
            for original in requests:
                request = copy.deepcopy(original)
                if harness_fault:
                    request["questions"]["move"]["instructions"] = REVERSED_MOVE_INSTRUCTION
                selected, reply = client.decide(request)
                stream.write(
                    json.dumps(
                        {
                            "request_id": request["id"],
                            "source_request_sha256": _digest(original),
                            "request_sha256": _digest(request),
                            "request": request,
                            "reply": reply,
                            "selected_action": selected,
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n"
                )
                stream.flush()
                manifest["responses"] += 1
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "incomplete"
        raise
    finally:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest["probes_sha256"] = hashlib.sha256((out / "probes.jsonl").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _verified_rows(directory: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    data = (directory / "probes.jsonl").read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest.get("probes_sha256"):
        raise ValueError("probe file checksum differs from manifest")
    rows = [json.loads(line) for line in data.decode().splitlines()]
    if len(rows) != manifest["requests"]:
        raise ValueError("probe count differs from manifest")
    for row in rows:
        if row["request_sha256"] != _digest(row["request"]):
            raise ValueError("probe request checksum differs")
        if row["selected_action"] != validate_reply(row["request"], row["reply"]):
            raise ValueError("probe selected action differs from reply")
    return rows


def compare_probes(clean: Path, fault: Path, *, component: str) -> dict[str, Any]:
    """Require a single component boundary and count same-request probability/choice effects."""
    if component not in ("harness", "weights"):
        raise ValueError("same-state probe comparison needs harness or weights")
    a, b = (json.loads((path / "manifest.json").read_text()) for path in (clean, fault))
    if a["status"] != "complete" or b["status"] != "complete" or a["requests"] != b["requests"]:
        raise ValueError("probe records are incomplete or unpaired")
    if a["harness_fault"] is not None:
        raise ValueError("clean probe has a harness fault")
    if component == "harness":
        if b["harness_fault"] != HARNESS_REVERSE_MOVE or a["weights_sha256"] != b["weights_sha256"]:
            raise ValueError("harness comparison changed another component")
    elif b["harness_fault"] is not None or a["weights_sha256"] == b["weights_sha256"]:
        raise ValueError("weights comparison changed another component or no checkpoint")
    if a["source_decisions_sha256"] != b["source_decisions_sha256"]:
        raise ValueError("probe records came from different source datasets")
    clean_rows, fault_rows = _verified_rows(clean, a), _verified_rows(fault, b)
    changes: list[dict[str, Any]] = []
    for before, after in zip(clean_rows, fault_rows, strict=True):
        if (before["request_id"], before["source_request_sha256"]) != (
            after["request_id"],
            after["source_request_sha256"],
        ):
            raise ValueError("probe states are not paired")
        old, new = before["request"], after["request"]
        if before["source_request_sha256"] != _digest(old):
            raise ValueError("clean probe differs from source request")
        if component == "weights" and old != new:
            raise ValueError("weights comparison changed the request")
        if component == "harness":
            restored = copy.deepcopy(new)
            restored["questions"]["move"]["instructions"] = old["questions"]["move"]["instructions"]
            if restored != old or new["questions"]["move"]["instructions"] != REVERSED_MOVE_INSTRUCTION:
                raise ValueError("harness comparison changed more than the movement instruction")
        max_delta = max(
            abs(
                before["reply"]["answers"][name]["probabilities"][option]
                - after["reply"]["answers"][name]["probabilities"][option]
            )
            for name in old["questions"]
            for option in old["questions"][name]["criteria"]
        )
        changes.append(
            {
                "request_id": before["request_id"],
                "max_probability_delta": max_delta,
                "selected_action_changed": before["selected_action"] != after["selected_action"],
            }
        )
    return {
        "component": component,
        "states": len(changes),
        "probability_effects_over_0_01": sum(change["max_probability_delta"] > 0.01 for change in changes),
        "selected_action_changes": sum(change["selected_action_changed"] for change in changes),
        "max_probability_delta": max((change["max_probability_delta"] for change in changes), default=0),
        "changes": changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    record = modes.add_parser("record")
    record.add_argument("--dataset", type=Path, required=True)
    record.add_argument("--out", type=Path, required=True)
    record.add_argument("--decision-server", required=True)
    record.add_argument("--expected-revision", required=True)
    record.add_argument("--harness-fault", choices=(HARNESS_REVERSE_MOVE,))
    record.add_argument("--per-group", type=int, default=2)
    record.add_argument("--limit", type=int, default=32)
    compare = modes.add_parser("compare")
    compare.add_argument("--clean", type=Path, required=True)
    compare.add_argument("--fault", type=Path, required=True)
    compare.add_argument("--component", choices=("harness", "weights"), required=True)
    args = parser.parse_args()
    if args.mode == "record":
        result = record_probes(
            args.dataset,
            args.out,
            decision_server=args.decision_server,
            expected_revision=args.expected_revision,
            harness_fault=args.harness_fault,
            per_group=args.per_group,
            limit=args.limit,
        )
    else:
        result = compare_probes(args.clean, args.fault, component=args.component)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
