"""Build label-free Tank War pilot diagnosis packets from reviewed replay data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _reviewed(root: Path) -> dict[str, dict[str, str]]:
    reviews: dict[str, dict[str, str]] = {}
    for round_name in ("pilot1", "pilot2", "pilot3"):
        path = root / round_name / "private" / "review-zhuohang-submitted.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                sample_id = row["sample_id"]
                if sample_id in reviews:
                    raise ValueError(f"duplicate human review: {sample_id}")
                if row["judgment"] not in {"relevant", "irrelevant", "uncertain"}:
                    raise ValueError(f"invalid human judgment: {sample_id}")
                reviews[sample_id] = row
    return reviews


def _packets(root: Path, ids: set[str]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    verified_roots = (
        root / "pilot1/private/effective-evidence",
        root / "pilot2/private/effective-evidence",
        root / "pilot2/private/controller-extra-all",
    )
    identity = ("sample_id", "decision_index", "state_hash", "injected_layer", "fault_instance")
    for directory in verified_roots:
        pattern = "*/*/private-review-packet.json" if directory.name == "effective-evidence" else "*/private-review-packet.json"
        for path in directory.glob(pattern):
            packet = json.loads(path.read_text(encoding="utf-8"))
            sample_id = packet["sample_id"]
            if sample_id in ids:
                if sample_id in found and any(packet[key] != found[sample_id][key] for key in identity):
                    raise ValueError(f"inconsistent duplicate packet: {sample_id}")
                if sample_id in found and packet["source_run"] != found[sample_id]["source_run"]:
                    for name in ("states.jsonl", "decisions.jsonl", "execution.jsonl", "events.jsonl", "result.json"):
                        original = Path(found[sample_id]["source_run"]) / name
                        duplicate = Path(packet["source_run"]) / name
                        if hashlib.sha256(original.read_bytes()).digest() != hashlib.sha256(duplicate.read_bytes()).digest():
                            raise ValueError(f"duplicate match differs for {sample_id}: {name}")
                found[sample_id] = packet
    if set(found) != ids:
        raise ValueError(f"missing reviewed packets: {sorted(ids - set(found))}")
    return found


def _tank(tank: dict[str, Any]) -> dict[str, Any]:
    return {key: tank.get(key) for key in ("id", "x", "y", "angle", "alive", "cooldown", "assistActive")}


def _state(row: dict[str, Any]) -> dict[str, Any]:
    raw = row["raw_state"]
    return {
        "time": row["time"],
        "player": row["player"],
        "tanks": [_tank(tank) for tank in raw["tanks"]],
        "bullets": raw["bullets"],
        "shots": raw["shots"],
        "kills": raw["kills"],
        "assist_enabled": raw["assistEnabled"],
        "assist_profile": raw["assistProfile"],
        "fire_guard_enabled": raw["fireGuardEnabled"],
    }


REQUEST_FIELDS = (
    "protocol", "state", "weapon_state", "aim", "motions", "admissible_motion_ids",
    "context_revision", "avoidance_enabled", "avoidance_level", "predicted_response_delay_ms",
)


def _window_start(sample_id: str, index: int, count: int, width: int = 11) -> int:
    low = max(0, index - width + 1)
    high = min(index, max(0, count - width))
    if high < low:
        raise ValueError("selected decision is outside episode")
    offset = int(hashlib.sha256(f"tank183-window-v1:{sample_id}".encode()).hexdigest()[:8], 16)
    return low + offset % (high - low + 1)


def make_packet(packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    run = Path(packet["source_run"])
    states = _rows(run / "states.jsonl")
    decisions = _rows(run / "decisions.jsonl")
    execution = _rows(run / "execution.jsonl")
    events = _rows(run / "events.jsonl")
    if len(states) != len(execution) or [row["decision_index"] for row in states] != list(range(len(states))):
        raise ValueError(f"invalid decision timeline: {run}")
    by_index: dict[int, list[dict[str, Any]]] = {}
    for row in decisions:
        by_index.setdefault(row["decision_index"], []).append(row)
    if set(by_index) != set(range(len(states))):
        raise ValueError(f"missing model decision: {run}")
    index = packet["decision_index"]
    if not isinstance(index, int) or not 0 <= index < len(states):
        raise ValueError("invalid selected decision")
    start = _window_start(packet["sample_id"], index, len(states))
    end = min(start + 11, len(states))
    opaque_id = hashlib.sha256(f"tank183-pilot-read-v1:{packet['sample_id']}".encode()).hexdigest()[:20]
    timeline = [
        {
            "decision_index": i,
            "time": row["accepted_time"],
            "tanks": [_tank(tank) for tank in states[i]["raw_state"]["tanks"]],
            "selected_action": row["selected_action"],
            "acknowledged_command": row["acknowledged_command"],
            "executed_action": row["executed_action"],
        }
        for i, row in enumerate(execution)
    ]
    detail = []
    for i in range(start, end):
        state = states[i]
        detail.append(
            {
                "decision_index": i,
                "raw_state": _state(state),
                "rendered_request": {key: state["request"].get(key) for key in REQUEST_FIELDS if key in state["request"]},
                "model_answers": [
                    {
                        "questions": row["questions"],
                        "answers": row["answers"],
                        "chosen": row["chosen"],
                        "constraints": row["constraints"],
                        "applied": row["applied"],
                        "input_acknowledgement": row["input_acknowledgement"],
                    }
                    for row in by_index[i]
                ],
                "execution": execution[i],
            }
        )
    maze = states[0]["raw_state"]["maze"]
    cause = packet.get("injected_layer", packet.get("condition"))
    if cause not in {"harness", "weights", "controller", "no_injection"}:
        raise ValueError("invalid pilot condition")
    public = {
        "episode_id": opaque_id,
        "game": "Tank War",
        "agent_player": states[0]["player"],
        "outcome": json.loads((run / "result.json").read_text(encoding="utf-8"))["outcome"][packet["seat"]],
        "map_walls": maze["walls"],
        "timeline": timeline,
        "detail_window": detail,
        "events": events,
    }
    gold = {
        "episode_id": opaque_id,
        "sample_id": packet["sample_id"],
        "cause": cause,
        "cause_reference_source": (
            "verified_no_injection_manifest" if cause == "no_injection" else "single_layer_injection_manifest"
        ),
        "decision_index": index,
        "decision_reference_source": (
            "preselected_midpoint_control" if cause == "no_injection" else "first_locally_verified_execution_difference"
        ),
        "human_judgment": None if cause == "no_injection" else "relevant",
        "human_judgment_scope": (
            None if cause == "no_injection" else "whether_preselected_local_difference_matters_to_loss"
        ),
        "window_start": start,
        "window_end_exclusive": end,
        "fault_instance": packet.get("fault_instance"),
    }
    serialized = json.dumps(public)
    for forbidden in (packet["sample_id"], packet.get("fault_instance"), str(run), "injected_layer", "fault_instance"):
        if forbidden is None:
            continue
        if forbidden in serialized:
            raise ValueError(f"gold identifier leaked into diagnosis input: {opaque_id}")
    return public, gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output directory already exists")
    reviews = _reviewed(args.root)
    packets = _packets(args.root, set(reviews))
    selected = [packets[sample_id] for sample_id, row in reviews.items() if row["judgment"] == "relevant"]
    if len(selected) != 15 or {layer: sum(p["injected_layer"] == layer for p in selected) for layer in
                                  ("harness", "weights", "controller")} != dict.fromkeys(
                                      ("harness", "weights", "controller"), 5
                                  ):
        raise ValueError("expected five Zhuohang-relevant pilot examples per cause")
    args.out.mkdir(parents=True)
    (args.out / "inputs").mkdir()
    gold_rows = []
    for packet in selected:
        public, gold = make_packet(packet)
        (args.out / "inputs" / f"{public['episode_id']}.json").write_text(
            json.dumps(public, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        gold_rows.append(gold)
    (args.out / "gold.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in gold_rows), encoding="utf-8"
    )
    print(json.dumps({"reviewed": len(reviews), "selected": len(selected), "inputs": str(args.out / "inputs")}))


if __name__ == "__main__":
    main()
