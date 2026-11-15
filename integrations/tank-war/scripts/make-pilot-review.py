"""Build one self-contained, readable human review page from private #183 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

NAMES = {
    "disable_assistance-v1": "关闭辅助设置",
    "reverse_prior_scalar_edit-v1": "修改模型权重",
    "wrong_legal_projection-v1": "把合法动作映射错",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def build(
    root: Path,
    template: Path,
    out: Path,
    evidence_set: str,
    review_file: Path | None = None,
    expected_samples: int = 15,
    storage_key: str = "tank183-review-v3-zhuohang",
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    reviews: dict[str, dict[str, str]] = {}
    if review_file:
        with review_file.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        reviews = {row["sample_id"]: row for row in rows}
        if len(reviews) != len(rows) or any(row["reviewer"] != "zhuohang" for row in rows):
            raise ValueError("expected distinct samples reviewed by Zhuohang")
    if evidence_set not in {"evidence", "effective-evidence"}:
        raise ValueError("unknown evidence set")
    for arm in ("harness-disable-assistance", "weights-reverse-edit", "controller-wrong-projection"):
        for packet in _lines(root / "private" / evidence_set / arm / "samples.jsonl"):
            source = Path(packet["source_run"])
            comparisons = _lines(Path(packet["same_state_evidence"]))
            if not comparisons or not comparisons[-1]["locally_verified"]:
                raise ValueError(f"no local difference for {packet['sample_id']}")
            comparison = comparisons[-1]
            if evidence_set == "effective-evidence" and not comparison["local_execution_changed"]:
                raise ValueError(f"no actual execution change for {packet['sample_id']}")
            if comparison["decision_index"] != packet["decision_index"]:
                raise ValueError(f"window differs for {packet['sample_id']}")
            state = next(
                row for row in _lines(source / "states.jsonl") if row["decision_index"] == packet["decision_index"]
            )
            nearby_states = [
                row
                for row in _lines(source / "states.jsonl")
                if packet["window"][0] <= row["decision_index"] <= packet["window"][1]
            ]
            all_states = _lines(source / "states.jsonl")
            executions = {row["decision_index"]: row for row in _lines(source / "execution.jsonl")}
            if not nearby_states or any(row["decision_index"] not in executions for row in all_states):
                raise ValueError(f"incomplete nearby decisions for {packet['sample_id']}")
            if state["state_hash"] != packet["state_hash"]:
                raise ValueError(f"state hash differs for {packet['sample_id']}")
            manifest = _json(source / "manifest.json")
            gunlines = _json(root / "private" / evidence_set / arm / packet["sample_id"] / "gunlines.json")
            if (
                gunlines["source_run"] != packet["source_run"]
                or gunlines["engine_sha256"] != manifest["sources"]["web/game/engine.js"]
                or gunlines["max_distance_px"] != 400
            ):
                raise ValueError(f"gunline source or physics differs for {packet['sample_id']}")
            line_by_index = {row["decision_index"]: row for row in gunlines["rows"]}
            if len(line_by_index) != len(all_states) or any(
                row["decision_index"] not in line_by_index
                or row["state_hash"] != line_by_index[row["decision_index"]]["state_hash"]
                or len(line_by_index[row["decision_index"]]["lines"]) != 2
                for row in all_states
            ):
                raise ValueError(f"gunlines do not match recorded states for {packet['sample_id']}")
            review = reviews.get(packet["sample_id"])
            if review_file:
                if not review or (
                    int(review["seed"]),
                    review["seat"],
                    int(review["decision_index"]),
                    review["injected_layer"],
                ) != (packet["seed"], packet["seat"], packet["decision_index"], packet["injected_layer"]):
                    raise ValueError(f"submitted review does not match {packet['sample_id']}")
                if review["judgment"] not in {"relevant", "irrelevant", "uncertain"}:
                    raise ValueError(f"invalid review judgment for {packet['sample_id']}")
            reference = _json(Path(packet["reference_action"]))
            if reference["state_hash"] != packet["state_hash"]:
                raise ValueError(f"planner state differs for {packet['sample_id']}")
            raw = state["raw_state"]
            result = _json(source / "result.json")
            episode_end = result["sim_seconds"]
            clip_start = 0 if episode_end <= 20 else max(0, min(state["time"] - 5, episode_end - 10))
            clip_end = episode_end if episode_end <= 20 else clip_start + 10
            playback_states = [row for row in all_states if clip_start <= row["time"] <= clip_end]
            if not playback_states or not any(
                row["decision_index"] == packet["decision_index"] for row in playback_states
            ):
                raise ValueError(f"playback misses candidate window for {packet['sample_id']}")
            samples.append(
                {
                    "id": packet["sample_id"],
                    "claim": NAMES[packet["fault_instance"]],
                    "layer": packet["injected_layer"],
                    "seed": packet["seed"],
                    "seat": packet["seat"],
                    "decision": packet["decision_index"],
                    "time": state["time"],
                    "window": packet["window"],
                    "nearby": [
                        {
                            "decision": row["decision_index"],
                            "time": row["time"],
                            "selected": executions[row["decision_index"]]["selected_action"],
                            "executed": executions[row["decision_index"]]["executed_action"],
                        }
                        for row in nearby_states
                    ],
                    "stratum": packet["task_stratum"],
                    "state_hash": packet["state_hash"],
                    "submitted_review": (
                        {
                            "judgment": review["judgment"],
                            "seconds": int(review["seconds_spent"]),
                            "note": review["notes"],
                        }
                        if review
                        else None
                    ),
                    "scene": {
                        "walls": raw["maze"]["walls"],
                        "player": state["player"],
                    },
                    "playback": {
                        "start": clip_start,
                        "end": clip_end,
                        "episode_end": episode_end,
                        "frames": [
                            {
                                "time": row["time"],
                                "decision": row["decision_index"],
                                "tanks": [
                                    {key: tank[key] for key in ("x", "y", "angle", "alive")}
                                    for tank in row["raw_state"]["tanks"]
                                ],
                                "bullets": [
                                    {key: bullet[key] for key in ("id", "x", "y", "owner", "bounces")}
                                    for bullet in row["raw_state"]["bullets"]
                                ],
                                "selected": executions[row["decision_index"]]["selected_action"],
                                "executed": executions[row["decision_index"]]["executed_action"],
                                "gunlines": line_by_index[row["decision_index"]]["lines"],
                            }
                            for row in playback_states
                        ],
                        "events": [
                            {key: event[key] for key in ("type", "time", "player") if key in event}
                            | {key: event[key] for key in ("victim", "self") if key in event}
                            for event in _lines(source / "events.jsonl")
                            if event["type"] in {"shot", "kill", "assist"}
                            and clip_start - 0.001 <= event["time"] <= clip_end + 0.001
                        ],
                    },
                    "faulty": comparison["faulty"],
                    "clean": comparison["clean"],
                    "changed": {
                        "choice": comparison["local_decision_changed"],
                        "command": comparison["local_command_changed"],
                        "execution": comparison["local_execution_changed"],
                    },
                    "planner": reference["acceptable_actions"],
                    "model_input": state["observation"]["state"],
                    "fault_manifest": {
                        "harness_sha256": manifest["harness_sha256"],
                        "controller_sha256": manifest["controller_sha256"],
                        "weights_sha256": manifest["model"]["weights_sha256"],
                    },
                }
            )
    if len(samples) != expected_samples or len({item["id"] for item in samples}) != expected_samples:
        raise ValueError(f"expected {expected_samples} distinct samples")
    if review_file and set(reviews) != {item["id"] for item in samples}:
        raise ValueError("submitted review does not cover the selected samples exactly")
    if re.fullmatch(r"[A-Za-z0-9_-]+", storage_key) is None:
        raise ValueError("invalid local review storage key")
    html = (
        template.read_text(encoding="utf-8")
        .replace("__SAMPLE_COUNT__", str(len(samples)))
        .replace("__REVIEW_STORAGE_KEY__", storage_key)
        .replace("__SAMPLE_DATA__", json.dumps(samples, ensure_ascii=False).replace("<", "\\u003c"))
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return {
        "samples": len(samples),
        "out": str(out),
        "bytes": out.stat().st_size,
        "review_sha256": hashlib.sha256(review_file.read_bytes()).hexdigest() if review_file else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence-set", choices=("evidence", "effective-evidence"), default="effective-evidence")
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--expected-samples", type=int, default=15)
    parser.add_argument("--storage-key", default="tank183-review-v3-zhuohang")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.root,
                Path(__file__).with_name("pilot-review-template.html"),
                args.out,
                args.evidence_set,
                args.review_file,
                args.expected_samples,
                args.storage_key,
            )
        )
    )


if __name__ == "__main__":
    main()
