"""Freeze unseen critical-window losses and construction-blind System 0 inputs."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

SOURCE_FILES = ("manifest.json", "result.json", "states.jsonl", "decisions.jsonl", "execution.jsonl", "events.jsonl")
FORBIDDEN = ("inject", "fault", "clean", "gold", "manifest", "source_run", "seed")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(item: dict) -> str:
    return hashlib.sha256(f"tank183-critical-v1:{item['layer']}:{item['seed']}:{item['seat']}".encode()).hexdigest()


def select(
    pairs: Path, harness_pairs: Path | None, old_harness: Path | None, extra_pairs: list[Path]
) -> tuple[list[dict], dict]:
    candidates: dict[str, list[dict]] = {layer: [] for layer in ("harness", "weights", "controller")}
    controls = []
    counts = {layer: {"blind_pairs": 0, "blind_fault_losses": 0, "technical_eligible": 0} for layer in candidates}
    roots = (pairs, *extra_pairs) if harness_pairs is None else (pairs, harness_pairs, *extra_pairs)
    for root in roots:
        frozen = read(root / "freeze.json")
        for entry in frozen["entries"]:
            if (
                (harness_pairs or old_harness)
                and entry["layer"] == "harness"
                and root != harness_pairs
                and root not in extra_pairs
            ):
                continue
            if harness_pairs and entry["layer"] != "harness" and root == harness_pairs:
                continue
            source = root / entry["layer"] / f"s{entry['seed']}-{entry['seat']}"
            evidence_file = source / "pair_evidence.json"
            if not evidence_file.exists():
                continue
            evidence = read(evidence_file)
            counts[entry["layer"]]["blind_pairs"] += 1
            if evidence["fault_loss"]:
                counts[entry["layer"]]["blind_fault_losses"] += 1
            if (
                evidence["fault_loss"]
                and evidence["triggered"]
                and evidence.get("same_prefix")
                and evidence.get("layer_isolated")
                and evidence.get("local_action_changed")
                and (entry["layer"] != "harness" or evidence.get("selected_action_changed"))
            ):
                counts[entry["layer"]]["technical_eligible"] += 1
                candidates[entry["layer"]].append({**entry, "source_run": str(source / "fault"), "evidence": evidence})
            if evidence["clean_outcome"] == -1:
                controls.append(
                    {**entry, "layer": "control", "source_run": str(source / "clean"), "evidence": evidence}
                )
    if old_harness:
        old_freeze = read(old_harness / "freeze.json")
        verified = []
        for entry in old_freeze["entries"]:
            summary_path = old_harness / f"s{entry['seed']}-{entry['seat']}" / "summary.json"
            if not summary_path.exists():
                continue
            summary = read(summary_path)
            if summary["status"] != "verified":
                continue
            control = (
                old_harness.parent.parent
                / "v0-pilot"
                / "formal-controls-final-v1"
                / "games"
                / f"s{entry['seed']}-{entry['seat']}"
                / "result.json"
            )
            clean_outcome = read(control)["outcome"][entry["seat"]] if control.exists() else None
            verified.append(
                {
                    "layer": "harness",
                    "seed": entry["seed"],
                    "seat": entry["seat"],
                    "source_run": entry["run"],
                    "evidence": {
                        "decision_index": summary["technical_reference_index"],
                        "outcome_improved_without_fault": clean_outcome is not None and clean_outcome > -1,
                    },
                }
            )
        unique = []
        seen_seeds = set()
        for case in verified:
            if case["seed"] not in seen_seeds:
                unique.append(case)
                seen_seeds.add(case["seed"])
        counts["harness"].update(
            old_losses=len(old_freeze["entries"]),
            verified=len(verified),
            verified_unique_seeds=len(unique),
            blind_reserved=min(10, len(unique)),
        )
        candidates["harness"] = unique[:10]
    chosen = []
    used = set()
    for layer in ("harness", "weights", "controller"):
        ordered = candidates[layer] if layer == "harness" and old_harness else sorted(candidates[layer], key=rank)
        for case in ordered:
            if case["seed"] in used:
                continue
            chosen.append(case)
            used.add(case["seed"])
            if sum(x["layer"] == layer for x in chosen) == 10:
                break
    for case in sorted(controls, key=rank):
        if case["seed"] in used:
            continue
        chosen.append(case)
        used.add(case["seed"])
        if sum(x["layer"] == "control" for x in chosen) == 10:
            break
    return chosen, counts


def public_case(case: dict) -> dict:
    source = Path(case["source_run"])
    state = rows(source / "states.jsonl")
    execution = rows(source / "execution.jsonl")
    calls = rows(source / "model_calls.jsonl")
    decisions = rows(source / "decisions.jsonl")
    events = rows(source / "events.jsonl")
    result = read(source / "result.json")
    if not state or len(state) != len(execution) or len(state) != len(calls):
        raise ValueError(f"incomplete decision trace: {source}")
    player = state[0]["player"]
    by_index: dict[int, list[dict]] = {}
    for decision in decisions:
        by_index.setdefault(decision["decision_index"], []).append(decision)
    timeline = []
    for index, (record, action, call) in enumerate(zip(state, execution, calls, strict=True)):
        if record["decision_index"] != index or action["decision_index"] != index:
            raise ValueError(f"decision indices diverge: {source}/{index}")
        raw = record["raw_state"]
        observed = record["observation"]
        motion = action["selected_action"]["motion"]
        branch = call["request"].get("branches", {}).get(motion, {})
        branch_reply = call["reply"].get("branches", {}).get(motion, {})
        motion_question = call["request"]["questions"].get("motion", {})
        weapon_question = branch.get("questions", {}).get("weapon", {})
        weapon_answer = branch_reply.get("answers", {}).get("weapon", {})
        context_lines = "\n".join(call["request"].get("parts", {}).get("essential", [])).splitlines()
        strategy_line = context_lines[0].split(" AIM AT CURRENT", 1)[0] if context_lines else ""
        situation_line = context_lines[1][:220] if len(context_lines) > 1 else ""
        forecasts = []
        for line in context_lines:
            if not line.startswith("motion/"):
                continue
            name, _, detail = line.partition(": ")
            relevant = [
                part.strip()
                for part in detail.split(";")
                if any(word in part for word in ("target progress", "shell collision", "shot after turn", "clearance"))
            ]
            forecasts.append(f"{name}: {'; '.join(relevant)}")
        constraints = next((d.get("constraints") for d in by_index.get(index, []) if d.get("constraints")), {})
        aim = observed.get("aim", {}).get("best") if isinstance(observed.get("aim"), dict) else None
        timeline.append(
            {
                "index": index,
                "time": record["time"],
                "tanks": [{key: tank[key] for key in ("x", "y", "angle", "alive")} for tank in raw["tanks"]],
                "bullets": [
                    {key: bullet[key] for key in ("id", "x", "y", "vx", "vy", "owner", "bounces")}
                    for bullet in raw["bullets"]
                ],
                "aim": (
                    {key: aim.get(key) for key in ("heading_deg", "delta_deg", "hit_ms", "bounces")} if aim else None
                ),
                "model_input": {
                    "strategy_guidance": strategy_line,
                    "observed_situation": situation_line,
                    "option_forecasts": forecasts,
                    "motion_instruction": motion_question.get("instructions"),
                    "weapon_instruction": weapon_question.get("instructions"),
                    "fire_description": weapon_question.get("criteria", {}).get("fire"),
                    "hold_description": weapon_question.get("criteria", {}).get("hold"),
                },
                "model_probabilities": {
                    "motion": call["reply"].get("answers", {}).get("motion", {}).get("probabilities"),
                    "weapon": weapon_answer.get("probabilities"),
                },
                "selected": action["selected_action"],
                "acknowledged": action["acknowledged_command"]["controls"],
                "executed": action["executed_action"],
                "fire_permitted": constraints.get("fire_permitted"),
            }
        )
    return {
        "episode_id": case["episode_id"],
        "outcome": "loss",
        "agent_player": player,
        "duration_seconds": result["sim_seconds"],
        "map_walls": state[0]["raw_state"]["maze"]["walls"],
        "timeline": timeline,
        "events": [
            {key: event[key] for key in ("type", "time", "player", "victim", "self") if key in event}
            for event in events
            if event["type"] in {"shot", "kill", "assist"}
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--harness-pairs", type=Path)
    parser.add_argument("--old-harness", type=Path)
    parser.add_argument("--extra-pairs", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "sample-freeze.json").exists() or (args.out / "inputs").exists():
        raise FileExistsError("Blind sample has already been frozen")
    for name in ("protocol.json", "prompt.txt", "schema.json"):
        if not (args.out / name).exists():
            raise FileNotFoundError(args.out / name)
    cutoff_path = args.out / "collection-cutoff.json"
    if not cutoff_path.exists():
        raise FileNotFoundError(cutoff_path)
    if args.harness_pairs and args.old_harness:
        parser.error("choose only one harness source")
    chosen, counts = select(args.pairs, args.harness_pairs, args.old_harness, args.extra_pairs)
    cutoff = read(cutoff_path)
    cutoff_time = datetime.datetime.fromisoformat(cutoff["freeze_at_or_before_utc"].replace("Z", "+00:00"))
    if datetime.datetime.now(datetime.timezone.utc) < cutoff_time and any(
        sum(case["layer"] == layer for case in chosen) < 10
        for layer in ("harness", "weights", "controller", "control")
    ):
        raise ValueError("pre-cutoff independent-case target not yet met")
    (args.out / "inputs").mkdir()
    cases = []
    for item in chosen:
        item_id = hashlib.sha256(
            f"tank183-critical-v1:episode:{item['layer']}:{item['seed']}:{item['seat']}".encode()
        ).hexdigest()[:20]
        case = {
            "episode_id": item_id,
            "layer": item["layer"],
            "seed": item["seed"],
            "seat": item["seat"],
            "source_run": item["source_run"],
            "technical_reference_index": item["evidence"].get("decision_index"),
            "outcome_improved_without_fault": item["evidence"].get("outcome_improved_without_fault"),
            "source_hashes": {name: sha(Path(item["source_run"]) / name) for name in SOURCE_FILES},
        }
        public = public_case({**item, "episode_id": item_id})
        serialized = json.dumps(public, separators=(",", ":"))
        if (
            len(serialized) > 500_000
            or any(term in serialized.lower() for term in FORBIDDEN)
            or any(value in serialized for value in (str(item["seed"]), item["source_run"]))
        ):
            raise ValueError(f"public replay leaks construction or is oversized: {item_id}")
        input_path = args.out / "inputs" / f"{item_id}.json"
        input_path.write_text(serialized)
        case["input_sha256"] = sha(input_path)
        case["decisions"] = len(public["timeline"])
        cases.append(case)
    freeze = {
        "status": "frozen_before_model_answers",
        "cases": cases,
        "source_freeze_sha256": {
            "main": sha(args.pairs / "freeze.json"),
            "harness": (
                sha(args.harness_pairs / "freeze.json")
                if args.harness_pairs
                else sha(args.old_harness / "freeze.json") if args.old_harness else None
            ),
            "extra": {str(root): sha(root / "freeze.json") for root in args.extra_pairs},
        },
        "protocol_sha256": sha(args.out / "protocol.json"),
        "prompt_sha256": sha(args.out / "prompt.txt"),
        "schema_sha256": sha(args.out / "schema.json"),
        "collection_cutoff_sha256": sha(cutoff_path),
        "scoring_script_sha256": sha(Path(__file__).with_name("score-critical-diagnosis.py")),
        "preparation_script_sha256": sha(Path(__file__)),
        "selection_counts": counts,
    }
    (args.out / "sample-freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(
        json.dumps(
            {
                "cases": len(cases),
                "by_layer": {
                    layer: sum(x["layer"] == layer for x in cases)
                    for layer in ("harness", "weights", "controller", "control")
                },
                "selection_counts": counts,
            }
        )
    )


if __name__ == "__main__":
    main()
