"""Audit every Exp 0 v0 attempt of one game, attach component evidence, select 10/10/10/10 and assign neutral ids.

Game-agnostic port of Griddly v0 ``audit_set.py`` for the episode contract (CONTRACT.md).

  python3 audit_set.py reask ROOT OUT.jsonl --endpoint CLEAN_URL [--decisions N]
      For every valid weights attempt, re-ask its logged requests (first N model decisions, stopping at the
      first that differs by more than the tolerance) under the CLEAN checkpoint. This is the saved-request
      evidence for games whose clean and weights attempts never share an identical request (#185: use
      saved-request probes when exact state restore is unavailable). It calls the clean service.

  python3 audit_set.py audit ROOT OUT_DIR [--reask OUT.jsonl] [--adapter file.py:attr]
      Validate every attempt against the frozen plan, attach evidence, choose the scored episodes with
      Griddly's rule and write OUT_DIR/attempts.jsonl (every attempt) and OUT_DIR/episodes.json (the
      evaluator index with gold labels; never a model input). Exit 1 when the attempts do not yet cover
      every condition.

Evidence (labels stay separate from it; see CONTRACT.md section 9):
  harness     the first changed row: request vs clean_request, with the changed parts and options;
  controller  the first changed row: model choice vs executed option, and requested-vs-sent windows;
  weights     the first model decision whose request (and native tick) is identical to the matched clean
              attempt's and whose normalized probabilities differ by more than the tolerance; else the
              saved-request re-ask record; else the attempt is excluded.
Selection (Griddly): per slot the first non-excluded attempt is the candidate; interventions take
non-wins first in plan order; controls take the first ones in plan order, swapping in one candidate of
the other outcome when all chosen ones share an outcome. Then a seeded shuffle assigns neutral ids.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import (
    CONDITIONS,
    ResponseRejected,
    ServiceClient,
    canonical,
    fingerprint_files,
    guard_decision,
    guard_response,
    keyed,
    load_run,
    max_question_difference,
    model_rows,
    multi,
    normalized,
    plan_question_keys,
    read_json,
    read_rows,
    same_request,
    unkeyed,
    write_json,
    write_rows,
)
from validate_episode import check_run, runs_of

HERE = Path(__file__).resolve().parent
TOOL_FILES = ("audit_set.py", "validate_episode.py", "contract.py", "schema_lite.py")


def verification_tool_sha256() -> str:
    return fingerprint_files([HERE / name for name in TOOL_FILES], HERE)


# ---------------------------------------------------------------------------------------- evidence
def _lines(request: dict[str, Any]) -> list[str]:
    return [f"{section}: {line}" for section, lines in request["parts"].items() for line in lines]


def _question_change(request: dict[str, Any], clean: dict[str, Any], key: str) -> dict[str, Any]:
    options_now = list(request["questions"][key]["criteria"])
    options_before = list(clean["questions"][key]["criteria"])
    return {
        "options_removed": [o for o in options_before if o not in options_now],
        "option_order_changed": [o for o in options_now if o in options_before]
        != [o for o in options_before if o in options_now],
        "options_added": [o for o in options_now if o not in options_before],
        "option_texts_changed": [
            o
            for o in options_now
            if o in options_before
            and request["questions"][key]["criteria"][o] != clean["questions"][key]["criteria"][o]
        ],
        "instructions_changed": request["questions"][key]["instructions"] != clean["questions"][key]["instructions"],
    }


def change_summary(request: dict[str, Any], clean: dict[str, Any], question_key) -> dict[str, Any]:
    """What the harness fault changed in the request, by part and by option (both requests stay in the run).

    ``question_key`` is the plan's question, or (0.2.0) the list of its questions: then the option changes are
    reported per question under ``questions``."""
    now, before = _lines(request), _lines(clean)
    sections = sorted({*request["parts"], *clean["parts"]})
    summary = {
        "parts_changed": [s for s in sections if request["parts"].get(s) != clean["parts"].get(s)],
        "lines_only_in_clean": [line for line in before if line not in now],
        "lines_only_in_actual": [line for line in now if line not in before],
    }
    if isinstance(question_key, str):
        return {**summary, **_question_change(request, clean, question_key)}
    per = {key: _question_change(request, clean, key) for key in question_key}
    summary["questions"] = per
    summary["questions_changed"] = [key for key, change in per.items() if any(change.values())]
    return summary


def harness_evidence(run: dict[str, Any], question_key: str) -> dict[str, Any]:
    row = next(row for row in run["rows"] if row["fault"] and row["fault"]["changed"])
    return {
        "method": "request_vs_clean_render",
        "index": row["index"],
        "native_tick": row["native_before"]["tick"],
        "request_differs": True,
        "request_ref": f"{run['name']}/decisions.jsonl#index={row['index']}:request",
        "clean_request_ref": f"{run['name']}/decisions.jsonl#index={row['index']}:clean_request",
        "change_summary": change_summary(row["request"], row["clean_request"], question_key),
        "changed_rows": run["result"]["fault_changed_rows"],
    }


def controller_evidence(run: dict[str, Any]) -> dict[str, Any]:
    row = next(row for row in run["rows"] if row["fault"] and row["fault"]["changed"])
    mismatches = [w for w in row["executed"]["windows"] if "intended_action" in w or "intended_ticks" in w]
    first = []
    for w in mismatches[:5]:
        item = {
            "start_tick": w["start_tick"],
            "requested": w.get("intended_action", w["native_action"]),
            "sent": w["native_action"],
        }
        if "intended_ticks" in w:  # 0.2.0: the window's length changed
            item.update(requested_ticks=w["intended_ticks"], sent_ticks=w["ticks"])
        first.append(item)
    return {
        "method": "selected_vs_executed",
        "index": row["index"],
        "native_tick": row["native_before"]["tick"],
        "selected": row["model_choice"],
        "executed": row["executed"]["option"],
        "window_mismatches": len(mismatches),
        "first_mismatches": first,
        "executed_ref": f"{run['name']}/decisions.jsonl#index={row['index']}:executed",
        "changed_rows": run["result"]["fault_changed_rows"],
    }


def _normalized_rows(probabilities: Any, keys: list[str]) -> dict[str, dict[str, float]]:
    """{question: normalized probabilities} of a row's recorded probabilities (either row form)."""
    return {key: normalized(value) for key, value in keyed(probabilities, keys).items()}


def _argmax_map(probabilities: dict[str, dict[str, float]]) -> dict[str, str]:
    return {key: max(values, key=values.get) for key, values in probabilities.items()}


def matched_weights_evidence(
    clean: dict[str, Any], changed: dict[str, Any], tolerance: float, keys: list[str] | None = None
) -> dict[str, Any] | None:
    """Griddly's rule: the first identical request at the same native tick whose probabilities differ.

    With several questions (0.2.0) the difference is the largest over all questions; the recorded clean and
    alternate probabilities are then keyed by question."""
    keys = keys or ["action"]
    for base, alternate in zip(model_rows(clean["rows"]), model_rows(changed["rows"])):  # noqa: B905
        if (
            not same_request(base["request"], alternate["request"])
            or base["native_before"]["tick"] != alternate["native_before"]["tick"]
        ):
            continue
        left, right = _normalized_rows(base["probabilities"], keys), _normalized_rows(alternate["probabilities"], keys)
        difference = max_question_difference(left, right)
        if difference > tolerance:
            return {
                "method": "matched_clean_identical_request",
                "index": alternate["index"],
                "native_tick": alternate["native_before"]["tick"],
                "clean_run": clean["name"],
                "clean_index": base["index"],
                "request_id": alternate["request"].get("id"),
                "clean_checkpoint": clean["manifest"]["pins"]["checkpoint"]["weightsSha256"],
                "alternate_checkpoint": changed["manifest"]["pins"]["checkpoint"]["weightsSha256"],
                "clean_probabilities": unkeyed(left, keys),
                "alternate_probabilities": unkeyed(right, keys),
                "max_probability_difference": difference,
                "tolerance": tolerance,
                "choice_changed": base["model_choice"] != alternate["model_choice"],
            }
    return None


def _guarded(request: dict[str, Any], raw: str, keys: list[str]) -> dict[str, dict[str, float]]:
    """{question: probabilities as returned} of one guarded reply (one or several questions)."""
    if multi(keys):
        return {key: value["probabilities"] for key, value in guard_decision(request, raw, keys)["questions"].items()}
    return {keys[0]: guard_response(request, raw, keys[0])["probabilities"]}


def reask_evidence(
    record: dict[str, Any], changed: dict[str, Any], tolerance: float, keys: list[str] | None = None
) -> dict[str, Any] | None:
    keys = keys or ["action"]
    by_index = {row["index"]: row for row in changed["rows"]}
    for item in record["rows"]:
        row = by_index[item["index"]]
        if not same_request(item["request"], row["request"]):
            raise ValueError(f"the re-ask record of {changed['name']} does not hold the logged request")
        left = {
            key: normalized(value) for key, value in _guarded(row["request"], item["clean_reply_raw"], keys).items()
        }
        right = _normalized_rows(row["probabilities"], keys)
        difference = max_question_difference(left, right)
        if difference > tolerance:
            return {
                "method": "same_request_reask",
                "index": row["index"],
                "native_tick": row["native_before"]["tick"],
                "clean_run": None,
                "request_id": row["request"].get("id"),
                "clean_checkpoint": record["clean_checkpoint"]["weightsSha256"],
                "alternate_checkpoint": changed["manifest"]["pins"]["checkpoint"]["weightsSha256"],
                "clean_probabilities": unkeyed(left, keys),
                "alternate_probabilities": unkeyed(right, keys),
                "max_probability_difference": difference,
                "tolerance": tolerance,
                "choice_changed": unkeyed(_argmax_map(left), keys) != row["model_choice"],
                "reask_ref": "weights-reask.jsonl",
            }
    return None


# ------------------------------------------------------------------------------------------- reask
def reask(root: Path, out: Path, endpoint: str, decisions: int | None) -> None:
    plan = read_json(root / "collection.json")
    limit = decisions or plan["probe"]["first_decisions"]
    tolerance = plan["weight_difference_tolerance"]
    client = ServiceClient(endpoint)
    start = client.checkpoint()
    if start["weightsSha256"] != plan["clean_weights_sha256"]:
        raise ValueError("the re-ask service does not serve the plan's clean checkpoint")
    with out.open("x", encoding="utf-8") as stream:
        for path in runs_of(root):
            report = check_run(path, plan)
            if report["status"] != "valid" or report["summary"]["condition"] != "weights":
                continue
            run = load_run(path)
            items = []
            keys = plan_question_keys(plan)
            for row in model_rows(run["rows"])[:limit]:
                raw, seconds = client.decide_raw(row["request"])
                text = raw.decode("utf-8")
                parsed = _guarded(row["request"], text, keys)
                difference = max_question_difference(
                    {key: normalized(value) for key, value in parsed.items()},
                    _normalized_rows(row["probabilities"], keys),
                )
                items.append(
                    {
                        "index": row["index"],
                        "request": row["request"],
                        "clean_reply_raw": text,
                        "seconds": seconds,
                        "max_probability_difference": difference,
                    }
                )
                if difference > tolerance:
                    break
            portable = {key: value for key, value in start.items() if key not in ("path", "gpuUuid")}
            stream.write(canonical({"run": path.name, "clean_checkpoint": portable, "rows": items}) + "\n")
            stream.flush()
    if client.checkpoint() != start:
        raise ValueError("the clean service's identity changed while re-asking")


# ------------------------------------------------------------------------------------------- audit
def assess(root: Path, plan: dict[str, Any], reasks: dict[str, dict[str, Any]], adapter: Any) -> list[dict[str, Any]]:
    by_slot: dict[tuple[str, str], list[Path]] = {}
    for path in runs_of(root):
        manifest = read_json(path / "manifest.json")
        by_slot.setdefault((manifest["reset_group"], manifest["condition"]), []).append(path)
    known = {group["id"] for group in plan["groups"]}
    attempts: list[dict[str, Any]] = []
    for group_id, condition in sorted(set(by_slot) - {(g, c) for g in known for c in CONDITIONS}):
        for path in by_slot[(group_id, condition)]:
            attempts.append(  # noqa: PERF401
                {
                    "run": path.name,
                    "condition": condition,
                    "scenario_group": group_id,
                    "split": "train",
                    "attempt": None,
                    "annotation_status": "excluded",
                    "selected_for_scoring": False,
                    "exclusion_reason": "reset group is not in the frozen plan",
                }
            )
    keys = plan_question_keys(plan)
    question_key = keys if multi(keys) else keys[0]
    tolerance = plan["weight_difference_tolerance"]
    probed = plan["probe"]["first_decisions"]
    tool_sha = verification_tool_sha256()
    for group in plan["groups"]:
        clean_runs: list[dict[str, Any]] = []
        for condition in CONDITIONS:
            paths = sorted(
                by_slot.get((group["id"], condition), []),
                key=lambda p: read_json(p / "manifest.json")["attempt"]["number"],
            )
            for path in paths:
                report = check_run(path, plan)
                manifest = read_json(path / "manifest.json")
                item: dict[str, Any] = {
                    "run": path.name,
                    "condition": condition,
                    "scenario_group": group["id"],
                    "split": group["split"],
                    "attempt": manifest["attempt"]["number"],
                    "selected_for_scoring": False,
                    "validation": {
                        "status": report["status"],
                        "errors": report["errors"][:10],
                        "warnings": len(report["warnings"]),
                    },
                }
                attempts.append(item)
                if report["status"] != "valid":
                    item.update(
                        annotation_status="excluded",
                        exclusion_reason=f"attempt is {report['status']}"
                        + (f": {report['errors'][0]}" if report["errors"] else ""),
                    )
                    continue
                run = load_run(path)
                result = run["result"]
                item["native_outcome"] = {"status": result["status"], "reason": result["reason"]}
                try:
                    item["replay"] = replay_record(adapter, path, result)
                    if condition == "no_injection":
                        clean_runs.append(run)
                        item["annotation_status"] = "control"
                    else:
                        if condition == "harness":
                            if result["trigger_index"] is None:
                                raise ValueError("the harness fault never changed a request")
                            evidence = harness_evidence(run, question_key)
                        elif condition == "controller":
                            if result["trigger_index"] is None:
                                raise ValueError("the controller fault never changed an execution")
                            evidence = controller_evidence(run)
                        else:
                            evidence = None
                            for clean in clean_runs:
                                evidence = matched_weights_evidence(clean, run, tolerance, keys)
                                if evidence:
                                    break
                            if evidence is None and path.name in reasks:
                                evidence = reask_evidence(reasks[path.name], run, tolerance, keys)
                            if evidence is None:
                                raise ValueError("no identical-request decision differs by more than the tolerance")
                        position = [row["index"] for row in model_rows(run["rows"])].index(evidence["index"])
                        paired = clean_runs[0]["result"] if clean_runs else None
                        item.update(
                            annotation_status="verified",
                            verified_component=condition,
                            fault_instance=manifest["intervention"]["instance"],
                            evidence={
                                **evidence,
                                "model_decision_number": position,
                                "within_probed_decisions": position < probed,
                            },
                            paired_clean=(
                                None
                                if paired is None
                                else {
                                    "run": clean_runs[0]["name"],
                                    "status": paired["status"],
                                    "reason": paired["reason"],
                                }
                            ),
                            outcome_effect="not_established" if paired is not None else "untested",
                            release_level_score_drop="untested",
                        )
                except (ValueError, KeyError, TypeError, ResponseRejected) as error:
                    item.update(annotation_status="excluded", exclusion_reason=str(error))
                    continue
                item["verification_procedure"] = (
                    "contract validation (validate_episode.py), native outcome from recorded native events"
                    + (
                        ", native replay"
                        if item["replay"]["method"] == "native_replay"
                        else (
                            ", strict harness replay on the recorded observations (the game is not replayed)"
                            if item["replay"]["method"] == "harness_replay_on_recorded_observations"
                            else ""
                        )
                    )
                    + ", and the condition-specific evidence rule of audit_set.py"
                )
                item["verification_tool_sha256"] = tool_sha
    return attempts


def replay_record(adapter: Any, path: Path, result: dict[str, Any]) -> dict[str, Any]:
    if adapter is not None and hasattr(adapter, "replay"):
        replayed = adapter.replay(path)
        if replayed.get("status") != result["status"] or not replayed.get("verified"):
            raise ValueError("native replay differs from the recorded outcome")
        # 0.2.0: an adapter may name its own procedure, e.g. the full HK's strict harness replay on the recorded
        # observations (the game itself cannot be replayed); without a name it is a native replay (Griddly)
        return {
            "verified": True,
            "status": result["status"],
            **replayed,
            "method": replayed.get("method", "native_replay"),
        }
    return {
        "method": "contract_record",
        "verified": True,
        "status": result["status"],
        "terminal_tick": result["terminal_tick"],
        "native_ticks": result["native_ticks"],
        "wall_seconds": result["wall_seconds"],
    }


def choose(attempts: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Griddly's selection, with one candidate per slot (its first non-excluded attempt)."""
    seen: set[tuple[str, str]] = set()
    candidates_by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for row in attempts:
        if row["annotation_status"] == "excluded":
            continue
        slot = (row["scenario_group"], row["condition"])
        if slot in seen:
            row["unselected_reason"] = "a later attempt of a slot that already has an eligible attempt"
            continue
        seen.add(slot)
        candidates_by_condition[row["condition"]].append(row)
    selected = []
    for condition in CONDITIONS:
        candidates = candidates_by_condition[condition]
        if len(candidates) < count:
            continue
        win = [row["native_outcome"]["status"] == "win" for row in candidates]
        if condition == "no_injection":
            chosen = candidates[:count]
            if all(flag == win[0] for flag in win[:count]):
                swap = next(
                    (row for row, flag in zip(candidates[count:], win[count:]) if flag != win[0]), None  # noqa: B905
                )
                if swap is not None:
                    chosen[-1] = swap
        else:
            chosen = sorted(candidates, key=lambda row: row["native_outcome"]["status"] == "win")[:count]
        selected.extend(chosen)
    return selected


def index_row(root: Path, row: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    manifest = read_json(root / row["run"] / "manifest.json")
    result = read_json(root / row["run"] / "result.json")
    pins = manifest["pins"]
    intervention = manifest["intervention"]
    entry = {
        "episode_id": row["episode_id"],
        "run": row["run"],
        "game_id": manifest["game_id"],
        "condition": row["condition"],
        "scenario_group": row["scenario_group"],
        "split": row["split"],
        "attempt": row["attempt"],
        "identity": {
            "code": pins["code"],
            "game_build": pins["game_build"],
            "scenario": manifest["scenario"],
            "selection": manifest["selection"],
            "weights_sha256": pins["checkpoint"]["weightsSha256"],
            "checkpoint": {k: pins["checkpoint"].get(k) for k in ("repo", "revision", "role")},
            "harness": pins["harness"],
            "controller": pins["controller"],
            "observation": pins["observation"],
            "contract_version": manifest["contract_version"],
        },
        "injection_layer": row["condition"],
        "fault_family": intervention["family"] if intervention else None,
        "fault_instance": intervention["instance"] if intervention else None,
        "changed_artifact": intervention["changed_artifact"] if intervention else None,
        "intervention": (
            None
            if intervention is None
            else {
                **intervention,
                "trigger_index": (
                    row["evidence"]["index"] if row["condition"] == "weights" else result["trigger_index"]
                ),
            }
        ),
        "annotation_status": row["annotation_status"],
        "annotation_provenance": {
            "verifier": "exp0 common audit_set.py (automated, contract evidence rules)",
            "revisions": [],
            "disagreements": [],
        },
        "verification_procedure": row["verification_procedure"],
        "verification_tool_sha256": row["verification_tool_sha256"],
        "reference_actions": {"status": "unavailable"},
        "outcome_effect": row.get("outcome_effect", "not_applicable"),
        "native_outcome": row["native_outcome"],
        "replay": row["replay"],
    }
    if row["annotation_status"] == "verified":
        entry.update(
            verified_component=row["verified_component"],
            evidence=row["evidence"],
            paired_clean=row["paired_clean"],
            release_level_score_drop=row["release_level_score_drop"],
        )
    return entry


def audit(root: Path, out: Path, reask_path: Path | None, adapter_spec: str | None) -> int:
    plan = read_json(root / "collection.json")
    reasks = {record["run"]: record for record in read_rows(reask_path)} if reask_path else {}
    adapter = None
    if adapter_spec:
        from adapter import load_adapter

        adapter = load_adapter(adapter_spec)
    attempts = assess(root, plan, reasks, adapter)
    selected = choose(attempts, plan["per_condition"])
    chosen = {row["run"] for row in selected}
    for row in attempts:
        row["selected_for_scoring"] = row["run"] in chosen
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out / "attempts.jsonl", attempts)
    counts = {
        condition: sum(r["annotation_status"] != "excluded" for r in attempts if r["condition"] == condition)
        for condition in CONDITIONS
    }
    print(canonical({"eligible": counts, "attempts": len(attempts)}), flush=True)
    if len(selected) != len(CONDITIONS) * plan["per_condition"]:
        print("Verified attempts do not yet cover each condition; see attempts.jsonl.", file=sys.stderr)
        return 1
    random.Random(plan["shuffle_seed"]).shuffle(selected)
    for number, row in enumerate(selected, start=1):
        row["episode_id"] = f"{plan['neutral_id_prefix']}{number:03d}"
    write_json(out / "episodes.json", [index_row(root, row, plan) for row in selected])
    outside = sum(1 for row in selected if row.get("evidence") and not row["evidence"]["within_probed_decisions"])
    print(canonical({"selected": len(selected), "verified_evidence_outside_probed_decisions": outside}), flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    ask = commands.add_parser("reask")
    ask.add_argument("root", type=Path)
    ask.add_argument("out", type=Path)
    ask.add_argument("--endpoint", required=True)
    ask.add_argument("--decisions", type=int)
    run = commands.add_parser("audit")
    run.add_argument("root", type=Path)
    run.add_argument("out", type=Path)
    run.add_argument("--reask", type=Path)
    run.add_argument("--adapter")
    args = parser.parse_args()
    if args.command == "reask":
        reask(args.root, args.out, args.endpoint, args.decisions)
    else:
        sys.exit(audit(args.root, args.out, args.reask, args.adapter))


if __name__ == "__main__":
    main()
