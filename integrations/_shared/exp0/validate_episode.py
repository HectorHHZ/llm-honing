"""Validate Exp 0 v0 episode attempts against the contract (CONTRACT.md), one run directory or a whole collection.

Usage:
  python3 validate_episode.py RUN_OR_COLLECTION [--plan collection.json] [--json]

A collection is a directory whose subdirectories with a ``manifest.json`` are attempts. With ``--plan``
(default: ``collection.json`` in the collection root when it exists) the attempt is also checked against
the frozen collection plan. Each attempt gets one status:

- ``valid``: complete (``result.json``, no ``error.json``) and consistent with the contract;
- ``incomplete``: ``error.json`` is present (a failed or interrupted attempt; never scored, always kept);
- ``invalid``: a contract violation (listed in ``errors``).

``triggered`` says whether a harness or controller fault changed anything (an untriggered intervention
attempt is valid but cannot be a verified example). Exit status 1 when any attempt is invalid.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import (
    COMPONENTS,
    CONTRACT_VERSION,
    FORMAT,
    ResponseRejected,
    Sampler,
    canonical,
    guard_decision,
    guard_response,
    keyed,
    load_run,
    multi,
    plan_faults,
    plan_question_keys,
    question_options,
    read_json,
    request_options,
    request_question_keys,
    same_request,
    schema,
)
from schema_lite import errors as schema_errors

COMPATIBLE_VERSIONS = {"0.1", "0.2"}
GENERIC_LEAK_WORDS = ("inject", "fault", "alternate checkpoint", "clean checkpoint", "intervention")


class Report:
    def __init__(self, run: str) -> None:
        self.run = run
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.summary: dict[str, Any] = {}

    def error(self, message: str) -> None:
        if len(self.errors) < 200:
            self.errors.append(message)

    def warn(self, message: str) -> None:
        if len(self.warnings) < 200:
            self.warnings.append(message)

    def schema(self, instance: Any, name: str, where: str) -> bool:
        found = schema_errors(instance, SCHEMAS[name])
        for message in found[:10]:
            self.error(f"{where}: {message}")
        return not found


SCHEMAS = {name: schema(name) for name in ("manifest", "decision", "result", "error")}


def _leak_terms(manifest: dict[str, Any], plan: dict[str, Any] | None) -> list[str]:
    """Strings that would reveal the label if they reached the model or the diagnoser."""
    terms = set()
    intervention = manifest.get("intervention") or {}
    for key in ("family", "instance"):
        if intervention.get(key):
            terms.add(intervention[key])
    if plan:
        for component in COMPONENTS:
            for fault in plan_faults(plan, component):
                terms.update(value for key, value in fault.items() if key in ("family", "instance") and value)
        terms.update((plan.get("clean_weights_sha256", ""), plan.get("alternate_weights_sha256", "")))
    terms.add((manifest.get("pins", {}).get("checkpoint") or {}).get("weightsSha256", ""))
    terms.add(manifest.get("clean_weights_sha256", ""))
    return sorted(term for term in terms if term and len(term) >= 4)


def _model_text(row: dict[str, Any]) -> str:
    """Everything of a row that reaches the decision model or the diagnoser."""
    windows = (row.get("executed") or {}).get("windows", [])
    return canonical(
        [
            row.get("request"),
            row.get("public_observation"),
            [window.get("native_action") for window in windows],
            [event.get("text") for event in row.get("events", [])],
        ]
    )


def check_run(run_dir: Path, plan: dict[str, Any] | None) -> dict[str, Any]:
    report = Report(run_dir.name)
    run = load_run(run_dir)
    manifest, rows, result, error = run["manifest"], run["rows"], run["result"], run["error"]
    if manifest is None:
        report.error("manifest.json is missing")
        return _finish(report, "invalid")
    if not report.schema(manifest, "manifest", "manifest.json"):
        return _finish(report, "invalid")
    if manifest["format"] != FORMAT:
        report.error(f"format {manifest['format']!r} is not {FORMAT!r}")
    major_minor = CONTRACT_VERSION.rsplit(".", 1)[0]  # noqa: F841
    # 0.2.0 only adds forms (several questions, window lengths, several fault instances): 0.1.x attempts stay valid
    if manifest["contract_version"].rsplit(".", 1)[0] not in COMPATIBLE_VERSIONS:
        report.error(
            f"contract_version {manifest['contract_version']} is not one of {sorted(COMPATIBLE_VERSIONS)} (.x)"
        )
    if manifest["run"] != run_dir.name:
        report.error(f"manifest run {manifest['run']!r} differs from the directory name {run_dir.name!r}")
    condition = manifest["condition"]
    intervention = manifest["intervention"]
    if (intervention is None) != (condition == "no_injection"):
        report.error("an intervention is recorded exactly when the condition is not no_injection")
    elif intervention is not None and intervention["component"] != condition:
        report.error("the intervention component differs from the condition")
    checkpoint = manifest["pins"]["checkpoint"]
    clean_sha = manifest["clean_weights_sha256"]
    if condition == "weights":
        if checkpoint["role"] != "alternate" or checkpoint["weightsSha256"] == clean_sha:
            report.error("a weights episode must be served by a distinct alternate checkpoint")
    elif checkpoint["role"] != "clean" or checkpoint["weightsSha256"] != clean_sha:
        report.error(f"a {condition} episode must be served by the clean checkpoint")
    if manifest["attempt"]["slot"] != f"{manifest['reset_group']}/{condition}":
        report.error("attempt.slot must be '<reset_group>/<condition>'")
    if plan is not None:
        _check_plan(report, manifest, plan)
    # 0.2.0: the plan names the questions of every request (one or several); without a plan each request's own
    question_keys = plan_question_keys(plan) if plan else None

    if error is not None:
        report.schema(error, "error", "error.json")
    elif result is None:
        report.error("result.json is missing (and there is no error.json)")
    if not rows and error is None:
        report.error("decisions.jsonl is missing or empty")

    leak_terms = _leak_terms(manifest, plan)
    sampler = Sampler(manifest["selection"]["seed"]) if manifest["selection"]["rule"] == "sample" else None
    states = run["public_states"]
    counts = Counter()
    events = Counter()
    changed_rows: list[int] = []
    previous_end = None
    windows_total = 0
    for position, row in enumerate(rows):
        where = f"decisions.jsonl row {position}"
        if not report.schema(row, "decision", where):
            continue
        if row["index"] != position:
            report.error(f"{where}: index {row['index']} is not its position")
        counts[row["kind"]] += 1
        before = row["native_before"]["tick"]
        if previous_end is not None and before < previous_end:
            report.error(f"{where}: native_before.tick {before} precedes the previous row's end {previous_end}")
        executed = row["executed"]
        last_row = position == len(rows) - 1
        if executed is None:
            if not (last_row and error is not None):
                report.error(f"{where}: only the last row of a failed attempt may be unexecuted")
        else:
            end = _check_windows(report, where, row, before)
            windows_total += len(executed["windows"])
            if manifest["shape"] == "native" and row["kind"] != "auto" and len(executed["windows"]) != 1:
                report.error(f"{where}: a native-shape decision executes exactly one window")
            if row["kind"] == "auto" and not executed["windows"]:
                report.error(f"{where}: an auto row records the native windows it ran")
            after = row["native_after"]
            if after is None:
                report.error(f"{where}: an executed row needs native_after")
            else:
                if after["tick"] < end:
                    report.error(f"{where}: native_after.tick {after['tick']} precedes the last window's end {end}")
                previous_end = after["tick"]
        if row["public_state"] is not None:
            if states is None or row["public_state"] >= len(states):
                report.error(f"{where}: public_state {row['public_state']} is not a line of public_states.jsonl")
            elif states[row["public_state"]].get("index") != row["index"]:
                report.error(f"{where}: public_states.jsonl line {row['public_state']} belongs to another row")
        elif row["kind"] == "model":
            report.warn(f"{where}: no saved public state, so this decision cannot be re-rendered for probes")
        for event in row["events"]:
            events[event["kind"]] += 1
            if event["tick"] < before:
                report.error(f"{where}: an event at tick {event['tick']} precedes the decision")
        text = _model_text(row)
        for term in leak_terms:
            if term in text:
                report.error(f"{where}: model- or diagnoser-facing text contains {term!r}")
        lowered = text.lower()
        for word in GENERIC_LEAK_WORDS:
            if word in lowered:
                report.warn(f"{where}: model- or diagnoser-facing text contains the word {word!r}")
        for window in (executed or {}).get("windows", []):
            for term in leak_terms:
                if term in window.get("reason", ""):
                    report.error(f"{where}: a window reason contains {term!r}; fault code must not write reasons")
        if row["kind"] == "model":
            request_id = str((row["request"] or {}).get("id", ""))
            if run_dir.name in request_id or (condition != "no_injection" and condition in request_id.lower()):
                report.error(f"{where}: the request id {request_id!r} names the run or its condition")
            _check_model_row(report, where, run_dir, row, question_keys, sampler)
            if row["request"] is not None:
                counts["questions"] += len((row["request"] or {}).get("questions") or {})
        changed = _check_fault(report, where, row, condition)
        if changed:
            changed_rows.append(row["index"])

    if rows and counts["model"] == 0 and error is None:
        report.error("the episode has no model decision")
    triggered = bool(changed_rows) if condition in ("harness", "controller") else None
    report.summary = {
        "condition": condition,
        "reset_group": manifest["reset_group"],
        "attempt": manifest["attempt"]["number"],
        "rows": len(rows),
        "model_decisions": counts["model"],
        "windows": windows_total,
        "questions": counts["questions"],
        "triggered": triggered,
        "trigger_index": changed_rows[0] if changed_rows else None,
    }
    if result is not None and error is None:  # noqa: SIM102
        if report.schema(result, "result", "result.json"):
            _check_result(
                report,
                result,
                rows,
                counts,
                events,
                windows_total,
                changed_rows,
                checkpoint,
                plan,
                manifest["scenario"]["caps"],
            )
            report.summary.update(
                status=result["status"],
                reason=result["reason"],
                wall_seconds=result["wall_seconds"],
                native_ticks=result["native_ticks"],
            )
    if report.errors:
        return _finish(report, "invalid")
    return _finish(report, "incomplete" if error is not None else "valid")


def _check_plan(report: Report, manifest: dict[str, Any], plan: dict[str, Any]) -> None:
    for key, expected in (
        ("game_id", plan["game_id"]),
        ("shape", plan["shape"]),
        ("clean_weights_sha256", plan["clean_weights_sha256"]),
    ):
        if manifest[key] != expected:
            report.error(f"manifest {key} {manifest[key]!r} differs from the plan's {expected!r}")
    if manifest["pins"]["observation"]["condition"] != plan["observation"]:
        report.error("the observation condition differs from the plan")
    expected = dict(plan["selection"])
    path = expected.pop("per_group_seed", None)
    if path:  # the plan names where each reset group keeps its own policy seed, e.g. "scenario.policy_seed"
        value: Any = next((g for g in plan["groups"] if g["id"] == manifest["reset_group"]), None)
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        expected["seed"] = value
    actual = {key: value for key, value in manifest["selection"].items() if key != "per_group_seed"}
    if actual != expected:
        report.error(f"the selection {actual} differs from the plan's {expected} for this reset group")
    expected_sha = plan["alternate_weights_sha256" if manifest["condition"] == "weights" else "clean_weights_sha256"]
    if manifest["pins"]["checkpoint"]["weightsSha256"] != expected_sha:
        report.error("the serving checkpoint differs from the plan's checkpoint for this condition")
    groups = {group["id"] for group in plan["groups"]}
    if manifest["reset_group"] not in groups:
        report.error(f"reset group {manifest['reset_group']!r} is not in the plan")
    if manifest["attempt"]["number"] > plan["attempt_cap_per_slot"]:
        report.error("the attempt number exceeds the plan's attempt cap per slot")
    intervention = manifest["intervention"]
    if intervention is not None:
        # 0.2.0: a component may list several validated instances; the attempt's must be one of them
        definitions = plan_faults(plan, intervention["component"])
        fault = next((item for item in definitions if item["instance"] == intervention["instance"]), None)
        if fault is None:
            report.error(f"intervention instance {intervention['instance']!r} is not a fault definition of the plan")
            return
        for key in ("family", "instance", "trigger_rule", "changed_artifact"):
            if intervention[key] != fault[key]:
                report.error(f"intervention {key} differs from the plan's fault definition")


def _check_windows(report: Report, where: str, row: dict[str, Any], before: int) -> int:
    end = before
    for number, window in enumerate(row["executed"]["windows"]):
        if window["start_tick"] < end:
            report.error(f"{where}: window {number} starts at {window['start_tick']}, before tick {end}")
        end = window["start_tick"] + window["ticks"]
        if "intended_action" in window and window["intended_action"] == window["native_action"]:
            report.error(f"{where}: window {number} records an intended action equal to the sent one")
        if "intended_ticks" in window and window["intended_ticks"] == window["ticks"]:
            report.error(f"{where}: window {number} records an intended length equal to the sent one")
    return end


def _check_model_row(
    report: Report,
    where: str,
    run_dir: Path,
    row: dict[str, Any],
    question_keys: list[str] | None,
    sampler: Sampler | None,
) -> None:
    request = row["request"]
    try:
        keys = question_keys or request_question_keys(request)
        if multi(keys):  # noqa: SIM108
            options = question_options(request, keys)
        else:
            options = {keys[0]: request_options(request, keys[0])}
    except ResponseRejected as error:
        report.error(f"{where}: {error}")
        return
    shape_ok = True
    for field in ("probabilities", "server_choice", "model_choice"):
        value = row[field]
        if value is None:
            continue
        if multi(keys) and (not isinstance(value, dict) or list(value) != keys):
            report.error(f"{where}: with several questions {field} is keyed by question in request order {keys}")
            shape_ok = False
        elif not multi(keys) and field != "probabilities" and not isinstance(value, str):
            report.error(f"{where}: with one question {field} is the option itself")
            shape_ok = False
    if row["executed"] is not None and row["executed"]["option"] is not None:
        value = row["executed"]["option"]
        if multi(keys) and (not isinstance(value, dict) or list(value) != keys):
            report.error(f"{where}: with several questions executed.option is keyed by question in request order")
            shape_ok = False
        elif not multi(keys) and not isinstance(value, str):
            report.error(f"{where}: with one question executed.option is the option itself")
            shape_ok = False
    if not shape_ok:
        return
    tries = row["response"]["tries"]
    accepted = [number for number, attempt in enumerate(tries) if attempt["status"] == "accepted"]
    executed = row["executed"] is not None
    if executed and accepted != [len(tries) - 1]:
        report.error(f"{where}: an executed decision has exactly one accepted try, its last")
    if executed and row["model_choice"] is None:
        report.error(f"{where}: an executed model decision records the model's choice")
    if not executed and len(accepted) > 1:
        report.error(f"{where}: more than one accepted try")
    for number, attempt in enumerate(tries):
        path = attempt["file"]
        if path is None:
            if attempt["status"] != "transport_error":
                report.error(f"{where}: try {number} has no raw response file")
            continue
        raw_path = (run_dir / path).resolve()
        if run_dir.resolve() not in raw_path.parents or not raw_path.is_file():
            report.error(f"{where}: raw response file {path} is missing or outside the run")
            continue
        raw = raw_path.read_bytes()
        try:
            if multi(keys):
                parsed = guard_decision(request, raw, keys)["questions"]
                reply = json.loads(raw)
            else:
                single = guard_response(request, raw, keys[0])
                parsed, reply = {keys[0]: single}, single["reply"]
        except ResponseRejected as rejection:
            if attempt["status"] == "accepted":
                report.error(f"{where}: try {number} was executed but fails the guard: {rejection}")
            continue
        if attempt["status"] == "rejected":
            report.warn(f"{where}: try {number} was rejected by the collector but passes the common guard")
        if attempt["status"] == "accepted":
            for key in keys:
                answered = list(reply["answers"][key]["probabilities"])
                if answered != options[key]:
                    report.error(
                        f"{where}: the recorded request offers {options[key]} for {key!r} but the server "
                        f"answered in the order {answered}; rows must keep the request's key order (never "
                        f"write them with sort_keys)"
                    )
            recorded_probabilities = keyed(row["probabilities"], keys) or {}
            recorded_server = keyed(row["server_choice"], keys) or {}
            if any(parsed[key]["probabilities"] != recorded_probabilities.get(key) for key in keys):
                report.error(f"{where}: recorded probabilities differ from the raw response")
            if any(parsed[key]["server_choice"] != recorded_server.get(key) for key in keys):
                report.error(f"{where}: recorded server_choice differs from the raw response")
            if row["response"]["file"] != path:
                report.error(f"{where}: response.file must name the accepted try's file")
            model = keyed(row["model_choice"], keys) or {}
            if sampler is None:
                if any(model.get(key) != parsed[key]["server_choice"] for key in keys):
                    report.error(f"{where}: with argmax selection the model choice is the server's choice")
            else:
                if multi(keys):
                    expected = sampler.choose_all(request, {key: parsed[key]["probabilities"] for key in keys}, keys)
                else:
                    expected = {keys[0]: sampler.choose(request, parsed[keys[0]]["probabilities"], keys[0])}
                if model != expected:
                    report.error(
                        f"{where}: the sampled model choice {row['model_choice']!r} is not the seeded draw {expected!r}"
                    )
    if row["model_choice"] is not None:
        model = keyed(row["model_choice"], keys)
        if any(model[key] not in options[key] for key in keys):
            report.error(f"{where}: model_choice {row['model_choice']!r} is not an offered option")
    if (
        not executed
        and not accepted
        and any(row[key] is not None for key in ("probabilities", "server_choice", "model_choice"))
    ):
        report.error(f"{where}: a decision without an accepted response has no probabilities or choice")
    if executed:
        done = keyed(row["executed"]["option"], keys)
        if done is None or any(done[key] not in options[key] for key in keys):
            report.error(f"{where}: executed option {row['executed']['option']!r} is not an offered option")


def _check_fault(report: Report, where: str, row: dict[str, Any], condition: str) -> bool:
    """Check the row's fault record against the condition; return whether a fault changed this row."""
    fault, clean = row["fault"], row["clean_request"]
    executed = row["executed"] or {}
    mismatched_windows = [w for w in executed.get("windows", []) if "intended_action" in w or "intended_ticks" in w]
    option_changed = row["kind"] in ("model", "forced") and executed and executed.get("option") != row["model_choice"]
    if fault is not None and fault["component"] != condition:
        report.error(f"{where}: a {fault['component']} fault record in a {condition} episode")
        return False
    if fault is not None and row["kind"] == "auto":
        report.error(f"{where}: faults act on model or forced decisions, not on auto rows")
    if condition != "harness" and clean is not None:
        report.error(f"{where}: clean_request is recorded only by the harness fault")
    if condition != "controller" and (option_changed or mismatched_windows):
        report.error(f"{where}: selected and executed actions differ outside a controller-fault episode")
    if condition == "no_injection" and fault is not None:
        report.error(f"{where}: a no_injection episode has no fault record")
    if condition == "harness" and fault is not None:
        if row["kind"] != "model" or clean is None:
            report.error(f"{where}: an applied harness fault records the clean request of a model decision")
            return False
        differs = not same_request(clean, row["request"])
        if fault["changed"] is not differs:
            report.error(f"{where}: fault.changed must say whether the request differs from clean_request")
        return differs
    if condition == "harness" and clean is not None:
        report.error(f"{where}: clean_request without a harness fault record")
    if condition == "controller":
        differs = bool(option_changed or mismatched_windows)
        if fault is None:
            if differs:
                report.error(f"{where}: an execution change without a controller fault record")
            return False
        if fault["changed"] is not differs:
            report.error(f"{where}: fault.changed must say whether the executed action differs from the selection")
        return differs
    if condition == "weights" and fault is not None and fault["changed"] is not None:
        report.error(f"{where}: a weights fault's per-decision effect is unknown at collection (changed: null)")
    return False


def _check_result(
    report: Report,
    result: dict[str, Any],
    rows: list[dict[str, Any]],
    counts: Counter,
    events: Counter,
    windows_total: int,
    changed_rows: list[int],
    checkpoint: dict[str, Any],
    plan: dict[str, Any] | None,
    caps: dict[str, Any],
) -> None:
    expected = {
        "rows": len(rows),
        "model_decisions": counts["model"],
        "forced_decisions": counts["forced"],
        "auto_rows": counts["auto"],
        "windows": windows_total,
        "trigger_index": changed_rows[0] if changed_rows else None,
        "fault_changed_rows": changed_rows,
        "event_counts": dict(sorted(events.items())),
    }
    for key, value in expected.items():
        recorded = result[key]
        if key == "event_counts":
            recorded = dict(sorted((k, v) for k, v in recorded.items() if v))
        if recorded != value:
            report.error(f"result.json {key} {result[key]!r} differs from the decision rows ({value!r})")
    if rows:
        start = rows[0]["native_before"]["tick"]
        if result["native_ticks"] != result["terminal_tick"] - start:
            report.error("result.json native_ticks must equal terminal_tick minus the first row's tick")
        last = rows[-1]["native_after"]
        if last is not None and result["terminal_tick"] < last["tick"]:
            report.error("result.json terminal_tick precedes the last row's end")
    end = result["checkpoint_end"]
    if end["weightsSha256"] != checkpoint["weightsSha256"] or end["role"] != checkpoint["role"]:
        report.error("the decision server's checkpoint changed during the episode")
    if plan is not None and result["reason"] not in plan["outcomes"][result["status"]]:
        report.error(f"outcome reason {result['reason']!r} is not a {result['status']} reason in the plan")
    if caps.get("decisions") and counts["model"] + counts["forced"] > caps["decisions"]:
        report.error(f"the episode exceeds its decision cap {caps['decisions']}")
    if caps.get("native_ticks") and result["native_ticks"] > caps["native_ticks"]:
        report.error(f"the episode exceeds its native tick cap {caps['native_ticks']}")
    if caps.get("wall_seconds") and result["wall_seconds"] > 1.25 * caps["wall_seconds"]:
        report.warn(f"the episode ran {result['wall_seconds']:.0f} s, well over its wall cap {caps['wall_seconds']}")


def _finish(report: Report, status: str) -> dict[str, Any]:
    return {
        "run": report.run,
        "status": status,
        "errors": report.errors,
        "warnings": report.warnings,
        "summary": report.summary,
    }


def runs_of(path: Path) -> list[Path]:
    if (path / "manifest.json").is_file():
        return [path]
    return sorted(child for child in path.iterdir() if (child / "manifest.json").is_file())


def validate(path: Path, plan_path: Path | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    if plan_path is None:
        root = path if not (path / "manifest.json").is_file() else path.parent
        plan_path = root / "collection.json" if (root / "collection.json").is_file() else None
    plan = None
    if plan_path is not None:
        plan = read_json(plan_path)
        problems = schema_errors(plan, schema("collection"))
        if problems:
            raise ValueError(f"the collection plan violates its schema: {problems[:5]}")
    return [check_run(run, plan) for run in runs_of(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--json", action="store_true", help="print the full per-attempt reports as JSON")
    args = parser.parse_args()
    reports = validate(args.path, args.plan)
    if args.json:
        print(json.dumps(reports, indent=2, sort_keys=True))
    else:
        for item in reports:
            summary = item["summary"]
            print(f"{item['run']}: {item['status']} {canonical(summary)}")
            for message in item["errors"][:20]:
                print(f"  error: {message}")
            for message in item["warnings"][:5]:
                print(f"  warning: {message}")
        print(canonical(dict(Counter(item["status"] for item in reports))))
    sys.exit(1 if any(item["status"] == "invalid" for item in reports) else 0)


if __name__ == "__main__":
    main()
