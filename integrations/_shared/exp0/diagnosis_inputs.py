"""Make label-free Exp 0 v0 diagnosis inputs (``read`` mode) with the shared #168 decision-trace renderer.

  python3 diagnosis_inputs.py ROOT INDEX OUT.jsonl [--max-tokens N]

ROOT is the collection root, INDEX the evaluator ``episodes.json``; OUT gets one ``{"episode_id", "text"}``
line per scored episode, in index order. Game-agnostic port of Griddly v0 ``diagnosis_inputs.py``.

Explicit allowlist. Of each attempt only these reach the text:
  - the plan's ``diagnosis.preamble`` (game-neutral description of the time unit and the options);
  - per model decision: the request's rendered parts, plus one more state section "offered options" with the
    question's instructions and every option's text (as the model saw them; the #168 renderer alone prints only
    the first decision's instructions and the option keys, and HK-style harnesses put state into option texts),
    the option probabilities (normalized) and the model's choice;
  - per model decision, as the renderer's ``constraints``: the ``public_observation`` (what a human player
    sees; cut at ``public_observation_max_chars`` with a visible mark);
  - per model decision, as ``applied``: the executed option and a summary of its native windows: runs of
    the same native action with window and tick counts, the first ``windows.max_runs_listed`` runs in order,
    then per-action totals;
  - events: "First model decision." (key, priority 0), selected-vs-executed differences (key, 0),
    requested-vs-sent window differences (key, 0; the first ``windows.max_mismatches_listed`` runs, then a
    count), forced rows ("only X was available"), auto rows (native windows run without a
    decision), native events with ``visible: true``, "Native episode ended." (key, 0);
  - the native outcome: status and reason.
Never: run names, file paths, condition, fault fields, ``clean_request``, checkpoint ids or roles,
fingerprints, window ``reason``, ``extra``, invisible events, internal native state. The token budget
(9,000 estimated tokens as Griddly) is the renderer's; it keeps the key events (priority 0) and drops the
decisions farthest from them first, then state fields, and cuts as a last resort. A final check refuses a
text that contains a run name, a fault family or instance name, a label field name or a 64-hex digest.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream
from contract import canonical, format_choice, load_run, multi, plan_fault_instances, plan_question_keys, read_json

LABEL_FIELDS = (
    "verified_component",
    "annotation_status",
    "injection_layer",
    "clean_request",
    "intended_action",
    "no_injection",
    "fault_family",
    "fault_instance",
    "weightsSha256",
)
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
DEFAULT_RUNS, DEFAULT_MISMATCHES = 12, 6


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def windows_summary(windows: list[dict[str, Any]], max_runs: int) -> str:
    """Runs of the same sent native action, then totals per action (requested-vs-sent pairs are events)."""
    runs: list[list[Any]] = []
    for window in windows:
        if runs and runs[-1][0] == window["native_action"]:
            runs[-1][1] += 1
            runs[-1][2] += window["ticks"]
        else:
            runs.append([window["native_action"], 1, window["ticks"]])
    if not windows:
        return "no native input was sent"
    if len(windows) == 1:
        return f"{windows[0]['native_action']} for {_plural(windows[0]['ticks'], 'tick')}"
    shown = [f"{action} x{count} ({_plural(ticks, 'tick')})" for action, count, ticks in runs[:max_runs]]
    text = ", ".join(shown)
    if len(runs) > max_runs:
        text += f", then {_plural(len(runs) - max_runs, 'more run')}"
    totals: Counter = Counter()
    counts: Counter = Counter()
    for window in windows:
        totals[window["native_action"]] += window["ticks"]
        counts[window["native_action"]] += 1
    total_text = ", ".join(
        f"{action} {_plural(counts[action], 'window')} {_plural(totals[action], 'tick')}"
        for action in sorted(totals, key=lambda a: (-totals[a], a))
    )
    return f"{_plural(len(windows), 'native window')} in {_plural(len(runs), 'run')}: {text}; totals: {total_text}"


def _time(row: dict[str, Any], unit: str) -> float:
    if unit == "native_tick":
        return float(row["native_before"]["tick"])
    value = row["native_before"].get("sim_seconds")
    if value is None:
        raise ValueError("the plan's time unit is sim_seconds but a row has no native_before.sim_seconds")
    return float(value)


def _tick_time(tick: int, row: dict[str, Any], unit: str) -> float:
    """An event's renderer time: its tick, or the row's sim time scaled by its tick offset when in seconds."""
    if unit == "native_tick":
        return float(tick)
    before = row["native_before"]
    after = row.get("native_after") or before
    span = (after["tick"] - before["tick"]) or 1
    seconds = (after.get("sim_seconds") or before["sim_seconds"]) - before["sim_seconds"]
    return float(before["sim_seconds"] + seconds * (tick - before["tick"]) / span)


def mismatch_events(core: Any, row: dict[str, Any], unit: str, max_listed: int) -> list[Any]:
    executed = row["executed"] or {}
    events = []
    tick = row["native_before"]["tick"]
    if row["kind"] in ("model", "forced") and executed and executed.get("option") != row["model_choice"]:
        events.append(
            core.TraceEvent(
                _tick_time(tick, row, unit),
                f"Selected {format_choice(row['model_choice'])}; executed "
                f"{format_choice(executed['option'])} at native tick {tick}.",
                "selected and executed differ",
                0,
            )
        )
    runs: list[list[Any]] = []
    for window in executed.get("windows", []):
        if "intended_action" not in window:
            continue
        key = (window["intended_action"], window["native_action"])
        if runs and runs[-1][0] == key and runs[-1][3] == window["start_tick"]:
            runs[-1][1] += 1
            runs[-1][3] = window["start_tick"] + window["ticks"]
        else:
            runs.append([key, 1, window["start_tick"], window["start_tick"] + window["ticks"]])
    for (intended, sent), count, start, _end in runs[:max_listed]:
        events.append(
            core.TraceEvent(
                _tick_time(start, row, unit),
                f"Requested {intended}; sent {sent} in {_plural(count, 'window')} from native tick {start}.",
                "requested and sent input differ",
                0,
            )
        )
    if len(runs) > max_listed:
        hidden = sum(run[1] for run in runs[max_listed:])
        events.append(
            core.TraceEvent(
                _tick_time(runs[max_listed][2], row, unit),
                f"Requested and sent input also differ in {_plural(hidden, 'more window')}.",
            )
        )
    # 0.2.0: a window whose length the controller changed (the same keys held longer or shorter)
    longer = [window for window in executed.get("windows", []) if "intended_ticks" in window]
    for window in longer[:max_listed]:
        events.append(  # noqa: PERF401
            core.TraceEvent(
                _tick_time(window["start_tick"], row, unit),
                f"Requested {window.get('intended_action', window['native_action'])} for "
                f"{_plural(window['intended_ticks'], 'tick')}; sent {window['native_action']} for "
                f"{_plural(window['ticks'], 'tick')} from native tick {window['start_tick']}.",
                "requested and sent input differ",
                0,
            )
        )
    if len(longer) > max_listed:
        events.append(
            core.TraceEvent(
                _tick_time(longer[max_listed]["start_tick"], row, unit),
                f"Requested and sent window lengths also differ in "
                f"{_plural(len(longer) - max_listed, 'more window')}.",
            )
        )
    return events


def episode_traces(run_dir: Path, episode_id: str, plan: dict[str, Any]) -> list[Any]:
    core = upstream.core()
    run = load_run(run_dir)
    rows, result = run["rows"], run["result"]
    if result is None or run["error"] is not None:
        raise ValueError("diagnosis inputs are made from complete attempts only")
    settings = plan["diagnosis"]
    unit = settings["time"]["unit"]
    max_runs = settings.get("windows", {}).get("max_runs_listed", DEFAULT_RUNS)
    max_mismatches = settings.get("windows", {}).get("max_mismatches_listed", DEFAULT_MISMATCHES)
    max_chars = settings.get("public_observation_max_chars")
    keys = plan_question_keys(plan)
    question_key = keys if multi(keys) else keys[0]
    outcome = f"{result['status']} ({result['reason']})"
    events: list[Any] = []
    records: list[dict[str, Any]] = []
    for row in rows:
        executed = row["executed"] or {"option": None, "windows": []}
        tick = row["native_before"]["tick"]
        if row["kind"] == "model":
            if records:
                records[-1]["events"] = events
                events = []
            else:
                events.append(core.TraceEvent(_time(row, unit), "First model decision.", "episode start", 0))
            observation = canonical(row["public_observation"])
            if max_chars and len(observation) > max_chars:
                observation = observation[:max_chars] + f" [public observation cut at {max_chars} characters]"
            records.append(
                {
                    "row": row,
                    "time": _time(row, unit),
                    "constraints": "public observation " + observation,
                    "applied": (
                        (
                            f"{'options' if isinstance(executed['option'], dict) else 'option'} "
                            f"{format_choice(executed['option'])}; {windows_summary(executed['windows'], max_runs)}"
                        )
                        if executed["option"] is not None or executed["windows"]
                        else None
                    ),
                }
            )
        elif row["kind"] == "forced":
            events.append(
                core.TraceEvent(
                    _tick_time(tick, row, unit),
                    f"Native tick {tick}: only {format_choice(row['model_choice'])} was available; "
                    f"executed {format_choice(executed['option'])}: "
                    f"{windows_summary(executed['windows'], max_runs)}.",
                )
            )
        else:
            events.append(
                core.TraceEvent(
                    _tick_time(tick, row, unit),
                    # 2026-11-26: neutral wording; "the harness ran ..." named an answer class in
                    # inputs whose auto rows follow deaths (VVVVVV), i.e. depend on the condition
                    f"Native tick {tick}: native input ran without a model decision: "
                    f"{windows_summary(executed['windows'], max_runs)}.",
                )
            )
        events.extend(mismatch_events(core, row, unit, max_mismatches))
        for event in row["events"]:
            if event["visible"]:
                events.append(
                    core.TraceEvent(
                        _tick_time(event["tick"], row, unit), f"Native tick {event['tick']}: {event['text']}"
                    )
                )
    if not records:
        raise ValueError("diagnosis input has no model decision")
    last = records[-1]["row"]
    end_time = (
        _tick_time(result["terminal_tick"], last, unit)
        if unit == "native_tick"
        else float((last.get("native_after") or last["native_before"]).get("sim_seconds") or records[-1]["time"])
    )
    events.append(core.TraceEvent(max(end_time, records[-1]["time"]), "Native episode ended.", "episode end", 0))
    records[-1]["events"] = events
    traces = []
    for step, record in enumerate(records):
        ordered = tuple(sorted(record.get("events", ()), key=lambda event: event.time_seconds))
        traces.append(
            upstream.decision_trace(
                episode_id,
                step,
                record["time"],
                record["row"],
                question_key,
                events=ordered,
                constraints=record["constraints"],
                applied=record["applied"],
                outcome=outcome,
                show_options=True,
            )
        )
    return traces


def forbidden_markers(index: list[dict[str, Any]], plan: dict[str, Any]) -> list[str]:
    markers = set(LABEL_FIELDS)
    # Fault instance ids are specific; family names are generic descriptions (Griddly's weights family is
    # "alternate_checkpoint", which the probe inputs' stable key names contain by design), so only instances.
    for row in index:
        markers.add(row["run"])
        if row.get("fault_instance"):
            markers.add(row["fault_instance"])
        repo = (row.get("identity", {}).get("checkpoint") or {}).get("repo")
        if repo:
            markers.add(repo)
    markers.update(plan_fault_instances(plan))
    return sorted(marker for marker in markers if marker)


def check_isolation(text: str, markers: list[str]) -> None:
    found = [marker for marker in markers if marker in text]
    if found or HEX64.search(text):
        raise ValueError(f"evaluator metadata would reach the diagnoser: {found or 'a 64-hex digest'}")


def input_record(run_dir: Path, episode_id: str, plan: dict[str, Any], max_tokens: int) -> dict[str, str]:
    renderer = upstream.traces()
    time = plan["diagnosis"]["time"]
    text = (
        plan["diagnosis"]["preamble"].strip()
        + "\n"
        + renderer.render_traces(
            episode_traces(run_dir, episode_id, plan),
            max_tokens=max_tokens,
            before_seconds=time["before"],
            after_seconds=time["after"],
        )
    )
    return {"episode_id": episode_id, "text": text}


def build(root: Path, index_path: Path, out: Path, max_tokens: int | None = None) -> None:
    plan = read_json(root / "collection.json")
    index = read_json(index_path)
    if len({row["episode_id"] for row in index}) != len(index):
        raise ValueError("duplicate episode ID in evaluator index")
    pattern = re.compile(rf"{plan['neutral_id_prefix']}[0-9]{{3}}")
    markers = forbidden_markers(index, plan)
    budget = max_tokens or plan["diagnosis"]["max_tokens"]
    with out.open("x", encoding="utf-8") as stream:
        for row in index:
            if not pattern.fullmatch(row["episode_id"]):
                raise ValueError("episode ID must be neutral")
            source = (root / row["run"]).resolve()
            if source.parent != root.resolve():
                raise ValueError("episode path leaves the collection root")
            record = input_record(source, row["episode_id"], plan, budget)
            check_isolation(record["text"], markers)
            stream.write(canonical(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()
    build(args.root, args.index, args.out, args.max_tokens)


if __name__ == "__main__":
    main()
