"""Probe the first model decisions of every scored episode under both checkpoints (Griddly v0 ``probes.py``, #170).

python3 probes.py selfcheck ROOT --adapter FILE.py:ATTR [--max-runs N]
    No model call. For every valid attempt, re-render each model decision from its saved public state
    with the game's clean harness and require the logged request (or, where the harness fault acted,
    the logged clean_request). A failure means probes would misreport harness changes.

python3 probes.py capture ROOT INDEX OUT.jsonl --endpoint URL --checkpoint-role clean|alternate --adapter A
    For the first ``probe.first_decisions`` model decisions of every indexed episode, send the exact
    logged request and the standard re-render (only when it differs) to one checkpoint's service; keep
    the raw replies. Checks the service's weights SHA-256 against the plan before and after.

python3 probes.py compare ROOT INDEX OUT_DIR --reference-capture R.jsonl --alternate-capture A.jsonl --adapter A
    Re-derive the requests, check both captures hold exactly them, run the pinned #170 identity check
    (same request + the episode's own checkpoint reproduces the logged probabilities within 0.01, tie
    0.01) and write OUT_DIR/identity.jsonl (private audit) and OUT_DIR/probe-inputs.jsonl (what the
    ``probe`` mode adds for the diagnoser: per decision the normalized probabilities of the same request
    under the "reference" and "alternate" checkpoints, whether the standard re-render changed the request,
    and the re-render's probabilities under both; no labels, run names or checkpoint ids). Exit 1 if
    any identity check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import upstream
from adapter import load_adapter
from contract import (
    ResponseRejected,
    ServiceClient,
    canonical,
    guard_decision,
    guard_response,
    load_run,
    model_rows,
    multi,
    normalized,
    plan_question_keys,
    public_history_entry,
    read_json,
    read_rows,
    same_request,
)
from validate_episode import check_run, runs_of

ROLES = {"clean": "clean_weights_sha256", "alternate": "alternate_weights_sha256"}


def rerender_row(adapter: Any, run: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    if row["public_state"] is None or run["public_states"] is None:
        raise ValueError(f"{run['name']} row {row['index']} has no saved public state to re-render")
    state = run["public_states"][row["public_state"]]["state"]
    history = [public_history_entry(earlier) for earlier in run["rows"][: row["index"]]]
    return adapter.rerender(state, history)


def requests(run_dir: Path, adapter: Any, first: int) -> list[dict[str, Any]]:
    run = load_run(run_dir)
    chosen = []
    for row in model_rows(run["rows"])[:first]:
        chosen.append(  # noqa: PERF401
            {
                "index": row["index"],
                "native_tick": row["native_before"]["tick"],
                "logged": row["request"],
                "rerendered": rerender_row(adapter, run, row),
            }
        )
    if not chosen:
        raise ValueError(f"{run_dir.name} has no model decision to probe")
    return chosen


def selfcheck(root: Path, adapter: Any, max_runs: int | None) -> dict[str, Any]:
    plan = read_json(root / "collection.json")
    checked, mismatches, runs = 0, [], 0
    for path in runs_of(root):
        if check_run(path, plan)["status"] != "valid":
            continue
        runs += 1
        if max_runs is not None and runs > max_runs:
            break
        run = load_run(path)
        for row in model_rows(run["rows"]):
            expected = (
                row["clean_request"] if row["fault"] and row["fault"]["component"] == "harness" else row["request"]
            )
            try:
                rendered = rerender_row(adapter, run, row)
            except Exception as error:
                mismatches.append(
                    {"run": path.name, "index": row["index"], "error": f"{type(error).__name__}: {error}"}
                )
                continue
            checked += 1
            if not same_request(rendered, expected):
                mismatches.append(
                    {"run": path.name, "index": row["index"], "error": _first_difference(rendered, expected)}
                )
    return {
        "runs": min(runs, max_runs) if max_runs else runs,
        "decisions": checked,
        "mismatches": len(mismatches),
        "first_mismatches": mismatches[:10],
        "ok": not mismatches and checked > 0,
    }


def _first_difference(left: Any, right: Any, where: str = "request") -> str:
    if type(left) is not type(right):
        return f"{where}: {type(left).__name__} vs {type(right).__name__}"
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return f"{where}.{key}: present on one side only"
            if left[key] != right[key]:
                return _first_difference(left[key], right[key], f"{where}.{key}")
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{where}: {len(left)} vs {len(right)} items"
        for number, (a, b) in enumerate(zip(left, right)):  # noqa: B905
            if a != b:
                return _first_difference(a, b, f"{where}[{number}]")
    return f"{where}: {str(left)[:120]!r} vs {str(right)[:120]!r}"


def capture(root: Path, index: list[dict[str, Any]], endpoint: str, role: str, adapter: Any, out: Path) -> None:
    plan = read_json(root / "collection.json")
    checksum = plan[ROLES[role]]
    client = ServiceClient(endpoint)
    start = client.checkpoint()
    if start.get("weightsSha256") != checksum:
        raise ValueError(f"the service at {endpoint} does not serve the plan's {role} checkpoint")
    first = plan["probe"]["first_decisions"]
    with out.open("x", encoding="utf-8") as stream:
        for row in index:
            decisions = requests(root / row["run"], adapter, first)
            for choice in decisions:
                raw, seconds = client.decide_raw(choice["logged"])
                choice["logged_reply_raw"], choice["logged_seconds"] = raw.decode("utf-8"), seconds
                if same_request(choice["rerendered"], choice["logged"]):
                    choice["rerendered_reply_raw"], choice["rerendered_seconds"] = choice["logged_reply_raw"], 0.0
                else:
                    raw, seconds = client.decide_raw(choice["rerendered"])
                    choice["rerendered_reply_raw"], choice["rerendered_seconds"] = raw.decode("utf-8"), seconds
            stream.write(
                canonical({"episode_id": row["episode_id"], "checkpoint": checksum, "role": role, "rows": decisions})
                + "\n"
            )
            stream.flush()
    if client.checkpoint() != start:
        raise ValueError("the checkpoint identity changed while probing")


def read_capture(path: Path, index: list[dict[str, Any]], checksum: str) -> dict[str, list[dict[str, Any]]]:
    rows = read_rows(path)
    captured = {row["episode_id"]: row["rows"] for row in rows}
    if (
        len(rows) != len(index)
        or set(captured) != {row["episode_id"] for row in index}
        or any(row["checkpoint"] != checksum for row in rows)
    ):
        raise ValueError(f"{path.name} must cover exactly the indexed episodes under the expected checkpoint")
    return captured


def _normalized_answers(request: dict[str, Any], raw: str, keys: list[str]) -> dict[str, dict[str, float]]:
    """{question: normalized probabilities} of one guarded reply (one or several questions)."""
    if multi(keys):
        return {key: value["normalized"] for key, value in guard_decision(request, raw, keys)["questions"].items()}
    return {keys[0]: normalized(guard_response(request, raw, keys[0])["probabilities"])}


def compare(
    root: Path, index: list[dict[str, Any]], reference_path: Path, alternate_path: Path, adapter: Any, out: Path
) -> bool:
    plan = read_json(root / "collection.json")
    keys = plan_question_keys(plan)
    question_key = keys if multi(keys) else keys[0]
    reference_sha, alternate_sha = plan["clean_weights_sha256"], plan["alternate_weights_sha256"]
    reference = read_capture(reference_path, index, reference_sha)
    alternate = read_capture(alternate_path, index, alternate_sha)
    identity_report = upstream.shared_probe().identity_report
    first = plan["probe"]["first_decisions"]
    ok = True
    out.mkdir(parents=True, exist_ok=False)
    with (
        (out / "identity.jsonl").open("x", encoding="utf-8") as audit,
        (out / "probe-inputs.jsonl").open("x", encoding="utf-8") as stream,
    ):
        for row in index:
            episode_id = row["episode_id"]
            run_dir = root / row["run"]
            run = load_run(run_dir)
            expected = requests(run_dir, adapter, first)
            own_sha = run["manifest"]["pins"]["checkpoint"]["weightsSha256"]
            by_index = {r["index"]: r for r in run["rows"]}
            traces, answers, summary = [], [], []
            for step, (req, ref, alt) in enumerate(
                zip(expected, reference[episode_id], alternate[episode_id], strict=True)
            ):
                for key in ("index", "native_tick"):
                    if ref[key] != req[key] or alt[key] != req[key]:
                        raise ValueError(f"the captured probe request differs for {episode_id}")
                for key in ("logged", "rerendered"):
                    if not same_request(ref[key], req[key]) or not same_request(alt[key], req[key]):
                        raise ValueError(f"the captured probe request differs for {episode_id}")
                try:
                    same_ref = _normalized_answers(req["logged"], ref["logged_reply_raw"], keys)
                    same_alt = _normalized_answers(req["logged"], alt["logged_reply_raw"], keys)
                    rendered_ref = _normalized_answers(req["rerendered"], ref["rerendered_reply_raw"], keys)
                    rendered_alt = _normalized_answers(req["rerendered"], alt["rerendered_reply_raw"], keys)
                except ResponseRejected as rejection:
                    raise ValueError(f"a probe reply for {episode_id} fails the guard: {rejection}") from None
                logged_row = by_index[req["index"]]
                traces.append(
                    upstream.decision_trace(episode_id, step, float(req["native_tick"]), logged_row, question_key)
                )
                answers.append(same_ref if own_sha == reference_sha else same_alt)
                if not multi(keys):  # the 0.1.x probe-input form: one question's probabilities, unkeyed
                    same_ref, same_alt = same_ref[keys[0]], same_alt[keys[0]]
                    rendered_ref, rendered_alt = rendered_ref[keys[0]], rendered_alt[keys[0]]
                summary.append(
                    {
                        "native_tick": req["native_tick"],
                        "same_request_reference_checkpoint": same_ref,
                        "same_request_alternate_checkpoint": same_alt,
                        "standard_rerender_request_changed": not same_request(req["logged"], req["rerendered"]),
                        "standard_rerender_reference_checkpoint": rendered_ref,
                        "standard_rerender_alternate_checkpoint": rendered_alt,
                    }
                )
            report = identity_report(
                traces, answers, tolerance=plan["probe"]["identity_tolerance"], tie=plan["probe"]["tie"]
            )
            ok &= report.ok
            audit.write(
                canonical(
                    {
                        "episode_id": episode_id,
                        "shared_revision": upstream.PROBE_REV,
                        "checkpoint_role": "reference" if own_sha == reference_sha else "alternate",
                        **report.to_json(),
                    }
                )
                + "\n"
            )
            stream.write(canonical({"episode_id": episode_id, "text": canonical(summary)}) + "\n")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("selfcheck")
    check.add_argument("root", type=Path)
    check.add_argument("--adapter", required=True)
    check.add_argument("--max-runs", type=int)
    for name in ("capture", "compare"):
        sub = commands.add_parser(name)
        sub.add_argument("root", type=Path)
        sub.add_argument("index", type=Path)
        sub.add_argument("out", type=Path)
        sub.add_argument("--adapter", required=True)
        if name == "capture":
            sub.add_argument("--endpoint", required=True)
            sub.add_argument("--checkpoint-role", choices=tuple(ROLES), required=True)
        else:
            sub.add_argument("--reference-capture", type=Path, required=True)
            sub.add_argument("--alternate-capture", type=Path, required=True)
    args = parser.parse_args()
    adapter = load_adapter(args.adapter)
    if args.command == "selfcheck":
        report = selfcheck(args.root, adapter, args.max_runs)
        print(canonical(report))
        sys.exit(0 if report["ok"] else 1)
    index = read_json(args.index)
    if args.command == "capture":
        capture(args.root, index, args.endpoint, args.checkpoint_role, adapter, args.out)
    elif not compare(args.root, index, args.reference_capture, args.alternate_capture, adapter, args.out):
        print("same-request same-checkpoint identity failed; see identity.jsonl", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
