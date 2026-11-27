"""Check what an end-to-end campaign run wrote, as a reviewer would read it.

Usage: python scripts/e2e/check_campaign.py <work-dir> [--expect MOVE=RESULT ...]

Checks that hold for any run of scripts/e2e/ra2_campaign.py, scripts/e2e/tank_campaign.py, or
scripts/e2e/honing_pipeline.py:

- summary.json names a result for every step (published, rolled back, skipped, or restored for a
  rollback step);
- the release chain has the initial release plus one record per step, and the head
  is the last published or restored release (or r0000 when there is none);
- every gated step's episodes are on disk: summary.json lists each one's run
  directory, and each holds a result.json. A method's own minibatches, rollouts,
  and demonstrations leave run directories too, and are not counted. A summary that lists no
  workspaces (the pipeline's) is not checked for them;
- a gate that records the seeds it played (``seed_first`` to ``seed_last``, as
  the tank gate does) shares none with an earlier gate of the run;
- every move's commit record carries a cost, the same cost its summary step reports, and the
  costs sum to summary.json's cost_total, an unknown part (null) making that part of the sum
  unknown; with a budget, the spend it reports is the total in its unit, a run it stopped at
  its limit spent at least that limit, and a run stopped because a move's dollars were
  unknown fails. A run whose records hold no cost at all was written before costs
  were recorded, and is not checked for them.

``--expect harness="rolled back"`` also requires that step's result, in order.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

RESULTS = {"published", "rolled back", "skipped", "restored"}
MOVE_RESULTS = {"published", "rolled back", "skipped"}
PURPOSES = ("gate", "method", "rollout", "demonstration")
COST_FIELDS = ("gpu_seconds", "serving_gpu_seconds", "input_tokens", "output_tokens", "api_usd", "usd", "wall_seconds")


def _plus(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else left + right


def _same(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def cost_problems(summary: dict, records: list[dict], steps: list[dict]) -> list[str]:
    """The cost checks: every move's record has one, the summary reports it, and the costs add up."""
    moves = [s for s in steps if s["result"] in MOVE_RESULTS]
    costed = [r for r in records[1:] if r["op"] in ("publish", "reject", "skip")]
    if not any("cost" in r for r in records) and "cost_total" not in summary:
        return []  # written before costs were recorded
    found = []
    missing = [i + 1 for i, r in enumerate(costed) if "cost" not in r]
    if missing:
        found.append(f"move records {missing} carry no cost")
        return found
    if len(costed) != len(moves):
        return [f"the chain has {len(costed)} move records for {len(moves)} moves"]
    total: dict[str, object] = {"episodes": dict.fromkeys(PURPOSES, 0), **dict.fromkeys(COST_FIELDS, 0)}
    for index, (record, step) in enumerate(zip(costed, moves, strict=True), start=1):
        cost = record["cost"]
        if step.get("cost") != cost:
            found.append(f"move {index} ({step['move']}) reports cost {step.get('cost')}, its record {cost}")
        episodes = total["episodes"]
        for purpose, count in cost["episodes"].items():
            episodes[purpose] = episodes.get(purpose, 0) + count  # type: ignore[union-attr]
        for name in COST_FIELDS:
            total[name] = _plus(total[name], cost[name])  # type: ignore[arg-type]
    reported = summary.get("cost_total")
    if not isinstance(reported, dict):
        return [*found, "summary.json has no cost_total"]
    found.extend(
        f"the moves' {name} sum to {total[name]}, but cost_total says {reported.get(name)}"
        for name in ("episodes", *COST_FIELDS)
        if not _same(total[name], reported.get(name))
    )
    budget = summary.get("budget")
    if budget:
        spent = total["usd" if budget["unit"] == "usd" else "wall_seconds"]
        if not _same(budget["spent"], spent):
            found.append(f"the budget reports {budget['spent']} spent, but the moves cost {spent} {budget['unit']}")
        stopped = budget["stopped"]
        if stopped == "limit":
            if spent is None or spent < budget["limit"]:  # type: ignore[operator]
                found.append(f"the budget stopped the run at {spent} of {budget['limit']} {budget['unit']}")
        elif stopped == "unknown_cost":
            # Recorded and consistent, but a run whose spend cannot be priced has not kept to its budget.
            found.append(f"the run stopped because its spend could not be priced: {budget['reason']}")
        elif stopped is not None:
            found.append(f"the budget reports an unknown stop {stopped!r}")
    return found


def problems(work: Path, expected: list[tuple[str, str]]) -> list[str]:
    found: list[str] = []
    summary = json.loads((work / "summary.json").read_text())
    steps = summary["steps"]
    if not steps:
        found.append("the campaign ran no steps")
    for index, step in enumerate(steps):
        if step["result"] not in RESULTS:
            found.append(f"step {index + 1} ({step['move']}) has result {step['result']!r}")
        if step["result"] == "published" and not step["release"]:
            found.append(f"step {index + 1} was published without a release id")
    if expected and [(s["move"], s["result"]) for s in steps] != expected:
        found.append(f"expected steps {expected}, got {[(s['move'], s['result']) for s in steps]}")

    records = [json.loads(line) for line in (work / "releases/commits.jsonl").read_text().splitlines() if line]
    gated = [s for s in steps if s["result"] in ("published", "rolled back")]
    if len(records) != 1 + len(steps):
        found.append(f"the release chain has {len(records)} records for 1 initial release and {len(steps)} steps")
    # A restored rollback step names the release it made the head again.
    moved = [s["release"] for s in steps if s["result"] in ("published", "restored")]
    head = moved[-1] if moved else "r0000"
    if summary["head"] != head:
        found.append(f"the head is {summary['head']}, but the last published or restored release is {head}")

    for index, step in enumerate(gated):
        if "workspaces" not in step:
            continue
        workspaces = step["workspaces"]
        if len(workspaces) != step["episodes"]:
            found.append(
                f"step {index + 1} ({step['move']}) lists {len(workspaces)} of its {step['episodes']} episodes"
            )
        missing = [w for w in workspaces if not w or not (Path(w) / "result.json").exists()]
        if missing:
            found.append(
                f"step {index + 1} ({step['move']}) has {len(missing)} episodes without a result: {missing[:3]}"
            )

    played: list[tuple[int, int]] = []
    for record in records:
        metrics = (record.get("decision") or {}).get("metrics", {})
        if "seed_first" in metrics:
            first, last = int(metrics["seed_first"]), int(metrics["seed_last"])
            shared = [(a, b) for a, b in played if a <= last and first <= b]
            if shared:
                found.append(f"a gate played seeds {first}-{last}, which overlap an earlier gate's {shared[0]}")
            played.append((first, last))
    found += cost_problems(summary, records, steps)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--expect", action="append", default=[], help="MOVE=RESULT, in step order")
    args = parser.parse_args(argv)
    expected = [tuple(item.split("=", 1)) for item in args.expect]
    found = problems(args.work_dir, expected)  # type: ignore[arg-type]
    summary = json.loads((args.work_dir / "summary.json").read_text())
    for step in summary["steps"]:
        print(f"{step['move']}: {step['result']} ({step.get('episodes', 0)} episodes): {step['reason']}")
    if "cost_total" in summary:
        print(f"cost total: {json.dumps(summary['cost_total'], sort_keys=True)}")
    if summary.get("budget"):
        print(f"budget: {json.dumps(summary['budget'], sort_keys=True)}")
    for problem in found:
        print(f"FAIL: {problem}", file=sys.stderr)
    print("campaign end to end: " + ("FAILED" if found else "ok"))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
