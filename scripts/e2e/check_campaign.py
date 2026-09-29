"""Check what an end-to-end campaign run wrote, as a reviewer would read it.

Usage: python scripts/e2e/check_campaign.py <work-dir> [--expect MOVE=RESULT ...]

Checks that hold for any run of scripts/e2e/ra2_campaign.py:

- summary.json names a result for every step (published, rolled back, or skipped);
- the release chain has the initial release plus one record per gated step, and
  the head is the last published release (or r0000 when nothing was published);
- every gated step's episodes are on disk: a run directory with result.json each.

``--expect harness="rolled back"`` also requires that step's result, in order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS = {"published", "rolled back", "skipped"}


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
    gated = [s for s in steps if s["result"] != "skipped"]
    if len(records) != 1 + len(steps):
        found.append(f"the release chain has {len(records)} records for 1 initial release and {len(steps)} steps")
    published = [s["release"] for s in steps if s["result"] == "published"]
    head = published[-1] if published else "r0000"
    if summary["head"] != head:
        found.append(f"the head is {summary['head']}, but the last published release is {head}")

    episodes = sum(s["episodes"] for s in gated)
    finished = [p for p in (work / "runs").iterdir() if (p / "result.json").exists()]
    if len(finished) != episodes:
        found.append(f"{episodes} gated episodes, but {len(finished)} run directories with a result")
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
        print(f"{step['move']}: {step['result']} ({step['episodes']} episodes): {step['reason']}")
    for problem in found:
        print(f"FAIL: {problem}", file=sys.stderr)
    print("campaign end to end: " + ("FAILED" if found else "ok"))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
