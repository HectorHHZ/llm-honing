"""Re-ask recorded NanoJev requests and check every option probability.

This checks the model/request identity gate before any diagnostic probe. The
recorded calls are kept separate from private cause labels.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


def probabilities(reply: dict[str, Any]) -> dict[str, float]:
    result = {
        f"answers/{question}/{option}": value
        for question, answer in reply["answers"].items()
        for option, value in answer["probabilities"].items()
    }
    for branch, branch_reply in reply.get("branches", {}).items():
        result.update({f"branches/{branch}/{key}": value for key, value in probabilities(branch_reply).items()})
    return result


def compare_call(record: dict[str, Any], url: str, tolerance: float = 0.01) -> dict[str, Any]:
    request = record["request"]
    original = record["reply"]
    payload = json.dumps(request, separators=(",", ":")).encode()
    http_request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/decide", payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(http_request, timeout=120) as response:
        repeated = json.load(response)
    before, after = probabilities(original), probabilities(repeated)
    if before.keys() != after.keys():
        raise ValueError(f"Probability options changed for {request['id']}")
    differences = {key: abs(before[key] - after[key]) for key in before}
    audits = [original["tokenAudit"], repeated["tokenAudit"]]
    for reply in (original, repeated):
        audits.extend(branch["tokenAudit"] for branch in reply.get("branches", {}).values())
    truncated = any(audit[question].get("stateTruncated") for audit in audits for question in audit)
    too_long = any(audit[question].get("maxInputTokens", 0) > 8192 for audit in audits for question in audit)
    largest = max(differences.values(), default=0.0)
    return {
        "request_id": request["id"],
        "question_count": len(differences),
        "maximum_probability_difference": largest,
        "truncated": truncated,
        "too_long": too_long,
        "passed": largest <= tolerance and not truncated and not too_long,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()
    calls = [json.loads(line) for run in args.run for line in (run / "model_calls.jsonl").read_text().splitlines()]
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.out.open("x") as output:
        for call in calls:
            try:
                row = compare_call(call, args.url, args.tolerance)
            except Exception as error:
                row = {"request_id": call["request"]["id"], "passed": False, "error": repr(error)}
            rows.append(row)
            output.write(json.dumps(row, sort_keys=True) + "\n")
            output.flush()
    summary = {
        "calls": len(rows),
        "probability_checks": sum(row.get("question_count", 0) for row in rows),
        "failed_calls": sum(not row["passed"] for row in rows),
        "maximum_difference": max((row.get("maximum_probability_difference", 0) for row in rows), default=0),
        "tolerance": args.tolerance,
    }
    print(json.dumps(summary, sort_keys=True))
    if summary["failed_calls"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
