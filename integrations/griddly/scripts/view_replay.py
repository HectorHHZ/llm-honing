"""Build an offline, labeled replay from audited native game records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from honing.backends.griddly.replay import audit_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="a completed Griddly run directory")
    args = parser.parse_args()
    run = args.run.resolve()
    manifest, steps, result = audit_run(run)
    decisions = [json.loads(line) for line in (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    payload = {
        "name": run.name,
        "controller": manifest["controller"],
        "model": (manifest.get("model") or {}).get("metadata", {}).get("family"),
        "scenario": manifest["scenario"],
        "result": result,
        "frames": [
            {key: row["frame"][key] for key in ("step", "observation", "legal_actions", "status")} for row in steps
        ],
        "decisions": [
            {
                "action": row["applied"],
                "probabilities": (
                    row["response"]["answers"]["action"]["probabilities"] if row["response"] is not None else None
                ),
                "latency_ms": row["decision_ms"],
                "forced": row["forced_action"],
            }
            for row in decisions
        ],
    }
    # Records are data, including any strings coming from a server. Never allow
    # them to close the JSON script element or become executable markup.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("replay-template.html").read_text(encoding="utf-8")
    output = run / "replay.html"
    output.write_text(template.replace("__REPLAY_JSON__", encoded), encoding="utf-8")
    print(json.dumps({"replay": str(output), "steps": result["steps"]}))


if __name__ == "__main__":
    main()
