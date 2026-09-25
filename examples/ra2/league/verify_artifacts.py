"""Recompute evidence from match artifacts instead of trusting caller-supplied booleans."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from gates import tick_budget_matches
from storage import compress, digest

ROOT = Path(__file__).resolve().parents[1]


def verify(match, split, replay_python):
    result = json.loads((match / "result.json").read_text())
    protocol = json.loads((ROOT / "league/protocol.json").read_text())
    spec = protocol[split]
    if not tick_budget_matches(result, spec):
        raise ValueError("Tick budget differs from the requested evaluation split, or is not recorded")
    if result["map"] not in spec["maps"] or result["seed"] not in spec["seeds"]:
        raise ValueError("Match does not belong to claimed split")
    if (match / "error.json").exists() or not result.get("normalView") or not result.get("observerDataExcluded"):
        raise ValueError("Invalid match or observation provenance")
    if (
        result["cadenceTicks"] != protocol["cadenceTicks"]
        or result["decisionInterval"] != protocol["cadenceTicks"]
        or result["engine"]["engineCommit"] != protocol["engineCommit"]
    ):
        raise ValueError("Protocol mismatch")
    for rel, checksum in result["implementationSha256"].items():
        if digest(match / "source" / rel) != checksum:
            raise ValueError(f"Source snapshot differs: {rel}")
    pins = json.loads((match / "source/pins.json").read_text())
    if pins["supalosa"]["revision"] != protocol["baselineRevision"]:
        raise ValueError("Baseline revision mismatch")
    runtime = result["runtime"]
    if digest(Path(runtime["path"]) / "model.safetensors") != runtime["weightsSha256"]:
        raise ValueError("Checkpoint artifact mismatch")
    manifest = json.loads((match / "manifest.json").read_text())
    if any(
        manifest.get(key) != result.get(key)
        for key in ("map", "seed", "layaSeat", "runtime", "cadenceTicks", "maxTicks")
    ):
        raise ValueError("Result differs from initial manifest")
    subprocess.run(
        ["node", str(ROOT / "integration/laya/audit.mjs"), str(match)], check=True, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        [
            replay_python,
            str(ROOT / "league/verify_replay.py"),
            "--match",
            str(match),
            "--game-dir",
            os.environ["RA2_GAME_DIR"],
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    audit = json.loads((match / "audit.json").read_text())
    replay = json.loads((match / "replay-verification.json").read_text())
    if not audit["passed"] or not replay["verified"]:
        raise ValueError("Audit/replay failed")
    evidence = {
        "map": result["map"],
        "seed": result["seed"],
        "seat": result["layaSeat"],
        "split": split,
        "controller": "model",
        "normalView": True,
        "fallbackCount": 0,
        "auditPassed": True,
        "replayVerified": True,
        "error": None,
        "checkpointSha256": runtime["weightsSha256"],
        "ended": result["ended"],
        "truncated": result["truncated"],
        "outcome": result["outcome"],
        "tick": result["tick"],
        "maxTicks": result["maxTicks"],
        "engineCommit": result["engine"]["engineCommit"],
        "baselineRevision": pins["supalosa"]["revision"],
        "cadenceTicks": result["cadenceTicks"],
        "decisionInterval": result["decisionInterval"],
        "engineActionCap": result["maxActionsPerSeat"],
        "verifierSha256": digest(Path(__file__)),
        "path": str(match),
        "artifacts": {
            name: digest(match / name)
            for name in (
                "result.json",
                "manifest.json",
                "actions.jsonl",
                "decisions.jsonl",
                "match.rplx",
                "audit.json",
                "replay-verification.json",
            )
        },
    }
    evidence["logStorage"] = compress(match)
    (match / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "dev", "layaQualification", "sealedTest"], required=True)
    parser.add_argument(
        "--replay-python",
        default=os.environ.get("RA2_REPLAY_PYTHON"),
        required=not os.environ.get("RA2_REPLAY_PYTHON"),
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.match.resolve(), args.split, args.replay_python)))


if __name__ == "__main__":
    main()
