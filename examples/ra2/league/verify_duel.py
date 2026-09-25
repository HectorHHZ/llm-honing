"""Verify both model policies and one replay without treating either as Supalosa."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from storage import digest

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def verify(match):
    result, manifest = read(match / "result.json"), read(match / "manifest.json")
    if result.get("kind") != "model-duel-v1" or (match / "error.json").exists():
        raise ValueError("Not a successful two-model duel")
    if any(result.get(k) != value for k, value in manifest.items()):
        raise ValueError("Root result differs from its initial manifest")
    if not all(
        result.get(k) is True
        for k in [
            "normalView",
            "observerDataExcluded",
            "samePreStepObservation",
            "opponentChoicesExcluded",
            "noFallbackPolicy",
            "strictCurrentEnemyTargets",
        ]
    ):
        raise ValueError("Missing fairness provenance")
    if (
        result["maxActionsPerSeat"] != 1
        or result["decisionInterval"] != result["cadenceTicks"]
        or result["candidateMode"] != "intent"
        or result["decisionMode"] != "staged"
    ):
        raise ValueError("Duel budget differs between players")
    for rel, expected in result["implementationSha256"].items():
        if digest(match / "source" / rel) != expected:
            raise ValueError("Changed duel source snapshot: " + rel)
    evidence = {}
    for seat in ["red", "blue"]:
        directory = match / seat
        player, initial = read(directory / "result.json"), read(directory / "manifest.json")
        if any(player.get(k) != value for k, value in initial.items()):
            raise ValueError("Player result differs from its initial manifest")
        if (
            player.get("policySeat") != seat
            or player.get("opponentKind") != "decision-model"
            or "baselineSeat" in player
            or "layaSeat" in player
        ):
            raise ValueError("Misidentified player/opponent")
        if (
            player["runtime"] != result["runtimes"][seat]
            or player["modelFamily"] != result["modelFamilies"][seat]
            or player["inferenceCalls"] != result["inferenceCalls"][seat]
        ):
            raise ValueError("Player runtime inconsistent with duel manifest")
        for key in [
            "map",
            "seed",
            "tick",
            "hash",
            "ended",
            "truncated",
            "outcome",
            "stats",
            "cadenceTicks",
            "decisionInterval",
            "maxActionsPerSeat",
            "implementationSha256",
        ]:
            if player[key] != result[key]:
                raise ValueError("Player differs from shared duel: " + key)
        if (directory / "actions.jsonl").resolve() != (match / "actions.jsonl").resolve() or (
            directory / "source"
        ).resolve() != (match / "source").resolve():
            raise ValueError("Player does not use shared action/source evidence")
        runtime, family = player["runtime"], player["modelFamily"]
        if family == "laya":
            if digest(Path(runtime["path"]) / "model.safetensors") != runtime["weightsSha256"]:
                raise ValueError("Laya checkpoint digest mismatch")
            checkpoint = runtime["weightsSha256"]
        elif family in {"nanojev", "jev-omni"}:
            native_python = os.environ.get("RA2_NATIVE_PYTHON") or os.environ.get("LAYA_PYTHON")
            if not native_python:
                raise ValueError("Native duel verification requires RA2_NATIVE_PYTHON or LAYA_PYTHON")
            subprocess.run(
                [native_python, str(ROOT / "league/verify_native_encoding.py"), "--match", str(directory)],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            native = read(directory / "native-verification.json")
            if not native["verified"] or native["weightSetSha256"] != runtime["weightSetSha256"]:
                raise ValueError("Native encoding/weights not verified")
            checkpoint = native["weightSetSha256"]
        else:
            raise ValueError("Unsupported model family")
        subprocess.run(
            ["node", str(ROOT / "integration/laya/audit.mjs"), str(directory)], check=True, stdout=subprocess.DEVNULL
        )
        audit = read(directory / "audit.json")
        if not audit["passed"] or audit["policySeat"] != seat:
            raise ValueError("Seat action audit failed")
        trace = read(directory / "decision-log.json")
        if trace["contentSha256"] != digest(directory / "decisions.jsonl") or trace["records"] != audit["decisions"]:
            raise ValueError("Decision log was changed or incomplete")
        evidence[seat] = {
            "modelFamily": family,
            "checkpointSha256": checkpoint,
            "auditPassed": True,
            "nativeEncodingVerified": family != "laya",
            "inferenceCalls": player["inferenceCalls"],
            "decisions": audit["decisions"],
            "selectedActions": audit["selectedActions"],
            "fallbackCount": 0,
            "artifacts": {
                name: digest(directory / name)
                for name in ["result.json", "manifest.json", "audit.json", "decisions.jsonl", "decision-log.json"]
            },
        }
    subprocess.run(
        [
            os.environ["RA2_REPLAY_PYTHON"],
            str(ROOT / "league/verify_replay.py"),
            "--match",
            str(match),
            "--game-dir",
            os.environ["RA2_GAME_DIR"],
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    replay = read(match / "replay-verification.json")
    if not replay["verified"]:
        raise ValueError("Shared duel replay mismatch")
    report = {
        "kind": "verified-model-duel-v1",
        "verified": True,
        "map": result["map"],
        "seed": result["seed"],
        "tick": result["tick"],
        "maxTicks": result["maxTicks"],
        "outcome": result["outcome"],
        "normalView": True,
        "equalActionBudget": 1,
        "cadenceTicks": result["cadenceTicks"],
        "replayVerified": True,
        "players": evidence,
        "notBaselineEvaluation": True,
        "learningPerformed": False,
        "verifierSha256": digest(Path(__file__)),
        "artifacts": {
            name: digest(match / name)
            for name in ["result.json", "manifest.json", "actions.jsonl", "match.rplx", "replay-verification.json"]
        },
    }
    (match / "evidence.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.match.resolve())))
