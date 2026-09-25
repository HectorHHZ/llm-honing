"""Verify native files and re-encode each logged request with its pinned tokenizer."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from storage import open_content

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_prediction(parts, questions, prediction, measure, limit):
    from integration.models.choice_protocol import pack_request

    state, audit, trimmed = pack_request(parts, questions, measure, limit)
    if prediction["stateText"] != state or prediction["tokenAudit"] != audit or prediction["trimmed"] != trimmed:
        raise ValueError("Native request differs from independently reconstructed lossless encoding")


def verify(match):
    from integration.models.provenance import make_measurer, verify_runtime

    result = json.loads((match / "result.json").read_text())
    runtime = result["runtime"]
    if result.get("modelFamily") != runtime.get("family"):
        raise ValueError("Native family missing or inconsistent")
    pins = json.loads((match / "source/league/model-sources.json").read_text())
    verify_runtime(runtime, pins)
    adapter = make_measurer(runtime)
    requests = 0
    with open_content(match / "decisions.jsonl") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            for stage in row.get("stages", [row] if row.get("questions") else []):
                if not stage["questions"]:
                    continue
                if "ensemble" in stage:
                    entries = [
                        (stage["questions"], stage["ensemble"]["original"]),
                        (stage["ensemble"]["permutedQuestions"], stage["ensemble"]["swapped"]),
                    ]
                else:
                    entries = [(stage["questions"], stage)]
                for questions, prediction in entries:
                    verify_prediction(stage["contextParts"], questions, prediction, adapter.measure, runtime["maxLen"])
                    requests += 1
    if requests != result["inferenceCalls"]:
        raise ValueError("Native request count differs from recorded inference calls")
    report = {
        "verified": True,
        "verifierSourceSha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in [
                "league/verify_native_encoding.py",
                "league/storage.py",
                "integration/models/provenance.py",
                "integration/models/worker.py",
                "integration/models/head_checkpoint.py",
                "integration/models/choice_protocol.py",
                "integration/models/upstream_adapters.py",
            ]
        },
        "modelFamily": runtime["family"],
        "weightSetSha256": runtime["weightSetSha256"],
        "checkpointKind": runtime.get("checkpointKind", "prepared-upstream-model"),
        "learnedHead": runtime.get("learnedHead"),
        "verifiedFiles": len(runtime["verifiedFilesSha256"]),
        "requests": requests,
        "modelObjectsLoadedByVerifier": False,
        "gpuAllocatedByVerifier": False,
        "exactNativeTokenizationRecomputed": True,
    }
    (match / "native-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.match.resolve())))
