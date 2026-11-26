"""Label-isolation check of the diagnosis inputs (read and probe), stricter than the common builders' own check.

  python3 -B isolation_check.py ROOT [--out FILE]

Searches every `read-inputs.jsonl` and `probes/probe-inputs.jsonl` text for: every fault family and instance name of
the plan (with '-', '_' and ' ' treated as equal, case-insensitive; the PR #187 review's request), every run and
reset-group-condition name, the checkpoint repos, revisions and weights SHA-256s, any 64-hex digest, the evaluator's
label field names, and the condition names. A marker found in some but not all episodes of a file is an error (it
could correlate with the label, as the fault-named strategy line of PR #187 did); a marker found in every episode of a
file alike (e.g. the weights family name `alternate-checkpoint` against the probe fields `..._alternate_checkpoint`,
which every probe input has) is reported as uniform, not as an error; any 64-hex digest is an error. Reports the generic
words "fault", "inject", "intervention", "alternate checkpoint", "clean checkpoint" and "harness", and counts the shared
renderer's fixed auto-row sentence ("native input ran without a model decision"; VVVVVV: the death animation) per
collection condition. Exit 1 on any error. No model call.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

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
    "checkpoint_role",
    "trigger_index",
)
GENERIC = ("fault", "inject", "intervention", "alternate checkpoint", "clean checkpoint", "harness")
AUTO_SENTENCE = "native input ran without a model decision"  # common renderer wording since 2026-11-26
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")


def norm(text: str) -> str:
    return re.sub(r"[-_ ]+", "-", text.lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    root = a.root
    plan = json.loads((root / "collection.json").read_text())
    index = json.loads((root / "audit" / "episodes.json").read_text())
    condition_of = {row["episode_id"]: row["condition"] for row in index}
    markers: set[str] = set()
    for fault in plan["faults"].values():
        for item in (fault if isinstance(fault, list) else [fault]):
            markers.update({item["family"], item["instance"]})
    attempts = [
        json.loads(line) for line in (root / "audit" / "attempts.jsonl").read_text().splitlines() if line.strip()
    ]
    markers.update(row["run"] for row in attempts)
    for row in index:
        checkpoint = row["identity"]["checkpoint"]
        markers.update(v for v in (checkpoint.get("repo"), checkpoint.get("revision")) if v)
        markers.add(row["identity"]["weights_sha256"])
    markers.update({plan["clean_weights_sha256"], plan["alternate_weights_sha256"], *LABEL_FIELDS})
    normalized = {norm(m): m for m in markers if len(m) >= 4}
    report = {
        "files": {},
        "forbidden_hits": [],
        "uniform_markers": [],
        "generic_words": Counter(),
        "auto_sentence_by_condition": Counter(),
        "markers_checked": len(normalized),
    }
    for name in ("read-inputs.jsonl", "probes/probe-inputs.jsonl"):
        rows = [json.loads(line) for line in (root / name).read_text().splitlines() if line.strip()]
        report["files"][name] = {
            "episodes": len(rows),
            "chars": sum(len(r["text"]) for r in rows),
            "keys_only_episode_id_and_text": all(set(r) == {"episode_id", "text"} for r in rows),
        }
        found: dict[str, list[str]] = {}
        for row in rows:
            text = row["text"]
            folded = norm(text)
            for key, marker in normalized.items():
                if key in folded:
                    found.setdefault(marker, []).append(row["episode_id"])
            if HEX64.search(text):
                report["forbidden_hits"].append(
                    {"file": name, "episode_id": row["episode_id"], "marker": "64-hex digest"}
                )
            lowered = text.lower()
            for word in GENERIC:
                report["generic_words"][f"{name}: {word}"] += lowered.count(word)
            if name == "read-inputs.jsonl":
                report["auto_sentence_by_condition"][condition_of[row["episode_id"]]] += lowered.count(AUTO_SENTENCE)
        for marker, episodes in sorted(found.items()):
            by_condition = dict(Counter(condition_of[e] for e in episodes))
            if len(set(episodes)) == len(rows):
                report["uniform_markers"].append({"file": name, "marker": marker, "episodes": len(rows)})
            else:
                report["forbidden_hits"].append(
                    {"file": name, "marker": marker, "episodes_by_condition": by_condition}
                )
    report["generic_words"] = dict(report["generic_words"])
    report["auto_sentence_by_condition"] = dict(report["auto_sentence_by_condition"])
    report["ok"] = not report["forbidden_hits"] and all(
        f["keys_only_episode_id_and_text"] for f in report["files"].values()
    )
    text = json.dumps(report, indent=1)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
