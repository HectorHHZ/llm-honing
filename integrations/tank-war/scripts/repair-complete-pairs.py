"""Rebuild missing pair evidence when both Tank War arms already finished."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    args = parser.parse_args()
    specification = importlib.util.spec_from_file_location("critical_pairs", args.collector)
    if specification is None or specification.loader is None:
        raise ValueError("cannot load pair comparator")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    journal = args.root / "recovered_pair_evidence.jsonl"
    for layer in ("harness", "weights", "controller"):
        for family in sorted((args.root / layer).glob("s*")):
            clean, fault = family / "clean", family / "fault"
            evidence_path = family / "pair_evidence.json"
            if evidence_path.exists() or not (clean / "result.json").exists() or not (fault / "result.json").exists():
                continue
            try:
                evidence = module.compare(clean, fault, layer)
                evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
                record = {
                    "family": str(family),
                    "status": "recovered",
                    "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                }
            except (OSError, KeyError, ValueError) as error:
                record = {"family": str(family), "status": "error", "error": repr(error)}
            with journal.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
