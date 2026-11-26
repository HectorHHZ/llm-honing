"""Write game-local controller candidates for Tank War Exp 0 screening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KINDS = (
    "swap_selected_turn",
    "drop_fire_after_selection",
    "wrong_legal_projection",
    "suppress_selected_turn",
    "invert_selected_drive",
)


def build(out: Path) -> list[dict[str, str]]:
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for kind in KINDS:
        record = {"layer": "controller", "kind": kind, "family": kind, "instance": f"{kind}-v1"}
        (out / f"{kind}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        records.append(record)
    (out / "catalog.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"candidates": len(build(args.out)), "out": str(args.out)}))


if __name__ == "__main__":
    main()
