"""Build label-free diagnosis inputs for ten replay-verified no-injection controls."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _pilot_builder():
    path = Path(__file__).with_name("make-blind-pilot.py")
    spec = importlib.util.spec_from_file_location("tank_blind_pilot", path)
    if not spec or not spec.loader:
        raise RuntimeError("pilot packet builder missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("control output directory already exists")
    source = args.root / "pilot1/private/controls"
    packets = [json.loads(path.read_text()) for path in sorted(source.glob("*/private-control-packet.json"))]
    if len(packets) != 10 or any(packet["condition"] != "no_injection" for packet in packets):
        raise ValueError("expected ten replay-verified no-injection controls")
    builder = _pilot_builder()
    args.out.mkdir()
    (args.out / "inputs").mkdir()
    gold = []
    for packet in packets:
        public, private = builder.make_packet(packet)
        if len(public["detail_window"]) != 11:
            raise ValueError("control episode does not have a complete eleven-decision window")
        (args.out / "inputs" / f"{public['episode_id']}.json").write_text(
            json.dumps(public, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        gold.append(private)
    (args.out / "gold.jsonl").write_text("".join(json.dumps(row) + "\n" for row in gold))
    print(json.dumps({"controls": len(gold), "outcomes": {
        str(outcome): sum(json.loads((args.out / "inputs" / f"{row['episode_id']}.json").read_text())["outcome"] == outcome
                          for row in gold)
        for outcome in (-1, 0, 1)
    }}))


if __name__ == "__main__":
    main()
