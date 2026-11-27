"""Send tank runs played before recording to Reef, as decision records and episode reports (#227).

    python scripts/reef/import_tank_runs.py RUN_DIR... --out records.jsonl \
        [--release R] [--harness H] [--url http://127.0.0.1:8977 --scenario tank]

Without ``--url`` it only writes the records. With it, ``reef-client``
(the ``reef`` extra) imports them, resumably, with the token in
``REEF_TOKEN``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from honing.backends.reef.tank import write_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release")
    parser.add_argument("--harness")
    parser.add_argument("--url")
    parser.add_argument("--scenario", default="tank")
    args = parser.parse_args(argv)
    count = write_records(sorted(args.runs), args.out, release=args.release, harness=args.harness)
    print(f"wrote {count} records of {len(args.runs)} runs to {args.out}")
    if args.url:
        from reef_client.record_import import import_records_file

        acknowledged = import_records_file(
            args.out,
            url=args.url,
            scenario=args.scenario,
            token=os.environ.get("REEF_TOKEN"),
            max_batch_bytes=9 * 1024 * 1024,
        )
        print(f"Reef acknowledged {acknowledged} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
