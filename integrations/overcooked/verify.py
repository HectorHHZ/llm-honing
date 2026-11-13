"""Verify a saved run by executing every action again in the pinned native engine."""

import argparse
import json
from pathlib import Path

from integrations.overcooked.recording import replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    audit = replay(args.run)
    print(json.dumps({k: v for k, v in audit.items() if k != "frames"}))


if __name__ == "__main__":
    main()
