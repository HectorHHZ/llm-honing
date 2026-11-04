"""Run GEPA end to end against real model APIs, then check the run; see honing/e2e/gepa_live.py and README.md.

Run from the repository root, with the key in the environment:
    python scripts/e2e/gepa_live.py --config configs/e2e/gepa_live.json
"""

import sys
from pathlib import Path

# The config names fakes under tests/ for the planes a harness move never touches.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from honing.e2e.gepa_live import main

if __name__ == "__main__":
    sys.exit(main())
