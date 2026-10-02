"""Run a Honing campaign through every plane from one config; see honing/e2e/pipeline.py.

Run from the repository root:
    python scripts/e2e/honing_pipeline.py --config configs/e2e/all_planes.json
"""

import sys
from pathlib import Path

# The default config names fakes under tests/, which import from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from honing.e2e.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
