"""Check the trainer a pipeline config names against Honing's contract; see honing/e2e/trainer_conformance.py.

Run from the repository root:
    python scripts/e2e/check_trainer.py --config configs/e2e/all_planes.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from honing.e2e.check import trainer_main

if __name__ == "__main__":
    sys.exit(trainer_main())
