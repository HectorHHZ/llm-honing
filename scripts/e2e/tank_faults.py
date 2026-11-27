"""Build the Tank War fault library (#171) from one config, and gate it; see honing/backends/tank/faults.py.

Run from the repository root:
    python scripts/e2e/tank_faults.py --config configs/e2e/tank_faults_dry_run.json
"""

import sys
from pathlib import Path

# The dry-run config names fakes under tests/, which import from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from honing.backends.tank.faults import main

if __name__ == "__main__":
    sys.exit(main())
