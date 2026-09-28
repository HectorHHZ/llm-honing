"""Run a short Honing campaign on RA2 from one config; see honing/backends/ra2/e2e.py.

Run from the repository root:
    python scripts/e2e/ra2_campaign.py --config configs/e2e/ra2_dry_run.json
"""

import sys
from pathlib import Path

# The dry-run config names fakes under tests/, which import from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from honing.backends.ra2.e2e import main

if __name__ == "__main__":
    sys.exit(main())
