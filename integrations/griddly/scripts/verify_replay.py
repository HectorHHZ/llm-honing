"""Audit or re-execute a native Spiders recording."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from honing.backends.griddly.replay import main

if __name__ == "__main__":
    main()
