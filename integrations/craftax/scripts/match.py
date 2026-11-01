"""Run from a repository checkout using the separate Craftax Python runtime."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from honing.backends.craftax.match import main

if __name__ == "__main__":
    main()
