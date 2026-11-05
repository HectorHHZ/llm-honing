"""Run from the checkout without installing Honing into the game environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from honing.backends.griddly.match import main

if __name__ == "__main__":
    main()
