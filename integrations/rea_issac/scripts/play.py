"""Launch REA-Issac from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from integrations.rea_issac.ui import main

if __name__ == "__main__":
    main()
