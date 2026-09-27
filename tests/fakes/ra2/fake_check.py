"""Stands in for the audit and the replay check. A run directory containing a file named
after the check (audit-fails or replay-fails) fails it."""

import json
import sys
from pathlib import Path

kind = sys.argv[1]
run = Path(sys.argv[-1])
if (run / f"{kind}-fails").exists():
    print(f"{kind}: mismatch at decision 3")
    sys.exit(0 if kind == "replay" else 1)
print(json.dumps({"verified": True}) if kind == "replay" else "audit ok")
