"""Stands in for integrations/ra2/integration/models/duel.mjs: same flags, a result.json, no engine.

The seat whose decision server URL contains "strong" wins; if neither does, the duel is truncated.
FAKE_DUEL_EXIT sets the exit code.
"""

import json
import os
import sys
from pathlib import Path

args = dict(a.removeprefix("--").split("=", 1) for a in sys.argv[1:])
out = Path(args["out"])
out.mkdir(parents=True)
strong = [seat for seat in ("red", "blue") if "strong" in args[f"{seat}-decision-server"]]
if len(strong) == 1:
    winner = strong[0]
    outcome = {winner: 1, ("blue" if winner == "red" else "red"): -1}
else:
    outcome = {"red": 0, "blue": 0}
(out / "result.json").write_text(json.dumps({"outcome": outcome, "truncated": len(strong) != 1, "arguments": args}))
sys.exit(int(os.environ.get("FAKE_DUEL_EXIT", "0")))
