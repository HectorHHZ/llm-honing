"""Check the included historical sample using the current adapter, offline."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    source = json.loads((here / "SOURCE.json").read_text())
    for name, expected in source["includedSha256"].items():
        if hashlib.sha256((here / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Sample bytes changed: {name}")
    integration = here.parents[1]
    sys.path.insert(0, str(integration))
    spec = importlib.util.spec_from_file_location("sample_adapter", integration / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the local adapter")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    request = json.loads((here / "request.json").read_text())
    response = json.loads((here / "response.raw").read_bytes())
    action, probability = adapter.checked_action(response, request)
    row = json.loads((here / "transition.json").read_text())
    before, after = row["before"], row["after"]
    if before["id"] != request["id"] or after["id"] != before["id"] + 1 or row["action"] != action:
        raise ValueError("Decision/native sequence differs")
    game = source["game"]
    if game == "vvvvvv":
        ticks = row["ticks"]
        if after["tick"] - before["tick"] != ticks or after["engineFrame"] - before["engineFrame"] != ticks:
            raise ValueError("VVVVVV native clock differs")
        if after["lastAction"] != action or after["lastActionTicks"] != ticks:
            raise ValueError("VVVVVV native command differs")
        if after["flips"] != before["flips"] + 1:
            raise ValueError("The included native flip did not occur")
        if hashlib.sha256((here / "response.raw").read_bytes()).hexdigest() != row["responseSha256"]:
            raise ValueError("Original VVVVVV response receipt differs")
    elif game == "taisei":
        if after["frame"] - before["frame"] != row["frames"] or after["player"]["y"] >= before["player"]["y"]:
            raise ValueError("The included upward movement did not occur")
    elif game == "hurrican":
        if (
            after["tick"] - before["tick"] != row["frames"]
            or after["primaryProjectiles"] <= before["primaryProjectiles"]
        ):
            raise ValueError("The included original-engine shot did not occur")
    elif game == "celeste":
        adapter.check_transition(before, after, adapter.MASKS[action], after["lastFrames"])
        if before["player"]["dashes"] != 1 or after["player"]["dashes"] != 0:
            raise ValueError("The included first native dash did not consume its charge")
    else:
        raise ValueError("Unknown historical sample")
    print(
        json.dumps(
            {
                "game": game,
                "historical": True,
                "choice": action,
                "probability": probability,
                "sha256CheckedFiles": len(source["includedSha256"]),
                "checks": "passed",
                "freshModelCall": False,
                "postFixNativeRun": False,
            }
        )
    )


if __name__ == "__main__":
    main()
