"""Check the included real decision offline, with the adapter in this directory; no model or game is started."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_adapter(path: Path):
    spec = importlib.util.spec_from_file_location("hallowdale_sample_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the local adapter")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def main() -> None:
    here = Path(__file__).resolve().parent
    adapter = load_adapter(here.parents[1] / "run.py")
    request = json.loads((here / "request.json").read_text(encoding="utf-8"))
    response = adapter.strict_json_loads((here / "response.raw").read_bytes())
    row = json.loads((here / "transition.json").read_text(encoding="utf-8"))
    before, after, frames = row["before"], row["after"], row["frames"]

    adapter.require_visible(before)
    adapter.require_visible(after)
    if adapter.request_for(before, frames) != request:
        raise ValueError("The request is not what the adapter builds from the visible observation")
    choice, probability = adapter.checked_action(response, request)
    command = adapter.native_command(choice, frames)
    if row["command"] != {"id": before["frame"]["id"], **command}:
        raise ValueError("The native command is not the model's choice")
    adapter.check_transition(before, after, command, row["ack"])

    start, end = row["debugBefore"], row["debugAfter"]
    if end["frame"] - start["frame"] != frames or end["fixedSteps"] - start["fixedSteps"] != frames * 5:
        raise ValueError("The game clock did not advance by the requested window")
    moved = end["x"] - start["x"]
    if ("right" in choice and moved <= 0) or ("left" in choice and moved >= 0):
        raise ValueError("The player did not move the way the model chose")

    print(
        json.dumps(
            {
                "game": "hallowdale",
                "choice": choice,
                "probability": probability,
                "observationId": before["frame"]["id"],
                "framesPlayed": frames,
                "playerMovedTiles": round(moved, 4),
                "checks": "passed",
                "modelCalled": False,
                "gameStarted": False,
            }
        )
    )


if __name__ == "__main__":
    main()
