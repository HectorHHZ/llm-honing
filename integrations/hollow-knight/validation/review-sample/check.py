"""Check the included real decision offline: the model's response, then what the game did with it."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    source = json.loads((here / "SOURCE.json").read_text())
    for name, expected in source["includedSha256"].items():
        if hashlib.sha256((here / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Sample bytes changed: {name}")
    sys.path.insert(0, str(here.parents[2]))
    from _shared.decision import checked_action, strict_json_loads

    request = json.loads((here / "request.json").read_text())
    response = strict_json_loads((here / "response.raw").read_bytes())
    offered = request["questions"]["action"]["criteria"]
    choice, probability = checked_action(response, request, offered)

    row = json.loads((here / "transition.json").read_text())
    decision, windows = row["decision"], row["windows"]
    before, after, visible = row["debugBefore"], row["debugAfter"], row["visibleBefore"]
    if (decision["choice"], decision["probability"]) != (choice, probability):
        raise ValueError("The recorded choice differs from the validated response")
    if list(offered) != decision["offered"]:
        raise ValueError("The recorded menu differs from the submitted options")
    if not request["id"] == before["id"] == visible["frame"]["id"] == windows[0]["obs_id"]:
        raise ValueError("Request, observations and first native window are not the same moment")
    if visible["schema"] != "honing.visible.v1" or visible["frame"]["tick"] != before["frame"]:
        raise ValueError("The model-side observation is not the visible profile of this frame")

    identity, tick = before["id"], before["frame"]
    for window in windows:
        if (window["obs_id"], window["tick"]) != (identity, tick):
            raise ValueError("Native windows are not contiguous")
        if window["label"] != f"Laya chose {choice}" or window["decision"] != decision["decision"]:
            raise ValueError("A native window is not attributed to this decision")
        identity, tick = identity + 1, tick + window["frames"]
    if (after["id"], after["frame"]) != (identity, tick):
        raise ValueError("The game state after the last window does not follow from the windows")

    moved = [round(after["x"] - before["x"], 2), round(after["y"] - before["y"], 2)]
    if row["skillResult"]["skill"] != choice or after["x"] <= before["x"]:
        raise ValueError(f"{choice} did not move the knight right")
    print(
        json.dumps(
            {
                "choice": choice,
                "probability": probability,
                "offered": len(offered),
                "nativeWindows": len(windows),
                "renderFrames": after["frame"] - before["frame"],
                "debugMoved": moved,
                "health": [before["health"], after["health"]],
                "checks": "passed",
                "freshModelCall": False,
            }
        )
    )


if __name__ == "__main__":
    main()
