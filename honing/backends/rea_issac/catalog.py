"""Independent movement, shooting and bomb decisions for REA-Issac."""

from __future__ import annotations

from typing import Any

from honing.backends.rea_issac.engine import Action

PROTOCOL = "rea-issac-typed-v1"
DIRECTIONS = {
    "NONE": (0, 0),
    "N": (0, -1),
    "NE": (1, -1),
    "E": (1, 0),
    "SE": (1, 1),
    "S": (0, 1),
    "SW": (-1, 1),
    "W": (-1, 0),
    "NW": (-1, -1),
}
OPTIONS = {"move": tuple(DIRECTIONS), "shoot": tuple(DIRECTIONS), "bomb": ("NO", "YES")}


def criteria(question: str) -> dict[str, str]:
    if question == "bomb":
        return {
            "NO": "Release the bomb action.",
            "YES": "Place a bomb if available; its explosion also hurts the player.",
        }
    verb = "Move" if question == "move" else "Shoot"
    return {name: f"{verb} toward {name}; NONE means no {question}." for name in DIRECTIONS}


def action_for(choices: dict[str, str]) -> dict[str, Any]:
    mx, my = DIRECTIONS[choices["move"]]
    sx, sy = DIRECTIONS[choices["shoot"]]
    return Action(mx, my, sx, sy, choices["bomb"] == "YES").__dict__.copy()
