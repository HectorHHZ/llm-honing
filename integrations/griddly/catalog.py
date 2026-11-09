"""Pinned Spiders rules and relative actions, without loading the optional engine."""

PROTOCOL = "spiders-full-state-v1"
FORMAT = "griddly-match-v1"
GAME = "Single-Player/Mini-Grid/minigrid-spiders.yaml"
# The Windows and Linux wheels use different line endings for the same GDY.
GAME_SHA256 = "c33b12711b5b92ece3a8d9af835e57d9053bd66b7d1e3d5fd332bf2d845ddb01"
VERSIONS = {
    "griddly": "1.6.7",
    "gym": "0.23.1",
    "numpy": "1.23.5",
    "PyYAML": "6.0.2",
    "imageio": "2.34.2",
    "pillow": "10.4.0",
    "cloudpickle": "2.2.1",
    "gym-notices": "0.0.8",
}
ACTIONS = {
    "WAIT": 0,
    "TURN_LEFT": 1,
    "FORWARD": 2,
    "TURN_RIGHT": 3,
    "BACKWARD": 4,
}
DESCRIPTIONS = {
    "WAIT": "Wait for one tick; enemies still act",
    "TURN_LEFT": "Rotate left without moving",
    "FORWARD": "Move one cell forward relative to facing",
    "TURN_RIGHT": "Rotate right without moving",
    "BACKWARD": "Move one cell backward, keeping facing",
}
