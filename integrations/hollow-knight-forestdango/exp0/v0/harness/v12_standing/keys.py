"""v10_sword_cadence (from v9_keys): the two questions and the keys each answer holds (N159-N168; design P60).
Unchanged from v9_keys except that the length of a decision's last bridge window is a parameter (`tail`).

Every decision asks both questions and offers every option, always in this order. Nothing is hidden, ranked or
recommended. The keys that go to the game in a decision's window are exactly the ones of the two chosen options:

  move    direction key: LEFT / RIGHT / none, down for the whole window
          jump key:      no jump    = up for the whole window (letting go while rising ends the rise, as in the game)
                         short jump = down for the first SHORT frames of the window (0.1 s), then up
                         long jump  = down for the whole window
          A jump key that goes down is a new press, with one exception: when the previous window ended with the key
          down (it was a long jump) and the knight is still rising, the key simply stays down, which is how a player
          holds a jump for more height. So jump height is chosen window by window: short = 1.7 units; long = 4.4;
          long, long = 6.8; long, long, long = 7.05, the full jump (measured, runs/probe01_jump, window = 9 frames).
          In the air (a new game has no second jump) a new press does nothing.
  attack  none           = no attack key
          strike forward = NAIL pressed in the window's first frame. The strike goes the way the knight faces; a
                           direction key held in the same window turns the knight first (the bridge then presses NAIL
                           one frame later, see HoningBridge.cs "Turn before a sideways strike")
          strike up      = UP down in the first frame together with the NAIL press
          strike down    = DOWN down in the first frame together with the NAIL press (a down strike exists only in
                           the air; on the ground the same keys strike forward)
          focus          = CAST down for the whole window (heals one mask after about 1.2 s of holding on the ground
                           with enough SOUL)

A decision's window is played as two or three bridge windows that carry these keys and nothing else; the last
one is `tail` frames long (v9_keys: 1; v10 default 2), so that the harness can read the knight's and the enemies'
speed from two observations (information only; two frames halve the pixel noise of an enemy's speed).
"""
from __future__ import annotations

SHORT = 3        # frames the jump key is down in a short jump

# id, direction, jump
MOVES = (
    ("left", -1, "none"), ("left_short_jump", -1, "short"), ("left_long_jump", -1, "long"),
    ("right", 1, "none"), ("right_short_jump", 1, "short"), ("right_long_jump", 1, "long"),
    ("stay", 0, "none"), ("short_jump", 0, "short"), ("long_jump", 0, "long"),
)
ATTACKS = ("none", "forward", "up", "down", "focus")
MOVE = {m[0]: m for m in MOVES}
DIRKEY = {-1: "LEFT", 1: "RIGHT"}


def bridge_windows(move: str, attack: str, frames: int, jump_was_down: bool, rising: bool, tail: int = 1) -> tuple[list[dict], bool]:
    """The bridge windows of one decision and whether the jump key is down at its end."""
    _, direction, jump = MOVE[move]
    if attack not in ATTACKS:
        raise ValueError(attack)
    base = [DIRKEY[direction]] if direction else []
    if attack == "focus":
        base.append("CAST")
    press, first = [], []
    if attack in ("forward", "up", "down"):
        press.append("NAIL")
        if attack != "forward":
            first = [attack.upper()] + ([DIRKEY[direction]] if direction else [])
    keep = jump != "none" and jump_was_down and rising          # the same press goes on
    if jump == "short":
        lead = SHORT if keep or not jump_was_down else SHORT + 1    # a key still down is let go for one frame first
        cuts = [lead, frames - lead - tail, tail]
    else:
        cuts = [frames - tail, tail]
    out = []
    for i, n in enumerate(cuts):
        if n <= 0:
            continue
        hold, pr = list(base), []
        if jump != "none" and (i == 0 or jump == "long"):
            if keep or i > 0:
                hold.append("JUMP")
            else:
                pr.append("JUMP")
        w = {"frames": n, "hold": hold, "press": pr + (press if i == 0 else []), "first": first if i == 0 else []}
        out.append(w)
    return out, jump == "long"


def frame_plan(move: str, frames: int, jump_was_down: bool, rising: bool, after: str = "hold"):
    """The same keys per frame, for the forecast: k -> (direction, jump key down, new press). Frames after the
    window follow the stated assumption: the direction stays held; the jump key stays down after a long jump
    (after="hold") and is up otherwise."""
    _, direction, jump = MOVE[move]
    keep = jump != "none" and jump_was_down and rising
    delay = 1 if (jump != "none" and jump_was_down and not keep) else 0

    def plan(k: int):
        if jump == "none":
            return direction, False, False
        if k >= frames:
            return direction, jump == "long" and after == "hold", False
        if jump == "short":
            lead = SHORT if keep or not jump_was_down else SHORT + 1
            down = delay <= k < lead
        else:
            down = k >= delay
        return direction, down, (not keep) and k == delay
    return plan
