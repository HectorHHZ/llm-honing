"""v10_sword_cadence (from v9_keys): the words the model reads. Plain, short English, the way a player would say it.
Facts only: no line says what to do or where to go, and the options are never reordered, hidden or marked as better
(N168).

All numbers are world units (the knight is 0.5 wide and 1.3 tall) relative to the knight, as "3.0 right, 1.5 up".

Changed from v9_keys: the sentence about jump heights follows the cadence (JUMPS; the heights are measured, see the
v10 entry of harness/CHANGELOG.md); the legend explains the 'then ... or ...' places of an AIR tag when the reach
forecast is on; a decision's length is said to two decimals when it needs them ("0.43"); an option text that would be
longer than the decision model's limit of 48 tokens per option loses detail by a fixed rule (`fit`).
"""
from __future__ import annotations

from .keys import ATTACKS, MOVES

BRIEFING = (
    "You play a small knight in a 2D cave platformer and explore to find the way onward. Each decision plays {secs} "
    "seconds; the game waits while you decide. Answer both questions every time: the movement keys and the attack key. "
    "Rules a player learns: a room is left through an opening at its border, and the way can lead up, down, left or right. "
    "{jumps} "
    "Enemies and spikes hurt on touch and take one mask; with no masks left the knight dies. A sword strike reaches about "
    "2 units, needs 0.4 seconds between strikes, and each hit on an enemy fills some SOUL. Holding focus on the ground for "
    "about 1.2 seconds turns a third of the SOUL into one mask. Some doors and walls break when struck. "
    "Distances are in units; the knight is 0.5 wide and 1.3 tall."
)
# the jump heights per cadence (frames per decision), measured in the game (9: v9_keys probe01_jump; 13: v10 probe)
JUMPS = {
    9: "A long jump held for two or three decisions in a row rises higher (up to 7 units); a short jump rises 1.7.",
    13: "A long jump rises 5.6 units, or 7 when it is held for two decisions in a row; a short jump rises 1.7.",
}
JUMPS_OTHER = "A long jump rises higher the more decisions in a row it is held (up to 7 units); a short jump rises 1.7."

MOVE_TEXT = {
    "left": "Hold left, no jump", "left_short_jump": "Hold left, short jump", "left_long_jump": "Hold left, long jump",
    "right": "Hold right, no jump", "right_short_jump": "Hold right, short jump", "right_long_jump": "Hold right, long jump",
    "stay": "No direction, no jump", "short_jump": "No direction, short jump", "long_jump": "No direction, long jump",
}
ATTACK_TEXT = {
    "none": "Do not attack", "forward": "Strike forward", "up": "Strike upward",
    "down": "Strike downward (only in the air)", "focus": "Hold focus to heal",
}
BRIEFING_SHORT = (
    "You play a small knight in a 2D cave platformer and explore to find the way onward. Each decision plays {secs} "
    "seconds; the game waits while you decide. Answer every question: they are the keys you hold. "
    "Distances are in units; the knight is 0.5 wide and 1.3 tall."
)
LEGEND = (
    "Each option starts with what it leads to. Movement: WALK, LAND, AIR (still in the air after this decision; {air}), "
    "STAY, WALL (blocked), HURT (you lose a mask), UNSEEN (out of what you have seen), "
    "NEW PLACE (you have not stood there) or SEEN and how many times you stood there. Attack: HIT, MISS, NOT READY, "
    "NO EFFECT, HEAL, NO KEY."
)
AIR_HOLD = "'then' says where holding the keys ends"                       # v9_keys
AIR_REACH = ("'then' lists, top to bottom and in whole units, the places you can still land on by holding or letting go "
             "of these keys at the next decisions; '+N more' when not all fit")
FRONTIER = ("After a place you have stood on, a sideways option may add what lies further that way on this floor: "
            "'new ground N further' (the nearest part you have not stood on), 'not seen beyond N', or that all is seen "
            "up to a wall or an edge.")
DIR_TEXT = {"left": "Hold left", "right": "Hold right", "none": "No direction key"}
JUMP_TEXT = {"none": "No jump", "short": "Short jump", "long": "Long jump (hold the jump key)"}
SPLIT_TO_MOVE = {("left", "none"): "left", ("left", "short"): "left_short_jump", ("left", "long"): "left_long_jump",
                 ("right", "none"): "right", ("right", "short"): "right_short_jump", ("right", "long"): "right_long_jump",
                 ("none", "none"): "stay", ("none", "short"): "short_jump", ("none", "long"): "long_jump"}
DIR_ASK = "Choose the direction key for the next {secs} seconds."
JUMP_ASK = "Choose the jump key for the next {secs} seconds."
MOVE_ASK = "Choose the movement keys for the next {secs} seconds."
ATTACK_ASK = "Choose the attack key for the next {secs} seconds."


STEP = 0.1     # distances are said to this step (set from the knob `round` by the agent)
OPTION_TOKENS = 48      # the decision model (Laya) refuses a request with a longer option text

_TOKENIZER = None
_TOKENIZER_FILE = "<set by the Exp 0 adapter>"


def option_tokens(key: str, body: str) -> int:
    """Tokens of one option as Laya counts it (" <id>: <text>"). With Laya's tokenizer when it can be loaded;
    otherwise a count that is never below the real one in the texts of this harness (one token per word, number
    part or punctuation mark)."""
    global _TOKENIZER
    line = f" {key}: {body}"
    if _TOKENIZER is None:
        try:
            from tokenizers import Tokenizer
            _TOKENIZER = Tokenizer.from_file(_TOKENIZER_FILE)
        except Exception:
            _TOKENIZER = False
    if _TOKENIZER:
        return len(_TOKENIZER.encode(line, add_special_tokens=False).ids)
    import re
    return len(re.findall(r"[A-Za-z]+|\d|[^\sA-Za-z\d]", line)) + 2


def q(v: float) -> float:
    """A distance as it is said. With step 0.5 the text no longer flips between "1.9" and "2.0" from run to run
    (the knight's place differs by a few hundredths with the game's frame phase); the default is 0.1."""
    return round(v / STEP) * STEP + 0.0


def small() -> float:
    """A distance below this is said as none."""
    return max(0.15, STEP / 2)


def rel(dx: float, dy: float, zero: str = "here") -> str:
    parts = []
    if abs(dx) >= small():
        parts.append(f"{abs(q(dx)):.1f} {'right' if dx > 0 else 'left'}")
    if abs(dy) >= small():
        parts.append(f"{abs(q(dy)):.1f} {'up' if dy > 0 else 'down'}")
    return ", ".join(parts) if parts else zero


def rel_whole(dx: float, dy: float, zero: str = "here") -> str:
    """As `rel`, to the nearest whole unit (the places of a reach forecast)."""
    parts = []
    if round(abs(dx)) >= 1:
        parts.append(f"{round(abs(dx))} {'right' if dx > 0 else 'left'}")
    if round(abs(dy)) >= 1:
        parts.append(f"{round(abs(dy))} {'up' if dy > 0 else 'down'}")
    return ", ".join(parts) if parts else zero


def visited(n: int) -> str:
    return "a new place" if n == 0 else ("been there once" if n == 1 else f"been there {n} times")


def secs(frames: int) -> str:
    """0.3 for 9 frames (as v9_keys), 0.43 for 13."""
    s = f"{frames / 30:.2f}"
    return s[:-1] if s.endswith("0") else s


def jumps(frames: int) -> str:
    return JUMPS.get(frames, JUMPS_OTHER)


def fit(key: str, tag: str, keys: str, legacy: bool) -> tuple[str, list[str]]:
    """The tag of an option, shortened when the option would be too long; returns (tag, what was cut).
    legacy = v9_keys' rule (by characters). Otherwise by tokens, in this fixed order: the ceiling note, the 'or'
    places of an AIR tag, a bracketed remark."""
    cut = []
    if not tag:
        return tag, cut
    if legacy:
        if len(tag) + len(keys) > 96:
            tag = tag.replace(", head hits ceiling", "")
        if len(tag) + len(keys) > 96:
            tag = tag.split(" (")[0].rstrip(".") + "."
        return tag, cut
    if option_tokens(key, f"{tag} {keys}") > OPTION_TOKENS and ", head hits ceiling" in tag:
        tag = tag.replace(", head hits ceiling", "")
        cut.append("ceiling")
    if option_tokens(key, f"{tag} {keys}") > OPTION_TOKENS and "; or " in tag:
        tag = tag.split("; or ")[0].rstrip(".") + "."
        cut.append("or")
    if option_tokens(key, f"{tag} {keys}") > OPTION_TOKENS and " (" in tag:
        tag = tag.split(" (")[0].rstrip(".") + "."
        cut.append("bracket")
    return tag, cut


def questions(frames: int, forecasts: dict | None, attack_tags: dict | None, split: str = "one",
              legacy_fit: bool = True, cuts: dict | None = None) -> dict:
    """The questions. With forecasts / tags given, each option text starts with its own tag (placement "option"):
    "<TAG>. <keys>". Every option is there, in the fixed order of keys.py. `cuts` collects what `fit` removed."""
    s = secs(frames)

    def opt(key, tag, keys):
        tag, cut = fit(key, tag, keys, legacy_fit)
        if cut and cuts is not None:
            cuts[key] = cut
        return f"{tag} {keys}" if tag else keys
    attack = {a: opt(a, (attack_tags or {}).get(a), ATTACK_TEXT[a]) for a in ATTACKS}
    out = {}
    if split == "two":
        fc = forecasts or {}
        out["direction"] = {"type": "choice", "instructions": DIR_ASK.format(secs=s),
                            "criteria": {d: opt(d, fc.get(SPLIT_TO_MOVE[(d, "none")]), DIR_TEXT[d]) for d in ("left", "right", "none")}}
        out["jump"] = {"type": "choice", "instructions": JUMP_ASK.format(secs=s),
                       "criteria": {j: opt(j, fc.get(SPLIT_TO_MOVE[("none", j)]), JUMP_TEXT[j]) for j in ("none", "short", "long")}}
    else:
        out["move"] = {"type": "choice", "instructions": MOVE_ASK.format(secs=s),
                       "criteria": {m: opt(m, (forecasts or {}).get(m), MOVE_TEXT[m]) for m, _, _ in MOVES}}
    out["attack"] = {"type": "choice", "instructions": ATTACK_ASK.format(secs=s), "criteria": attack}
    return out
