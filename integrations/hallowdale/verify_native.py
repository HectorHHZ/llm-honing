"""Model-free native checks of the hallowdale bridge: controls, refused commands, watchdog and action effects."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from run import ACTIONS, Game, decide_and_step, native_command, write_json

HERE = Path(__file__).resolve().parent
SAVES = HERE / "validation/debug-saves"
TICKS_PER_FRAME = 5
STATE_FIELDS = ("x", "y", "vx", "vy", "health", "soul", "grounded", "frozen", "facing", "frame", "fadeAlpha")

# The keys every offered choice must apply: held for the whole window, pressed at its start.
EXPECTED_KEYS: dict[str, tuple[set[str], set[str]]] = {
    "wait": (set(), set()),
    "left": ({"LEFT"}, set()),
    "right": ({"RIGHT"}, set()),
    "up": ({"UP"}, set()),
    "down": ({"DOWN"}, set()),
    "jump": (set(), {"JUMP"}),
    "jump_left": ({"LEFT"}, {"JUMP"}),
    "jump_right": ({"RIGHT"}, {"JUMP"}),
    "short_jump": (set(), {"JUMP"}),
    "short_jump_left": ({"LEFT"}, {"JUMP"}),
    "short_jump_right": ({"RIGHT"}, {"JUMP"}),
    "attack": (set(), {"NAIL"}),
    "attack_left": ({"LEFT"}, {"NAIL"}),
    "attack_right": ({"RIGHT"}, {"NAIL"}),
    "attack_up": ({"UP"}, {"NAIL"}),
    "attack_down": ({"DOWN"}, {"NAIL"}),
    "dash": (set(), {"DASH"}),
    "dash_left": ({"LEFT"}, {"DASH"}),
    "dash_right": ({"RIGHT"}, {"DASH"}),
    "interact": (set(), {"INTERACT"}),
    "focus": ({"FOCUS"}, set()),
    "cast": (set(), {"CAST"}),
    "cast_left": ({"LEFT"}, {"CAST"}),
    "cast_right": ({"RIGHT"}, {"CAST"}),
    "cast_up": ({"UP"}, {"CAST"}),
    "cast_down": ({"DOWN"}, {"CAST"}),
    "c_dash": ({"C_DASH"}, set()),
    "dream_nail": ({"D_NAIL"}, set()),
}


class Play:
    """One booted game, with the evaluation-only debug state read next to every window."""

    def __init__(self, game: Game, first: dict[str, Any]) -> None:
        self.game = game
        self.observation = first
        self.steps: list[dict[str, Any]] = []

    def debug(self) -> dict[str, Any]:
        return json.loads((self.game.session / "debug_observation.json").read_text())

    def state(self) -> dict[str, Any]:
        debug = self.debug()
        return {key: debug[key] for key in STATE_FIELDS}

    def play(self, action: str, frames: int, *, keep: bool = True, **extra: Any) -> dict[str, Any]:
        """Play one window; keep=False leaves routine waiting out of the printed evidence."""
        before = self.state()
        self.observation, ack = self.game.step(self.observation, {"action": action, "frames": frames, **extra})
        step = {
            "action": action,
            "frames": frames,
            **extra,
            "started": ack["started"],
            "children": ack["children"],
            "keys": ack["keys"],
            "before": before,
            "after": self.state(),
        }
        if keep:
            self.steps.append(step)
        return step

    def wait_until(self, ready: Callable[[dict[str, Any]], bool], *, limit: int = 240) -> dict[str, Any]:
        """Let the game run until the debug state satisfies a condition."""
        for _ in range(limit):
            state = self.state()
            if ready(state):
                return state
            self.play("wait", 3, keep=False)
        raise AssertionError("The expected game state did not arrive")

    def settled(self) -> dict[str, Any]:
        return self.wait_until(lambda s: s["grounded"] and not s["frozen"] and s["fadeAlpha"] == 0 and s["vy"] == 0)

    def blocking_text(self) -> list[dict[str, Any]]:
        return [item for item in self.observation["hud"]["text"] if item.get("blocking")]


class SingleTickGame(Game):
    """One bridge frame per game tick at the normal step size, so that a window can end on any tick."""

    def environment(self, display: str) -> dict[str, str]:
        return {**super().environment(display), "HONING_FRAME_HZ": "150", "HONING_TICKS_PER_FRAME": "1"}


@contextmanager
def booted(
    exe: Path,
    output: Path,
    name: str,
    *,
    save: str | None = None,
    idle: float = 300,
    game_class: type[Game] = Game,
) -> Iterator[Play]:
    """Boot one owned game; a debug start loads a hand-made save and says so in the evidence."""
    scene = f"save:{SAVES / save}" if save else "new"
    game = game_class(exe, output / name, idle_seconds=idle, scene=scene)
    try:
        yield Play(game, game.start())
    finally:
        game.close()


def result(name: str, start: str, play: Play, measured: dict[str, Any], checks: dict[str, bool]) -> dict[str, Any]:
    failed = [check for check, passed in checks.items() if not passed]
    return {
        "name": name,
        "start": start,
        "passed": not failed,
        "failedChecks": failed,
        "measured": measured,
        "steps": play.steps,
    }


# ---- controls, edges and refusals ---------------------------------------------------------------------------
def check_controls(exe: Path, output: Path) -> dict[str, Any]:
    """Every offered choice applies exactly its documented keys, and the paused game does not advance."""
    with booted(exe, output, "controls") as play:
        paused = (play.game.session / "visible_observations.jsonl").read_bytes()
        time.sleep(1)
        checks = {
            "pausedGameDoesNotAdvance": paused == (play.game.session / "visible_observations.jsonl").read_bytes()
        }
        for choice, (hold, press) in EXPECTED_KEYS.items():
            step = play.play(**native_command(choice, 12))
            keys, total = step["keys"], 12 * TICKS_PER_FRAME
            extra = set(keys) - hold - press
            checks[f"{choice}:keys"] = hold | press <= set(keys) and extra <= {"JUMP"}
            checks[f"{choice}:held"] = all(keys[key]["ticks"] == total for key in hold)
            expected_press = 3 * TICKS_PER_FRAME if choice.startswith("short_jump") else TICKS_PER_FRAME
            checks[f"{choice}:press"] = all(
                (
                    keys[key]["ticks"] in (expected_press, expected_press - 1, total - 1, total)
                    if key == "JUMP"
                    else keys[key]["ticks"] == TICKS_PER_FRAME
                )
                for key in press
            )
        checks["everyChoiceCovered"] = set(EXPECTED_KEYS) == set(ACTIONS)
        moved = {step["action"]: step["after"]["x"] - step["before"]["x"] for step in play.steps}
        checks["leftMovesLeft"] = moved["left"] < 0
        checks["rightMovesRight"] = moved["right"] > 0
        return result("controls", "new game", play, {"offeredChoices": len(ACTIONS)}, checks)


def jump_height(play: Play) -> tuple[float, dict[str, Any]]:
    """Play one full jump frame by frame and return its height in tiles with the jump's own window."""
    ground = play.settled()["y"]
    step = play.play("jump", 1)
    top = max(ground, step["after"]["y"])
    for _ in range(90):
        state = play.play("wait", 1, keep=False)["after"]
        top = max(top, state["y"])
        if state["grounded"]:
            break
    return round(top - ground, 3), step


def check_repeats(exe: Path, output: Path) -> dict[str, Any]:
    """A repeated jump or strike gets a new press edge and happens again."""
    with booted(exe, output, "repeats") as play:
        first, _ = jump_height(play)
        second, again = jump_height(play)
        play.settled()
        strikes = [play.play("attack_right", 15), play.play("attack_right", 15)]
        checks = {
            "firstJumpRose": first > 5,
            "secondJumpRoseAsHigh": abs(second - first) < 0.05,
            "secondJumpReleasedTheKeyFirst": again["keys"]["JUMP"]["first_tick"] == 1,
            "bothStrikesStarted": all("NAIL" in step["started"] for step in strikes),
        }
        return result("repeated presses", "new game", play, {"jumpHeightsInTiles": [first, second]}, checks)


def check_refusals(exe: Path, output: Path) -> dict[str, Any]:
    """Invalid requests are refused before execution and leave the paused game where it was."""
    with booted(exe, output, "refusals") as play:
        identity = play.observation["frame"]["id"]
        bad = {
            "malformed JSON": "junk",
            "stale id": json.dumps({"id": identity - 1, "action": "wait", "frames": 1}),
            "future id": json.dumps({"id": identity + 1, "action": "wait", "frames": 1}),
            "string id": json.dumps({"id": str(identity), "action": "wait", "frames": 1}),
            "duplicate key": f'{{"id":{identity},"id":{identity},"action":"wait","frames":1}}',
            "missing frames": json.dumps({"id": identity, "action": "wait"}),
            "zero frames": json.dumps({"id": identity, "action": "wait", "frames": 0}),
            "too many frames": json.dumps({"id": identity, "action": "wait", "frames": 601}),
            "text frames": json.dumps({"id": identity, "action": "wait", "frames": "12"}),
            "unknown action": json.dumps({"id": identity, "action": "fly", "frames": 1}),
            "pause key": json.dumps({"id": identity, "action": "keys", "press": ["PAUSE"], "frames": 1}),
            "map key": json.dumps({"id": identity, "action": "keys", "hold": ["OPEN_MAP"], "frames": 1}),
            "unknown key": json.dumps({"id": identity, "action": "keys", "hold": ["F12"], "frames": 1}),
            "button in first": json.dumps({"id": identity, "action": "left", "first": ["NAIL"], "frames": 1}),
            "jump_frames without jump": json.dumps({"id": identity, "action": "left", "jump_frames": 1, "frames": 2}),
            "zero jump_frames": json.dumps({"id": identity, "action": "jump", "jump_frames": 0, "frames": 2}),
            "jump_frames above frames": json.dumps({"id": identity, "action": "jump", "jump_frames": 3, "frames": 2}),
        }
        stream = play.game.session / "visible_observations.jsonl"
        before = stream.read_bytes()
        refused = {}
        for name, text in bad.items():
            count = (
                len((play.game.session / "acks.jsonl").read_text().splitlines())
                if (play.game.session / "acks.jsonl").exists()
                else 0
            )
            temporary = play.game.session / "action.tmp"
            temporary.write_text(text)
            temporary.replace(play.game.session / "action.json")
            deadline = time.monotonic() + 20
            ack: dict[str, Any] = {}
            while time.monotonic() < deadline:
                lines = (
                    (play.game.session / "acks.jsonl").read_text().splitlines()
                    if (play.game.session / "acks.jsonl").exists()
                    else []
                )
                if len(lines) > count:
                    ack = json.loads(lines[-1])
                    break
                time.sleep(0.01)
            refused[name] = ack.get("reason") if ack.get("status") == "refused" else None
        checks = {name: reason is not None for name, reason in refused.items()}
        checks["nothingWasPlayed"] = stream.read_bytes() == before and play.state()["frame"] == 0
        after, _ = play.game.step(play.observation, {"action": "wait", "frames": 1})
        checks["stillAcceptsAValidCommand"] = after["frame"]["tick"] == 1
        return result("refused commands", "new game", play, {"refusals": refused}, checks)


def check_watchdog(exe: Path, output: Path) -> dict[str, Any]:
    """Stale and invalid commands neither run nor extend the idle deadline."""
    with booted(exe, output, "watchdog", idle=3) as play:
        started = time.monotonic()
        sent = 0
        process = play.game.process
        assert process is not None
        while process.poll() is None and time.monotonic() - started < 60:
            sent += 1
            temporary = play.game.session / "action.tmp"
            temporary.write_text(json.dumps({"id": 0, "action": "wait", "frames": 1, "attempt": sent}))
            temporary.replace(play.game.session / "action.json")
            time.sleep(0.02)
        seconds = time.monotonic() - started
        log = (play.game.session / "bridge.log").read_text()
        checks = {
            "exitedByItself": process.poll() is not None,
            "exitedNearTheDeadline": seconds < 30,
            "loggedTheTimeout": "timed out waiting for an action" in log,
            "nothingWasPlayed": play.debug()["frame"] == 0,
        }
        return result(
            "idle watchdog under a stale-command flood",
            "new game",
            play,
            {"idleSeconds": 3, "secondsToExit": round(seconds, 2), "staleCommandsSent": sent},
            checks,
        )


def check_rejected_model(exe: Path, output: Path) -> dict[str, Any]:
    """A response that fails validation keeps its raw bytes and plays nothing."""

    class BadModel:
        def decide(self, request: dict[str, Any], *, raw_path: Path) -> dict[str, Any]:
            response = {"id": request["id"], "stateText": "missing required current state"}
            raw_path.write_text(json.dumps(response))
            return response

    with booted(exe, output, "rejected-model") as play:
        identity = play.observation["frame"]["id"]
        refused = False
        try:
            decide_and_step(play.game, BadModel(), play.observation, 12)  # type: ignore[arg-type]
        except ValueError:
            refused = True
        time.sleep(0.5)
        checks = {
            "responseWasRejected": refused,
            "rawBytesKept": (play.game.session / f"responses/{identity:06d}.raw").is_file(),
            "noCommandWritten": not (play.game.session / "action.json").exists(),
            "nothingWasPlayed": play.state()["frame"] == 0,
        }
        return result("rejected model response", "new game", play, {}, checks)


# ---- what each action does in the game --------------------------------------------------------------------
def probe_strike_breaks_door(exe: Path, output: Path) -> dict[str, Any]:
    with booted(exe, output, "door") as play:
        play.settled()
        blocked_at = None
        for _ in range(60):
            step = play.play("right", 6, keep=False)
            if step["after"]["x"] - step["before"]["x"] < 0.05:
                blocked_at = step["after"]["x"]
                break
        doors = {item["id"] for item in play.observation["objects"] if item["look"] == "door"}
        strike = play.play("attack_right", 3)
        play.play("wait", 12, keep=False)
        remaining = {item["id"] for item in play.observation["objects"] if item["look"] == "door"}
        walked = play.play("right", 12)
        checks = {
            "aDoorBlockedTheWay": blocked_at is not None and bool(doors),
            "strikeStarted": "NAIL" in strike["started"],
            "thatDoorIsGone": bool(doors - remaining),
            "theWayIsOpen": walked["after"]["x"] > (blocked_at or 0) + 0.5,
        }
        measured = {"blockedAtX": blocked_at, "doorsBroken": sorted(doors - remaining)}
        return result("attack_right breaks a door", "new game", play, measured, checks)


def probe_jump_heights(exe: Path, output: Path) -> dict[str, Any]:
    """Jump height follows how long the key is held; without jump_frames a jump is a full jump."""

    def height(play: Play, **extra: Any) -> float:
        ground = play.settled()["y"]
        frames = extra.get("jump_frames", 1)
        top = max(ground, play.play("jump", frames, **extra)["after"]["y"])
        for _ in range(90):
            state = play.play("wait", 1, keep=False)["after"]
            top = max(top, state["y"])
            if state["grounded"]:
                break
        return round(top - ground, 3)

    with booted(exe, output, "jump-heights") as play:
        heights = {
            "jump_frames=1": height(play, jump_frames=1),
            "jump_frames=3": height(play, jump_frames=3),
            "jump_frames=6": height(play, jump_frames=6),
            "jump_frames=12": height(play, jump_frames=12),
            "no jump_frames (full jump)": height(play),
            "jump_frames=3 again": height(play, jump_frames=3),
        }
        values = list(heights.values())
        checks = {
            "heightGrowsWithHold": values[0] < values[1] < values[2] < values[3] <= values[4],
            "fullJumpLeavesShortJumpUnchanged": values[5] == values[1],
        }
        return result("variable jump height", "new game", play, {"heightInTiles": heights}, checks)


def probe_focus(exe: Path, output: Path) -> dict[str, Any]:
    with booted(exe, output, "focus", save="spikes.save") as play:
        play.settled()
        for _ in range(60):
            if play.play("right", 6, keep=False)["after"]["health"] < 5:
                break
        hurt = play.settled()
        step = play.play("focus", 60)
        healed = step["after"]
        checks = {
            "tookDamageFirst": hurt["health"] == 4,
            "healthRestored": healed["health"] == hurt["health"] + 1,
            "soulSpent": healed["soul"] == hurt["soul"] - 33,
            "focusStarted": "FOCUS" in step["started"],
        }
        measured = {
            "healthBefore": hurt["health"],
            "healthAfter": healed["health"],
            "soulBefore": hurt["soul"],
            "soulAfter": healed["soul"],
        }
        return result("focus heals with soul", "debug save spikes.save", play, measured, checks)


def probe_spells(exe: Path, output: Path) -> dict[str, Any]:
    with booted(exe, output, "spells", save="abilities.save") as play:
        play.settled()
        spells = {}
        for action, name in (
            ("cast_right", "CAST_VENGEFUL_SPIRIT"),
            ("cast_up", "CAST_HOWLING_WRAITHS"),
            ("cast_down", "DIVE_HOP"),
        ):
            soul = play.settled()["soul"]
            step = play.play(action, 30)
            spells[action] = {
                "gameStarted": step["started"],
                "soulBefore": soul,
                "soulAfter": step["after"]["soul"],
                "expected": name,
            }
        checks = {
            f"{action}:started": spell["expected"] in spell["gameStarted"] for action, spell in spells.items()
        } | {f"{action}:soul": spell["soulAfter"] == spell["soulBefore"] - 33 for action, spell in spells.items()}
        return result("spells spend soul", "debug save abilities.save", play, {"spells": spells}, checks)


def probe_interact(exe: Path, output: Path) -> dict[str, Any]:
    with booted(exe, output, "interact", save="lore.save") as play:
        play.settled()
        labels = [item for item in play.observation["hud"]["text"] if item["kind"] == "label"]
        before = play.blocking_text()
        play.play("interact", 3)
        for _ in range(40):
            if play.blocking_text():
                break
            play.play("wait", 3, keep=False)
        opened = play.blocking_text()
        frozen = play.play("left", 12)
        presses = 0
        for _ in range(60):
            if play.blocking_text():
                play.play("interact", 3, keep=False)
                presses += 1
            elif not play.debug()["interacting"]:
                break
            play.play("wait", 6, keep=False)
        walked = play.play("left", 12)
        checks = {
            "labelShownAtTheTrigger": bool(labels),
            "noTextBeforeInteract": not before,
            "textOpenedOnInteract": bool(opened),
            "playerHeldWhileTextIsOpen": frozen["after"]["x"] == frozen["before"]["x"],
            "textClosedByInteract": not play.blocking_text(),
            "playerMovesAfterwards": walked["after"]["x"] < walked["before"]["x"],
        }
        measured = {
            "label": [item["content"] for item in labels[:1]],
            "openedTextKind": [item["kind"] for item in opened[:1]],
            "openedTextCharacters": sum(len(item.get("content", "")) for item in opened[:1]),
            "pressesToClose": presses,
        }
        return result("interact opens and closes text", "debug save lore.save", play, measured, checks)


def probe_c_dash(exe: Path, output: Path) -> dict[str, Any]:
    with booted(exe, output, "c-dash", save="abilities.save") as play:
        play.settled()
        walk = play.play("left", 12)["after"]["vx"]
        play.settled()
        short = play.play("c_dash", 10)
        after_short = play.play("wait", 6)
        play.settled()
        charge = play.play("c_dash", 30)
        launch = play.play("wait", 6)
        cancel = play.play("jump", 3)
        checks = {
            "shortHoldStartsCharge": "C_DASH_CHARGE" in short["started"],
            "shortHoldDoesNotLaunch": "C_DASH" not in after_short["started"],
            "longHoldCharges": "C_DASH_CHARGE_CRYSTALS" in charge["children"],
            "releaseLaunches": "C_DASH" in launch["started"],
            "launchIsFasterThanWalking": abs(launch["after"]["vx"]) > 2 * abs(walk),
            "jumpEndsTheDash": "C_DASH_COOLDOWN" in cancel["started"],
        }
        measured = {"walkSpeed": walk, "cDashSpeed": launch["after"]["vx"]}
        return result("c-dash charge and release", "debug save abilities.save", play, measured, checks)


def probe_dream_nail(exe: Path, output: Path) -> dict[str, Any]:
    with booted(exe, output, "dream-nail", save="abilities.save") as play:
        play.settled()
        short = play.play("dream_nail", 10)
        play.play("wait", 15, keep=False)
        play.settled()
        swing = play.play("dream_nail", 32)
        checks = {
            "holdStartsTheAction": "DREAM_NAIL" in short["started"],
            "shortHoldDoesNotSwing": "DREAM_NAIL" not in short["children"],
            "fullHoldSwings": "DREAM_NAIL" in swing["children"],
        }
        return result("dream nail swing", "debug save abilities.save", play, {}, checks)


def probe_movement_abilities(exe: Path, output: Path) -> dict[str, Any]:
    with booted(exe, output, "movement", save="abilities.save") as play:
        play.settled()
        walk = play.play("right", 12)["after"]["vx"]
        play.settled()
        dash = play.play("dash_right", 4)
        play.play("wait", 24, keep=False)
        repeated = [play.play("dash_left", 20), play.play("dash_left", 20)]
        play.play("wait", 24, keep=False)
        play.settled()
        rise = play.play("jump", 6)
        flap = play.play("jump", 6)
        checks = {
            "dashStarted": "DASH" in dash["started"],
            "dashIsFasterThanWalking": abs(dash["after"]["vx"]) > 1.5 * abs(walk),
            "repeatedDashStartsAgain": all("DASH" in step["started"] for step in repeated),
            "firstJumpLeftTheGround": not rise["after"]["grounded"],
            "secondJumpInTheAirFlaps": "FLAP" in flap["started"],
        }
        measured = {"walkSpeed": walk, "dashSpeed": dash["after"]["vx"]}
        return result("dash and double jump", "debug save abilities.save", play, measured, checks)


def probe_strike_directions(exe: Path, output: Path) -> dict[str, Any]:
    """The strike goes the way the action says; a downward strike needs the air."""
    with booted(exe, output, "strikes") as play:
        play.settled()
        strikes = {}
        for action in ("attack_left", "attack_right", "attack_up"):
            strikes[action] = play.play(action, 3)["children"]
            play.play("wait", 15, keep=False)
            play.settled()
        play.play("jump", 6)
        strikes["attack_down"] = play.play("attack_down", 3)["children"]
        checks = {
            action: f"NAIL_{action.removeprefix('attack_').upper()}" in children
            for action, children in strikes.items()
        }
        return result("strike directions", "new game", play, {"attachedStrike": strikes}, checks)


def probe_death_reload(exe: Path, output: Path) -> dict[str, Any]:
    """A death on the last tick of a window plays the game's reload tick once and repeats no command."""
    with booted(exe, output, "death", save="spikes.save", game_class=SingleTickGame) as play:
        hits = 0
        for _ in range(6000):
            state = play.state()
            if state["health"] <= 0:
                break
            health = state["health"]
            if state["frozen"] or state["fadeAlpha"] > 0 or not state["grounded"]:
                play.play("wait", 10, keep=False)
            else:
                play.play("right" if state["x"] < 92 else "left", 10, keep=False)
            hits += int(play.state()["health"] < health)
        windows = reload_ticks = 0
        while play.debug()["deaths"] == 0 and windows < 3000:
            before = play.debug()["fixedSteps"]
            play.play("wait", 1, keep=False)
            windows += 1
            reload_ticks += int(play.debug()["fixedSteps"] - before == 2)
        final = play.debug()
        for _ in range(30):
            play.play("wait", 1, keep=False)
        log = (play.game.session / "bridge.log").read_text()
        checks = {
            "playerDied": final["deaths"] == 1,
            "healthRestoredByTheReload": final["health"] == final["healthMax"],
            "reloadTickPlayedOnce": reload_ticks == 1,
            "gameContinuesAfterTheReload": play.debug()["deaths"] == 1,
            "noWindowEndedWithoutAGame": "no game in progress" not in log,
        }
        measured = {"spikeHits": hits, "windowsUntilTheReload": windows, "windowsWithTheReloadTick": reload_ticks}
        return result("death on the last tick of a window", "debug save spikes.save", play, measured, checks)


def probe_locked(exe: Path, output: Path) -> dict[str, Any]:
    """At the start of a new game the locked abilities do nothing, as for a keyboard player."""
    with booted(exe, output, "locked") as play:
        play.settled()
        abilities = play.debug()["abilities"]
        soul = play.state()["soul"]
        locked = {
            "dash_right": "DASH",
            "cast": "CAST_VENGEFUL_SPIRIT",
            "c_dash": "C_DASH_CHARGE",
            "dream_nail": "DREAM_NAIL",
        }
        steps = {action: play.play(action, 30) for action in locked}
        checks = {f"{action}:nothingStarted": name not in steps[action]["started"] for action, name in locked.items()}
        checks["noAbilitiesUnlocked"] = abilities == []
        checks["soulUnchanged"] = play.state()["soul"] == soul
        return result("locked abilities do nothing", "new game", play, {"abilities": abilities}, checks)


CHECKS = (check_controls, check_repeats, check_refusals, check_watchdog, check_rejected_model)
PROBES = (
    probe_strike_breaks_door,
    probe_strike_directions,
    probe_jump_heights,
    probe_focus,
    probe_spells,
    probe_interact,
    probe_c_dash,
    probe_dream_nail,
    probe_movement_abilities,
    probe_death_reload,
    probe_locked,
)


def run_all(exe: Path, output: Path, only: set[str]) -> bool:
    """Run every selected check, keeping going after a failure so that the report is complete."""
    output.mkdir(parents=True, exist_ok=False)
    passed = True
    for label, group in (("native-checks", CHECKS), ("action-probes", PROBES)):
        reports = []
        for check in group:
            if only and check.__name__ not in only:
                continue
            try:
                report = check(exe, output)
            except Exception as error:
                report = {"name": check.__name__, "passed": False, "error": f"{type(error).__name__}: {error}"}
            passed = passed and report["passed"]
            reports.append(report)
            print(
                f"{'PASS' if report['passed'] else 'FAIL'} {report['name']} {report.get('failedChecks', '')}",
                flush=True,
            )
        write_json(output / f"{label}.json", {"passed": all(item["passed"] for item in reports), "results": reports})
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True, help="Executable printed by build.py")
    parser.add_argument("--output", type=Path, required=True, help="New local directory")
    parser.add_argument("--only", action="append", default=[], help="Run only this check or probe function")
    args = parser.parse_args()
    raise SystemExit(0 if run_all(args.game, args.output, set(args.only)) else 1)


if __name__ == "__main__":
    main()
