"""Compile the bridge's real input and command modules; only the game and its renderer are left out."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import load_script

HERE = Path(__file__).parents[1] / "integrations/hallowdale"
pytestmark = pytest.mark.unit
PIN = "a330d3462afe5fa7b8e65573ad8a7c7c79de17e9"

DRIVER = r"""
(* Drives the production parser, validation, per-tick schedule and key edges; prints JSON. *)
let ticks_per_frame = 5

let names (selected : string -> bool) : Yojson.Safe.t =
  `List (List.filter_map (fun k -> if selected k then Some (`String k) else None) Honing_input.key_names)

let tick (keys : string list) : Yojson.Safe.t =
  Honing_input.set_tick keys;
  `Assoc
    [
      ("down", names Honing_input.is_down);
      ("pressed", names Honing_input.is_pressed);
      ("released", names Honing_input.is_released);
    ]

let scenario (json : Yojson.Safe.t) : Yojson.Safe.t =
  let open Yojson.Safe.Util in
  let sticky = json |> member "sticky" |> to_bool in
  let latched = ref false in
  Honing_input.set_tick [];
  Honing_input.set_tick [];
  `List
    (List.map
       (fun command ->
         match Honing_command.parse (Yojson.Safe.to_string command) with
         | Honing_command.Window (_, action, w) ->
           let down_before = List.filter Honing_input.is_down Honing_input.key_names in
           latched := Honing_input.latch_after ~sticky ~latched:!latched w;
           let keys_at =
             Honing_input.window_keys ~ticks_per_frame ~sticky ~down_before ~latched:!latched w
           in
           let total = w.Honing_input.frames * ticks_per_frame in
           `Assoc
             [
               ("action", `String action);
               ("ticks", `List (List.init total (fun i -> tick (keys_at i))));
               ( "keys",
                 `Assoc
                   (List.map
                      (fun (key, (first, last, count)) -> (key, `List [ `Int first; `Int last; `Int count ]))
                      (Honing_input.key_ticks ~ticks_per_frame ~keys_at w)) );
             ]
         | Honing_command.Stop _ -> `Assoc [ ("stop", `Bool true) ]
         | Honing_command.Refused (_, reason) -> `Assoc [ ("refused", `String reason) ])
       (json |> member "windows" |> to_list))

let text (raw : string) : Yojson.Safe.t =
  match Honing_command.parse raw with
  | Honing_command.Window (id, action, _) -> `Assoc [ ("window", `String action); ("id", `Int id) ]
  | Honing_command.Stop id -> `Assoc [ ("stop", `Int id) ]
  | Honing_command.Refused (_, reason) -> `Assoc [ ("refused", `String reason) ]

let () =
  let open Yojson.Safe.Util in
  let input = Yojson.Safe.from_channel stdin in
  let table =
    List.map
      (fun (name, (hold, press)) ->
        let strings keys = `List (List.map (fun k -> `String k) keys) in
        (name, `Assoc [ ("hold", strings hold); ("press", strings press) ]))
      Honing_input.actions
  in
  print_string
    (Yojson.Safe.to_string
       (`Assoc
         [
           ("actions", `Assoc table);
           ("scenarios", `List (List.map scenario (input |> member "scenarios" |> to_list)));
           ("texts", `List (List.map (fun t -> text (to_string t)) (input |> member "texts" |> to_list)));
         ]))
"""

# The documented action table of API.md, written out independently of the OCaml source.
DOCUMENTED = {
    "wait": ([], []),
    "left": (["LEFT"], []),
    "right": (["RIGHT"], []),
    "up": (["UP"], []),
    "down": (["DOWN"], []),
    "jump": ([], ["JUMP"]),
    "jump_left": (["LEFT"], ["JUMP"]),
    "jump_right": (["RIGHT"], ["JUMP"]),
    "attack": ([], ["NAIL"]),
    "attack_left": (["LEFT"], ["NAIL"]),
    "attack_right": (["RIGHT"], ["NAIL"]),
    "attack_up": (["UP"], ["NAIL"]),
    "attack_down": (["DOWN"], ["NAIL"]),
    "dash": ([], ["DASH"]),
    "dash_left": (["LEFT"], ["DASH"]),
    "dash_right": (["RIGHT"], ["DASH"]),
    "interact": ([], ["INTERACT"]),
    "focus": (["FOCUS"], []),
    "cast": ([], ["CAST"]),
    "cast_left": (["LEFT"], ["CAST"]),
    "cast_right": (["RIGHT"], ["CAST"]),
    "cast_up": (["UP"], ["CAST"]),
    "cast_down": (["DOWN"], ["CAST"]),
    "c_dash": (["C_DASH"], []),
    "dream_nail": (["D_NAIL"], []),
}


@pytest.fixture(scope="module")
def driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the unmodified bridge modules with a small test driver."""
    ocamlfind = shutil.which("ocamlfind")
    if ocamlfind is None or subprocess.run([ocamlfind, "query", "yojson"], capture_output=True).returncode != 0:
        pytest.skip("ocamlfind and yojson are required to compile the bridge modules")
    build = tmp_path_factory.mktemp("hallowdale-bridge")
    for name in ("honing_input.ml", "honing_command.ml"):
        shutil.copyfile(HERE / name, build / name)
    (build / "driver.ml").write_text(DRIVER)
    command = [
        ocamlfind,
        "ocamlopt",
        "-w",
        "-a",
        "-package",
        "yojson",
        "-linkpkg",
        "honing_input.ml",
        "honing_command.ml",
        "driver.ml",
        "-o",
        "driver",
    ]
    compiled = subprocess.run(command, cwd=build, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return build / "driver"


def drive(driver: Path, scenarios: list[dict[str, Any]] | None = None, texts: list[str] | None = None) -> dict:
    payload = json.dumps({"scenarios": scenarios or [], "texts": texts or []})
    environment = {key: value for key, value in os.environ.items() if not key.startswith("HONING_")}
    result = subprocess.run([str(driver)], input=payload, capture_output=True, text=True, env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def windows(driver: Path, *commands: dict[str, Any], sticky: bool = False) -> list[dict[str, Any]]:
    numbered = [{"id": index, **command} for index, command in enumerate(commands)]
    return drive(driver, [{"sticky": sticky, "windows": numbered}])["scenarios"][0]


def ticks_with(window: dict[str, Any], kind: str, key: str) -> list[int]:
    return [index for index, tick in enumerate(window["ticks"]) if key in tick[kind]]


def test_compiled_action_table_is_the_documented_interface(driver: Path) -> None:
    table = drive(driver)["actions"]
    assert {name: (keys["hold"], keys["press"]) for name, keys in table.items()} == DOCUMENTED


def test_every_offered_choice_is_accepted_by_the_bridge_parser(driver: Path) -> None:
    run = load_script("integrations/hallowdale/run.py")
    texts = [json.dumps({"id": 1, **run.native_command(choice, 12)}) for choice in run.ACTIONS]
    parsed = drive(driver, texts=texts)["texts"]
    assert all("window" in item for item in parsed), parsed


def test_repeated_jump_is_released_for_one_tick_and_pressed_again(driver: Path) -> None:
    first, second = windows(driver, {"action": "jump", "frames": 2}, {"action": "jump", "frames": 2})
    assert ticks_with(first, "pressed", "JUMP") == [0]
    assert ticks_with(second, "released", "JUMP") == [0]
    assert ticks_with(second, "pressed", "JUMP") == [1]


def test_window_mode_holds_a_jump_to_the_end_of_its_window_only(driver: Path) -> None:
    jump, walk = windows(driver, {"action": "jump", "frames": 2}, {"action": "right", "frames": 1})
    assert ticks_with(jump, "down", "JUMP") == list(range(10))
    assert ticks_with(walk, "released", "JUMP") == [0]
    assert ticks_with(walk, "down", "JUMP") == []


def test_sticky_jump_stays_down_until_the_next_jump_press(driver: Path) -> None:
    jump, walk, again = windows(
        driver,
        {"action": "jump", "frames": 1},
        {"action": "right", "frames": 2},
        {"action": "jump_left", "frames": 1},
        sticky=True,
    )
    assert ticks_with(jump, "pressed", "JUMP") == [1]
    assert ticks_with(walk, "down", "JUMP") == list(range(10))
    assert ticks_with(walk, "released", "JUMP") == []
    assert ticks_with(again, "released", "JUMP") == [0]
    assert ticks_with(again, "pressed", "JUMP") == [1]


@pytest.mark.parametrize("sticky", [False, True])
def test_jump_frames_end_the_hold_and_leave_nothing_latched(driver: Path, sticky: bool) -> None:
    short, after = windows(
        driver,
        {"action": "jump", "frames": 4, "jump_frames": 2},
        {"action": "wait", "frames": 2},
        sticky=sticky,
    )
    start = 1 if sticky else 0
    assert ticks_with(short, "down", "JUMP") == list(range(start, start + 10))
    assert ticks_with(short, "released", "JUMP") == [start + 10]
    assert ticks_with(after, "down", "JUMP") == []


def test_jump_without_jump_frames_is_unchanged_by_the_new_field(driver: Path) -> None:
    for sticky in (False, True):
        plain = windows(driver, {"action": "jump", "frames": 3}, sticky=sticky)[0]
        null = windows(driver, {"action": "jump", "frames": 3, "jump_frames": None}, sticky=sticky)[0]
        assert plain == null


@pytest.mark.parametrize("action", ["attack_right", "dash_left", "cast", "interact"])
def test_repeated_button_in_one_frame_windows_gets_a_new_edge(driver: Path, action: str) -> None:
    results = windows(driver, *[{"action": action, "frames": 1}] * 3)
    key = DOCUMENTED[action][1][0]
    assert ticks_with(results[0], "pressed", key) == [0]
    for window in results[1:]:
        assert ticks_with(window, "released", key) == [0]
        assert ticks_with(window, "pressed", key) == [1]


@pytest.mark.parametrize("action", ["attack_up", "dash", "cast_down"])
def test_button_press_lasts_one_bridge_frame(driver: Path, action: str) -> None:
    window = windows(driver, {"action": action, "frames": 3})[0]
    assert ticks_with(window, "down", DOCUMENTED[action][1][0]) == list(range(5))


@pytest.mark.parametrize("action", ["focus", "c_dash", "dream_nail"])
def test_hold_actions_stay_down_across_windows_and_release_afterwards(driver: Path, action: str) -> None:
    key = DOCUMENTED[action][0][0]
    first, second, third = windows(
        driver, {"action": action, "frames": 2}, {"action": action, "frames": 2}, {"action": "wait", "frames": 1}
    )
    assert ticks_with(first, "pressed", key) == [0]
    assert ticks_with(second, "down", key) == list(range(10))
    assert ticks_with(second, "released", key) == []
    assert ticks_with(third, "released", key) == [0]


def test_first_replaces_the_held_direction_in_the_first_tick_only(driver: Path) -> None:
    window = windows(driver, {"action": "right", "first": ["LEFT"], "frames": 1})[0]
    assert window["ticks"][0]["down"] == ["LEFT"]
    assert all(tick["down"] == ["RIGHT"] for tick in window["ticks"][1:])


def test_keys_command_combines_any_gameplay_keys(driver: Path) -> None:
    window = windows(driver, {"action": "keys", "hold": ["RIGHT", "UP"], "press": ["NAIL"], "frames": 2})[0]
    assert window["keys"]["RIGHT"] == [0, 9, 10]
    assert window["keys"]["UP"] == [0, 9, 10]
    assert window["keys"]["NAIL"] == [0, 4, 5]


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("junk", "malformed"),
        ("[1]", "object"),
        ('{"action":"wait","frames":1}', "id must be"),
        ('{"id":"1","action":"wait","frames":1}', "id must be"),
        ('{"id":99999999999999999999,"action":"wait","frames":1}', "id must be"),
        ('{"id":1,"id":1,"action":"wait","frames":1}', "duplicate"),
        ('{"id":1,"frames":1}', "action must"),
        ('{"id":1,"action":3,"frames":1}', "action must"),
        ('{"id":1,"action":"wait"}', "frames must"),
        ('{"id":1,"action":"wait","frames":"1"}', "frames must"),
        ('{"id":1,"action":"wait","frames":1.5}', "frames must"),
        ('{"id":1,"action":"wait","frames":0}', "within"),
        ('{"id":1,"action":"wait","frames":601}', "within"),
        ('{"id":1,"action":"fly","frames":1}', "unknown action"),
        ('{"id":1,"action":"keys","press":["PAUSE"],"frames":1}', "cannot be used"),
        ('{"id":1,"action":"keys","hold":["OPEN_MAP"],"frames":1}', "cannot be used"),
        ('{"id":1,"action":"keys","hold":["DEBUG_1"],"frames":1}', "cannot be used"),
        ('{"id":1,"action":"right","hold":"LEFT","frames":1}', "list of key names"),
        ('{"id":1,"action":"right","press":[1],"frames":1}', "list of key names"),
        ('{"id":1,"action":"right","first":["NAIL"],"frames":1}', "direction"),
        ('{"id":1,"action":"left","jump_frames":1,"frames":2}', "presses JUMP"),
        ('{"id":1,"action":"jump","jump_frames":0,"frames":2}', "1..frames"),
        ('{"id":1,"action":"jump","jump_frames":3,"frames":2}', "1..frames"),
        ('{"id":1,"action":"jump","jump_frames":"2","frames":2}', "jump_frames must"),
    ],
)
def test_invalid_commands_are_refused_with_a_reason(driver: Path, text: str, reason: str) -> None:
    assert reason in drive(driver, texts=[text])["texts"][0]["refused"]


def test_stop_and_display_only_fields_are_accepted(driver: Path) -> None:
    texts = ['{"id":4,"action":"stop"}', '{"id":4,"action":"left","frames":2,"model":"harness","probability":0.5}']
    assert drive(driver, texts=texts)["texts"] == [{"stop": 4}, {"window": "left", "id": 4}]


def test_native_patch_changes_only_the_five_documented_files() -> None:
    patch = (HERE / "native.patch").read_text()
    changed = re.findall(r"^\+\+\+ b/(.+)$", patch, flags=re.MULTILINE)
    assert sorted(changed) == ["bin/main.ml", "src/controls.ml", "src/env.ml", "src/file.ml", "src/render.ml"]


def test_input_enters_the_game_only_through_the_four_key_functions() -> None:
    patch = (HERE / "native.patch").read_text()
    section = patch[patch.index("+++ b/src/controls.ml") : patch.index("diff --git a/src/env.ml")]
    assert section.count("+  if Honing_input.active then") == 4
    for function in ("key_up", "key_down", "key_pressed", "key_released"):
        assert re.search(rf"^ let {function} ", section, flags=re.MULTILINE), function
    gameplay = [path for path in ("src/player.ml", "src/state.ml", "src/interactions.ml") if f"b/{path}" in patch]
    assert gameplay == []


def test_patch_applies_to_the_pinned_source() -> None:
    source = os.environ.get("HONING_HALLOWDALE_SOURCE")
    if not source:
        pytest.skip("set HONING_HALLOWDALE_SOURCE to an unpatched checkout of the pinned commit")
    assert subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip() == PIN
    subprocess.run(["git", "-C", source, "apply", "--check", str(HERE / "native.patch")], check=True)
