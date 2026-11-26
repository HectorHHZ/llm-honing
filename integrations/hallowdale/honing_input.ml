(* SPDX-License-Identifier: Apache-2.0 *)
(* Honing bridge, part 1 of 3: what the game's own modules need to see, and the native action interface.

   This file has no dependencies, so that Env / File / Controls / Render can use it and the tests can
   compile it alone. The bridge is active only when the environment variable HONING_SESSION_DIR is set;
   without it the game behaves exactly like the original.

   * virtual keyboard: in bridge mode Controls.key_down / key_pressed / key_released / key_up read these
     tables instead of raylib. One call of [set_tick] per game tick gives the keys that are down in that
     tick; pressed/released edges follow from the previous tick, as for a real keyboard. This is the only
     place where input enters the game, so scripted sequences and cutscenes run the game's own code.
     Key names are the names of Controls.show_game_action.
   * [before_end_drawing]: called by Render.tick right before Raylib.end_drawing (frame capture).
   * [window_scale]: optional fixed window scale (HONING_WINDOW_SCALE); the original derives it from the
     monitor size.
   * the action table, the validation of a requested window and the keys of every tick of a window. *)

let session_dir : string = match Sys.getenv_opt "HONING_SESSION_DIR" with Some s -> s | None -> ""
let active : bool = session_dir <> ""

let window_scale : float option =
  if not active then
    None
  else (
    match Sys.getenv_opt "HONING_WINDOW_SCALE" with
    | Some s when s <> "" -> Some (float_of_string s)
    | _ ->
      (* default: the 44 x 24 tile view is 960 px wide (tile = 240/11 px) *)
      Some (20. /. 33.))

let now : (string, unit) Hashtbl.t = Hashtbl.create 16
let before : (string, unit) Hashtbl.t = Hashtbl.create 16
let is_down (k : string) : bool = Hashtbl.mem now k
let is_pressed (k : string) : bool = Hashtbl.mem now k && not (Hashtbl.mem before k)
let is_released (k : string) : bool = (not (Hashtbl.mem now k)) && Hashtbl.mem before k

let set_tick (keys : string list) : unit =
  Hashtbl.reset before;
  Hashtbl.iter (fun k () -> Hashtbl.replace before k ()) now;
  Hashtbl.reset now;
  List.iter (fun k -> Hashtbl.replace now k ()) keys

let before_end_drawing : (unit -> unit) ref = ref (fun () -> ())

(* ---- the native action interface ------------------------------------------------------------------- *)

(* Every control of the game that acts during play. PAUSE and OPEN_MAP are not here: they open menus that
   stop the game. The DEBUG_* controls have no key outside the game's development build. *)
let key_names =
  [ "JUMP"; "NAIL"; "DASH"; "FOCUS"; "C_DASH"; "D_NAIL"; "CAST"; "INTERACT"; "UP"; "DOWN"; "LEFT"; "RIGHT" ]

let directions = [ "UP"; "DOWN"; "LEFT"; "RIGHT" ]
let max_frames = 600

(* named action -> keys held during the whole window, keys pressed at its start *)
let actions : (string * (string list * string list)) list =
  [
    ("wait", ([], []));
    ("left", ([ "LEFT" ], []));
    ("right", ([ "RIGHT" ], []));
    ("up", ([ "UP" ], []));
    ("down", ([ "DOWN" ], []));
    ("jump", ([], [ "JUMP" ]));
    ("jump_left", ([ "LEFT" ], [ "JUMP" ]));
    ("jump_right", ([ "RIGHT" ], [ "JUMP" ]));
    ("attack", ([], [ "NAIL" ]));
    ("attack_left", ([ "LEFT" ], [ "NAIL" ]));
    ("attack_right", ([ "RIGHT" ], [ "NAIL" ]));
    ("attack_up", ([ "UP" ], [ "NAIL" ]));
    ("attack_down", ([ "DOWN" ], [ "NAIL" ]));
    ("dash", ([], [ "DASH" ]));
    ("dash_left", ([ "LEFT" ], [ "DASH" ]));
    ("dash_right", ([ "RIGHT" ], [ "DASH" ]));
    ("interact", ([], [ "INTERACT" ]));
    ("focus", ([ "FOCUS" ], []));
    ("cast", ([], [ "CAST" ]));
    ("cast_left", ([ "LEFT" ], [ "CAST" ]));
    ("cast_right", ([ "RIGHT" ], [ "CAST" ]));
    ("cast_up", ([ "UP" ], [ "CAST" ]));
    ("cast_down", ([ "DOWN" ], [ "CAST" ]));
    ("c_dash", ([ "C_DASH" ], []));
    ("dream_nail", ([ "D_NAIL" ], []));
  ]

(* one accepted window *)
type window = {
  hold : string list;  (** down in every tick *)
  press : string list;  (** a fresh press at the start of the window *)
  first : string list;  (** directions down in the first tick instead of the held directions *)
  frames : int;
  jump_frames : int option;  (** bridge frames for which a pressed JUMP stays down *)
}

(* [action] is a named action, or "keys" for the keys given in [hold] / [press] alone. Nothing of an
   invalid request is executed. *)
let resolve ~(action : string) ~(frames : int) ~(hold : string list) ~(press : string list) ~(first : string list)
    ~(jump_frames : int option) : (window, string) result =
  let named = if action = "keys" then Some ([], []) else List.assoc_opt action actions in
  match named with
  | None -> Error (Printf.sprintf "unknown action %S" action)
  | Some (named_hold, named_press) -> (
    let hold = named_hold @ hold and press = named_press @ press in
    match List.filter (fun k -> not (List.mem k key_names)) (hold @ press @ first) with
    | k :: _ -> Error (Printf.sprintf "key %S cannot be used" k)
    | [] ->
      if frames < 1 || frames > max_frames then
        Error (Printf.sprintf "frames must be within 1..%d" max_frames)
      else if List.exists (fun k -> not (List.mem k directions)) first then
        Error "first may name direction keys only"
      else (
        match jump_frames with
        | Some _ when not (List.mem "JUMP" press) -> Error "jump_frames needs a command that presses JUMP"
        | Some n when n < 1 || n > frames -> Error "jump_frames must be within 1..frames"
        | _ -> Ok { hold; press; first; frames; jump_frames }))

(* Sticky mode keeps the jump key down after a press until the next jump, so that every jump is a full
   jump. A jump with [jump_frames] ends by itself and leaves nothing latched. *)
let latch_after ~(sticky : bool) ~(latched : bool) (w : window) : bool =
  if List.mem "JUMP" w.press then sticky && Option.is_none w.jump_frames else latched

(* The keys that are down in tick [i] of window [w].

   [press]: a fresh press at the start of the window. A key that is still down from before is let go for
   the first tick, so that a repeated jump, dash or strike gets a new press edge; the press then lasts one
   bridge frame. JUMP lasts [jump_frames] bridge frames when they are given. Otherwise it stays down to the
   end of the window (mode "window": the jump is cut when the window ends while still rising) or until the
   next jump press (mode "sticky"; [latched] is the result of [latch_after] for this window).
   [first]: direction keys that are down in the very first tick instead of the directions of [hold] (the
   direction of a strike is taken in the tick of the press; a window may strike one way and walk the other). *)
let window_keys ~(ticks_per_frame : int) ~(sticky : bool) ~(down_before : string list) ~(latched : bool) (w : window)
    (i : int) : string list =
  let total = w.frames * ticks_per_frame in
  let start_of k = if List.mem k down_before || (k = "JUMP" && sticky) then 1 else 0 in
  let pressed =
    List.filter
      (fun k ->
        let start = start_of k in
        let stop =
          if k <> "JUMP" then
            start + ticks_per_frame
          else (
            match w.jump_frames with
            | None -> total
            | Some n -> min total (start + (n * ticks_per_frame)))
        in
        i >= start && i < stop)
      w.press
  in
  let held_jump = if latched && not (List.mem "JUMP" w.press) then [ "JUMP" ] else [] in
  let held =
    if i = 0 && w.first <> [] then
      w.first @ List.filter (fun k -> not (List.mem k directions)) w.hold
    else
      w.hold
  in
  List.sort_uniq compare (held @ pressed @ held_jump)

(* what a window applied: key -> first tick, last tick and number of ticks it was down *)
let key_ticks ~(ticks_per_frame : int) ~(keys_at : int -> string list) (w : window) : (string * (int * int * int)) list =
  let table = Hashtbl.create 12 in
  for i = 0 to (w.frames * ticks_per_frame) - 1 do
    List.iter
      (fun k ->
        match Hashtbl.find_opt table k with
        | None -> Hashtbl.replace table k (i, i, 1)
        | Some (first, _, count) -> Hashtbl.replace table k (first, i, count + 1))
      (keys_at i)
  done;
  List.sort compare (Hashtbl.fold (fun k v acc -> (k, v) :: acc) table [])
