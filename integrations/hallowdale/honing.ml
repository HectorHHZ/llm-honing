(* SPDX-License-Identifier: Apache-2.0 *)
(* Honing bridge, part 3 of 3: the file protocol, the fixed-step loop and the observations.

     <session>/observation.json        latest human-visible observation, schema honing.visible.v1
     <session>/visible_observations.jsonl  every visible observation, one per line
     <session>/debug_observation.json  latest global/debug state (plumbing and evaluation only)
     <session>/observations.jsonl      every debug observation, one per line
     <session>/action.json             written by the client: one native action window
     <session>/ack.json, acks.jsonl    what the bridge applied or refused, for every command
     <session>/frames/%06d.png         one captured frame per bridge frame of an action window (HONING_CAPTURE=1)

   The game is paused between windows: nothing ticks until an action with the id of the latest
   observation arrives. A window of N bridge frames = N * ticks_per_frame calls of the ORIGINAL
   State.tick |> Render.tick with a fixed dt (the original uses the real frame time). Gameplay code is
   not changed; the bridge only (1) supplies key states through Honing_input, (2) fixes dt and the random
   seed, (3) reads game state to write observations.

   The visible observation is computed from a second render of the same scene with the same camera in
   which every drawable is painted in a flat colour that encodes its class and object (the original
   draw order is kept, so occlusion is what the player sees). No pixels are given to the model: the
   label image is reduced here to a grid of class unions and object bounds. *)
open Utils
open Types

let getenv name default =
  match Sys.getenv_opt name with
  | Some v when v <> "" -> v
  | _ -> default

let session = Honing_input.session_dir
let capture = getenv "HONING_CAPTURE" "0" = "1"
let capture_boot = getenv "HONING_CAPTURE_BOOT" "0" = "1"
let positive_int name default =
  let value = getenv name default |> int_of_string in
  if value <= 0 then failwith (name ^ " must be positive");
  value

let positive_float name default =
  let value = getenv name default |> float_of_string in
  if (not (Float.is_finite value)) || value <= 0. then failwith (name ^ " must be positive and finite");
  value

let one_of name default allowed =
  let value = getenv name default in
  if not (List.mem value allowed) then failwith (name ^ " must be one of " ^ String.concat ", " allowed);
  value

let frame_hz = positive_int "HONING_FRAME_HZ" "30"
let ticks_per_frame = positive_int "HONING_TICKS_PER_FRAME" "5"
let dt = 1. /. Int.to_float (frame_hz * ticks_per_frame)
let cell = positive_int "HONING_VISIBLE_CELL_PIXELS" "20"
let jump_mode = one_of "HONING_JUMP_MODE" "window" [ "window"; "sticky" ]
let scene = getenv "HONING_SCENE" "new"
let seed = getenv "HONING_SEED" "42" |> int_of_string

(* how long the bridge waits for a valid command before it exits; a client sets this to its own model
   timeout plus a margin *)
let model_timeout = positive_float "HONING_MODEL_TIMEOUT_SECONDS" "180"
let mask_capture = getenv "HONING_VISIBLE_MASK_CAPTURE" "0" = "1"
let self_box = getenv "HONING_SELF_BOX" "1" = "1"
let profile = one_of "HONING_OBSERVATION_PROFILE" "visible" [ "visible"; "global_debug" ]

(* View knobs. By default the observation's screen is the game window. With HONING_VIEW_ANCHOR=feet:X,Y the
   observation's screen is a player-centred cut of the window instead: the point under the middle of the
   player's feet is always at (X, Y) of a HONING_VIEW_SIZE screen, magnified by HONING_VIEW_ZOOM. Parts of
   that screen that lie outside the window are reported as unknown; parts of the window outside that
   screen are not reported. This can only remove information. *)
let view_anchor : (float * float) option =
  match getenv "HONING_VIEW_ANCHOR" "" with
  | "" -> None
  | spec ->
    let spec = if String.length spec > 5 && String.sub spec 0 5 = "feet:" then String.sub spec 5 (String.length spec - 5) else spec in
    let x, y = String.split_at_first ',' spec in
    Some (float_of_string x, float_of_string y)

let view_zoom = positive_float "HONING_VIEW_ZOOM" "1"

let view_size : (int * int) option =
  match getenv "HONING_VIEW_SIZE" "" with
  | "" -> None
  | spec ->
    let w, h = String.split_at_first 'x' spec in
    Some (int_of_string w, int_of_string h)

(* which ticks are drawn: "all" = every tick, as the original does; "frame" = the last tick of every
   bridge frame; "window" = the last tick of every action window. Ticks that are not drawn still get
   every change the original Render.tick makes to the game state (see [render_state_only]). *)
let render_mode = one_of "HONING_RENDER" "all" [ "all"; "frame"; "window" ]

(* ---- clock and counters ------------------------------------------------------------------------ *)
let sim_time = ref 10.0 (* the original starts its clock at program start; loading takes a few seconds *)
let frames_played = ref 0 (* bridge frames played inside action windows ("tick" of the observation) *)
let ticks_played = ref 0
let sequence = ref 0
let deaths = ref 0
let capture_path : string option ref = ref None
let jump_latched = ref false

let path name = Filename.concat session name

let write_atomic name (text : string) =
  let final = path name in
  let tmp = final ^ ".tmp" in
  let ch = open_out_bin tmp in
  output_string ch text;
  close_out ch;
  Sys.rename tmp final

let append name (text : string) =
  let ch = open_out_gen [ Open_append; Open_creat; Open_binary ] 0o644 (path name) in
  output_string ch text;
  output_char ch '\n';
  close_out ch

let log_line fmtstr =
  Printf.ksprintf
    (fun s ->
      prerr_endline ("[honing] " ^ s);
      append "bridge.log" s)
    fmtstr

(* ---- a tick that is not drawn ------------------------------------------------------------------------ *)
(* Render.tick is not only drawing: it also moves sprite rectangles to their entities, places the things
   attached to the player (the nail swing, whose rectangle the next tick's hit test reads) and advances
   the screen-fade timer. This function makes exactly those changes, in the order of Render.tick, without
   drawing, so that a run with HONING_RENDER=window gives the same observations as HONING_RENDER=all. *)
let render_state_only (state : state) : state =
  match state.context with
  | IN_PROGRESS game ->
    let fade_step ~under_ghost : bool =
      match state.screen_fade with
      | None -> false
      | Some fade ->
        if under_ghost = fade.show_ghost then (
          (match fade.timer with
          | None -> ()
          | Some timer ->
            timer.left <- { seconds = timer.left.seconds -. state.frame.dt };
            if not (fade.target_alpha > fade.starting_alpha) then (
              let target' =
                fade.starting_alpha
                - ((fade.starting_alpha - fade.target_alpha |> Int.to_float)
                   *. (1. -. (timer.left.seconds /. timer.total.seconds))
                  |> Float.to_int)
              in
              if target' < 0 && timer.left.seconds < 0. then
                state.screen_fade <- None));
          true)
        else
          false
    in
    let adjust (e : entity) = Entity.adjust_sprite_dest e in
    let fragments (drawn : layer -> bool) =
      List.iter
        (fun (layer : layer) -> if drawn layer && not layer.hidden then List.iter adjust layer.spawned_fragments)
        game.room.layers
    in
    fragments (fun layer -> layer.config.render.bg);
    fragments (fun layer -> layer.config.collides_with_ghost || layer.config.hazard);
    List.iter (fun (n : npc) -> adjust n.entity) game.room.npcs;
    List.iter (fun (pg : party_ghost) -> adjust pg.ghost.entity) game.party;
    let screen_faded = fade_step ~under_ghost:true in
    let player = game.player in
    let place_child (child : ghost_child) =
      child.sprite.facing_right <- player.ghost.entity.sprite.facing_right;
      child.sprite.dest.pos <-
        Entity.get_child_pos player.ghost.entity child.relative_pos child.sprite.dest.w
          child.sprite.dest.h
    in
    List.iter (fun (p : projectile) -> adjust p.entity) player.spawned_vengeful_spirits;
    let in_front, behind = Ghost_child_kind.Map.partition (fun _ child -> child.in_front) player.children in
    Ghost_child_kind.Map.iter (fun _ child -> place_child child) behind;
    adjust player.ghost.entity;
    Ghost_child_kind.Map.iter (fun _ child -> place_child child) in_front;
    if not screen_faded then (
      let on_top = ref [] in
      List.iter
        (fun (enemy : enemy) ->
          if enemy.entity.dest.pos.x > -1. && enemy.entity.dest.pos.y > -1. then (
            adjust enemy.entity;
            List.iter
              (fun (p : projectile) -> if p.draw_on_top then on_top := [ p ] @ !on_top else adjust p.entity)
              enemy.projectiles))
        game.room.enemies;
      List.iter (fun (p : projectile) -> adjust p.entity) game.room.loose_projectiles;
      (match game.reflection_x with
      | None -> ()
      | Some reflection_x ->
        let e = player.ghost.entity in
        let original_pos = e.dest.pos in
        e.dest.pos <- { original_pos with x = reflection_x -. (original_pos.x -. reflection_x) };
        e.sprite.facing_right <- not e.sprite.facing_right;
        adjust e;
        e.dest.pos <- original_pos;
        e.sprite.facing_right <- not e.sprite.facing_right);
      fragments (fun layer -> layer.config.render.fg);
      List.iter (fun (p : projectile) -> adjust p.entity) !on_top;
      ignore (fade_step ~under_ghost:false));
    state
  | _ -> state

let draw_this_tick = ref true

(* ---- one game tick ------------------------------------------------------------------------------- *)
let tick (state : state) (keys : string list) : state =
  Honing_input.set_tick keys;
  sim_time := !sim_time +. dt;
  state.frame.time <- !sim_time;
  state.frame.idx <- 1 + state.frame.idx;
  state.frame.dt <- dt;
  incr ticks_played;
  let state' = State.tick state in
  (* a death shows as the game reloading its last save (the context lasts one tick) *)
  (match state'.context with
  | RELOAD_LAST_SAVED_GAME _ -> incr deaths
  | _ -> ());
  if !draw_this_tick then Render.tick state' else render_state_only state'

let () =
  Honing_input.before_end_drawing :=
    fun () ->
      match !capture_path with
      | None -> ()
      | Some file ->
        capture_path := None;
        let image = Raylib.load_image_from_screen () in
        ignore (Raylib.export_image image file);
        Raylib.unload_image image

(* [frame state keys_of_tick n]: one bridge frame = [ticks_per_frame] ticks; the last tick's picture is
   the frame's picture. *)
let last_frame_file : string option ref = ref None

let frame ?save ?(last_of_window = true) (state : state) (keys_of_tick : int -> string list) : state =
  let s = ref state in
  for i = 0 to ticks_per_frame - 1 do
    let last = i = ticks_per_frame - 1 in
    if last then capture_path := save;
    draw_this_tick :=
      render_mode = "all"
      || (last && (Option.is_some save || render_mode = "frame" || last_of_window));
    s := tick !s (keys_of_tick i)
  done;
  draw_this_tick := true;
  (match (!capture_path, save) with
  | Some file, _ ->
    (* nothing was drawn in the last tick (the one tick in which the game reloads a save): the frame
       shows the previous picture *)
    capture_path := None;
    (match !last_frame_file with
    | Some previous when Sys.file_exists previous -> File.copy_file previous file
    | _ -> ())
  | None, Some file -> last_frame_file := Some file
  | None, None -> ());
  !s

(* ---- game state access ----------------------------------------------------------------------------- *)
let game_of (state : state) : game option =
  match state.context with
  | IN_PROGRESS game -> Some game
  | _ -> None

let tile_px () = Config.window.dest_tile_size
let room_name (game : game) = Room.get_filename game.room

(* the alpha of the black screen fade as Render.tick would draw it now (read only) *)
let fade_alpha (state : state) : int =
  match state.screen_fade with
  | None -> 0
  | Some fade -> (
    match fade.timer with
    | None -> 255
    | Some timer ->
      let done_ = 1. -. (timer.left.seconds /. timer.total.seconds) in
      if fade.target_alpha > fade.starting_alpha then
        Int.bound 0 (Int.min ((fade.target_alpha |> Int.to_float) *. done_ |> Float.to_int) fade.target_alpha) 255
      else
        Int.bound 0
          (fade.starting_alpha
          - ((fade.starting_alpha - fade.target_alpha |> Int.to_float) *. done_ |> Float.to_int))
          255)

let strip_markup (s : string) : string =
  (* "{{red}} lava {{white}}" -> "lava" *)
  Str.global_replace (Str.regexp "{{[a-z]+}} ?") "" s

(* ---- label render ------------------------------------------------------------------------------------ *)
(* class codes (red channel of the label image); green/blue carry an object index *)
let c_occluder = 0 (* shaded near-foreground layers are drawn black by the original code *)
let c_bg = 10
let c_wall = 20 (* terrain: the "floors" tile layer *)
let c_wall_like = 21 (* looks like terrain but has no collision (shadow layers that hide secrets) *)
let c_solid = 23 (* other solid tile layers: lever doors, boss doors, benches, temporary floors, conveyor belts *)
let c_platform = 30
let c_hazard_tiles = 40
let c_hazard_rect = 41
let c_hazard_obj = 45
let c_enemy = 50
let c_player = 55
let c_equipment = 56
let c_door = 60 (* breakable tile groups of the "doors" layer: solid and can be struck open *)
let c_lever = 65
let c_scenery = 70
let c_water = 71
let c_npc = 72
let c_corpse = 73
let c_unknown = 80
let bit_bg, bit_wall, bit_plat, bit_haz, bit_actor, bit_inter, bit_unk, bit_scen = (1, 2, 4, 8, 16, 32, 64, 128)

let bit_of_class (c : int) : int =
  if c = c_bg then bit_bg
  else if c = c_wall || c = c_wall_like || c = c_solid then bit_wall
  else if c = c_platform then bit_plat
  else if c = c_hazard_tiles || c = c_hazard_rect || c = c_hazard_obj then bit_haz
  else if c = c_enemy || c = c_player then bit_actor
  else if c = c_door then bit_wall lor bit_inter
  else if c = c_lever then bit_inter
  else if c = c_scenery || c = c_water || c = c_npc || c = c_corpse || c = c_equipment || c = c_occluder then bit_scen
  else bit_unk

let color (cls : int) (idx : int) : Raylib.Color.t =
  Raylib.Color.create cls ((idx lsr 8) land 255) (idx land 255) 255

let vertex_shader =
  "#version 330\n\
   in vec3 vertexPosition;\n\
   in vec2 vertexTexCoord;\n\
   in vec4 vertexColor;\n\
   out vec2 fragTexCoord;\n\
   out vec4 fragColor;\n\
   uniform mat4 mvp;\n\
   void main() {\n\
  \  fragTexCoord = vertexTexCoord;\n\
  \  fragColor = vertexColor;\n\
  \  gl_Position = mvp*vec4(vertexPosition, 1.0);\n\
   }\n"

(* a texel counts as visible when it is (almost) opaque; soft shadow edges of the terrain art do not *)
let fragment_shader =
  "#version 330\n\
   in vec2 fragTexCoord;\n\
   in vec4 fragColor;\n\
   out vec4 finalColor;\n\
   uniform sampler2D texture0;\n\
   uniform vec4 colDiffuse;\n\
   void main() {\n\
  \  vec4 texel = texture(texture0, fragTexCoord);\n\
  \  if (texel.a < 0.9) discard;\n\
  \  finalColor = vec4(fragColor.rgb, 1.0);\n\
   }\n"

let label_shader = lazy (Raylib.load_shader_from_memory vertex_shader fragment_shader)
let label_target : (int * int * Raylib.RenderTexture.t) option ref = ref None

let get_target w h =
  match !label_target with
  | Some (w', h', t) when w = w' && h = h' -> t
  | _ ->
    let t = Raylib.load_render_texture w h in
    label_target := Some (w, h, t);
    t

(* stable ids for projectiles (they live in lists that change): physical identity -> number *)
let projectile_ids : (projectile * int) list ref = ref []
let next_projectile_id = ref 0

let projectile_id (p : projectile) : int =
  match List.assq_opt p !projectile_ids with
  | Some id -> id
  | None ->
    incr next_projectile_id;
    projectile_ids := (p, !next_projectile_id) :: !projectile_ids;
    !next_projectile_id

let base_layer_name (name : string) : string =
  (* "shadow4" -> "shadow", "bg-iso2" -> "bg-iso" (Room.init does the same to pick a layer config) *)
  try Str.first_chars name (Str.search_forward (Str.regexp "[0-9]$") name 0) with
  | Not_found -> name

type obj = {
  oid : string;
  kind : string;
  look : string;
  mutable x0 : int;
  mutable y0 : int;
  mutable x1 : int;
  mutable y1 : int; (* inclusive pixel bounds; x1 < x0 = not seen *)
}

let new_obj oid kind look = { oid; kind; look; x0 = max_int; y0 = max_int; x1 = -1; y1 = -1 }

(* a rectangle in world pixels, to be checked against the label image afterwards *)
type rect_obj = {
  o : obj;
  r : rect;
  classes : int list;
}

let draw_labels (state : state) (game : game) (sprite_objs : (int, obj) Hashtbl.t) : unit =
  let camera_x, camera_y =
    ( Raylib.Vector2.x (Raylib.Camera2D.target state.camera.raylib) -. Config.window.center.x,
      Raylib.Vector2.y (Raylib.Camera2D.target state.camera.raylib) -. Config.window.center.y )
  in
  let room = game.room in
  let tag = room_name game in
  let register cls idx oid kind look =
    Hashtbl.replace sprite_objs ((cls lsl 16) lor idx) (new_obj (tag ^ "/" ^ oid) kind look)
  in
  let draw_layer ?(parallax = false) (layer : layer) (cls : int) =
    (* Render.draw_tiled_layer also draws the fragments and stubs of destroyed tiles with the layer's
       tint; they are loose debris, so here they are drawn as scenery *)
    let fragments, stubs = (layer.spawned_fragments, layer.spawned_stub_sprites) in
    layer.spawned_fragments <- [];
    layer.spawned_stub_sprites <- [];
    let parallax' =
      if parallax then
        Some
          {
            x = camera_x *. (1. -. layer.json.parallax_x);
            y = camera_y *. (1. -. layer.json.parallax_y);
          }
      else
        None
    in
    Render.draw_tiled_layer room camera_x camera_y state ~tint:(color cls 0) ~parallax:parallax' layer;
    layer.spawned_fragments <- fragments;
    layer.spawned_stub_sprites <- stubs;
    if not layer.hidden then (
      List.iter (fun (f : entity) -> Render.draw_entity ~tint:(color c_scenery 0) f) fragments;
      List.iter
        (fun ((sprite : sprite), bits) ->
          Render.draw_texture ~tint:(color c_scenery 0) sprite.texture sprite.dest bits)
        stubs)
  in
  let class_of_layer (layer : layer) : int option =
    let name = base_layer_name layer.name in
    if layer.config.shaded then Some c_occluder
    else if layer.config.hazard then Some c_hazard_tiles
    else if layer.config.collides_with_ghost then
      if name = "doors" then Some c_door else Some c_solid
    else (
      match name with
      | "floors" -> Some c_wall
      | "hazard" -> Some c_hazard_tiles
      | "water" -> Some c_water
      | "fast-conveyor-belt"
      | "slow-conveyor-belt" ->
        Some c_solid
      | "shadow" ->
        (* a layer that a "hide:<layer>" trigger removes covers a secret and looks like terrain; any
           other "shadow" layer is decoration in front of the scene *)
        if List.exists (fun (t : trigger) -> t.name_suffix = layer.name) room.triggers.shadows then
          Some c_wall_like
        else
          Some c_scenery
      | "fg"
      | "fg-jugs"
      | "fg-flowers"
      | "bg-jugs"
      | "bg-flowers" ->
        Some c_scenery
      | _ -> (* plain background layers: nothing is in front of them *) None)
  in
  let draw_projectile (p : projectile) =
    let id = projectile_id p in
    register c_hazard_obj id (fmt "j%d" id) "visible_hazard" "projectile";
    Render.draw_entity ~tint:(color c_hazard_obj id) p.entity
  in
  (* --- the order below is the order of Render.tick --- *)
  (* background layers *)
  List.iter
    (fun (layer : layer) ->
      if layer.config.render.bg then (
        match class_of_layer layer with
        | Some cls -> draw_layer ~parallax:true layer cls
        | None -> ()))
    room.layers;
  (* levers *)
  List.iteri
    (fun i (lever : lever) ->
      let struck = lever.sprite.texture == state.global.textures.door_lever_struck in
      register c_lever i (fmt "l%d" i) "interactable" (if struck then "lever-struck" else "lever");
      Render.draw_texture ~tint:(color c_lever i) lever.sprite.texture lever.sprite.dest
        lever.transformation)
    room.triggers.levers;
  (* solid tile layers *)
  List.iter
    (fun (layer : layer) ->
      if layer.config.collides_with_ghost || layer.config.hazard then (
        match class_of_layer layer with
        | Some cls -> draw_layer layer cls
        | None -> ()))
    room.layers;
  (* npcs and party ghosts *)
  List.iteri
    (fun i (n : npc) ->
      register c_npc i (fmt "n%d" i) "visible_scenery" ("npc " ^ Show.npc_id n.id);
      Render.draw_entity ~tint:(color c_npc i) n.entity)
    room.npcs;
  List.iter
    (fun (pg : party_ghost) ->
      Render.draw_entity ~tint:(color c_scenery 0) pg.ghost.entity;
      Render.draw_ghost_head ~tint:(color c_scenery 0) pg.ghost)
    game.party;
  (* the player: things attached behind, body and head, things attached in front *)
  let player = game.player in
  let draw_child (child : ghost_child) =
    child.sprite.facing_right <- player.ghost.entity.sprite.facing_right;
    child.sprite.dest.pos <-
      Entity.get_child_pos player.ghost.entity child.relative_pos child.sprite.dest.w
        child.sprite.dest.h;
    Render.draw_sprite ~tint:(color c_equipment 0) child.sprite
  in
  List.iter
    (fun (p : projectile) -> Render.draw_entity ~tint:(color c_equipment 0) p.entity)
    player.spawned_vengeful_spirits;
  let in_front, behind = Ghost_child_kind.Map.partition (fun _ child -> child.in_front) player.children in
  Ghost_child_kind.Map.iter (fun _ child -> draw_child child) behind;
  register c_player 0 "player" "player" (Show.ghost_id player.ghost.id);
  Render.draw_entity ~tint:(color c_player 0) player.ghost.entity;
  Render.draw_ghost_head ~tint:(color c_player 0) player.ghost;
  Ghost_child_kind.Map.iter (fun _ child -> draw_child child) in_front;
  (* enemies and their projectiles *)
  let on_top = ref [] in
  List.iteri
    (fun i (enemy : enemy) ->
      if enemy.entity.dest.pos.x > -1. && enemy.entity.dest.pos.y > -1. then (
        let name = Show.enemy_id enemy.id in
        if Enemy.is_dead enemy then (
          register c_corpse i (fmt "e%d" i) "visible_scenery" (name ^ " dead");
          Render.draw_entity ~tint:(color c_corpse i) enemy.entity)
        else (
          register c_enemy i (fmt "e%d" i) "enemy" name;
          Render.draw_entity ~tint:(color c_enemy i) enemy.entity);
        List.iter
          (fun (p : projectile) -> if p.draw_on_top then on_top := p :: !on_top else draw_projectile p)
          enemy.projectiles))
    room.enemies;
  (* floating platforms *)
  List.iteri
    (fun i (platform : platform) ->
      register c_platform i (fmt "p%d" i) "platform" platform.sprite.ident;
      Render.draw_sprite ~tint:(color c_platform i) platform.sprite)
    room.platforms;
  List.iter draw_projectile room.loose_projectiles;
  (* foreground layers, the terrain among them *)
  List.iter
    (fun (layer : layer) ->
      if layer.config.render.fg then (
        match class_of_layer layer with
        | Some cls -> draw_layer ~parallax:true layer cls
        | None -> draw_layer ~parallax:true layer c_scenery))
    room.layers;
  List.iter draw_projectile !on_top;
  (* heads-up display: it hides what is behind it *)
  let padding = Config.other.hud_padding in
  let pod_image = player.shared_textures.energon_pod_1.image in
  let pod_w = (Raylib.Texture.width pod_image / 2 |> Int.to_float) *. Config.scale.soul in
  let pod_h = (Raylib.Texture.height pod_image |> Int.to_float) *. Config.scale.soul in
  let base_image = player.shared_textures.energon_pod_base.image in
  let base_w = (Raylib.Texture.width base_image |> Int.to_float) *. Config.scale.soul in
  let base_h = (Raylib.Texture.height base_image |> Int.to_float) *. Config.scale.soul in
  let unknown (r : rect) = Render.Draw.rect r (color c_unknown 0) in
  unknown { pos = { x = camera_x +. padding; y = camera_y +. padding }; w = pod_w; h = pod_h };
  unknown { pos = { x = camera_x +. padding; y = camera_y +. padding +. pod_h }; w = base_w; h = base_h };
  (match game.mode with
  | CLASSIC
  | DEMO ->
    let image = player.shared_textures.health.image in
    let w = (Raylib.Texture.width image / 2 |> Int.to_float) *. Config.scale.health in
    let h = (Raylib.Texture.height image |> Int.to_float) *. Config.scale.health in
    for idx = 0 to player.health.max - 1 do
      let x, y =
        if idx < 10 then
          (camera_x +. pod_w +. padding +. ((idx |> Int.to_float) *. (w +. padding)), camera_y +. padding)
        else
          ( camera_x +. pod_w +. padding +. ((idx - 10 |> Int.to_float) *. (w +. padding)),
            camera_y +. (2. *. padding) +. h )
      in
      unknown { pos = { x; y }; w; h }
    done
  | STEEL_SOLE -> ());
  (* text boxes that block the view *)
  let box (config : Interaction.text_config) =
    unknown
      {
        pos = { x = camera_x +. config.margin_x; y = camera_y +. config.margin_y_top };
        w = Config.window.w -. (2. *. config.margin_x);
        h = Config.window.h -. (config.margin_y_top +. config.margin_y_bottom);
      }
  in
  (match (state.pause_menu, game.interaction.text) with
  | Some _, _ ->
    unknown { pos = { x = camera_x; y = camera_y }; w = Config.window.w; h = Config.window.h }
  | None, None -> ()
  | None, Some (ABILITY _)
  | None, Some (FOCUS_ABILITY _) ->
    box Config.text.base_config
  | None, Some (DIALOGUE _) -> box Config.text.dialogue_config
  | None, Some (PLAIN (_, lines)) ->
    let margin_y_bottom =
      if String.length (List.last lines) > 700 then
        Config.text.tall_margin_y_bottom
      else
        Config.text.short_margin_y_bottom
    in
    box (Config.get_plain_text_config margin_y_bottom)
  | None, Some (MENU _) ->
    unknown { pos = { x = camera_x; y = camera_y }; w = Config.window.w; h = Config.window.h })

(* observation screen = (window - anchor_r) * zoom + anchor_v *)
type view = {
  zoom : float;
  ax_r : float;
  ay_r : float;
  ax_v : float;
  ay_v : float;
  vw : int;
  vh : int;
}

let view_x (v : view) (x : float) : float = ((x -. v.ax_r) *. v.zoom) +. v.ax_v
let view_y (v : view) (y : float) : float = ((y -. v.ay_r) *. v.zoom) +. v.ay_v

type visible = {
  view : view;
  cols : int;
  rows : int;
  masks : int array;
  objects : obj list;
}

(* the rectangles of room.floors without repetitions (the game lists a rectangle once per area it
   touches), in a fixed order, so that "f<index>" names one rectangle of the room *)
let floor_cache : (string * rect list) option ref = ref None

let floor_rects (game : game) : rect list =
  let name = room_name game in
  match !floor_cache with
  | Some (name', rects) when name = name' -> rects
  | _ ->
    let all = List.map snd game.room.floors |> List.concat in
    let same (a : rect) (b : rect) = a.pos.x = b.pos.x && a.pos.y = b.pos.y && a.w = b.w && a.h = b.h in
    let rects =
      List.fold_left (fun acc r -> if List.exists (same r) acc then acc else r :: acc) [] all |> List.rev
    in
    floor_cache := Some (name, rects);
    rects

let label_pass (state : state) (game : game) : visible =
  let w, h = (Raylib.get_screen_width (), Raylib.get_screen_height ()) in
  let target = get_target w h in
  let sprite_objs : (int, obj) Hashtbl.t = Hashtbl.create 64 in
  Raylib.begin_texture_mode target;
  Raylib.clear_background (color c_bg 0);
  Raylib.begin_mode_2d state.camera.raylib;
  Raylib.begin_shader_mode (Lazy.force label_shader);
  draw_labels state game sprite_objs;
  Raylib.end_shader_mode ();
  Raylib.end_mode_2d ();
  Raylib.end_texture_mode ();
  let image = Raylib.load_image_from_texture (Raylib.RenderTexture.texture target) in
  if mask_capture then (
    (* human verification only; never read by a harness. Rows are bottom-up, as OpenGL stores them. *)
    (try Sys.mkdir (path "labels") 0o775 with
    | Sys_error _ -> ());
    ignore (Raylib.export_image image (path (fmt "labels/%06d.png" (!sequence + 1)))));
  let colors = Raylib.load_image_colors image in
  let bytes =
    Ctypes.bigarray_of_ptr Ctypes.array1 (w * h * 4) Bigarray.char
      (Ctypes.coerce (Ctypes.ptr Raylib.Color.t) (Ctypes.ptr Ctypes.char) colors)
  in
  let camera_x, camera_y =
    ( Raylib.Vector2.x (Raylib.Camera2D.target state.camera.raylib)
      -. Raylib.Vector2.x (Raylib.Camera2D.offset state.camera.raylib),
      Raylib.Vector2.y (Raylib.Camera2D.target state.camera.raylib)
      -. Raylib.Vector2.y (Raylib.Camera2D.offset state.camera.raylib) )
  in
  let view : view =
    match view_anchor with
    | None -> { zoom = 1.; ax_r = 0.; ay_r = 0.; ax_v = 0.; ay_v = 0.; vw = w; vh = h }
    | Some (ax_v, ay_v) ->
      let d = game.player.ghost.entity.dest in
      let vw, vh = match view_size with Some size -> size | None -> (960, 540) in
      {
        zoom = view_zoom;
        ax_r = d.pos.x +. (d.w /. 2.) -. camera_x;
        ay_r = d.pos.y +. d.h -. camera_y;
        ax_v;
        ay_v;
        vw;
        vh;
      }
  in
  let cols, rows = ((view.vw + cell - 1) / cell, (view.vh + cell - 1) / cell) in
  let masks = Array.make (cols * rows) 0 in
  let classes = Bytes.create (w * h) in
  let faded = fade_alpha state >= 200 in
  let last_key = ref (-1) in
  let last_obj = ref None in
  (* window pixel -> cell column / row of the observation's screen (-1 = outside it) *)
  let col_of =
    Array.init w (fun x ->
        let vx = Float.to_int (Float.floor (view_x view (Int.to_float x +. 0.5))) in
        if vx < 0 || vx >= view.vw then -1 else vx / cell)
  in
  let row_of =
    Array.init h (fun y ->
        let vy = Float.to_int (Float.floor (view_y view (Int.to_float y +. 0.5))) in
        if vy < 0 || vy >= view.vh then -1 else vy / cell)
  in
  for y = 0 to h - 1 do
    let src = (h - 1 - y) * w * 4 in
    let row = Array.unsafe_get row_of y in
    for x = 0 to w - 1 do
      let o = src + (x * 4) in
      let cls = Char.code (Bigarray.Array1.unsafe_get bytes o) in
      let cls = if faded then c_unknown else cls in
      Bytes.unsafe_set classes ((y * w) + x) (Char.unsafe_chr cls);
      let col = Array.unsafe_get col_of x in
      if row >= 0 && col >= 0 then (
        let ci = (row * cols) + col in
        Array.unsafe_set masks ci (Array.unsafe_get masks ci lor bit_of_class cls));
      if cls >= c_platform && cls <> c_hazard_tiles && cls <> c_hazard_rect && cls < c_unknown then (
        let idx =
          (Char.code (Bigarray.Array1.unsafe_get bytes (o + 1)) lsl 8)
          lor Char.code (Bigarray.Array1.unsafe_get bytes (o + 2))
        in
        let key = (cls lsl 16) lor idx in
        if key <> !last_key then (
          last_key := key;
          last_obj := Hashtbl.find_opt sprite_objs key);
        match !last_obj with
        | None -> ()
        | Some ob ->
          if x < ob.x0 then ob.x0 <- x;
          if x > ob.x1 then ob.x1 <- x;
          if y < ob.y0 then ob.y0 <- y;
          if y > ob.y1 then ob.y1 <- y)
    done
  done;
  Raylib.unload_image_colors colors;
  Raylib.unload_image image;
  (* cells that reach beyond the window (or beyond the observation's screen) are partly unknown *)
  let back_x vx = ((vx -. view.ax_v) /. view.zoom) +. view.ax_r in
  let back_y vy = ((vy -. view.ay_v) /. view.zoom) +. view.ay_r in
  for cy = 0 to rows - 1 do
    for cx = 0 to cols - 1 do
      let x0, x1 = (back_x (Int.to_float (cx * cell)), back_x (Int.to_float ((cx + 1) * cell))) in
      let y0, y1 = (back_y (Int.to_float (cy * cell)), back_y (Int.to_float ((cy + 1) * cell))) in
      if
        x0 < -0.01
        || y0 < -0.01
        || x1 > Int.to_float w +. 0.01
        || y1 > Int.to_float h +. 0.01
        || (cx + 1) * cell > view.vw
        || (cy + 1) * cell > view.vh
      then
        masks.((cy * cols) + cx) <- masks.((cy * cols) + cx) lor bit_unk
    done
  done;
  (* objects that are rectangles in the world: the part of the rectangle in which their class is seen *)
  let seen_in (r : rect) (cls_ok : int -> bool) (ob : obj) : unit =
    let fx0, fy0 = (r.pos.x -. camera_x, r.pos.y -. camera_y) in
    let x0 = Int.max 0 (Float.to_int (Float.round fx0)) in
    let y0 = Int.max 0 (Float.to_int (Float.round fy0)) in
    let x1 = Int.min (w - 1) (Float.to_int (Float.round (fx0 +. r.w)) - 1) in
    let y1 = Int.min (h - 1) (Float.to_int (Float.round (fy0 +. r.h)) - 1) in
    if x0 <= x1 && y0 <= y1 then (
      let at x y = cls_ok (Char.code (Bytes.unsafe_get classes ((y * w) + x))) in
      let row_has y =
        let found = ref false in
        let x = ref x0 in
        while (not !found) && !x <= x1 do
          if at !x y then found := true;
          incr x
        done;
        !found
      in
      let top = ref y0 in
      while !top <= y1 && not (row_has !top) do
        incr top
      done;
      if !top <= y1 then (
        let bottom = ref y1 in
        while !bottom > !top && not (row_has !bottom) do
          decr bottom
        done;
        let col_has x =
          let found = ref false in
          let y = ref !top in
          while (not !found) && !y <= !bottom do
            if at x !y then found := true;
            incr y
          done;
          !found
        in
        let left = ref x0 in
        while !left < x1 && not (col_has !left) do
          incr left
        done;
        let right = ref x1 in
        while !right > !left && not (col_has !right) do
          decr right
        done;
        ob.x0 <- !left;
        ob.x1 <- !right;
        ob.y0 <- !top;
        ob.y1 <- !bottom))
  in
  let tag = room_name game in
  let rect_objs = ref [] in
  let add oid kind look (r : rect) cls_ok =
    let ob = new_obj (tag ^ "/" ^ oid) kind look in
    seen_in r cls_ok ob;
    if ob.x1 >= ob.x0 then rect_objs := ob :: !rect_objs
  in
  let room = game.room in
  if not faded then (
    let terrain c = c = c_wall in
    List.iteri
      (fun i (r : rect) -> add (fmt "f%d" i) "wall" "terrain" r terrain)
      (floor_rects game);
    List.iteri
      (fun i ((r : rect), speed) ->
        add (fmt "c%d" i) "wall" (if speed > 0. then "conveyor-right" else "conveyor-left") r (fun c ->
            c = c_solid))
      room.conveyor_belts;
    List.iteri
      (fun li (layer : layer) ->
        if not layer.hidden then (
          let name = base_layer_name layer.name in
          if layer.config.collides_with_ghost then
            List.iter
              (fun (tg : tile_group) ->
                let first = match tg.tile_idxs with i :: _ -> i | [] -> 0 in
                if name = "doors" then (
                  let look =
                    match List.assoc_opt first room.idx_configs with
                    | Some (PURPLE_PEN _) -> "jar"
                    | _ -> "door"
                  in
                  add (fmt "d%d_%d" li first) "interactable" look tg.dest (fun c -> c = c_door))
                else
                  add (fmt "t%d_%d" li first) "wall" name tg.dest (fun c -> c = c_solid))
              layer.tile_groups
          else if name = "acid" then
            List.iter
              (fun (tg : tile_group) ->
                let first = match tg.tile_idxs with i :: _ -> i | [] -> 0 in
                add (fmt "a%d_%d" li first) "visible_hazard" "acid" tg.dest (fun c -> c = c_hazard_tiles))
              layer.tile_groups
          else if name = "water" then
            List.iter
              (fun (tg : tile_group) ->
                let first = match tg.tile_idxs with i :: _ -> i | [] -> 0 in
                add (fmt "w%d_%d" li first) "visible_scenery" "water" tg.dest (fun c -> c = c_water))
              layer.tile_groups))
      room.layers;
    (* spikes and other hazard rectangles: what is drawn there (terrain art, hazard tiles, scenery) *)
    let drawn c = c = c_wall || c = c_hazard_tiles || c = c_scenery || c = c_wall_like || c = c_solid in
    List.iteri (fun i (r : rect) -> add (fmt "s%d" i) "visible_hazard" "spikes" r drawn) room.spikes;
    List.iteri (fun i (r : rect) -> add (fmt "h%d" i) "visible_hazard" "hazard" r drawn) room.hazards;
    List.iteri
      (fun i (_id, (r : rect)) ->
        add (fmt "q%d" i) "visible_hazard" "platform-spikes" r (fun c -> drawn c || c = c_platform))
      (String.Map.bindings room.platform_spikes));
  let sprites =
    if faded then
      []
    else
      Hashtbl.fold (fun _ (ob : obj) acc -> if ob.x1 >= ob.x0 then ob :: acc else acc) sprite_objs []
  in
  (* window pixels -> the observation's screen; what falls outside it is dropped *)
  let to_view (ob : obj) : obj option =
    let x0 = Int.max 0 (Float.to_int (Float.floor (view_x view (Int.to_float ob.x0)))) in
    let y0 = Int.max 0 (Float.to_int (Float.floor (view_y view (Int.to_float ob.y0)))) in
    let x1 = Int.min (view.vw - 1) (Float.to_int (Float.ceil (view_x view (Int.to_float (ob.x1 + 1)))) - 1) in
    let y1 = Int.min (view.vh - 1) (Float.to_int (Float.ceil (view_y view (Int.to_float (ob.y1 + 1)))) - 1) in
    if x0 <= x1 && y0 <= y1 then
      Some { ob with x0; y0; x1; y1 }
    else
      None
  in
  let objects = List.filter_map to_view (sprites @ !rect_objs) in
  (* hazard rectangles also mark the grid: the cells they are seen in get the hazard bit *)
  List.iter
    (fun (ob : obj) ->
      if ob.kind = "visible_hazard" && ob.look <> "projectile" then
        for cy = ob.y0 / cell to ob.y1 / cell do
          for cx = ob.x0 / cell to ob.x1 / cell do
            masks.((cy * cols) + cx) <- masks.((cy * cols) + cx) lor bit_haz
          done
        done)
    objects;
  let objects = List.sort (fun (a : obj) (b : obj) -> compare a.oid b.oid) objects in
  { view; cols; rows; masks; objects }

(* ---- JSON ------------------------------------------------------------------------------------------------ *)
let utf8_cell (mask : int) : string =
  (* the Hollow Knight bridge writes each cell as the character U+0100 + mask *)
  let cp = 0x100 + mask in
  let b = Bytes.create 2 in
  Bytes.set b 0 (Char.chr (0xC0 lor (cp lsr 6)));
  Bytes.set b 1 (Char.chr (0x80 lor (cp land 0x3F)));
  Bytes.to_string b

let num (f : float) : Yojson.Safe.t = if Float.is_finite f then `Float f else `Float 0.

let round2 (f : float) : Yojson.Safe.t = num (Float.round (f *. 100.) /. 100.)
let round4 (f : float) : Yojson.Safe.t = num (Float.round (f *. 10000.) /. 10000.)
let round6 (f : float) : Yojson.Safe.t = num (Float.round (f *. 1000000.) /. 1000000.)

let rules : Yojson.Safe.t =
  `List
    (List.map
       (fun s -> `String s)
       [
         "Explore: leave the room through an opening at one of its edges. No direction to an offscreen exit is supplied.";
         "Walk left/right; jump height depends on how long jump is held. More abilities (dash, wall jump, double jump, spells) are found later in the game.";
         "The nail strikes left/right, or up/down while up/down is held (down only in the air); a downward strike on spikes or an enemy bounces you up.";
         "Doors and jars break when struck. Enemies and their projectiles hurt on touch. Spikes hurt and put you back on the last safe ground.";
         "Screen x increases right and screen y increases down. Each action holds for the requested window of frames; the game pauses while the next action is chosen.";
       ])

let legend : Yojson.Safe.t =
  `Assoc
    [
      ("unknown", `String "? unknown: outside the window, behind the HUD or a text box, or the screen is faded; never safe");
      ("background", `String ". visible background/no visible obstacle; not a collision guarantee");
      ("solid", `String "# visible solid-looking terrain (black areas of the picture, solid doors)");
      ("platform", `String "= visible floating platform (furniture and the like): solid");
      ("hazard", `String "! visible spikes/acid/projectile");
      ("actor", `String "@ visible player/enemy sprite");
      ("interactable", `String "+ visible thing that reacts to a nail strike (door, jar, lever)");
      ("scenery", `String ": visible scenery, debris, water, a dead enemy, the player's own nail swing; no collision");
    ]

let visible_json (state : state) (game : game) (v : visible) : Yojson.Safe.t =
  let tp = tile_px () in
  let camera = state.camera.raylib in
  let left =
    Raylib.Vector2.x (Raylib.Camera2D.target camera) -. Raylib.Vector2.x (Raylib.Camera2D.offset camera)
  in
  let top =
    Raylib.Vector2.y (Raylib.Camera2D.target camera) -. Raylib.Vector2.y (Raylib.Camera2D.offset camera)
  in
  let rows =
    List.init v.rows (fun cy ->
        `String (String.concat "" (List.init v.cols (fun cx -> utf8_cell v.masks.((cy * v.cols) + cx)))))
  in
  let faded = fade_alpha state >= 200 in
  let vw = v.view in
  let player_seen = List.exists (fun (ob : obj) -> ob.kind = "player") v.objects in
  let objects =
    List.map
      (fun (ob : obj) ->
        `Assoc
          [
            ("id", `String ob.oid);
            ("kind", `String ob.kind);
            ("bounds", `List [ `Int ob.x0; `Int ob.y0; `Int (ob.x1 - ob.x0 + 1); `Int (ob.y1 - ob.y0 + 1) ]);
            ("look", `String ob.look);
          ])
      v.objects
  in
  let self =
    let d = game.player.ghost.entity.dest in
    let base = [ ("object_id", if player_seen then `String (room_name game ^ "/player") else `Null) ] in
    if self_box && player_seen then
      `Assoc
        (base
        @ [
            ( "screen_box",
              `List
                [
                  round2 (view_x vw (d.pos.x -. left));
                  round2 (view_y vw (d.pos.y -. top));
                  round2 (d.w *. vw.zoom);
                  round2 (d.h *. vw.zoom);
                ] );
            ("facing", `String (if game.player.ghost.entity.sprite.facing_right then "right" else "left"));
          ])
    else
      `Assoc base
  in
  let texts =
    let entry kind blocking fields = `Assoc ([ ("kind", `String kind); ("blocking", `Bool blocking) ] @ fields) in
    let blocking =
      match (state.pause_menu, game.interaction.text) with
      | Some _, _ -> [ entry "menu" true [] ]
      | None, None -> []
      | None, Some (PLAIN (_, lines)) ->
        [ entry "text" true [ ("content", `String (strip_markup (String.concat " / " lines))) ] ]
      | None, Some (ABILITY t)
      | None, Some (FOCUS_ABILITY t) ->
        [
          entry "text" true
            [ ("content", `String (strip_markup (String.concat " / " (t.top_paragraphs @ t.bottom_paragraphs)))) ];
        ]
      | None, Some (DIALOGUE (speaker, text)) ->
        [ entry "dialogue" true [ ("speaker", `String speaker); ("content", `String (strip_markup text)) ] ]
      | None, Some (MENU _) -> [ entry "menu" true [] ]
    in
    let other (t : Interaction.non_blocking_text option) kind =
      match t with
      | None -> []
      | Some t -> [ entry kind false [ ("content", `String (strip_markup t.content)) ] ]
    in
    let label =
      match game.room.interaction_label with
      | None -> []
      | Some (label, dest) ->
        [
          entry "label" false
            [
              ("content", `String label);
              ( "bounds",
                `List
                  [
                    `Int (Float.to_int (Float.floor (view_x vw (dest.pos.x -. left))));
                    `Int (Float.to_int (Float.floor (view_y vw (dest.pos.y -. top))));
                    `Int (Float.to_int (dest.w *. vw.zoom));
                    `Int (Float.to_int (dest.h *. vw.zoom));
                  ] );
            ];
        ]
    in
    blocking @ other game.interaction.corner_text "corner" @ other game.interaction.floating_text "floating" @ label
  in
  let hud =
    let health =
      match game.mode with
      | STEEL_SOLE -> []
      | CLASSIC
      | DEMO ->
        if faded then
          []
        else
          [
            ("health", `Assoc [ ("value", `Int (Int.max 0 game.player.health.current)); ("evidence", `String "hud") ]);
            ("health_max", `Assoc [ ("value", `Int game.player.health.max); ("evidence", `String "hud") ]);
          ]
    in
    let soul =
      if faded then
        []
      else
        [
          ( "soul",
            `Assoc
              [
                ( "fraction",
                  round2 ((game.player.soul.current |> Int.to_float) /. (game.player.soul.max |> Int.to_float)) );
                ("evidence", `String "hud gauge");
              ] );
        ]
    in
    `Assoc (health @ soul @ [ ("text", `List (if faded then [] else texts)) ])
  in
  `Assoc
    [
      ("schema", `String "honing.visible.v1");
      ("adapter", `String "hallowdale-visible-v1");
      ("profile", `String "visible");
      ("game", `String "hallowdale");
      ( "frame",
        `Assoc
          [ ("id", `Int !sequence); ("tick", `Int !frames_played); ("simulation_time", round6 !sim_time) ] );
      ( "view",
        `Assoc
          [
            ("screen_rect", `List [ `Int 0; `Int 0; `Int vw.vw; `Int vw.vh ]);
            ( "world_from_screen",
              `List
                [
                  num (1. /. (tp *. vw.zoom));
                  num 0.;
                  num 0.;
                  num (-1. /. (tp *. vw.zoom));
                  num ((left +. vw.ax_r -. (vw.ax_v /. vw.zoom)) /. tp);
                  num (-.(top +. vw.ay_r -. (vw.ay_v /. vw.zoom)) /. tp);
                ] );
            ( "window",
              `Assoc
                [
                  ("size", `List [ `Int (Raylib.get_screen_width ()); `Int (Raylib.get_screen_height ()) ]);
                  (* where the real game window lies on the observation's screen (x, y, w, h); everything of
                     that screen outside this rectangle is not on the player's screen: reported as unknown,
                     no objects *)
                  ( "rect",
                    `List
                      [
                        round2 (view_x vw 0.);
                        round2 (view_y vw 0.);
                        round2 (Int.to_float (Raylib.get_screen_width ()) *. vw.zoom);
                        round2 (Int.to_float (Raylib.get_screen_height ()) *. vw.zoom);
                      ] );
                  ( "player_centred",
                    match view_anchor with
                    | None -> `Bool false
                    | Some (x, y) -> `Assoc [ ("feet_at", `List [ num x; num y ]); ("zoom", num vw.zoom) ] );
                ] );
            ( "world_units",
              `String
                "1 unit = 1 map tile (12 source px); x right, y up; origin = top-left corner of the current room; screen x right/y down" );
          ] );
      ( "terrain",
        `Assoc
          [
            ("status", `String "available");
            ("coverage", `String "full");
            ("origin", `List [ `Int 0; `Int 0 ]);
            ("cell_size", `List [ `Int cell; `Int cell ]);
            ("rows", `List rows);
            ("legend", legend);
          ] );
      ("objects", `List objects);
      ("self", self);
      ("hud", hud);
      ("rules", rules);
      ( "capabilities",
        `Assoc [ ("global_debug", `Bool true); ("pixels_to_model", `Bool false); ("hidden_own_state", `Bool false) ] );
      ( "disclosure",
        `Assoc
          [
            ( "visibility",
              `String
                "label render of the scene with the game's camera and draw order; what is covered by terrain, shadow layers, the HUD or a text box is not reported" );
            ( "geometry",
              `String
                "terrain/door/hazard objects are the game's collision rectangles clipped to where their art is seen; sprites are their visible pixel bounds" );
            ( "unsupported",
              `String "texels below 0.9 alpha are treated as not drawn; pause menu and map screens are reported as unknown" );
            ("resolution", `String (fmt "%d px cells on a %d x %d screen" cell vw.vw vw.vh));
            ( "debug",
              `String "debug_observation.json / observations.jsonl hold global state; plumbing and evaluation only" );
          ] );
    ]

let debug_json (state : state) (game : game) : Yojson.Safe.t =
  let tp = tile_px () in
  let e = game.player.ghost.entity in
  let ux v = v /. tp in
  let enemies =
    List.filter_map
      (fun (enemy : enemy) ->
        if enemy.entity.dest.pos.x > -1. && enemy.entity.dest.pos.y > -1. then
          Some
            (`Assoc
              [
                ("name", `String (Show.enemy_id enemy.id));
                ("x", round2 (ux (rect_center_x enemy.entity.dest)));
                ("y", round2 (-.ux (enemy.entity.dest.pos.y +. enemy.entity.dest.h)));
                ("health", `Int enemy.health.current);
              ])
        else
          None)
      game.room.enemies
  in
  let abilities =
    let a = game.player.abilities in
    List.filter_map
      (fun (name, on) -> if on then Some (`String name) else None)
      [
        ("dash", a.mothwing_cloak);
        ("wall_jump", a.mantis_claw);
        ("double_jump", a.monarch_wings);
        ("c_dash", a.crystal_heart);
        ("dream_nail", a.dream_nail);
        ("ismas_tear", a.ismas_tear);
        ("vengeful_spirit", a.vengeful_spirit);
        ("desolate_dive", a.desolate_dive);
        ("howling_wraiths", a.howling_wraiths);
      ]
  in
  let camera = state.camera.raylib in
  `Assoc
    [
      ("id", `Int !sequence);
      ("frame", `Int !frames_played);
      ("health", `Int (Int.max 0 game.player.health.current));
      ("jumps", `Int (if Entity.on_ground e then 1 else 0));
      ("fixedSteps", `Int !ticks_played);
      ("scene", `String (room_name game));
      ("x", round4 (ux (rect_center_x e.dest)));
      ("y", round4 (-.ux (e.dest.pos.y +. e.dest.h)));
      ("vx", round4 (ux e.v.x));
      ("vy", round4 (-.ux e.v.y));
      ("simulationTime", round4 !sim_time);
      ("grounded", `Bool (Entity.on_ground e));
      ("facing", `String (if e.sprite.facing_right then "right" else "left"));
      ("frozen", `Bool e.frozen);
      ("healthMax", `Int game.player.health.max);
      ("soul", `Int game.player.soul.current);
      ("deaths", `Int !deaths);
      ("interacting", `Bool (List.length game.interaction.steps > 0));
      ("fadeAlpha", `Int (fade_alpha state));
      ("inBossFight", `Bool game.in_boss_fight);
      ("mode", `String (Show.game_mode game.mode));
      ("abilities", `List abilities);
      ("pixel", `List [ round2 e.dest.pos.x; round2 e.dest.pos.y; round2 e.dest.w; round2 e.dest.h ]);
      ( "camera",
        `List
          [
            round2 (Raylib.Vector2.x (Raylib.Camera2D.target camera));
            round2 (Raylib.Vector2.y (Raylib.Camera2D.target camera));
          ] );
      ("tilePixels", round4 tp);
      ("enemies", `List enemies);
      ("exits", `List []);
    ]

let observe (state : state) : unit =
  match game_of state with
  | None -> log_line "observe: no game in progress"
  | Some game ->
    incr sequence;
    let v = label_pass state game in
    let visible = Yojson.Safe.to_string (visible_json state game v) in
    let debug = Yojson.Safe.to_string (debug_json state game) in
    append "visible_observations.jsonl" visible;
    append "observations.jsonl" debug;
    write_atomic "debug_observation.json" debug;
    write_atomic "observation.json" (if profile = "global_debug" then debug else visible)

(* ---- actions ------------------------------------------------------------------------------------------------ *)
(* Play one accepted window; the keys of every tick come from Honing_input.window_keys. Returns the state
   and the keys applied (key -> first tick, last tick, number of ticks). *)
let play_window (state : state) (w : Honing_input.window) : state * (string * (int * int * int)) list =
  let down_before = List.filter Honing_input.is_down Honing_input.key_names in
  let sticky = jump_mode = "sticky" in
  jump_latched := Honing_input.latch_after ~sticky ~latched:!jump_latched w;
  let latched = !jump_latched in
  let keys_at = Honing_input.window_keys ~ticks_per_frame ~sticky ~down_before ~latched w in
  let s = ref state in
  let frames = w.Honing_input.frames in
  for f = 0 to frames - 1 do
    let save =
      if capture then
        Some (path (fmt "frames/%06d.png" !frames_played))
      else
        None
    in
    s := frame ?save ~last_of_window:(f = frames - 1) !s (fun i -> keys_at ((f * ticks_per_frame) + i));
    incr frames_played
  done;
  (!s, Honing_input.key_ticks ~ticks_per_frame ~keys_at w)

(* The player died in the last tick of the window: the game reloads its save in the next tick, which
   takes no input. Play that tick, so that the observation shows the game as it continues and the command
   cannot run twice. Any other state without a game in progress ends the session. *)
let settle (state : state) : state option =
  match state.context with
  | IN_PROGRESS _ -> Some state
  | RELOAD_LAST_SAVED_GAME _ -> (
    let s = tick state [] in
    match s.context with
    | IN_PROGRESS _ -> Some s
    | _ -> None)
  | _ -> None

(* ---- acknowledgements ------------------------------------------------------------------------------------- *)
(* the game's own record of what the player started during a window (plumbing, never sent to a model) *)
let started_since (game : game) (time : float) : Yojson.Safe.t =
  let h = game.player.history in
  let all =
    [
      ("CAST_VENGEFUL_SPIRIT", h.cast_vs);
      ("CAST_DESOLATE_DIVE", h.cast_dive);
      ("CAST_HOWLING_WRAITHS", h.cast_wraiths);
      ("DIVE_HOP", h.dive_hop);
      ("DASH", h.dash);
      ("SHADE_DASH", h.shade_dash);
      ("C_DASH_CHARGE", h.charge_c_dash);
      ("C_DASH", h.c_dash);
      ("C_DASH_COOLDOWN", h.c_dash_cooldown);
      ("FLAP", h.flap);
      ("WALL_KICK", h.wall_kick);
      ("TAKE_DAMAGE", h.take_damage);
      ("TAKE_DAMAGE_AND_RESPAWN", h.take_damage_and_respawn);
      ("DIE", h.die);
      ("NAIL", h.nail);
      ("DREAM_NAIL", h.dream_nail);
      ("FOCUS", h.focus);
      ("HARDFALL", h.hardfall);
    ]
  in
  `List
    (List.filter_map
       (fun (name, (action : ghost_action)) -> if action.started.at > time then Some (`String name) else None)
       all)

(* the player's attached things at the end of a window; a strike is named with its direction *)
let child_name (kind : ghost_child_kind) : string =
  match kind with
  | NAIL slash -> "NAIL_" ^ String.uppercase_ascii (Controls.show_direction slash.direction)
  | _ -> Show.ghost_child_kind kind

(* every acknowledgement carries a running number, so that a client can tell a new one from the last one *)
let acks_written = ref 0

let write_ack (fields : (string * Yojson.Safe.t) list) : unit =
  incr acks_written;
  let text = Yojson.Safe.to_string (`Assoc (("ack", `Int !acks_written) :: fields)) in
  append "acks.jsonl" text;
  write_atomic "ack.json" text

let refuse (id : int option) (reason : string) : unit =
  log_line "refused command: %s" reason;
  write_ack
    [
      ("id", match id with Some n -> `Int n | None -> `Null);
      ("status", `String "refused");
      ("reason", `String reason);
      ("observation_id", `Int !sequence);
    ]

let read_command_text () : string option =
  let file = path "action.json" in
  if not (Sys.file_exists file) then
    None
  else (
    try Some (File.read file) with
    | Sys_error _
    | End_of_file ->
      None)

(* ---- boot: from the title menu to the first moment the player has control ------------------------------ *)
let idle (state : state) (frames : int) : state =
  let s = ref state in
  for _ = 1 to frames do
    let save =
      if capture && capture_boot then (
        let file = path (fmt "boot_frames/%06d.png" !ticks_played) in
        Some file)
      else
        None
    in
    s := frame ?save !s (fun _ -> [])
  done;
  !s

let tap (state : state) (key : string) (frames_after : int) : state =
  let s = frame state (fun _ -> [ key ]) in
  idle s frames_after

let boot (state : state) : state =
  let s = ref (idle state 20) in
  (* title menu: Start Game *)
  s := tap !s "JUMP" 15;
  (* save files: slot 1 (a new game, or the save file put there by HONING_SCENE=save:<file>) *)
  s := tap !s "JUMP" 15;
  (match !s.context with
  | IN_PROGRESS _ -> ()
  | _ ->
    (* game mode menu: Classic is the first entry *)
    s := tap !s "JUMP" 15);
  (* opening: advance text with the interact key until the game hands over control *)
  let ready (state : state) =
    match game_of state with
    | None -> false
    | Some game ->
      List.length game.interaction.steps = 0
      && Option.is_none game.interaction.text
      && Entity.on_ground game.player.ghost.entity
      && Option.is_none state.screen_fade
  in
  let frames = ref 0 in
  while (not (ready !s)) && !frames < frame_hz * 90 do
    (match game_of !s with
    | Some game when Option.is_some game.interaction.text -> s := tap !s "INTERACT" 9
    | _ -> s := idle !s 10);
    frames := !frames + 10
  done;
  if not (ready !s) then log_line "boot: the game did not hand over control within 90 s of game time";
  s := idle !s 5;
  !s

(* ---- main loop ---------------------------------------------------------------------------------------------- *)
(* before State.init: the session directories, and the save file of a debug start
   (HONING_SCENE=save:<file> puts that file into slot 1; the menu then offers "File 1: Continue") *)
let prepare () : unit =
  (try Sys.mkdir session 0o775 with
  | Sys_error _ -> ());
  List.iter
    (fun d ->
      try Sys.mkdir (path d) 0o775 with
      | Sys_error _ -> ())
    ([ "saves" ] @ (if capture then [ "frames" ] else []) @ if capture && capture_boot then [ "boot_frames" ] else []);
  if String.length scene > 5 && String.sub scene 0 5 = "save:" then
    File.copy_file (String.sub scene 5 (String.length scene - 5)) (path "saves/1.json")

let run (state : state) : unit =
  log_line "bridge start: scene=%s frame_hz=%d ticks_per_frame=%d dt=%.6f jump_mode=%s capture=%b render=%s window=%.0fx%.0f scale=%.5f seed=%d"
    scene frame_hz ticks_per_frame dt jump_mode capture render_mode Config.window.w Config.window.h Config.window.scale seed;
  let s = ref (boot state) in
  log_line "boot done: ticks=%d sim_time=%.3f" !ticks_played !sim_time;
  observe !s;
  let waiting_since = ref (Unix.gettimeofday ()) in
  let running = ref true in
  (* a command is looked at once: the file keeps its last command until the client replaces it *)
  let last_text = ref "" in
  let execute (id : int) (action : string) (w : Honing_input.window) : unit =
    let frame_before = !frames_played and time_before = !sim_time in
    let played, keys = play_window !s w in
    match settle played with
    | None ->
      log_line "no game in progress after the window for command %d" id;
      running := false
    | Some settled ->
      s := settled;
      let game_fields =
        match game_of settled with
        | None -> []
        | Some game ->
          [
            ("started", started_since game time_before);
            ( "children",
              `List
                (List.map
                   (fun (kind, _) -> `String (child_name kind))
                   (Ghost_child_kind.Map.bindings game.player.children)) );
          ]
      in
      (* published before the observation: a client that sees the next observation finds this ack *)
      write_ack
        ([
           ("id", `Int id);
           ("status", `String "applied");
           ("action", `String action);
           ("frames", `Int w.Honing_input.frames);
           ("jump_frames", match w.Honing_input.jump_frames with Some n -> `Int n | None -> `Null);
           ( "keys",
             `Assoc
               (List.map
                  (fun (key, (first, last, count)) ->
                    (key, `Assoc [ ("first_tick", `Int first); ("last_tick", `Int last); ("ticks", `Int count) ]))
                  keys) );
           ("ticks_per_frame", `Int ticks_per_frame);
           ("frame_before", `Int frame_before);
           ("frame_after", `Int !frames_played);
           ("observation_id", `Int (!sequence + 1));
         ]
        @ game_fields);
      observe !s;
      waiting_since := Unix.gettimeofday ()
  in
  while !running do
    if Raylib.window_should_close () then
      running := false
    else (
      (match read_command_text () with
      | Some text when text <> !last_text -> (
        last_text := text;
        match Honing_command.parse text with
        | Honing_command.Stop id when id = !sequence ->
          write_ack [ ("id", `Int id); ("status", `String "stopped") ];
          running := false
        | Honing_command.Window (id, action, w) when id = !sequence -> execute id action w
        | Honing_command.Stop id
        | Honing_command.Window (id, _, _) ->
          refuse (Some id) (fmt "id %d is not the current observation id %d" id !sequence)
        | Honing_command.Refused (id, reason) -> refuse id reason)
      | _ -> ());
      (* checked on every pass: stale and invalid commands neither run nor extend this deadline *)
      if !running then
        if Unix.gettimeofday () -. !waiting_since > model_timeout then (
          log_line "timed out waiting for an action";
          running := false)
        else
          Unix.sleepf 0.002)
  done;
  log_line "bridge stop: frames=%d ticks=%d deaths=%d" !frames_played !ticks_played !deaths;
  Raylib.close_audio_device ();
  Raylib.close_window ()
