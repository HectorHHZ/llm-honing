(* SPDX-License-Identifier: Apache-2.0 *)
(* Honing bridge, part 2 of 3: one action.json text -> a validated request, or the reason it is refused.
   Depends only on yojson and Honing_input, so that the tests can compile it alone. *)

type parsed =
  | Stop of int
  | Window of int * string * Honing_input.window  (** id, action name, the accepted window *)
  | Refused of int option * string  (** id when it could be read, reason *)

let unique_keys (fields : (string * Yojson.Safe.t) list) : bool =
  let names = List.map fst fields in
  List.length names = List.length (List.sort_uniq compare names)

let strings (name : string) (fields : (string * Yojson.Safe.t) list) : (string list, string) result =
  match List.assoc_opt name fields with
  | None -> Ok []
  | Some (`List items) ->
    if List.for_all (function `String _ -> true | _ -> false) items then
      Ok (List.filter_map (function `String s -> Some s | _ -> None) items)
    else
      Error (name ^ " must be a list of key names")
  | Some _ -> Error (name ^ " must be a list of key names")

let parse (text : string) : parsed =
  match Yojson.Safe.from_string text with
  | exception _ -> Refused (None, "malformed JSON")
  | `Assoc fields when not (unique_keys fields) -> Refused (None, "duplicate JSON keys")
  | `Assoc fields -> (
    match List.assoc_opt "id" fields with
    | Some (`Int id) -> (
      let refuse reason = Refused (Some id, reason) in
      match List.assoc_opt "action" fields with
      | Some (`String "stop") -> Stop id
      | Some (`String action) -> (
        match (strings "hold" fields, strings "press" fields, strings "first" fields) with
        | Error reason, _, _
        | _, Error reason, _
        | _, _, Error reason ->
          refuse reason
        | Ok hold, Ok press, Ok first -> (
          let jump_frames =
            match List.assoc_opt "jump_frames" fields with
            | None
            | Some `Null ->
              Ok None
            | Some (`Int n) -> Ok (Some n)
            | Some _ -> Error "jump_frames must be an integer"
          in
          match (List.assoc_opt "frames" fields, jump_frames) with
          | _, Error reason -> refuse reason
          | Some (`Int frames), Ok jump_frames -> (
            match Honing_input.resolve ~action ~frames ~hold ~press ~first ~jump_frames with
            | Ok window -> Window (id, action, window)
            | Error reason -> refuse reason)
          | _ -> refuse "frames must be an integer"))
      | _ -> refuse "action must be a string")
    | _ -> Refused (None, "id must be an integer"))
  | _ -> Refused (None, "a command must be a JSON object")
