# hallowdale native decision demo

This standalone integration connects an existing Honing decision server to
[hallowdale](https://github.com/mega-dean/hallowdale), a 2D action platformer by mega-dean written in OCaml with
raylib. The game calls itself an homage to Hollow Knight. Its Classic mode has the cutscenes of the Community
episode "Geothermal Escapism"; Steel Sole is a "New Game +" mode with all abilities.

The player walks, jumps, strikes with a nail, collects abilities, casts spells and fights bosses across connected
rooms. The pinned source has 63 room files in 11 areas. The time to finish the game has not been measured.

The game's own code runs all movement, combat and collision. A small bridge compiled into the game supplies key
states, plays fixed time steps and publishes what a player can see. This is an input demo: it has no campaign task,
EpisodeRunner, reward, grader or training.

- [API.md](API.md) defines the observation, every action, timing, acknowledgements and errors.
- [validation/README.md](validation/README.md) lists the evidence in this directory.

## Get the pinned game and build it

The source is pinned to commit `a330d3462afe5fa7b8e65573ad8a7c7c79de17e9`. Keep the checkout, its assets, the binary
and all run output outside this repository.

Install the toolchain with [opam](https://opam.ocaml.org/doc/Install.html). These are the versions used here:

```bash
sudo apt-get install pkg-config libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev \
  libgl1-mesa-dev libasound2-dev xvfb ffmpeg
opam switch create 5.1.0
eval "$(opam env --switch=5.1.0)"
opam install dune.3.25.0 raylib.1.2.1 atdgen.2.15.0 yojson.2.1.2 ppx_deriving.6.2.0
```

Then fetch, patch and build:

```bash
python integrations/hallowdale/build.py --source /local/hallowdale
```

`build.py` clones the upstream repository when the directory does not exist, refuses any other revision, applies
`native.patch` once, copies the three bridge modules into `src/` and runs `dune build --profile release`. It prints
the path of the executable. The game looks for a parent directory with `hallowdale` in its name to find its assets,
so the checkout directory must be named that way.

Without `HONING_SESSION_DIR` the patched game behaves like the original.

Validated on Linux x86_64 (Ubuntu 22.04, GCC 11.3, OCaml 5.1.0) with Mesa software OpenGL under Xvfb. Other
platforms have not been run.

## Start a decision server

Start any server with Honing's `GET /v1/health` and `POST /v1/decide` routes. The
[Hollow Knight Laya CPU recipe](../hollow-knight/README.md#local-cpu-reproduction) starts one on CPU. `run.py` does
not start, configure or stop the server.

## Run

```bash
python integrations/hallowdale/run.py \
  --game /local/hallowdale/_build/default/bin/main.exe \
  --decision-url http://127.0.0.1:8011 \
  --output /local/hallowdale-run --decisions 12 --frames 12 --capture
python integrations/hallowdale/encode.py --session /local/hallowdale-run
```

`run.py` starts one Xvfb with `-displayfd` and one game process, and stops only those two. The game runs silently.
`--output` must be a new directory. It holds the private requests, raw responses and frames, so do not commit it.

`--timeout` is the model's HTTP timeout, 180 seconds by default. The bridge's idle watchdog receives that value plus
30 seconds. The game pauses while the model decides, so the video shows game time without the waiting.

## Observation and actions

The model receives what a human player can see on the real game window, as a grid plus a list of the things in view.
Cells outside the window or covered by the HUD or other opaque UI are unknown, and nothing behind terrain, shadow
layers or text boxes is reported. Global state goes to a separate debug file that the client never sends to a model.
No pixels are sent. The bridge's default cell is 20 pixels. `run.py` uses 40 pixels so that a request fits a 2048
token Laya context; `--cell-pixels` changes it.

Every control the game offers during play is available: walking, full and shortened jumps, directional strikes,
dash, interact, focus, all three spells, c-dash and dream nail, plus a `keys` command for any combination. Pause and
map are left out because they open menus that stop the game. Abilities the player has not found do nothing, as in
the game, and the bridge unlocks nothing. API.md has the complete table.

Every command is checked before anything is played. An invalid one is refused and its reason is acknowledged. An
accepted one is acknowledged with the keys that reached the game on each tick.

## Native checks

`verify_native.py` runs fixed inputs against the real game with no model:

```bash
python integrations/hallowdale/verify_native.py \
  --game /local/hallowdale/_build/default/bin/main.exe --output /local/hallowdale-checks
```

The author-side results are in [validation/native-checks.json](validation/native-checks.json) and
[validation/action-probes.json](validation/action-probes.json). A "debug start" loads a hand-made save from
[validation/debug-saves/](validation/debug-saves/), because the ability is found much later in the game.

| Probe | Start | Measured in the game |
| --- | --- | --- |
| Strike | New game | `attack_right` breaks the door that blocked the way; strikes go left, right, up and down |
| Jump height | New game | 1.413, 3.139, 5.313 and 8.183 tiles for 1, 3, 6 and 12 held frames; 9.093 for a full jump |
| Repeated jump | New game | Two jumps in a row rise 9.093 tiles each |
| Locked abilities | New game | Dash, cast, c-dash and dream nail start nothing; soul stays at 99 |
| Interact | Debug start | Text opens at a readable sign, holds the player and closes on interact |
| Focus | Debug start | After a spike hit: health 4 to 5, soul 99 to 66 |
| Spells | Debug start | Each of the three spells starts and spends 33 soul |
| C-dash | Debug start | A short hold does not launch; a full charge launches at 33.3 tiles/s against 11.1 walking |
| Dream nail | Debug start | A short hold does not swing; a full hold does |
| Dash, double jump | Debug start | Dash at 22.2 tiles/s, a repeated dash starts again, a second jump in the air flaps |
| Death | Debug start | A death on the last tick of a window plays the reload tick once and repeats no command |

[validation/replay-regression.json](validation/replay-regression.json) compares this bridge with the earlier one. The
1,978 action windows of a recorded run were replayed with no model, and all 1,979 visible and 1,979 debug observations
are byte-identical to the recording.

Portable tests need no game. Those that compile the bridge modules need `ocamlfind` and `yojson`, and are skipped
without them:

```bash
pytest tests/test_hallowdale.py tests/test_hallowdale_bridge.py
```

## Real model run

[validation/review-sample/](validation/review-sample/) holds one complete decision from a 12-decision run of `run.py`
against the frozen Laya typed-decisions model on CPU, in a new game, with 12 frames per decision. All 12 responses
passed the shared receiver. The model chose `attack_right` once and `attack` eleven times, with the top probability
between 0.09 and 0.10 over the 28 choices. The player walked 4.4 tiles to the right during the first decision and
stayed there with full health. This shows that requests, responses and native actions connect. It does not show that
the model can play the game.

## Demo video

[validation/demo-harness-laya-edited.mp4](validation/demo-harness-laya-edited.mp4) is an edited cut of one run in the
first area, Dean's Pass, with the frozen Laya typed-decisions model choosing. **It was not made with `run.py`.** The
choices in that run were offered by a separate human-in-the-loop harness, which is not part of this integration and
comes in a later pull request. The harness turns the observation into language and into a short menu of skills, so the
video shows what that harness and the model did together. It is no measure of how the model plays through `run.py`.

The run made 63 decisions with no deaths. The cut plays decisions 1 to 30 at double speed, repeated stretches at eight
times speed, and leaves out decisions 31 to 63, of which 32 repeat earlier ones. The end card says so, and
[validation/demo-cut-list.txt](validation/demo-cut-list.txt) lists every decision shown. The player does not finish
the area: from decision 20 on it loops at a dead end.

## Known limitations

- There is no reset, reward, grader, campaign runner or deterministic replay guarantee across machines.
- The checks above are author-side. A reviewer has not rerun the game.
- Wall kick, shade dash and swimming are reachable through the same keys but have no native probe here.
- A probe's debug start skips the part of the game where an ability is normally found.
- With the 40 pixel grid several classes can share one cell. The grid is what fits Laya's context, not a finding about
  the best resolution.
- `run.py` stops when the server refuses a request that does not fit its context. It does not shorten the state.
- The optional player-centred view settings in the bridge exist for the harness above, and `run.py` does not use them.

## Licensing and provenance

hallowdale's code is under the MIT license and its original creative assets under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), both by mega-dean and stated in the
[upstream LICENSE.md at the pinned commit](https://github.com/mega-dean/hallowdale/blob/a330d3462afe5fa7b8e65573ad8a7c7c79de17e9/LICENSE.md).
The upstream README says that most assets come from the original
[Journey to the Center of Hawkthorne](https://github.com/hawkthorne/hawkthorne-journey) game. The upstream
`dune-project` lists CC-BY-4.0 for the assets; LICENSE.md, the stricter text, is what this integration follows.

- The Python scripts, tests, documents and the bridge modules `honing.ml`, `honing_input.ml` and `honing_command.ml`
  are original to this repository and follow its [Apache-2.0 license](../../LICENSE).
- `native.patch` changes five upstream files and carries a few lines of their context. Those lines remain
  hallowdale's MIT-licensed code; nothing upstream is relicensed.
- The demo video shows hallowdale's artwork. It is shared for noncommercial use under CC BY-NC 4.0 with the credit
  above, and Apache-2.0 does not apply to it.
- No game source beyond the patch context, asset, binary, save from a player, model weight or uncut recording is
  included.
