# Hollow Knight imitation: harness v8 with the frozen Laya model

This experiment puts a hand-written harness between the
[Hollow Knight imitation](https://github.com/DanielDFY/Hollow-Knight-Imitation) and the frozen Laya decision model.
In one continuous run the knight went from the start of `Level1` through its exit to the end of `Level2` with no deaths
and a lowest health of 4 of 5. The harness read only what a player can see on the screen.

The result comes mostly from the harness. A control that always takes the first menu option also clears both levels, and
in the final evaluation Laya's choices did not change whether a level was cleared. The game has two levels and no boss, so
this says nothing about how the approach would do against a boss.

Three kinds of statement appear below and are kept apart. Measured numbers come from the run records in [`data/`](data).
Statements marked "from the code" can be checked in the committed source. Statements marked "our reading" are
interpretation.

| Section | Content |
| --- | --- |
| [Result and demo](#result-and-demo) | The run and its video |
| [The visible bridge](#the-visible-bridge) | What the grid contains, what is hidden, how it is switched on |
| [Who did what](#who-did-what) | The human-in-the-loop process and the versions v1 to v8 |
| [The abstraction](#the-abstraction) | From grid to world model to menu to skill to native inputs |
| [How the rules were found](#how-the-rules-were-found) | Failures, probes, offline tests, controls |
| [Why routing and planning work](#why-routing-and-planning-work) | The hand-engineered parts, with code references |
| [Results with and without Laya](#results-with-and-without-laya) | All run tables |
| [What Laya contributed](#what-laya-contributed) | Decision-level numbers |
| [Known limitations](#known-limitations) | What this does not show |
| [Reproduce](#reproduce) | Runtime build, decision server, commands, offline checks |
| [Files](#files) | What is in this directory |

## Result and demo

| Measure | Main run `eval_laya_v8_L1L2` |
| --- | --- |
| Span | `Level1` start, through the level exit, to `Level2` x at or beyond 124 (the `Victory` sign), in one process |
| Decisions by Laya | 47 (34 in `Level1`, 13 in `Level2`) |
| Native input windows | 1166 |
| Rendered game frames | 4877 (162.6 s of game time at 30 fps) |
| Deaths | 0 |
| Lowest health | 4 of 5 (one enemy touch in `Level1`) |
| Repeat `eval2_laya_v8_L1L2` | Same 47 requests, choices and probabilities, same 1166 windows, same 1167 game-state records |

The main run and the demo predate the last change to the harness, one fix in `control.py` that ignores a blank health display
right after a level exit. The repeat ran the code committed here.

[`data/eval_laya_v8_L1L2/demo_cut.mp4`](data/eval_laya_v8_L1L2/demo_cut.mp4) is the edited cut of the main run (75 s). Normal
play is shown at double speed and the 15 stretches in which the knight stands and waits (16.9 s of game time) at eight
times speed. Nothing is left out; [`demo_cut.mp4.txt`](data/eval_laya_v8_L1L2/demo_cut.mp4.txt) lists the stretches. The caption
in each frame shows the behaviour Laya chose, the native input of that moment and the probability Laya gave the behaviour.
The game is paused while Laya decides, so model time is not in the video. The file is encoded at 30 fps and 104 kbit/s to
stay under the repository's 1 MB file limit.

## The visible bridge

`HoningBridge.cs` gains an optional observation profile, implemented in `VisibleObservation.cs`. Setting
`HONING_OBSERVATION_PROFILE=visible` in the player's environment switches it on, and `core/env.py` does that for every
run here. Without that exact value the bridge behaves as it did before this change. The full field list is in
[API.md](../../API.md#visible-observation-profile). This profile is what this experiment used. It is not a settled
observation format for other games; how environments should present information is still undecided.

| Part of the observation | Content |
| --- | --- |
| Terrain grid | 48 x 27 cells of 20 x 20 screen pixels. Each cell is a bit union of the classes seen in it: background, wall, platform, hazard, actor, interactable, unknown, scenery |
| Objects | Every visible sprite with an id, a kind (`player`, `enemy`, `visible_hazard`, `platform`, `wall`, `interactable`, `visible_equipment`, `visible_scenery`) and the screen box of its visible pixels |
| HUD | Number of full hearts |
| Camera | The transform from screen pixels to world units, which also tells the harness how far the screen scrolled |
| Fixed text | The game's rules and a statement of what the observation leaves out |

Classes come from a second render of the real camera with flat label colors, so anything hidden behind another sprite is
not reported. Deliberately hidden: everything off screen including the level exit, enemy health, the knight's velocity,
grounded state and remaining jumps, colliders without a sprite, and what a lever or secret does. The game's scene state
is still written to `debug_observation.json` and `observations.jsonl` for evaluation. No harness code reads it. In
`run_agent.py` it decides when an episode ends and what is printed.

The model never sees the grid. The harness reads it and sends Laya a short text.

## Who did what

The project author set the goals and the design rules and reviewed each round. The rules were: the harness is everything
between the observation and the model plus the executor, it may hold human experience, it reads only what a player sees,
and the model's weights stay frozen and its choice is final. Claude Fable 5.2, an AI coding agent, wrote and tested the
harness in two sessions: v1 to v7 on 2026-11-15 for `Level1`, and v8 on 2026-11-17 for `Level2` while keeping `Level1`. Laya's
weights never changed, and no training of any kind ran.

Each version answered one observed stuck point. "Furthest x" is the best position in that version's Laya run (the level
starts at x = -6.56 and the `Level1` exit is near x = 113.7).

| Version | What it added | The failure it answered | Furthest x with Laya |
| --- | --- | --- | ---: |
| v0 | Nothing: raw grid and 11 native inputs | Baseline. 29 of 30 choices were `attack_down` | -6.6 |
| v1 | Perception, measured physics, jump planner, first behaviours, short English facts | The model could not use the raw grid | 26.2 |
| v2 | Terrain memory, a loop guard that first speaks and only later withdraws an option | The floor left the screen and the same option was chosen 21 times | 40.7 |
| v3 | Enemy handling | The first fight took health from 5 to 1 | 13.7 (worse: the enemy stayed on the landing spot) |
| v4 | Wall jumps, levers, falling traps, waiting out enemies, danger grid, ranked menu, fade guard | The way on past x = 40 needs a wall climb | 48.5 |
| v5 | Exploration memory: dead ends, going back, re-checking after a lever | A skipped lever blocked the upper floor for good | 30.4 (stopped early) |
| v6 | Two menu rules: no leaving an untried lever, no walking onto a guarded landing | No wording made the model pick the lever | 30.4 (stopped early) |
| v7 | Fewer pointless stops | The model hopped between two platforms for 13 decisions | Reached `Level2` |
| v8 | Motion memory, moving-hazard planning, spike bouncing, looking over edges, reflexes, scene change | `Level2` saws, shooters, spikes and moving platforms | Both levels |

[`harness/CHANGELOG.md`](harness/CHANGELOG.md) records every rule with the failure that caused it.

## The abstraction

```text
honing.visible.v1 observation: grid, sprite boxes, hearts, camera transform
    |
    |  perception.py   surfaces, walls, enemies, hazards, interactables on this screen
    v
World model
    worldmap.py        surfaces and danger cells seen earlier, what was visited, what did not react
    motion.py          each moving thing: still, back and forth, or flying, with predicted positions
    |
    |  language.py     survey of what is reachable, then rules that decide what may be offered
    v
Text for the model: briefing, a few sentences on the situation, 3 to 5 options with one line each
    |
    |  Laya (frozen)   scores the options; the guard validates the response; the argmax is taken
    v
One behaviour, for example advance_right, pass_hazard_right, climb_wall_left, cross_spikes_right
    |
    |  skills.py       the behaviour's own procedure and stop conditions
    |  physics.py      frame-by-frame simulation and search of jump plans
    |  control.py      closed-loop execution, reflexes, recovery
    v
Native windows: one of 11 inputs held for N frames, each logged with the reason it was sent
```

Laya receives about 264 input tokens per decision (204 to 344 in the main run) and answers in about one second on CPU. It
chooses a behaviour about every 25 native windows, or 3.5 s of game time. Harness code picks every native input after that.
Each record in `windows.jsonl` names the decision it belongs to and why it was sent, so a window chosen by a reflex
(`reflex`, `dodge-falling`, `recover-after-hit`, `fade-guard`, `hurt-lock`, `settle`) can be told apart from one that a
behaviour planned.

## How the rules were found

The work followed a loop.

1. Run the current version with the frozen model, and with a control that needs no model (a fixed priority or the first
   menu option).
2. Read the per-decision log and find where the run got stuck or lost health.
3. Work out what the harness saw at that moment and add a generic rule. No rule refers to level coordinates.
4. Before accepting a new version, re-check what already worked. v8 was re-checked on `Level1` while it was developed on
   `Level2`.

Physics constants were measured once with scripted probes that use no model. From the code
([`physics.py`](harness/v8/physics.py)): walking and air steering at 5 units/s with no inertia, jump speed 7 units/s, gravity
9.81 units/s², a fixed physics step of 0.02 s, dash 10 units/s for about 0.15 s, and a bounce speed of 4 units/s after a downward
strike hits something. Questions about Laya were tested offline against the frozen model: for example, the wording of the
`drop_and_look_right` option was chosen by asking the model about two recorded situations, and 30 rewordings of a lever
option showed that no wording made the model prefer it, which is why v6 uses a menu rule there.

The designer's own judgment is flagged in the changelog. Three judgments are not learned from the screen: a moving hazard
is treated as round, spikes below the level of the feet are never offered for bouncing, and the menu's rank order comes
from v7.

## Why routing and planning work

Our reading: the run succeeds because of the parts below. All of them are hand-written rules or measured constants. None
is learned from data, and the frozen model plays no part in them.

| What it does | From the code |
| --- | --- |
| Simulates every candidate jump frame by frame against remembered terrain and predicted hazards, before pressing anything | [`physics.simulate`](harness/v8/physics.py#L158) |
| Searches take-off delays of 0 to 45 frames and second-jump times of 3 to 45 frames in steps of 3, preferring plans that keep the second jump, leave clearance and land soon | [`physics.plan_jump`](harness/v8/physics.py#L284) |
| Re-observes after each native window, compares the position with the plan and replans from where the knight really is | [`Ctl.jump_to`](harness/v8/control.py#L666) |
| Learns from the screen how saws, moving platforms and projectiles move, and says how far a prediction can be trusted | [`motion.Track.predict`](harness/v8/motion.py#L293), [`certain`](harness/v8/motion.py#L297) |
| Plans a hop over a rolling saw with every frame checked against the prediction | [`physics.plan_cross`](harness/v8/physics.py#L339), [`skills.pass_hazard`](harness/v8/skills.py#L1306) |
| Lands on a moving platform by aiming each window at where its middle will be | [`Ctl.land_on`](harness/v8/control.py#L769) |
| Bounces across a 21-unit spike field with downward strikes | [`skills.bounce_across`](harness/v8/skills.py#L1498) |
| Remembers surfaces and danger cells that have left the screen; the camera shows only 2.3 units below the feet | [`worldmap.WorldMap`](harness/v8/worldmap.py#L45) |
| Reflexes inside every behaviour: strike a projectile, step out of a saw's way, dodge a falling trap, recover in the air | [`Ctl.strike_plan`](harness/v8/control.py#L479), [`skills.evade_mover`](harness/v8/skills.py#L1379), [`skills.dodge_falling`](harness/v8/skills.py#L1050), [`Ctl.recover`](harness/v8/control.py#L884) |
| Offers only options that make sense now, ranks them and keeps pointless ones off the menu | [`language.build_menu`](harness/v8/language.py#L355) |

Measured in the main run ([`decision_table.md`](data/eval_laya_v8_L1L2/decision_table.md)): rules kept at least one behaviour off
the menu in 37 of 47 decisions. In 28 of 47 decisions the menu held at most one option that leads on, so the rules had
already settled the route. Our reading: this is why the routing looks strong even though the model's own preferences are weak.

## Results with and without Laya

Every number below is read from that run's own `summary.json`, `manifest.json`, `launch.json` and session logs and is
collected in [`data/runs.json`](data/runs.json). A pass means the knight reached the `Level1` exit, or `Level2` x at or beyond
124 where the run has that goal.

Runtime labels: "original" is the Linux player whose bridge source is committed here, run with screen capture on, as the
evaluation runs were. "speed-study" is the same game and bridge rebuilt for faster stepping, with a fixed number of paused
frames between windows, and run without capture; it was used during development and is not part of this pull request. Harness labels: "v8" is the code committed here; "v8, before the HUD fix" lacks one change in
`control.py` that ignores a blank health display right after a level exit; "earlier v8" is a development state.

### Laya choosing, final evaluation

| Run | Who chose | Harness | Runtime | Span | Decisions | Windows | Frames | Deaths | Min HP | Result |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `eval_laya_v8_L1L2` | Laya | v8, before the HUD fix | original | Level1 to Level2 | 47 | 1166 | 4877 | 0 | 4 | pass |
| Level1 part of `eval_laya_v8_L1L2` | | | | | 34 | 708 | 3158 | 0 | 4 | |
| Level2 part of `eval_laya_v8_L1L2` | | | | | 13 | 458 | 1719 | 0 | 5 | |
| `eval2_laya_v8_L1L2` | Laya | v8 | original | Level1 to Level2 | 47 | 1166 | 4877 | 0 | 4 | pass |
| `eval_laya_v8_L2` | Laya | v8, before the HUD fix | original | Level2 | 13 | 453 | 1713 | 0 | 5 | pass |
| `eval2_laya_v8_L2` | Laya | v8 | original | Level2 | 13 | 453 | 1713 | 0 | 5 | pass |

### Without Laya: first option and random choice

| Run | Who chose | Harness | Runtime | Span | Decisions | Windows | Frames | Deaths | Min HP | Result |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `eval_first_v8_L1L2` | Always first option | v8, before the HUD fix | original | Level1 to Level2 | 29 | 912 | 3764 | 0 | 5 | pass |
| Level1 part of `eval_first_v8_L1L2` | | | | | 16 | 448 | 2018 | 0 | 5 | |
| Level2 part of `eval_first_v8_L1L2` | | | | | 13 | 464 | 1746 | 0 | 5 | |
| `first_11_v8_L1L2` | Always first option | v8 | speed-study | Level1 to Level2 | 29 | 912 | 3764 | 0 | 5 | pass |
| `orig_first_01_v8_L2` | Always first option | v8, before the HUD fix | original | Level2 | 16 | 538 | 1910 | 0 | 5 | pass |
| `first_10_v8_L2` | Always first option | v8, before the HUD fix | speed-study | Level2 | 16 | 538 | 1910 | 0 | 5 | pass |
| `random_s1_v8_L1` | Uniform random, seed 1 | v8, before the HUD fix | speed-study | Level1 | 30 | 540 | 2567 | 0 | 4 | pass |
| `random_s1_v8_L2` | Uniform random, seed 1 | earlier v8 | speed-study | Level2 | 80 | 393 | 2290 | 0 | 4 | no: decision budget reached |
| `random_s1b_v8_L2` | Uniform random, seed 1 | v8, before the HUD fix | speed-study | Level2 | 42 | 734 | 2960 | 1 | 0 | froze (game bug) |
| `random_s2_v8_L2` | Uniform random, seed 2 | earlier v8 | speed-study | Level2 | 36 | 658 | 2664 | 0 | 5 | pass |
| `random_s3_v8_L2` | Uniform random, seed 3 | v8, before the HUD fix | speed-study | Level2 | 52 | 810 | 3354 | 0 | 2 | pass |

The random controls ran once per seed, on development states of v8. On `Level2`, seeds 2 and 3 cleared the level. Seed 1
stalled near the start in `random_s1_v8_L2` because of a motion-memory bug that was then fixed, and in the rerun
`random_s1b_v8_L2` it died once and then froze on the game bug described under [limitations](#known-limitations).

### Round 1: v7 on Level1

These runs used the earlier version v7 on `Level1` only. They are the controls of the first round, including a run in which
the menu order shown to Laya was shuffled.

| Run | Who chose | Harness | Runtime | Span | Decisions | Windows | Frames | Deaths | Min HP | Result |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `run_09_v7_laya` | Laya | v7 | original | Level1 | 34 | 727 | 3471 | 0 | 4 | pass |
| `ctrl_v7_laya_shuffled_menu` | Laya, menu order shuffled | v7 | original | Level1 | 22 | 539 | 2637 | 0 | 3 | pass |
| `ctrl_v7_random_choice` | Uniform random, seed 1 | v7 | original | Level1 | 33 | 576 | 2940 | 0 | 4 | pass |
| `dev_12_v7_first_option` | Always first option | v7 | original | Level1 | 15 | 452 | 2246 | 0 | 5 | pass |
| `check_v7_first_L1` | Always first option | v7 | speed-study | Level1 | 15 | 456 | 2278 | 0 | 5 | pass |

### Runs that froze

| Run | Who chose | Harness | Runtime | Span | Decisions | Windows | Frames | Deaths | Min HP | Result |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `laya_02_v8_L1L2` | Laya | earlier v8 | speed-study | Level1 to Level2 | 36 | 782 | 3492 | 0 | 4 | froze (game bug) |
| `laya_03_v8_L1L2` | Laya | earlier v8 | speed-study | Level1 to Level2 | 36 | 782 | 3492 | 0 | 4 | froze (game bug) |
| `laya_04_v8_L1L2` | Laya | v8 | speed-study | Level1 to Level2 | 36 | 782 | 3492 | 0 | 4 | froze (game bug) |
| `random_s1b_v8_L2` | Uniform random, seed 1 | v8, before the HUD fix | speed-study | Level2 | 42 | 734 | 2960 | 1 | 0 | froze (game bug) |
| `first_02_v8_L2` | Always first option | earlier v8 | speed-study | Level2 | 4 | 92 | 341 | 0 | 4 | froze (game bug) |

In the first four rows an enemy walked into a saw off screen. In `first_02_v8_L2` a guard in an early v8 missed a kill the
knight made on screen; that hole was fixed in the code committed here.

## What Laya contributed

Measured in the main run ([`decision_table.md`](data/eval_laya_v8_L1L2/decision_table.md)):

| Measure | Value |
| --- | ---: |
| Decisions | 47 |
| Options offered per decision | 3.64 on average (3 to 5) |
| Decisions with at most one option leading on (the menu settled the route) | 28 |
| Of those, Laya took that option | 20 |
| Of those, Laya chose to step back or wait | 8 |
| Decisions with two or more options leading on | 19 |
| Of those, Laya took the harness's first-ranked option | 10 |
| Of those, Laya took another option leading on | 9 |
| Mean probability of the chosen option | 0.426 |

Laya needed 34 decisions on `Level1` where the first-option control needed 16, and took the only hit of the two runs there.
On `Level2` both used 13 decisions. Laya's choices affected how many decisions a run took and how much health it lost. In the runs above
they did not change whether a level was cleared.

Laya's choices can hurt when the menu allows it. In an early v8 development run (`laya_01_v8_L2`) Laya chose `go_back_left`
five times on a moving platform, four of those choices took damage, and the knight died once. The response was to change the
menu and its wording, so that going back is no longer offered while a way on exists. The model stayed the same.

## Known limitations

1. No boss. This game has two levels and no boss fight, so we do not know how this harness would do against one. A boss
   adds attack patterns that change with the player's position, a health bar to wear down and phases. That is why the work
   has moved on to a more complete Hollow Knight build
   ([ForestDango/Hollow-Knight-Demo](https://github.com/ForestDango/Hollow-Knight-Demo)), where we will keep trying harness
   evolution. That work is not part of this pull request.
2. Laya contributes little. The first-option control clears both levels, and random choice cleared every level at least
   once. These runs do not show that Laya plays well.
3. Single deterministic runs. The game replays the same way for the same inputs, so a repeat mostly shows the same run again.
   Each evaluation setting ran once or twice, and there are no success rates or confidence intervals.
4. The game can freeze. A dying enemy's fade-out coroutine in the game spins forever if its sprite is already transparent
   when the bridge is paused (`PatrolController.fadeCoroutine` in the pinned game source). v8 keeps windows from ending on
   that frame for deaths it can see, but an enemy that dies off screen cannot be guarded against. Four such freezes happened
   on the speed-study runtime ([`data/freeze_check.json`](data/freeze_check.json)). In all six evaluation runs on the original
   runtime the knight killed that enemy with the sword, and none froze. The bug is not fixed.
5. Captured and uncaptured runs differ. The recording's caption box hides the top-left strip of the screen from the harness.
   The same code took a different wall kick without capture (`laya_04_v8_L1L2`) and then froze, so a result from one setup does
   not carry over to the other.
6. One-frame windows. A window of N frames advances the game N-1 frames, so a one-frame window presses a button without game
   time passing. v8 relies on this for some presses and limits itself to what a player could do in one frame. If the bridge
   changes, these timings must be measured again.
7. Designer priors and hand-written reflexes. Moving hazards are assumed round, spikes in pits are not bounced on, the menu
   order is fixed by hand, and the reflexes were written for what these two levels contain.
8. The way on is assumed to lead right. The briefing tells the model that levels lead right, and walking on is offered only
   to the right; going left appears only as stepping back, going back from a dead end or climbing a wall. In a separate local
   experiment, not part of this pull request, a port of this harness to another game
   ([mega-dean/hallowdale](https://github.com/mega-dean/hallowdale)) stalled in a room whose real exit is up and to the left.
9. Known weak spots, left as they are: two projectiles closer together than the sword cooldown cost a hit, and looking over
   an edge can land next to an enemy.
10. The visible observation is not fully repeatable. While the game waits for an input the bridge keeps rendering paused
    frames, and sprite animation still advances in them, so identical game states can show slightly different sprite boxes in
    two runs. Sprite boxes are also not collision shapes.

## Reproduce

The game, its assets, Unity and the model weights are not in this repository. You need Linux, Python 3.10 or newer, `numpy`,
`Xvfb` and, for video, `ffmpeg`.

### 1. Build the player

The evaluated player was assembled from four inputs, and [`build_linux_player.py`](build_linux_player.py) repeats that assembly:

| Input | Source |
| --- | --- |
| `--source` | The game checkout at the revision in [`pins.json`](../../pins.json) |
| `--mac-data` | `Contents/Resources/Data` of a Mac Mono build of that project made with Unity 2019.4.2f1 (see the [integration README](../../README.md)) |
| `--linux-module` | `Variations/linux64_withgfx_nondevelopment_mono` from Unity's [Linux Mono build support 2019.4.2f1](https://download.unity3d.com/download_unity/20b4642a3455/MacEditorTargetInstaller/UnitySetup-Linux-Mono-Support-for-Editor-2019.4.2f1.pkg) |
| `--mono-root` | A Mono 6.8 tree with `usr/bin/mono-sgen`, `usr/lib/mono/4.5/mcs.exe` and `etc/mono/config` |

```sh
python integrations/hollow-knight/experiments/harness-v8/build_linux_player.py \
  --source work/Hollow-Knight-Imitation --mac-data work/hollow-knight/mac-data \
  --linux-module work/unity/linux64_withgfx_nondevelopment_mono --mono-root work/mono \
  --output work/hollow-knight/linux-player
```

### 2. Start the decision server

Follow [Local CPU reproduction](../../README.md#local-cpu-reproduction) in the integration README. It starts Honing's
`DecisionServer` with `LayaDecisionModel` on CPU. The runs used `convaiinnovations/laya-typed-decisions` at revision
`1a793eb568e6718f15941d08f85432581df534e3` (weights SHA-256
`4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`) with Laya 0.3.20 from source commit
`970dc8c5f63d7b886a68409493f37d569424f933`, torch 2.8.0, transformers 4.56.2 and 4 CPU threads, as recorded in the run's
[`manifest.json`](data/eval_laya_v8_L1L2/manifest.json). `HONING_DECISION_URL` tells the runner where the server listens.

### 3. Run

```sh
export HONING_HK_GAME=work/hollow-knight/linux-player/HollowKnight
export HONING_DECISION_URL=http://127.0.0.1:8011

# The main run: Laya chooses, Level1 start to Level2 end, capture on.
python integrations/hollow-knight/experiments/harness-v8/run_agent.py --harness v8 --policy laya \
  --max-decisions 110 --max-frames 60000 --goal Level2:124 --through --out work/hollow-knight/harness-v8/laya

# The control that needs no model: always the first menu option.
python integrations/hollow-knight/experiments/harness-v8/run_agent.py --harness v8 --policy first \
  --max-decisions 70 --max-frames 40000 --goal Level2:124 --through --out work/hollow-knight/harness-v8/first
```

Each run writes its trace, a copy of the harness source, `summary.json` and, when `ffmpeg` is available, `gameplay.mp4` under
`--out`. The runner starts its own `Xvfb` and game and stops both on every exit path. A response that fails the guard stops the
run before anything is executed, and the raw bytes stay in `session/responses/`.

### 4. Check the committed evidence offline

```sh
python integrations/hollow-knight/validation/review-sample/check.py
pytest tests/test_hollow_knight_harness_v8.py
```

[`validation/review-sample`](../../validation/review-sample/README.md) holds one complete real decision: the request, the exact
response bytes and the transition it caused. The tests replay all 47 recorded responses through the response guard,
regenerate the decision table from the trace, and check the launcher's process cleanup with stub processes.

### What was re-run for this pull request

The first-option control needs no model, so it was re-run from the committed code and compared with the recorded run
`eval_first_v8_L1L2`. The full comparison is in [`data/first_option_reproduction.json`](data/first_option_reproduction.json).

| Rerun | Player | Requests and choices | Native windows | Frames | Game-state records | Result |
| --- | --- | --- | --- | ---: | --- | --- |
| `first_pr_built_runtime` | Built from the bridge source committed here | 29 of 29 identical | 912 of 912 identical | 3764 | 913 of 913 identical | pass |
| `first_original_runtime_2` | The player the evaluation ran on | 29 of 29 identical | 912 of 912 identical | 3764 | 913 of 913 identical | pass |
| `first_original_runtime` | The player the evaluation ran on | 29 of 29 identical | First 887 identical, then one window one frame shorter (911 in all) | 3768 | First 888 identical | pass |

The third rerun shows the limit of this kind of check. When it parted from the recorded run the game state was still identical,
but the visible observation of that moment was not: two overlapping sprites had different visible boxes, and the harness
walked one frame less. Visible observations are not byte-repeatable between runs (limitation 10). Only 503 to 615 of about 913
lines were byte-identical in these reruns, and in most cases the harness's decisions did not depend on the difference.

The bridge source was checked in two more ways. Given the bridge sources of the evaluated player, `build_linux_player.py` rebuilds
that player's `Assembly-CSharp.dll` byte for byte (SHA-256 `4a86f2c5bda2b8b9c76683b78c93743ff5e2b3afdf23ad2daaf32234fa8852c7`).
With the profile off, [`verify_native.py`](../../verify_native.py) passes on a player built from the source committed here (11
inputs, 132 frames, 7 rejected commands), and all 11 fixed-input steps match a player built with the unmodified bridge.

## Files

| Path | Content |
| --- | --- |
| `run_agent.py`, `harness/v8/`, `core/evalwatch.py`, `tools/freeze_check.py` | The code that produced the results, byte for byte apart from three formatter directives at the top of each file |
| `core/env.py` | Launcher. Changed from the run's version only in how it finds the player and `ffmpeg` and in its handling of failed starts |
| `core/laya.py` | Laya client and response guard, now built on the shared receiver [`integrations/_shared/decision.py`](../../../_shared/decision.py). All 47 recorded responses give the same choice and probabilities through it |
| `tools/decision_table.py`, `tools/make_cut.py` | Evaluation and video tools. They also read compressed traces and write the small demo |
| `build_linux_player.py` | Assembles the Linux player |
| `harness/CHANGELOG.md` | Every version and rule, with its reason |
| `data/runs.json` | The numbers behind the tables |
| `data/freeze_check.json` | Enemy deaths and freezes per run |
| `data/first_option_reproduction.json` | The reruns made for this pull request, compared with the recorded control |
| `data/eval_laya_v8_L1L2/` | The main run: per-decision requests, raw responses and probabilities, native windows, skill results, game-state and visible observations, the decision table, the demo cut |

The trace files keep the layout this harness writes. They do not define a common trace format for other games.

The harness files carry `# fmt: off`, `# isort: skip_file` and `# ruff: noqa` because they are a record of the code that ran.
Reformatting them would make that comparison harder.
