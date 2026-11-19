# Harness versions

One entry per version: what changed, why, which blocker it addresses. Newest at the bottom. Only v8 is part of this
repository; the earlier versions are history. Run names (`run_09`, `dev_12`, `laya_02`, ...) and `offline/` or `probes/`
names refer to the authors' local experiment records. Numbers for the runs the README relies on are in `../data/runs.json`.

- **v1** (2026-11-15): first rule-based harness. Perception of the visible grid/objects into surfaces,
  hazards, enemies; exact physics model from a scripted probe; jump planner (single/double jump with air
  steering, closed-loop replanning); L3 skills `advance_right`, `retreat_left`, `wait`,
  `fight_nearest_enemy`, `jump_over_enemy`, `climb_up`, `drop_down`, `jump_in_anyway`; state-dependent menu
  with short English facts instead of the raw grid. Blocker addressed: v0 never left the start
  (29x `attack_down`), because the model had to read a 48x27 grid and pick single native inputs.
- **v2** (2026-11-15): (a) **memory component** `worldmap.py`: every platform / wall / hazard sprite that was
  on screen is remembered as a world rectangle (union of partial views, forgotten when its place is on
  screen again without it; cleared on death); built only from past *visible* observations; facts and
  logs mark landing zones that are not on the current screen as "remembered". Blocker: in run_01 the
  knight stood on a small platform and the floor it had come from was below the screen edge (the camera
  shows only 2.3 units below the feet), so `advance_right` found nothing to land on.
  (b) **loop guard, language first**: a behaviour that changed nothing in the same state is reported in
  language ("The last choice changed nothing. Still open here: ..."; the futile option moves to the end
  with "Tried Nx here, nothing happened."); withdrawn only after 2 ignored notes; mechanism logged per
  decision (`loop_guard`). Blocker: 21 identical `advance_right` choices in run_01.
  (c) option texts capped at 150 characters: the Laya service rejects options longer than 48 tokens
  (HTTP 500 "exceeds lossless option/instruction budget"). (d) "Last behaviour: <name>" replaced by
  "Last result: ..." without the option name, because naming an option in the situation text raises its
  score (offline/try02..04). (e) `advance_right` is offered only when it can act.
- **v3** (2026-11-15): enemy handling. (a) `jump_in_anyway` removed from the menu: in run_02 / dev_02 it cost
  2 health every time (the knight lands on the enemy at the platform edge). The menu now offers the lure
  instead (`retreat_left`: out of the enemy's sight range so that it wanders off; fact sentence says so).
  (b) air recovery never chooses a landing spot within 1.6 units of a visible enemy. (c) `fight`: when the
  enemy is on top of the knight and an edge or wall is behind, leap over it to the open side instead of
  trading hits; strike cadence 11 frames. (d) loop-guard memory counts futile attempts at the same state
  over the whole life, not only the latest consecutive run (run_02 looped edge -> step back -> edge).
  Blocker: dev_02 (fixed fight preference) went from 5 to 1 health in the first fight.
- **v4** (2026-11-15): the rest of Level1. All rules are generic (nothing is tied to level coordinates).
  (a) **wall mechanics**: cling / slide / wall kick measured with `probes/probe_wall.py`, added to the
  physics model (matches the probe frame by frame) and to the planner (`plan_to_wall`,
  `plan_from_wall_to_land`); skill `climb_wall_left/right` (named `explore_ledge_*` when it leads to a
  ledge not visited yet): cling, kick upward, skip hazards on the wall with the air jump, step onto the
  first higher ledge within reach. Blocker: every earlier version dead-ended at x~40 (wall-jump shaft).
  (b) **memory additions**: surfaces the knight stood on (`visited`), interactables that did not react
  (`inert`), the look of platforms that vanished after being stood on (`collapsing`), and a **danger
  grid** (world cells of 0.5 units marked from the hazard class of the screen grid, cleared when the
  place is on screen without danger) instead of hazard boxes. Blockers: a box around the long sloped
  spike strip swallows the platforms above it; "last seen box" forgot the spike pit when only a sliver
  of it was on screen (run_04: the knight dropped into the pit).
  (c) `hit_switch` (strike an interactable, walk away the moment it reacts, mark it inert after two
  strikes without reaction), falling-hazard dodge reflex. Blocker: the obstacle on the upper floor.
  (d) `outwait_enemy` (stay out of the enemy's sight range until it has left the landing spot, then
  jump) and `leap_past_enemy`. Blocker: run_03 looped retreat/advance in under a second per cycle, the
  enemy never left. (e) `advance_right` does not stop on small platforms (human experience: do not
  linger there), verifies that walking really moves, and on a platform of the collapsing kind jumps past
  a guard instead of waiting. Blocker: the five collapsing platforms.
  (f) **menu order = the harness's rule-based ranking** (danger, lever, unvisited ledge, right, unknown
  wall, back off), logged as `harness_order`; landing spots must be within the screen width (cannot see
  who waits further away). Reason: offline/try05 — the frozen model leans strongly to the first option.
  (g) **fade guard** (plumbing, not a tactic): one uninterrupted window over the fade-out of a killed
  enemy / fallen trap, because the game's fade coroutine spins forever if the bridge pauses on the frame
  after the sprite's alpha reached 0 (dev_07 froze 57 frames after the second enemy died).
  Level-specific knowledge used: none in the rules. Constants taken from the game source rather than
  learned on screen: enemy sight range 6.3 (prefab detectDistance = 6), 3 s / 1 s fade delays (fade
  guard), sword reach and cooldown.
- **v5** (2026-11-15): exploration memory, for the case that the model makes a strategically wrong choice.
  (a) dead ends: the surface where `advance_right` ended "blocked" is remembered; there `advance_right`
  is demoted and described as leading to the dead end seen before. (b) `go_back_left`: go back the way
  the knight came (deep drops allowed) and stop where something not explored yet comes within reach (a
  ledge not visited, a wall never climbed). (c) after a lever was thrown the old dead ends become places
  to re-check and the options leading back up/right rank above going back. (d) wall climbing crosses
  over to a wall on the other side when it cannot get higher on the current one (zigzag); walls that
  gave no hold are not offered again. (e) an unvisited higher ledge is offered as `explore_ledge_up`.
  Blocker: run_04 / run_06 (model choosing): after the first fight the model took `advance_right`
  (p=0.35, third in the menu) and skipped the lever ledge; on the upper floor the obstacle then blocked it
  for good (x=48.5) and it only oscillated advance/retreat. It also picked the spiked right wall of the
  shaft, where the one-wall climber failed.
- **v6** (2026-11-15): two menu rules from human experience, plus cycle detection.
  (a) while a lever that has not been tried is within reach on the knight's platform (no enemy on it),
  leaving (`advance_right`, `go_back_left`) is not on the menu; (b) when an enemy guards the landing of
  the next platform, `advance_right` is not on the menu (it cannot get across; `outwait_enemy` /
  `leap_past_enemy` are). (c) a wall climbed to the end without finding a ledge is not offered again.
  (d) loop guard also counts repeats of the same option in the same state while nothing new was learned
  in between (progress token), so two-step cycles get the language note and then the withdrawal.
  Blocker: run_07 (v5, model choosing): standing 4 steps from the lever the model chose `advance_right`
  (p=0.41 vs `hit_switch` 0.22) and no wording changed that offline (try07..09: 30 variants, `hit_switch`
  never won); afterwards it cycled `advance_right` (0.4 steps to the edge, enemy below) and a fruitless
  wall climb.
- **v7** (2026-11-15): fewer pointless decision points. (a) `advance_right` no longer returns after every
  landing on a wide floor: it keeps going unless there is something to decide there (a lever on the
  platform, something not explored yet within reach) or one of its stop conditions fires (enemy ahead,
  damage, dead end, time cap). (b) going back up to a ledge already visited (`climb_up`,
  `climb_wall_*` "visited before") is on the menu only when nothing leads on or a thrown lever makes old
  dead ends worth re-checking. Blocker: run_08 (v6, model choosing) threw the lever and killed the first
  enemy, then spent 13 decisions hopping between the small platform and the floor below it: every
  `advance_right` ended right after the jump down, and with 5-7 options at p~0.3 the model kept picking
  a way back up.
  Result of v7: `run_09_v7_laya` (frozen model choosing) reaches Level2 — 34 decisions, 0 deaths, min HP 4.
  Controls on the same v7: first option 15 decisions (`dev_12`), uniform random 33 (`ctrl_v7_random_choice`),
  model with shuffled menu order 22 (`ctrl_v7_laya_shuffled_menu`); all reach Level2 without a death.
  Decision logs of v7 runs record per decision the offered options in harness order, the withheld
  options with the rule that withheld them (`withheld`), and the loop-guard mechanism.
  Known and left as is: when the lever rule withholds `advance_right`, the "nothing leads on, so going
  back up to a visited ledge is allowed" rule fires, which gave the model about ten decisions of hopping
  between the small platform and the floor in run_09; `advance_right` is still offered when an enemy
  overlaps the knight from the left (it then stops at once).
- **v8** (2026-11-17, round 2: Level2, keeping Level1; v7 is left untouched as the reference). A copy of v7
  plus the generic abilities Level2 needs. Everything still reads only the visible observation.
  (a) **motion memory** `motion.py`: the same object id drawn at another place a moment later gives a velocity;
  things are told apart as still / back-and-forth along x (saw, moving platform: constant speed, the
  turning points are learned by intersecting the straight runs before and after a turn) / straight flyers
  (projectiles, which advance with the 0.02 s physics steps). `predict(t)` and `certain(t)`: a
  back-and-forth mover is only certain to go as far as it was already seen to go until its turning point
  has been seen. Offline check on recorded runs: prediction error <= 0.1 units over 0.5-3 s.
  Blocker: v7 treated the rolling saw as a wall ("blocked by hazard") and died waiting in its path.
  (b) **planner** (`physics.py`): moving hazards are kept clear of at the time the knight is there, moving
  platforms are solids where they will be, a landing zone on a moving platform moves with it; head bump
  under a low platform; dash. `plan_cross`: hop over a rolling hazard on the knight's own floor (walk in,
  jump, dash at the top, land behind it, keep walking), every frame checked against the prediction.
  Shape prior, flagged as designer knowledge: a moving hazard is taken to be round (a blade rolling along
  a floor shows only its upper part; the observation gives only a box and a 20 px grid). With a box the
  hop under the low platform of Level2 would not be feasible at all.
  (c) **skills**: `pass_hazard_right/left` (wait outside the stretch the blade covers, watch where it
  turns, hop over it when it comes, leave its stretch); `cross_spikes_right` (spikes on the floor too wide
  to jump: jump over their edge, then strike down each time the knight comes down toward them — the
  downward strike bounces off spikes — until something to land on is in reach; a chain that catches the
  knight on the way is slid off); `drop_and_look_right` (an edge with nothing known beyond: step off and
  look, both jumps kept to come back); jumps onto / from moving platforms wait for the moment
  (`plan.wait`) and first watch a platform whose turning points have not been seen.
  (d) **reflexes** inside every behaviour: a projectile that is going to hit and is within sword reach is
  struck away - standing, and since first_06 also in the air during ordinary jumps (a strike that hits
  sets the knight's velocity, measured: upward strike -> (0, -1), forward -> (-/+1, 0); the flight is planned
  again from there; when a projectile's line cannot be avoided the jump keeps its line and relies on the
  strike). Not during the saw hop and the spike bounce: there a strike would spoil the manoeuvre over
  something deadly, and a lost hit point (which also makes the knight untouchable for 1.5 s) is the lesser
  evil. A saw rolling at the knight -> step out of its way (or hop it when cornered); knocked into spikes
  by a hit -> use the 1.5 s of being untouchable to jump out; a decision is never handed over while the
  knight is still in the air.
  (e) **menu / language**: sentences for a rolling saw, a field of spikes, a moving platform, a shooter;
  `advance_right` is withheld right in front of a saw (`pass_hazard_right` is offered), `cross_spikes_right`
  is offered when spikes block the floor and nothing is in jump reach, a ledge with a saw rolling along it
  is not offered as "explore" (kept low as `climb_up`), wall climbing on something that hangs over spikes
  with no known ledge is withheld. The briefing text and the ranking of the v7 options are unchanged.
  (f) **fade guard reworked** (plumbing around the game freeze P37, not a tactic): instead of one 62-frame
  wait after a kill, the guard only makes sure that no window ends on the frame after the fade
  (measured twice: exactly 57 simulation frames after the start of the killing strike), and it also covers
  enemies seen to die on screen without a strike (walked into a saw / spikes). Deaths off screen cannot be
  guarded against by any harness; the first-look Laya run froze that way.
  (g) **zero-time windows**: v8 sends 1-render-frame windows to press one button without game time passing
  (sword strike while standing / steering, jump without steering, dash while steering). This relies on the
  bridge quirk "an N-frame window advances N-1 frames" (Honing P37 item 2): if the interface is fixed,
  these presses become 1-frame windows and the bounce / hop timing has to be re-measured. The harness
  itself enforces what a player could do in one frame: never the same press twice without game time in
  between, and the sword's 0.35 s cooldown is kept by the harness (the game enforces it only through a
  timer on game time; probe_05_zero_cooldown: three zero-time strikes took an enemy from 3 to 0 at one
  instant — v8 never does that).
  (h) scene change (level exit) is handled inside the harness (memory reset, new decision), so one run can
  go Level1 -> Level2; `run_agent.py` got `--goal SCENE:X` (evaluation-only end, Level2 has no exit) and
  `--through` (additive; v7 runs behave as before).
  (i) **motion memory rewritten to follow real edges** (after first_03): a thing is followed by its left
  and top edge where the screen, the HUD, the caption box or a fading sprite does not cut them (cells that
  are purely "unknown" next to a box mark such a cut); an edge that moves exactly as the knight does is a
  screen artefact, not motion; a moving platform is recognised from partial views (the one ahead is
  usually cut by the screen edge); a hazard need not be a rectangle, so the top edge of a hazard cut
  left/right (the sliver of a disc) is not used as a measurement. Blocker: a screen-cut thing taken for a
  rolling hazard made `advance_right` return "a moving hazard rolls ..." 40 times in a row at one spot
  (first_03), and a re-entering saw got a false upward speed (probe_01 replay).
  (j) **what cannot be seen is looked at, not waited for** (after first_04: 3 decisions spent "watching" a
  platform that is below the picture): the camera shows only 2.3 units below the feet. A remembered
  moving platform whose turning points are unknown and which is above/below the picture is not a jump
  to make from memory (`UNSEEN`, sentence "A moving platform seen earlier lies beyond it, lower down and
  out of sight from here"); `drop_and_look_right` now also works from a moving platform and lands on a
  moving platform that comes into view: `Ctl.land_on` is a closed-loop landing (stay beside its edge while
  below its top, rise with an air jump, then aim each window at where its middle will be when the feet
  come down; air steering 5 units/s beats the platform's 2 units/s, so its turning points need not be
  known). Jump candidates on a moving platform are no longer clipped to the screen width.
  (k) **dead-end memory keeps the place**: "going right ended here" is stored with the x where it ended;
  standing beyond that spot on the same long floor (after the spike field) is no dead end (first_05 chose
  `retreat_left` first there).
  (l) loop guard: a behaviour that did nothing at all (0 windows) on the same surface and spot counts as
  a repeat even when other details of the situation text flicker (a projectile in the air); `advance`
  hands back "a moving hazard rolls ..." only when the menu's own survey sees that hazard too; small
  things coming straight down are projectiles (strike reflex), not falling traps.
  (m) **after the first Laya runs** (laya_01 / laya_02): (1) an edge that has not been looked below yet is
  not a dead end: no "something was missed further back" sentence there and `go_back_left` is withheld
  while a way on (advance / pass the saw / cross the spikes / drop and look) is on the menu; an edge that
  was looked below without result is remembered (`Memory.peeked`) and then is one. Laya had chosen
  `go_back_left` three times on the second moving platform and died on the way back. (2) A moving
  platform that is off the picture is kept at the place the motion memory predicts for now (it was left
  where last seen while its future shift was counted from now: a jump aimed 1-2 units off, the death in
  laya_01). (3) The loop guard's "same place" on a moving platform is the place on the platform, not the
  world position (laya_02 went retreat_left / advance_right six times there unnoticed). (4) Option text of
  `drop_and_look_right` chosen by asking the model offline (try01, 14 requests): "Go on right: drop off
  the edge onto what lies below; the two jumps bring you back if nothing is there." is ranked first in
  both test situations (0.48 / 0.37); "Step off the right edge to see what is below ..." lost to
  `retreat_left`. (5) Revisiting an old ledge is withheld while any of the new ways on is offered.
  (n) **after the random-choice control** (random_s1): a rolling hazard that is not drawn in the picture
  now is not in the picture - its "could be anywhere by now" box is cut at the screen edge it lies beyond
  and may come in from there only at its speed (a saw glimpsed once at the screen edge had grown into a
  box over the whole level a minute later and `advance_right` kept "stepping out of its way" at the level
  start); the top of a partly visible hazard is never taken lower than the part that shows; the strike
  reflex counts in that a moving platform carries the knight.
  (o) the HUD of a newly loaded scene can show 0 hearts in the one picture after the level exit: a "0" seen
  within a second of the whole picture jumping is not a death (random_s1_v8_L1 reported the level exit as
  "died and restarted"). This is the only change made after the first three evaluation runs on the
  original runtime (eval_laya_v8_L1L2, eval_laya_v8_L2, eval_first_v8_L1L2: their harness_snapshot differs
  from the final code in control.py only); the repeats eval2_* use the final code.
  Results of the final version (original runtime, capture on, Laya choosing): one continuous run Level1
  start -> Level2 end, 47 decisions (34 + 13), 0 deaths, min HP 4 (eval_laya_v8_L1L2); Level2 alone 13
  decisions, 0 deaths, HP 5 (eval_laya_v8_L2, identical repeat eval2_laya_v8_L2). Controls: first option
  clears both levels in one run (29 decisions); random choice clears Level2 with 2 of 3 seeds and Level1
  with 1 of 1. Not the same on the fast runtime without capture: there the same code's continuous Laya run
  takes one different wall kick in Level1 (frame 2125; the captured picture has the caption box over its top
  strip, so the harness sees a different picture) and then freezes at the start of Level2 on the game bug
  P37 (an enemy dies off screen; laya_04_v8_L1L2, also laya_02 / laya_03 with earlier code). Within Level2
  the captured and uncaptured runs I compared are identical. See
  the README of this experiment.
  Level-specific knowledge in the rules: no coordinates anywhere. **Designer judgements that are not
  learned from the screen** (flagged): (1) a moving hazard is taken to be round, see (b); (2) spikes whose
  tips are below the level of the feet (the bottom of a pit) are not offered for bouncing - a pit is
  crossed on what leads over it; only spikes standing on the floor are (`hazard_field`). In Level2 this
  separates the spike field (bounce) from the platform pit (platforms); the game treats both alike
  (observed: a touch of a saw, of the spike field and of the pit spikes each took the knight from full
  health to 0 - probe_02_cross, probe_03_pogo, dev_03_pit; the source has `Trap/Deadly.cs` for this). (3) The order of the menu (ranks) as in v7.
  Constants from measurement or the game source rather than learned on screen: pogo speed 4 and strike
  reach 1.44, dash 10 units/s for 0.15 s, head-stop skin, 57-frame fade, 1.5 s untouchable after a hit,
  sword cooldown 0.35 s, strike recoil (see d), platform/saw speed is measured on screen (2 units/s).
