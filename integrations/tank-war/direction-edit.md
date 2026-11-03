# Editing NanoJev's head for Tank War, without training

This is the report behind `honing.backends.jev.direction_edit`, Honing's
`direction_edit` algorithm. It applies the weight-space editing of
["Sparks of Weight-Space Programming"](https://rikiyat.github.io/blog/weight-space-programming/)
(Takehi, Teoh, Zhang and Khattab, 2026) to the decision model that plays Tank
War: read what the model's head sees, find the direction behind one failure,
and change the weights along it. No gradients, no optimizer. The experiments
ran on 2026-11-02 on two H100 nodes; every number below comes from rounds
recorded with `scripts/match.mjs`.

## The setting

- **Model.** The pinned NanoJev release (`C-Tianyu/NanoJev`, revision in
  `integrations/ra2/pins.json`): a Qwen3-0.6B backbone and a decision head.
  Every option of a question is scored on its own text path; the backbone's
  hidden state at the path's final token goes through a LayerNorm (`norm`),
  the scalar head (`scalar`, 1,024 weights and a bias) and, for choice
  questions, a set-attention correction over the option set
  (`set_project`, `set_attention`, `set_output`, about 200,000 weights). A
  softmax over the options gives the probabilities.
- **Game.** `scripts/match.mjs` in lockstep against the planner opponent, the
  default harness (`config/controller.json`, planned steering, dodge assist
  `light`), 30 s rounds. A win scores 1, a loss 0, a timeout or double kill
  0.5. The decision server ran with `--max-len 4096`: at the default 2,048
  tokens half the rounds overflow and the server returns HTTP 500.
- **Tasks.** The suite's `val` seeds 2001-2008 and `heldout` seeds
  3001-3008, both seats, 32 rounds; and 16 further seeds 4001-4016, both
  seats, 32 rounds, played once at the end. A round is deterministic given the
  seed, the seat and the model, so repeats add nothing; only new seeds do.
- **Teacher data.** The planner teacher's rounds (`--teacher`) on the
  `train` seeds 1001-1016, both seats: 3,796 questions, half `motion` and
  half `weapon`, 442 of the weapon questions with a permitted shot.

## What the model can decide

The planner and the bridge leave the model two choices. For `motion`, the
planner forecasts all nine drive/turn plans, names the ones it admits (1.88
on average; exactly one in 44% of decisions), and the bridge executes the
most probable admitted plan. For `weapon`, the bridge fires only when the
planner has verified the shot (`Fire permitted=true`, 23% of weapon
questions), whatever the model says. The dodge assist can override the
motion. So a model can lose by hesitating on a permitted shot, or by ranking
the admitted plans badly; it cannot shoot itself on purpose or drive into a
wall.

## Diagnosis

From the released model's own rounds:

|                                         | released NanoJev                                           |
| --------------------------------------- | ---------------------------------------------------------- |
| entropy of the motion distribution      | 2.16 (the maximum over 9 options is 2.20)                  |
| most probable plan is an admitted one   | 27% (admitted sets average 1.78 of 9; chance is about 20%) |
| fires when the planner permits the shot | 33% (mean p(fire) 0.486)                                   |
| shots per round                         | 3.3                                                        |
| score on 64 rounds                      | 0.383                                                      |

The released model answers Tank War's questions almost at random; the
planner does the playing. Every round ends by a hit, so the race is who
shoots first, and the one failure a model can be blamed for is holding a
permitted shot two times out of three.

Reading the head's inputs on 1,024 teacher questions explains it. On the
weapon question the scalar head's fire-minus-hold gap is -0.016 with a
standard deviation of 0.09 logits, the set correction adds -0.005, and
p(fire) is 0.495 whether or not the shot is permitted. The difference between
the fire and hold option states is one consistent direction (its mean has
norm 0.66 against a per-row norm of 0.76, so rows agree), and the scalar head
is orthogonal to it: cosine -0.046. The model never learned that this
question has an answer. On the motion question the head already leans toward
the teacher's plan (cosine 0.43 with the teacher-minus-others direction, a
margin of +0.15 logits), too weakly to move a 9-way softmax.

## The edit

`scalar.weight += strength * d`, where `d` is the unit mean, over the
teacher's weapon questions, of `h[fire] - h[hold]`, and `h` is the head's
input after `norm`. Nothing else changes. Only `scalar.weight` matters for
such an edit: the bias adds the same constant to every option and cancels in
the softmax, and zeroing `set_output` leaves the score unchanged within noise.

Because the set correction does not depend on `scalar.weight`, the edited
logits on the recorded questions follow exactly from the recorded inputs, so
the strength can be chosen before any round is played. From `strength = 1`
on, fire is the argmax on 100% of permitted shots (the projection is at least
0.52 on every question, far above the head's 0.09 spread); larger strengths
only sharpen the probability (0.65 at 1, 0.87 at 3, 0.98 at 6) and change
the executed motion plan on 6% to 14% of motion questions as a side effect.

Two things that did not work: contrasting permitted with forbidden shots
(the linear separation is 0.08 and the direction wrecks the motion question
when scaled up; in play it is also unnecessary, since the bridge vetoes the
forbidden shots), and the motion direction (below).

## Results

Scores on rounds the edit was not chosen on. The standard error of a 64-round
score is about 0.06.

| model                                                              | val (16) | heldout (16) | seeds 4001-4016 (32) | all 64    | fires when permitted | most probable plan admitted |
| ------------------------------------------------------------------ | -------- | ------------ | -------------------- | --------- | -------------------- | --------------------------- |
| released NanoJev                                                   | 0.375    | 0.406        | 0.375                | **0.383** | 0.33                 | 0.29                        |
| fire direction, strength 1 (the protocol run below)                | 0.625    | 0.562        | 0.531                | **0.563** | 1.00                 | 0.30                        |
| fire direction, strength about 3                                   | 0.375    | 0.750        | 0.719                | **0.641** | 1.00                 | 0.32                        |
| fire and motion directions, strength 2 each                        | 0.562    | 0.625        | 0.578                | 0.586     | 0.98                 | 0.33                        |
| motion direction only, strength 2                                  | 0.562    | 0.438        | 0.438                | 0.469     | 0.51                 | 0.32                        |
| an early edit that mixed both questions and turned the trigger off | 0.125    | 0.125        | not run              | 0.125     | 0.03                 | 0.32                        |
| `set_output` zeroed                                                | 0.312    | 0.406        | not run              | 0.359     | 0.33                 | 0.27                        |

For reference, the planner teacher itself scores 0.609 on the train seeds
against the planner (it decides every 200 ms, the opponent every 300 ms).

The fire direction alone lifts the score from 0.38 to 0.56 or 0.64, the
teacher's level, about three to four standard errors. The edit that turned
the trigger off drops it to 0.125, so the score follows the firing rate and
not the noise. The motion direction raises agreement with the teacher's plan
within the admitted set from 0.82 to 0.87 and does not raise the score; the
planner's admitted set already holds the sensible plans, and the opponent is
that same planner. The two strengths of the fire direction differ by 0.08 on
these 64 rounds and by 0.13 the other way on the practice rounds below; the
data cannot order them.

### Against SFT on the same data

Honing's `NanoJevTrainer`, one epoch, the released recipe (backbone learning
rate 1e-5, heads 1e-4), seeds 1013-1016 held out, about 57 minutes on one
H100 each.

|                                             | offline validation accuracy | all 64 rounds     | fires when permitted | most probable plan admitted |
| ------------------------------------------- | --------------------------- | ----------------- | -------------------- | --------------------------- |
| released NanoJev                            | 0.37                        | 0.383             | 0.33                 | 0.29                        |
| SFT, head only (`norm`, `scalar`, `set_*`)  | 0.51                        | **0.016**         | 0.00                 | 0.36                        |
| SFT, every weight                           | 0.93                        | **0.531**         | 1.00                 | 1.00                        |
| fire direction, 1,024 weights, no gradients | not applicable              | **0.563 / 0.641** | 1.00                 | 0.30                        |

Head-only SFT learns never to fire: 77% of the teacher's weapon rows are
`hold`, and a cross-entropy fit with only the head to move settles on a
constant. The same tensors, read instead of fitted, win rounds. Full SFT
reproduces the teacher almost exactly offline and scores below the one
direction in play: being more like the teacher is not the same as winning
against it.

## The protocol run

The exploratory runs above chose edits with the game in the loop and without
a budget, which is not how the paper's experiments were run. The paper's
editor works inside a harness with a Python kernel over the weights, a budget
of forward passes, practice data only, and a log of every turn
([lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs),
Experiment 8). The same was rebuilt for Tank War and run once with a budget
of 25 passes: a `play` pass (up to 32 practice rounds, original, edited or
both, paired) or a `read` pass (the head's inputs on up to 1,024 logged
questions); practice seeds 1001-1064 only; `save_edit` refusing the
embedding, the final norm and whole-tensor rescaling; no training and no
model code in the kernel.

| turn  | pass | what                                          | result                                                                            |
| ----- | ---- | --------------------------------------------- | --------------------------------------------------------------------------------- |
| 1     |      | the practice data and the head's sizes        | admitted sets 1.88, permitted 23%, teacher 0.61                                   |
| 2     | 1    | the original plays seeds 1017-1032            | 0.375, fires on 28% of permitted shots                                            |
| 3     | 2    | read 1,024 teacher questions                  | the head's inputs                                                                 |
| 4     |      | analysis                                      | the gap -0.016 +- 0.09, cosine -0.046 with the fire direction                     |
| 5     |      | the dose predicted from the recorded inputs   | strength 1 flips every permitted shot; strength 3 saved                           |
| 6     | 3    | paired play, seeds 1033-1048                  | 0.250 to 0.531, gain +0.28 +- 0.14                                                |
| 7-8   | 4    | strength 1 on the same rounds                 | 0.656; +0.41 +- 0.10 against the original, +0.13 +- 0.10 against strength 3       |
| 9-10  | 5    | strength 1 plus the motion direction          | 0.625; -0.03 +- 0.07 against strength 1, withdrawn                                |
| 11-12 | 6    | paired confirmation on unused seeds 1049-1064 | 0.156 to 0.625, gain **+0.47 +- 0.12**, 18 rounds lost-to-won and 3 the other way |
| 13    |      | final answer                                  | `scalar.weight += 1.0 * d` submitted                                              |

The session was closed by the experimenters after 6 of the 25 passes, for
lack of time; the submitted edit is the strength 1 row in the results table.
The run's turns, code, outputs and notes are kept with the other run
artifacts on the cluster.

## In Honing

`NanoJevDirectionEditTrainer` is a `Trainer` for the `direction_edit`
algorithm on `typed_decisions` batches, so a campaign schedules it as a train
move and gates it like any other candidate. Its settings choose the question
(`weapon`), the target (an option such as `fire`, or `label` for the
direction toward the batch's own labels) and the strength. The checkpoint has
NanoJev's layout and a `checkpoint.json` manifest, which the decision server
loads with `--checkpoint-manifest` and
`honing.backends.jev.nanojev_serving.nanojev_endpoints` starts on demand, one
server per checkpoint. `configs/e2e/tank_direction_edit.json` runs the whole
thing on the tank campaign: the pinned release, teacher rounds on the train
split, the edit, and a gate on all 16 validation tasks with the candidate and
the incumbent each on their own GPU.

```bash
export HONING_NANOJEV_SOURCE=/path/to/NanoJev   # at the revision in integrations/ra2/pins.json
export HONING_NANOJEV_GPUS=0,1 HONING_NANOJEV_EDIT_GPU=2 HF_HOME=/path/with/the/pinned/release
python scripts/e2e/tank_campaign.py --config configs/e2e/tank_direction_edit.json
```

## Limits

- The gain is a repaired failure, not a new ability. The paper's editor could
  not improve a model that did not loop; here the motion direction did not
  improve a model whose admitted plans the planner already chose well.
- The harness bounds what any edit can do. To make weight editing hard, and
  to separate it from SFT by more than 0.03 to 0.11, play raw controls
  (`controllerMode: raw`), where the model answers for its own safety.
- 64 rounds give a standard error of 0.06. Ordering strengths, or editing
  against SFT with more than one epoch, needs more seeds.
