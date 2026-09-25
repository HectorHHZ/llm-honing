# RA2 automatic optimization league (in development)

The target is a learned champion that reliably beats pinned Supalosa, transfers
the workflow to Laya, NanoJev and Jev-Omni, and improves through repeated
cross-model self-play. `protocol.json` records the completion requirements.
**No checkpoint currently satisfies the full objective.**

## Implemented workflow

- A normal-view, training-only macro teacher and staged-label collector. Teacher
  demonstrations are never counted as model evaluations or deployed fallbacks.
- Offline DAgger annotations of a learner's training episodes, using only its
  recorded normal observations, legal candidates and actual action history.
  The annotation process cannot influence the recorded match. Development and
  sealed-test maps/seeds are rejected as training inputs.
- Laya cross-entropy training, optionally initialized from a verified trained
  parent. Source snapshots, data hashes, parent lineage and weight hashes are
  recorded. Token packing matches inference and retains complete options.
  Duplicate training inputs and validation overlaps are removed; conflicting
  labels fail the run. Validation also reports root waiting, root actions and
  parameter stages separately; frequent waits can conceal action-selection
  errors in overall accuracy. These diagnostics do not change the training loss
  or checkpoint-selection rule, and are not gameplay wins.
  The marker batching/model APIs follow the upstream
  Apache-2.0 [fine-tuning example](https://github.com/NandhaKishorM/laya/blob/970dc8c5f63d7b886a68409493f37d569424f933/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).
- A generation loop: train, evaluate both seats on two development maps, audit,
  collect new learner training episodes, annotate, aggregate and train again.
  No teacher participates in the learned policy's actions.
- Immutable job plans, Linux process-start identity checks, dependency ordering,
  GPU occupancy checks, recorded assignment within an explicit allowed GPU pool,
  per-GPU worker locks, cross-controller dispatch leases handed to workers
  across process startup, completed-job resume and failure
  records. A vanished job requires inspection. Partial optimizer-state resume
  is not implemented; completed checkpoints/jobs can be resumed.
- Independent action/context audits and replay hash verification. The evidence
  builder checks checkpoint/source artifacts, split membership, normal vision,
  engine/baseline versions and cadence. Decision logs are streamed because full
  episodes can exceed Node's single-string size limit. Verified completed logs
  are gzip-compressed without changing their decoded hashes; audits and DAgger
  readers accept either representation. Compression retains the original if
  an existing archive differs.
- Final-gate aggregation rejects incomplete matrices, teacher controllers,
  mismatched protocols, unverified matches, truncations and mixed checkpoints.
  The development loop never automatically consumes the sealed final test.
  A separate final driver first re-audits the full 24-case development matrix,
  requires its win-rate/map/seat floors, then reserves all held-out cases in a
  durable project ledger before launching any match. Only an identical
  checkpoint, source fingerprint, protocol and output directory may resume.
  A changed protocol name or threshold cannot recycle already consumed cases.

Both players use normal observations and a three-tick decision cadence. The
learned controller selects one command per decision; the original baseline can
submit up to sixteen. This is equal vision/cadence, not equal command throughput.

## Run and resume

First configure the external dependencies from the example README. Replay
verification requires the separately supplied protocol-2 Python SDK/runtime;
set `RA2_REPLAY_PYTHON` to that environment's Python. Assets, engine binaries,
weights, private paths and experiment outputs are not included in Git.

```bash
# Collect separate training episodes, including a validation episode.
node league/collect.mjs --map=mp03t4.map --seed=104 --seat=red --out=runs/demo-red
node league/collect.mjs --map=mp06t2.map --seed=105 --seat=blue --out=runs/demo-validation
node league/collect.mjs --map=mp06t2.map --seed=106 --seat=red --out=runs/demo-other-map

# Produce an initial checkpoint.
"$LAYA_PYTHON" league/train_laya.py \
  --matches runs/demo-red runs/demo-validation runs/demo-other-map \
  --validation-seed=105 --gpu=1 --out=runs/bootstrap

# Run three development learning generations. Repeat this exact command to
# resume completed jobs after inspecting any failed/vanished process.
python league/optimize.py --root runs/learning-cycle \
  --initial-checkpoint runs/bootstrap/checkpoint.json \
  --data runs/demo-red runs/demo-validation runs/demo-other-map \
  --validation-seed=105 --gpus 1 5 --generations 3

# Evaluate a selected trained checkpoint with no teacher/fallback.
LAYA_CHECKPOINT_MANIFEST="$PWD/runs/bootstrap/checkpoint.json" \
  npm run match -- --model=typed-decisions --gpu=1 --seed=1000 --out=runs/eval
python league/verify_artifacts.py --match runs/eval --split dev
```

A generic campaign root contains `plan.json` with `jobs`, each specifying `id`,
`command` as an argv list, `cwd`, and expected `result`. Optional `gpu` and
`dependsOn` fields constrain dispatch. Execute with
`python league/campaign.py run --root /absolute/campaign --concurrency 3`.
Add `--gpu-pool 1 5` to let a pending GPU job use either available device.
The requested device stays in the immutable plan; `status/*.execution.json`
records the assigned device and actual command. Running jobs are never moved.
Without a pool, each job waits for its requested GPU.
Process completion plus a result file is **not** a win or promotion.

## Explicit final evaluation

Only after a candidate passes all development cases, run:

```bash
python league/final_gate.py --root runs/final-candidate \
  --checkpoint runs/selected/checkpoint.json --gpus 1 5 \
  --development-matches runs/dev-matrix/*
```

The default protocol requires 24 unique development games (two maps, six seeds,
both seats), all from the same checkpoint and policy source. The final driver
recomputes audits/replay evidence before reservation. Its ledger is
`.cache/league/sealed-consumption.json`; preserve that file across attempts.
Errors also consume the reserved cases. Exact resumes keep the immutable plan;
using a new output directory is a new attempt and is rejected. This protects
experiment discipline, not against an operator deleting or replacing the ledger.
Final verification jobs run before dispatching additional ready matches so
completed large logs are compressed promptly. The driver currently supports
Laya; passing this final gate alone does not complete the cross-model objective.

## Evidence and limitations

Server experiment artifacts are under ignored `runs/`. The first pilot trained
on three demonstrations (1,307 staged labels). Its internal validation contained
172 exact overlaps among 451 unique inputs, so the reported high label accuracy
is not evidence of generalization. Its three actual learned-policy development
games all reached 18,000 ticks without winning and produced no military units.
It repeatedly expanded refineries. Subsequent training uses deduplication,
transient deployment/construction states and learner-state corrections.

The operational final gate is at least 90% wins in the complete 96-case matrix,
with per-map/per-seat floors and a Wilson lower-bound check. Repeated seeds can
be correlated, so the interval is descriptive; trajectory diversity must also
be reported. Scores cannot convert ties, caps or failures into wins.

Remaining before full delivery:

1. Establish stable Laya wins and test them on fresh held-out cases; improve
   combat/search labels as supported by development evidence.
2. Exercise the final driver on a qualifying learned checkpoint, complete
   cross-model promotion/reporting and partial-optimizer interruption recovery.
3. Integrate and train NanoJev and Jev-Omni after establishing Laya. Source/model
   pins in `model-sources.json` are discovery records, not completed integration.
   [Experimental native choice adapters](../integration/models/README.md) cover
   lossless packing and answer binding only; game execution and training are
   still pending.
4. Complete at least two actual learning generations from cross-model self-play,
   a full cross-play matrix, baseline regression checks and final champion gate.
5. Deliver the verified champion, full lineage and one-command reproduction.

Evaluation evidence must record the configured `maxTicks`, and its final tick
must fit the split's budget. Longer-horizon diagnostics are separate results;
they cannot satisfy a shorter development gate. Historical matches without a
recorded limit remain historical evidence and must be rerun for promotion.

The final report also groups identical recorded action traces by map and seat.
Different simulation hashes can reflect different RNG state despite identical
commands; neither seed counts nor distinct command traces prove independent
scenarios. The Wilson threshold is a descriptive operational check only.

### Separate Laya qualification and final champion evidence

Before any held-out games were launched, protocol v2 reserved an additional
Laya qualification stage: 3 different maps, 6 seeds and both seats (36 games,
18,000 ticks). The original final champion set remains 4 maps, 12 seeds and
both seats (96 games, 36,000 ticks). Both are disjoint from training/development
and from each other. This prevents Laya qualification from consuming the final
champion's cases before cross-model learning. No win thresholds were relaxed.

Use `final_gate.py --stage layaQualification` after the complete development
matrix passes. `--stage sealedTest` remains the default for the final champion.
Each stage is reserved in the same consumption ledger; changing weights or
sources cannot reuse cases already consumed by that stage. A failed held-out
stage remains consumed and must not silently become a new development set.
The driver currently supports Laya checkpoints; other model adapters and
cross-model training remain required work.

GPU dispatch also checks model-worker locks during CPU-side preparation, when
`nvidia-smi` can still report no process. A separate shared lease is acquired
before dispatch and handed to the campaign worker through an inherited file
descriptor; it lasts for the whole job. Already-running older controllers gain
the lease check when their next worker starts. This coordinates this user's
campaigns, not unrelated cluster users or an external scheduler reservation.

Development loops archive the exact protocol bytes under their run root. Resume
retains that original hash and permits unrelated held-out planning changes only
when the installed training/development settings and gameplay rules still match.
For a legacy run with no snapshot, recovery requires the exact file whose hash
was recorded in `config.json`; never rewrite the old configuration to bypass it.

Protocol v3 broadens the training map pool with six public stock/tournament maps
before any held-out evaluation is consumed. Development, qualification and final
test maps, seeds, budgets and thresholds stay fixed. Existing experiments use
their frozen protocol; new curriculum data must pass the updated training split
checks. Demonstration wins remain training evidence and never count as model
wins or qualification results.
