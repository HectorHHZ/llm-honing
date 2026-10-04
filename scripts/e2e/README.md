# End-to-end runs

Every script here runs on a CPU with no model or GPU, and CI's `e2e` job runs each on every pull request. The tank commands also need Node 22 and the `gepa` extra.

| Command | What it runs |
| --- | --- |
| `python scripts/e2e/honing_pipeline.py --config configs/e2e/all_planes.json` | one campaign through every plane: train, prune with recovery, quantize (rejected, then accepted after quantization-aware training), harness evolution, GRPO on rollouts, and a rollback |
| `python scripts/e2e/check_runner.py --config configs/e2e/all_planes.json` | the checks a harness runner must pass before it joins a campaign |
| `python scripts/e2e/check_trainer.py --config configs/e2e/all_planes.json` | the checks a trainer must pass before it joins a campaign |
| `python scripts/e2e/ra2_campaign.py --config configs/e2e/ra2_dry_run.json` | the RA2 runner and grader in a campaign, with stand-in match processes |
| `python scripts/e2e/tank_campaign.py --config configs/e2e/tank_dry_run.json` | a tank campaign on real rounds (the Node match process against the planner opponent, Honing's `DecisionServer`) with a stand-in decision model: a GEPA harness move, sft on planner-teacher demonstrations, and grpo on sampled rollouts |
| `python scripts/e2e/check_runner.py --config configs/e2e/tank_runner_check.json` | the harness runner checks on the tank runner |
| `python scripts/e2e/check_campaign.py <work-dir> [--expect MOVE=RESULT ...]` | what an RA2 or tank campaign run wrote: a result for every step, the release chain, and every gated episode's run directory |
| `scripts/e2e/serving.sh` | a decision server process and the RA2 match's Node client stack; `DECISION_SERVER=<url>` checks a running server instead, such as a real model on a GPU |
| `node scripts/e2e/tank-war.mjs` | a decision server, Tank War's HTTP bridge, and both decision interfaces; `HONING_URL=<url>` plays against a running server, and `TANK_SECONDS` sets how long each interface plays |

One script is the exception: `gepa_live.py` calls real model APIs, needs a key, and costs money, so CI runs it only against a local stub (below).

## The all-planes pipeline

`configs/e2e/all_planes.json` names one component per slot, by import path (`{"factory": "package.module:function", "args": {...}}`). A real implementation replaces a fake by changing its entry.

| Plane | Slots | Default |
| --- | --- | --- |
| Model | `model_source`, `endpoints` | the walking skeleton's fake model; a stand-in OpenAI-compatible policy server |
| Data and Train | `data`, `rollout`, `trainer` | fake dataset and rollouts; `SlimeLikeTrainer`, a fake with the support Slime is expected to declare |
| Compression | `scorer`, `selector`, `editor`, `quantizer`, `telemetry` | fakes on a small MoE |
| Harness Evolve | `runner`, `grader`, `method` | `WorkspaceEpisodeRunner` around a stand-in agent process; fake grader and method |
| Gate | `evaluator` (optional), `selectors` | the runner's episodes on `tasks`; win/loss, no regression for compression |

`schedule` lists the steps and `expect` their results; a run whose results differ exits 1. The run writes `summary.json`, the release chain, and the agents' workspaces under `--work-dir`.

## The Slime trainer

`honing.backends.slime.SlimeTrainer` runs `sft`, `opd` (on-policy distillation, and self-distillation with a `lineage` teacher), and `grpo` as Slime train-only jobs on the batch the campaign hands it.

`configs/e2e/slime.json` is the all-planes campaign with Slime's declared support (`honing.backends.slime.SUPPORT`) in the `trainer` slot: SFT on a dataset, then self-distillation and GRPO on harness rollouts, alternating with harness evolution, then a rollback. Its trainer is a stand-in that writes every batch in Slime's rollout layout, because the fake model's checkpoints are not Hugging Face directories. `check_trainer.py` and `honing_pipeline.py` both take it, and CI runs both.

The real trainer needs a GPU and Slime's image. On a cluster without Docker:

```bash
scripts/e2e/setup-slime-enroot.sh                     # once per node; creates the enroot container "slime"
git clone https://github.com/THUDM/slime.git /path/to/slime && git -C /path/to/slime checkout 41014d1f
python scripts/e2e/slime_train_step.py --algorithm sft --base /path/to/Qwen2.5-1.5B-Instruct \
    --work-dir /tmp/honing-slime --enroot slime --slime-source /path/to/slime --gpus 0
```

`--algorithm` is `grpo`, `sft`, `opd` (with `--teacher`, a checkpoint of the same architecture), or `opsd` (on-policy self-distillation: the base model scores each sampled answer again with the correct answer added to the question, privileged context the student never sees, and the batch carries those scores as `teacher_logprobs`; no teacher checkpoint is loaded, and Slime reads the scores with `--opd-type sglang` and takes the student's log-probabilities from the same engine with `--use-rollout-logprobs`). `opsd` defaults to `--prompt-style work`, because under the `boxed` prompt, which asks for the final number alone, a teacher that already sees the answer prefers a bare boxed number, per-token distillation rewards skipping the working, and over eight iterations of this script Qwen2.5-1.5B-Instruct fell from 0.50 to 0.07-0.11 held-out reward on two seeds. `work` holds the base's reward (0.80) without improving on it, while GRPO under either prompt reaches 0.96-0.98. The collapse stays reproducible with `--prompt-style boxed`. The script samples grouped arithmetic answers from the base model with SGLang, turns them into the batch the algorithm consumes (the rollout for GRPO, the correct answers for SFT, the unrewarded rollout for OPD), runs one job through `SlimeTrainer`, and samples the trained checkpoint on held-out questions. `--steps N` splits that one rollout into N optimizer steps, so steps after the first train on samples the base policy generated; `--max-new-tokens` sets the generation window (thinking models need far more than the default 256, otherwise every answer is truncated, every reward is 0, and the run stops). `report.json` has the reward before and after and Slime's metrics. One H100 is enough for a 1.5B model, and a run takes four to ten minutes, most of it starting SGLang and Megatron. Mount the filesystem that holds the checkpoints and the work directory as a whole (`--mount /data`, say), because on some clusters enroot creates the parents of a deeper mount point without read permission.

## Plugging in a harness runner (OpenHands, Reef, ...)

A harness framework enters as an `EpisodeRunner` (`honing.core`) with its `RunnerSupport`, graded by a `TaskGrader`. `honing.e2e.WorkspaceEpisodeRunner` is the reference: it writes the harness files into a fresh workspace, starts the agent with the policy's OpenAI-compatible endpoint in `HONING_POLICY_URL` and `HONING_POLICY_MODEL`, and collects `result.json` and `transcript.jsonl`.

1. Write the runner and a factory for it, and a `DeliveryProbe` that tells which harness files reached the agent (`WorkspaceDeliveryProbe` works if the files stay in the workspace).
2. Copy `configs/e2e/all_planes.json` and point `components.runner`, `components.grader` if the framework has its own, and `conformance.probe` at them. Name what the runner needs from the machine in `requires`.
3. `python scripts/e2e/check_runner.py --config <yours>` runs the contract:

   | Check | Passes when |
   | --- | --- |
   | `support` | the runner declares a `RunnerSupport`, and the seed harness is runnable on it |
   | `episodes` | every task runs and grades, and each outcome names the task, harness, and policy it ran |
   | `delivery` | every harness file reached the agent, including one probe file of each declared kind |
   | `concurrent` | a gate's worth of episodes at once all finish, in separate workspaces |
   | `repeat` | a second gate in a new event loop works (a campaign runs one loop per gate) |
   | `gate` | the real gate scores two harnesses on the same tasks |

4. `python scripts/e2e/honing_pipeline.py --config <yours>` runs the full campaign with your runner in it.

`tests/test_e2e_pipeline.py` shows each check failing on a runner that breaks it.

## GEPA against real model APIs

```
export OPENAI_API_KEY=...
python scripts/e2e/gepa_live.py --config configs/e2e/gepa_live.json --work-dir runs/gepa-live-1
```

It needs the `gepa` and `litellm` extras (`uv pip install -e ".[dev,gepa,litellm]"`) and `python3` on `PATH` for the grader. Nothing is served locally and no GPU is used.

| Part | What runs |
| --- | --- |
| Tasks | `examples/tasks/python-bugfix/gepa-live.jsonl`: 6 train and 5 val bug-fix tasks, graded by `CommandGrader` on the fraction of hidden cases passed |
| Agent | `scripts/e2e/minimal_agent.py`, run by the workspace runner: reads `AGENTS.md` and `skills/*.md`, sends the task's files to the policy, writes back the files it returns |
| Policy | an OpenAI-compatible API (`components.endpoints`), called by the agent with the key named by `api_key_env` |
| Reflection | `LiteLLMChatModel` inside `GEPAMethod` (`components.method.args.reflection`) |
| Schedule | 6 GEPA moves, a rollback to `r0001`, 16 more; `max_metric_calls` is 60. Every move charges at least 3 metric calls (the parent's minibatch), and `r0001` holds at least 16 from the first gate, so the run reaches the budget whatever the models reply |

The model is pinned in `models`, `components.endpoints` and `components.method`; change all three together. The key is read only from the environment variables the config names. The agent receives only that variable, and the run checks that no file it wrote holds a key.

Under `--work-dir`, the run writes the release chain (`releases/commits.jsonl`), `summary.json` (steps, checks, calls, tokens, cost), `run.log`, `episodes.jsonl`, `reflections.jsonl`, `faults.jsonl`, and every agent's workspace. It exits 0 only if every check passes:

| Check | Passes when |
| --- | --- |
| `1 extracted` | each gated candidate is its parent with one component replaced by the extracted reflection reply |
| `2 reflection prompt` | each reflection prompt holds the task prompt, the agent's output and the grader's feedback for every parent episode |
| `3 gate on real scores` | the archive stores the graded gate episodes' scores, and at least one candidate reached the gate |
| `4 rejected stays` | a candidate the gate rejected stays in the archive (needs at least one rejection) |
| `5a`-`5d` | `metric_calls` matches GEPA's charges per move and on the final path; every agent the runner started is in `commits.jsonl` or `faults.jsonl`; an abandoned move charges nothing |
| `6 rollback` | the rollback restores the archive and the served harness together, and the next move starts from them |
| `7a`, `7b` | a failed reflection call is a skip that charged only its parent's episodes; an abandoned move left no record |
| `budget stop` | the last move proposed nothing because `metric_calls` reached `max_metric_calls` |
| `8 no key written` | no file under the work directory holds a key |

A model's replies differ from run to run, so a run can end without a rejection or a selection; checks 4 and 6 then fail as "not exercised". Rerun, or lengthen the schedule.

**Failures.** A reflection call that fails after the client's retries (bad key, rate limit, timeout) logs one line and proposes nothing. If the policy cannot be reached after the agent's retries, the agent exits 75 and the move is abandoned: GEPA's state is unchanged, and the move's episodes and tokens go to `faults.jsonl`. After 3 abandoned moves in a row the run stops with exit code 3.

**Cost.** About 200 agent episodes with one policy call each, and about 10 reflection calls. At the prices in litellm's bundled map for `gpt-4.1-mini-2025-04-14` ($0.40 per million input tokens, $1.60 per million output tokens), that is roughly $0.10 to $0.30. `summary.json` reports the actual calls, tokens and cost.

**Tests.** `tests/test_gepa_live_offline.py` runs this script against a loopback stub for both models, including rate limits, bad keys, timeouts and an outage inside a move; CI's `test-litellm` job runs it. `tests/test_gepa_live.py` (marker `live`) runs it on the real provider and is skipped unless the key is set.

## Plugging in a trainer (Slime, ...)

A training backend enters as a `Trainer` (`honing.core`) with its `BackendSupport`. The campaign checks every job against that support before calling the trainer (`honing/campaign/rules.py`), draws the batch from `data` or, for on-policy algorithms, from `rollout`, and gates the `TrainResult`.

1. Write the trainer and a factory for it. `SlimeLikeTrainer` in `tests/fakes/pipeline.py` is a stand-in to start from: it declares Slime's expected support and refuses what the rules refuse; the Laya, Kev, and NanoJev trainers in `honing/backends/jev/` show real ones (a GPU lock, a pinned environment reached through a subprocess, a checkpoint manifest).
2. Copy `configs/e2e/all_planes.json` and point `components.trainer` (and `components.rollout`, if the backend brings one) at yours. List the jobs to try under `trainer_check.jobs`, in the schedule's job format.
3. `python scripts/e2e/check_trainer.py --config <yours>` runs the contract:

   | Check | Passes when |
   | --- | --- |
   | `support` | the trainer declares a `BackendSupport`, supports the base architecture, and the rules let every listed job reach it |
   | `train:<algorithm>` | each job trains on a batch drawn as the campaign draws it (checked by `batch_meets_contract` and `batch_from_one_release`) and returns new weights in Hugging Face safetensors, a supported architecture, and finite metrics |
   | `fresh` | training again does not overwrite the first result, which a release may point at |
   | `refuses` | the trainer refuses a missing batch and an algorithm it does not declare |

4. `python scripts/e2e/honing_pipeline.py --config <yours>` runs the full campaign with your trainer in it.

The default config's model, policy server, and harness runner key on the fake model's checksums, so a real backend's checkpoints cannot pass through that gate. Check a real backend on CPU with a stand-in in its place, and report its GPU run separately, as the Jev trainer pull requests did. `tests/test_e2e_trainer.py` shows each check failing on a trainer that breaks it.

## The tank campaign

`configs/e2e/tank_dry_run.json` builds the campaign in `honing/backends/tank/e2e.py`. Every round is real: `TankEpisodeRunner` starts `integrations/tank-war/scripts/match.mjs`, which plays one Tank War round in lockstep against the planner opponent and asks Honing's `DecisionServer` for every decision. Only the decision model (`tests/fakes/tank.py`), the trainer, and the chat model GEPA reflects with are stand-ins. The stand-in model takes no permitted shot below quality 0.7, unless its weapon instructions tell it to fire whenever the shot is permitted. So:

1. harness (GEPA): the scripted reflection rewrites `tank/questions.json` with such instructions; the child beats its parent on a train minibatch, and its validation mean (0.5) beats the seed's (0): published.
2. train (sft on teacher demonstrations): quality 0.5 to 0.7 changes which motions the model drives (the published instructions already make it fire): 1 win, 3 ties on the gate: published.
3. train (grpo on sampled rollouts): quality 0.7 to 0.9 ties on every gate task: rolled back.

The rounds are deterministic (the game clock stops while a seat waits for its decision), so these results repeat. A change to the game, the planner, or the bridge can change them; if it does, check the new rounds and update the expectations in CI and `tests/test_tank_e2e.py`.

To run a real model, replace `model_source`, `endpoints`, and `trainer` with the Jev ones (`honing/backends/jev`) and `method.args.chat` with a real chat model.
