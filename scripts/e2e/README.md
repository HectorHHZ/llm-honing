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
| `node scripts/e2e/tank-duel.mjs` | a decision server, the tank games' HTTP bridge, and both engines; `HONING_URL=<url>` plays against a running server, and `TANK_SECONDS` sets how long each engine plays |

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

`configs/e2e/tank_dry_run.json` builds the campaign in `honing/backends/tank/e2e.py`. Every round is real: `TankEpisodeRunner` starts `integrations/tank-duel/scripts/match.mjs`, which plays one Classic round in lockstep against the planner opponent and asks Honing's `DecisionServer` for every decision. Only the decision model (`tests/fakes/tank.py`), the trainer, and the chat model GEPA reflects with are stand-ins. The stand-in model takes no permitted shot below quality 0.7, unless its weapon instructions tell it to fire whenever the shot is permitted. So:

1. harness (GEPA): the scripted reflection rewrites `tank/questions.json` with such instructions; the child beats its parent on a train minibatch, and its validation mean (0.5) beats the seed's (0): published.
2. train (sft on teacher demonstrations): quality 0.5 to 0.7 changes which motions the model drives (the published instructions already make it fire): 1 win, 3 ties on the gate: published.
3. train (grpo on sampled rollouts): quality 0.7 to 0.9 ties on every gate task: rolled back.

The rounds are deterministic (the game clock stops while a seat waits for its decision), so these results repeat. A change to the game, the planner, or the bridge can change them; if it does, check the new rounds and update the expectations in CI and `tests/test_tank_e2e.py`.

To run a real model, replace `model_source`, `endpoints`, and `trainer` with the Jev ones (`honing/backends/jev`) and `method.args.chat` with a real chat model, or use the SGLang config below.

## Serving with SGLang

`honing/backends/sglang` serves any Hugging Face checkpoint SGLang loads (RFC #103). Build its environment once, then point `SGLANG_PYTHON` at it:

```
scripts/setup-sglang.sh /path/to/sglang-env        # exactly scripts/sglang.lock; prints the export line
export SGLANG_PYTHON=/path/to/sglang-env/bin/python
```

The configs read the checkpoint and GPUs from the environment: `HONING_SGLANG_CHECKPOINT` (a checkpoint directory), `HONING_SGLANG_GPUS` (comma-separated, one server per GPU; devices joined with `+`, such as `0+1+2+3`, form one server that shards its checkpoint over them with tensor parallelism, for a model that does not fit one GPU), and for GEPA's reflection `HONING_SGLANG_CHAT_CHECKPOINT` and `HONING_SGLANG_CHAT_GPU`.

| Command | What it checks |
| --- | --- |
| `check_runner.py --config configs/e2e/tank_runner_check_sglang.json` | the tank runner on real rounds, the checkpoint answering `/v1/decide` through the lettered readout |
| `check_runner.py --config configs/e2e/workspace_runner_check_sglang.json` | the workspace runner, the checkpoint answering OpenAI-compatible chat |
| `tank_campaign.py --config configs/e2e/tank_campaign_sglang.json` | a campaign whose policy and candidates are each started on demand; the trainer (`tests/fakes/serving.py`) writes real, servable checkpoints without training |
| `sglang_endpoints.py <checkpoint> <work-dir> <gpu> <gpu>` | on-demand start, reuse, least-recently-used stop, weights refused under another checksum, the chat route |
| `sglang_logprobs.py <ref-gpu> <gpu> <out.json> <checkpoint>...` | next-token log-probabilities against the checkpoint's transformers class in fp32 and bf16 |
| `sglang_lettered.py <ref-gpu> <gpu> <out.json> <checkpoint>...` | lettered decisions against the same readout on transformers, fp32 and bf16 |

`integrations/ra2/parity/startlux_sglang.py` compares StartLux-Decision on SGLang with Honing's native StartLux path and benchmarks both.
