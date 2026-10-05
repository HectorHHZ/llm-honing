# End-to-end runs

Every script here runs on a CPU with no model, GPU, or game, and CI's `e2e` job runs each on every pull request.

| Command | What it runs |
| --- | --- |
| `python scripts/e2e/honing_pipeline.py --config configs/e2e/all_planes.json` | one campaign through every plane: train, prune with recovery, quantize (rejected, then accepted after quantization-aware training), harness evolution, GRPO on rollouts, and a rollback |
| `python scripts/e2e/check_runner.py --config configs/e2e/all_planes.json` | the checks a harness runner must pass before it joins a campaign |
| `python scripts/e2e/check_trainer.py --config configs/e2e/all_planes.json` | the checks a trainer must pass before it joins a campaign |
| `python scripts/e2e/ra2_campaign.py --config configs/e2e/ra2_dry_run.json` | the RA2 runner and grader in a campaign, with stand-in match processes |
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
