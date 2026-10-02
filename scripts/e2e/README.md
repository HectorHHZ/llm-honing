# End-to-end runs

Every script here runs on a CPU with no model, GPU, or game, and CI's `e2e` job runs each on every pull request.

| Command | What it runs |
| --- | --- |
| `python scripts/e2e/honing_pipeline.py --config configs/e2e/all_planes.json` | one campaign through every plane: train, prune with recovery, quantize (rejected, then accepted after quantization-aware training), harness evolution, GRPO on rollouts, and a rollback |
| `python scripts/e2e/check_runner.py --config configs/e2e/all_planes.json` | the checks a harness runner must pass before it joins a campaign |
| `python scripts/e2e/ra2_campaign.py --config configs/e2e/ra2_dry_run.json` | the RA2 runner and grader in a campaign, with stand-in match processes |
| `scripts/e2e/serving.sh` | a decision server process and the RA2 match's Node client stack |
| `node scripts/e2e/tank-duel.mjs` | a decision server, the tank games' HTTP bridge, and both engines |

## The all-planes pipeline

`configs/e2e/all_planes.json` names one component per slot, by import path (`{"factory": "package.module:function", "args": {...}}`). A real implementation replaces a fake by changing its entry.

| Plane | Slots | Default |
| --- | --- | --- |
| Model | `model_source`, `endpoints` | the walking skeleton's fake model; a stand-in OpenAI-compatible policy server |
| Data and Train | `data`, `rollout`, `trainer` | fake dataset, rollouts, and trainer |
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
