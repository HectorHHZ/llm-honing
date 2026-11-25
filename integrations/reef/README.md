# Honing's Reef recipe

Reef records Honing's decisions and episode grades and turns them into job
data. Honing's campaign trains on that data, and Honing's gate decides what
ships (#137). This package is the part Reef runs. It imports Reef and never
Honing, and Reef loads it by import path in its own Python 3.12 environment.

| File | What it is |
| --- | --- |
| `processor.py` | `DecisionEpisodeSamples`: one trajectory sample per reported episode. `DecisionJobData`: job data unchanged |
| `recipe.py` | `HoningDecisionRecipe`, which names the two stages |
| `backend.py` | A training backend that spools each job's data for Honing and saves no checkpoint |
| `deployment.yaml` | A Reef deployment of the three |

The records it reads are defined in `honing/backends/reef/records.py`, and
`honing/backends/reef/README.md` describes them.

## Run it

Reef stores artifacts with Git LFS, so `git lfs` must be on `PATH` or `reef serve` exits at startup.

```bash
python3.12 -m venv .reef-venv
.reef-venv/bin/pip install "reef-infra @ git+https://github.com/Human-Agent-Society/reef@bc8c15081bfdbd2ea0a4ceacc045545f020da668"
REEF_TOKEN=reef-local PYTHONPATH=. .reef-venv/bin/reef serve -c integrations/reef/deployment.yaml
HONING_REEF_BIN=.reef-venv/bin/reef pytest tests/test_reef_recipe.py   # what the CI reef job runs
```

## The path of an episode

1. The decision server records every answered request, and the grader posts
   the episode's report (#223, #225).
2. `DecisionEpisodeSamples` makes one `TrajectorySample` per report: one ATIF
   step per decision, in play order, with the record's `honing` payload, its
   id and the policy's choice under `extra.honing`. The score is the reward in
   `extra.reef`, and the task is the group. A report of another format or
   with no score trains nothing.
3. `DecisionJobData` puts the samples into job data unchanged.
4. The spool backend writes `<spool-dir>/<job id>.json` durably and keeps the
   job running. Honing's `ReefDataProvider` (`honing/backends/reef/provider.py`)
   reads it at a train move and writes `<job id>.consumed`. The job then finishes with no checkpoint, so
   Reef ends it and publishes nothing. `<job id>.cancelled` marks a job Reef
   cancelled, which Honing skips.

Reef runs one job at a time, so the next job's data is spooled once Honing
consumed the last. Reef consumed a job's records when it submitted the job,
so they never train again, even after a restart. `tests/test_reef_recipe.py`
checks all of this against a real Reef, a restart included.

## What #137 asked, as Reef answered

- **A component with no checkpoint** (open question 3). Reef reads a finished
  job without checkpoints as one that trained nothing (`JobStatus`). It ends
  the job and publishes no snapshot. The scenario serves no model either:
  `inference.backend: none`. Reef still asks for a `model-path`, which names
  nothing here.
- **Hundreds of references in one report** (open question 2). Reef admits a
  report that references all 58 decisions of a tank round. Decision records
  hold no chat messages, so each is the last call of its own trajectory.
- **Staleness.** Reef's default rule is on-policy against the artifact Reef
  serves. Decision records name no Reef artifact, so every sample read as
  stale and no job ever started. The stage sets `goes_stale = False`, and
  Honing keeps a batch to one release itself (`batch_from_one_release`).
- **Key order.** Reef stores payloads with object keys sorted. The record
  format keeps order in lists of `[name, value]` pairs for that reason.

## The Reef code each module follows

Reef [`bc8c150`](https://github.com/Human-Agent-Society/reef/tree/bc8c15081bfdbd2ea0a4ceacc045545f020da668).
A pull request that pins a later Reef updates these links.

| Honing | Reef | Differs |
| --- | --- | --- |
| `processor.py`, `DecisionEpisodeSamples` | [`ReportSamples.make_sample`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/samples/reports.py#L293) | Builds the trajectory from decision records, not chat payloads, and never goes stale |
| `processor.py`, `DecisionJobData` | [`OpenClawRLJobData.make_job_data`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/recipes/openclawrl/processor.py#L140) | No record file and no source rewrite |
| `recipe.py`, `HoningDecisionRecipe` | [`OpenClawRLRecipe`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/recipes/openclawrl/recipe.py) | Its jobs go to the spool backend |
| `backend.py`, `SpoolComponent`, `SpoolFacts`, `Deployment` | [`TrainingComponent`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/train/component.py#L506), [`tests/support/training_deployment.py`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/tests/support/training_deployment.py) | A job finishes when Honing marks it consumed, and saves no checkpoint |
