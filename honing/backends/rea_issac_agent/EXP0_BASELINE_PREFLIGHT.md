# REA-Issac Exp 0 baseline preflight

Date: 2026-11-20. Issue: #184. Base revision: `fdde1534` (`origin/main`).

## Decision

Do not start the 40-episode attribution collection or train fault checkpoints.
The two-seed real-model check below found that the pinned NanoJev checkpoint
never left the empty starting room, while the rule controller reached and
cleared combat rooms. There is no viable model-backed baseline for a meaningful
score-drop benchmark, and a scripted controller alone cannot support the
required `weights` intervention.

## Evidence collected without project GPU use

The repository's `integrations.rea_issac.policy.Baseline` completed the
unmodified Warden task on seven of eight inspected seeds. The run used the
existing engine, a 6,000-tick limit, and three ticks per action:

| Seed | Result | Tick | Rooms cleared | Damage taken |
| --- | --- | ---: | ---: | ---: |
| 7 | won | 1761 | 7 | 5 |
| 11 | won | 899 | 3 | 3 |
| 1001 | truncated | 6000 | 5 | 0 |
| 1002 | won | 1364 | 5 | 3 |
| 2001 | won | 1033 | 3 | 1 |
| 2002 | won | 1388 | 5 | 0 |
| 3001 | won | 1071 | 3 | 4 |
| 3002 | won | 785 | 3 | 0 |

These eight convenient smoke seeds were not selected as an independent
performance benchmark; the 7/8 figure is a capability check only.

Reproduce from the repository root with Python 3.10 or later:

```python
from integrations.rea_issac.engine import Engine
from integrations.rea_issac.policy import Baseline

for seed in (7, 11, 1001, 1002, 2001, 2002, 3001, 3002):
    engine, policy = Engine(seed, 6000), Baseline()
    while engine.status == "running":
        engine.step(policy.act(engine.observe()), 3)
    print(seed, engine.result())
```

The REA-Issac README states that model integration has only a protocol stand-in
test and makes no real-model performance claim. A search of the repository and
the user's existing server artifacts found no REA-Issac model checkpoint or
result report. The public pinned NanoJev checkpoint and its source are cached
on `ds-serv2.ucsd.edu`, but no REA-Issac quality measurement was found.

A one-GPU NanoJev service preflight was first attempted on GPU 2. Honing's `gpu_lock`
rejected startup before loading the model because that GPU already had a
compute process. A subsequent occupancy check showed compute processes on all
seven GPUs. At that stage, no model service, training job, or experiment GPU
process had started for this issue.

## Resumption gate

Resume formal sampling only after a different model-backed agent or a revised
harness demonstrates room entry and effective combat on fixed seeds. Measure
its clean Warden and bounded-room scores before choosing faults. The rule
controller's seven wins are a reference, not a model baseline and not evidence
that a `weights` fault can lower the model agent's score.

No 40-episode dataset, real-model diagnosis, cross-game experiment, or #184
acceptance result exists from this preflight.

## Follow-up: requested small real-model batch

The reproducible runner is now `python -m honing.backends.rea_issac_agent.model_preflight`. It verifies the model
revision and weight checksum from `/v1/health`, runs the same fixed seeds with
the rule baseline and model, audits every replay, and incrementally writes
`summary.json`. Its local HTTP stand-in tests passed with Python 3.12.

The preflight uses 900 ticks and three frames per decision. On its two default
seeds, the rule controller cleared four rooms (seed 1002) and two rooms (seed
2001) before truncation. A six-frame exploratory cadence left the controller
in the start room on all eight smoke seeds, so the runner retains the existing
three-frame game cadence. The 900-tick result measures room progress, not
Warden win rate.

At the follow-up check, all seven `ds-serv2` GPUs still had compute processes
owned by other users. The pinned NanoJev upstream predictor also explicitly
rejected a CPU load, including with `precision="fp32"`. Direct access to the
previously used `ds-serv8` inference server timed out. No busy GPU was
preempted or shared at that stage, and no experiment GPU process had started.

## Automatic one-GPU resumption

On 2026-11-20, the user authorized continuous monitoring and immediate use of
the first available GPU. A background watcher on `ds-serv2.ucsd.edu` (PID at
launch: `830862`) checks every 10 seconds for a card with zero compute
processes and at least 8 GiB free memory. It also waits if two GPUs already
have this user's compute processes; the NanoJev service's `gpu_lock` repeats
the exclusive occupancy check before loading. The watcher runs the two-seed,
900-tick preflight with a 60-minute limit, then terminates the service. It
does not train a model or reserve a GPU while waiting.

The job's status and logs were retained in the private server run directory.
The watcher script's SHA-256 is
`26186be6bda14441e6e11d9d42c36055754d4b7941da0788cde482ee2ab299a9`;
the staged preflight runner's SHA-256 is
`51911d0a6200c912d48b80dd5e0def9780c4ccec529256ce152cf7a8153a9186`.
Both staged scripts compiled. This watcher was stopped when the user explicitly
authorized sharing GPU 5, which still had another user's 1.2 GiB compute
process. A one-off launcher restricted to GPU 5 used the same per-user file
lock and required at least 24 GiB free before loading. It left the other
process in place and stopped the NanoJev service after the preflight.

## Real-model capability result

The pinned NanoJev revision was
`047b927b30882a1138fc504821b82ac145a4b81a` with weight SHA-256
`f68c47d66998231b86b7e91b4ed5e82ae23acf104c8b7cd6d165c3ac7b7ffe1b`.
The test used 900 ticks and three frames per decision on two paired seeds:

| Seed | Controller | Rooms visited | Rooms cleared | Kills | Hits | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1002 | Rule baseline | 4 | 4 | 5 | — | Truncated at 900 |
| 1002 | NanoJev | 1 | 1 | 0 | 0 | Truncated at 900 |
| 2001 | Rule baseline | 3 | 2 | 3 | — | Truncated at 900 |
| 2001 | NanoJev | 1 | 1 | 0 | 0 | Truncated at 900 |

Both starting rooms had zero enemies. NanoJev never left them: it chose
south-east movement in 290/300 and 288/300 decisions, and south-east shooting
in 299/300 decisions on each seed. Its 113 shots per game did not reach an
enemy. The recorded `rooms_cleared=1` reflects the empty starting room; it is
not evidence of combat ability. This check establishes a navigation failure
under the current harness, not a measured battle win rate. No fault candidate,
40-episode collection, Sol diagnosis, or cross-game accuracy result was run.

The first attempt used NanoJev's default `reference` inference. Its observed
GPU 5 total allocation grew from 1,220 MiB before loading to at least 28,959
MiB around tick 255, with 19,711 MiB free. It was stopped early to preserve
headroom for the existing process; the partial trajectory and logs are marked
as an excluded attempt in the artifact. The completed retry used the same
checkpoint with `tree` inference. The highest total allocation seen in
periodic checks was 6,299 MiB, leaving 42,371 MiB free, versus 1,220 MiB
before loading. These observations come from empty-room trajectories and do
not establish the peak for a longer combat run. The one-off shared launcher
required 24 GiB free at startup; future sharing should retain a comparable
margin until combat-state memory is measured.

The complete archive, including all four audited runs, the excluded partial
reference attempt, launch scripts, and logs, is
`exp0_preflight_20261120.tar.gz` in the private evidence store (SHA-256
`859304a0fcc21fe8dcb2ebb4ff5faaf7cad0f5a1cbd0760e2c3cfeb2e4dd3709`).
All four replays passed on the server's Python 3.12.13 x86_64 environment.
After extracting the archive on a Python 3.12.1 arm64 host, both model runs
passed, but baseline runs failed exact replay comparison at ticks 639 and 123.
At the first mismatch, a bullet velocity differed by about `2e-14`; the
bitwise state hash consequently changed. Thus portable exact-hash replay is
not verified across these environments, even though the downloaded archive
hash matched. This limitation must be resolved before claiming the full
cross-environment acceptance criterion.
