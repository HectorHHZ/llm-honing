# #183 sustained-fault diagnosis follow-up

**Scoped result: go for controlled probes as a way to classify these three
constructed fault families.** On 30 previously undiagnosed, technically
verified fault losses, `gpt-6-sol` at `high` effort identified the component in
28 games with probes versus 16 from the same replays alone. The paired gain was
12/30, or 40 percentage points (predeclared class-stratified bootstrap 95%
interval: 30–50 points). All 12 changed fault answers improved; none regressed.
The predeclared scoped go rule passed.

**This is not a general go for autonomous failure attribution.** Probes still
called harness or controller on 8 of 10 unmodified losses. A clean loss may
contain ordinary model mistakes or normal execution rules; those eight answers
have no supporting constructed fault. The test also contains only one fault
family per component. It does not establish transfer to other faults or that a
particular intervention caused the final loss.

| Technical class | Read-only correct | With controlled probes correct |
| --- | ---: | ---: |
| Harness | 6/10 | 9/10 |
| Weights | 0/10 | 9/10 |
| Controller | 10/10 | 10/10 |
| **All verified faults** | **16/30** | **28/30** |
| Unmodified losses called harness/controller | 9/10 | 8/10 |

The two remaining probe errors were a harness game called controller and a
weights game called harness. In the first, the diagnostic explanation focused
on repeated differences between selected and executed turns. This is also a
real ambiguity in the harness variant: its strategy setting participates in
plan revalidation after choice. In the second, the explanation treated the
ordinary `Do not fire yet` input text as the cause, even though the technical
same-state comparison identifies the edited checkpoint as the changed layer.
Unmodified games also contain that text, which explains part of the control
false-positive pattern. These are interpretation errors, not missing probe
coverage.

## Frozen test and validity

- The 40 selected games have distinct seeds: 10 verified losses for each
  component and 10 unmodified losses. Selection and the technical labels were
  frozen before either diagnostic mode. The sample freeze SHA-256 is
  `583edbc1bc3055df91c1e0bf5c32d134dcce3794fa799614d7b089b64dcef091`;
  the diagnosis freeze SHA-256 is
  `d4be3911baed7c148cbc6e07d5659adcb812649029e1c2c37cc7c04bff5101fc`.
- The 40 games passed full deterministic replay. Controlled repeats covered
  all 1850 recorded decisions. The probe audit passed 40/40: the three fault
  classes had their expected local signatures at the technical reference,
  and all decisions in the 10 unmodified games reproduced the clean model
  input, choice, command, and execution within the frozen probability tolerance
  of 0.01.
- The same `gpt-6-sol` model ID, `high` effort, prompt, JSON schema, and
  anonymous game inputs were used for both modes. The probe mode added neutral
  same-state repeats through the pinned clean checkpoint. Neither mode received
  a fault manifest, seed, injected layer, technical reference, or gold label.
  Bubblewrap exposed only the anonymous input and files required by Codex CLI.
  All 80 diagnoses completed with valid schema, recorded token usage, and no
  tool calls. Answers were not inspected until both modes completed.
- A case scored correct only when its one game-level component matched the
  technical label. `uncertain` counted as wrong on verified faults. The paired
  95% interval used 10,000 class-stratified paired resamples with seed 183.
  The 10 controls were reported separately; a `weights` answer on an ordinary
  unmodified loss was not automatically scored false.

## Cost and execution

| Mode | Calls | Input tokens | Output tokens | Sum of call durations |
| --- | ---: | ---: | ---: | ---: |
| Read-only | 40 | 2,117,328 | 16,650 | 486.943 s |
| Probes | 40 | 2,303,755 | 21,948 | 584.435 s |

Both modes stayed below the frozen cap of 3 million input and 100,000 output
tokens per mode. Call durations are summed across at most two concurrent calls,
not elapsed wall time. Probe construction used one temporary GPU 6 service,
then GPU 0 and 6 together after both were verified idle. The first phase ran
approximately 21:30:34–21:37:30 UTC; the two-GPU phase ran approximately
21:38:50–21:45:27 UTC on 2026-11-17. This is about 0.34 GPU-hours of reserved
service capacity, not a utilization measurement. All temporary services exited.
No model training ran. The login-based CLI exposes token usage, but its internal
model revision and temperature cannot be independently fixed or verified; the
token counts are not an API bill.

The full result, per-game answers, probe files, requests, replies, and logs are
retained privately at
`/data/xinkaiz/system0/exp0/tank-war/v2-sustained-diagnosis-v1/` on ds-serv2.
The committed [aggregate summary](diagnosis-summary.json) excludes per-game
gold labels and raw games. The v3 dataset index points to the same source games
used by this separately frozen follow-up.

## Commands run

```text
bash /data/xinkaiz/system0/exp0/tank-war/v2-sustained-diagnosis-tools/run-sustained-probes-gpu0-gpu6.sh
python3 /data/xinkaiz/system0/exp0/tank-war/v2-sustained-diagnosis-tools/run-sustained-diagnosis.py --source-blind /data/xinkaiz/system0/exp0/tank-war/v2-sustained-diagnosis-v1 --out /data/xinkaiz/system0/exp0/tank-war/v2-sustained-diagnosis-v1/diagnosis --tools /data/xinkaiz/system0/exp0/tank-war/v2-sustained-diagnosis-tools --mode probe
python3 /data/xinkaiz/system0/exp0/tank-war/v2-sustained-diagnosis-tools/score-sustained-diagnosis.py --source-blind /data/xinkaiz/system0/exp0/tank-war/v2-sustained-diagnosis-v1 --diagnosis /data/xinkaiz/system0/exp0/tank-war/v2-sustained-diagnosis-v1/diagnosis --out /data/xinkaiz/system0/exp0/tank-war/v2-sustained-diagnosis-v1/result.json
```

The complete probe command and its startup, resume, and stop logs are in the
private store. The read-mode command was the same diagnosis command with
`--mode read`. No shared pipeline code changed; cross-game generalization was
not tested and is not claimed. The full #183 acceptance criteria still need
more independent fault families, a clean harness-only boundary, and a better
test of unmodified-loss false positives before a broad go/no-go conclusion.
