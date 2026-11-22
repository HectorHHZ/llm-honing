# REA-Issac #184 fault candidate validation

This records the candidate-building stage on the frozen PR #221 agent. It is
not the 40-episode attribution experiment. The clean agent and game are pinned
in [`FAULT_PROTOCOL.md`](FAULT_PROTOCOL.md) and `fault_freeze.json`. All four
short pilots used seed 8001, three ticks per decision, and a 300-tick limit.
Every pilot and every model-backed run passed engine replay and agent decision
audit. The pilot ended at its tick limit, so `success: false` below is not a
completed-task failure.

| Arm | Checkpoint | Room clears | Kills | Damage | Controller swap triggers |
| --- | --- | ---: | ---: | ---: | ---: |
| Clean | `a60db30e...` | 2 | 2 | 0 | 0 |
| Harness reversed movement instructions | `a60db30e...` | 2 | 2 | 0 | 0 |
| Weights reverse-movement SFT | `feea1950...` | 2 | 2 | 0 | 0 |
| Controller swapped movement axes | `a60db30e...` | 1 | 0 | 0 | 100/100 decisions |

The harness and weights comparisons used the same 32 preselected saved states
from the clean demonstration dataset. Neither probe received teacher labels.
The comparison requires matching source-request hashes and validates that only
the declared component changed. A movement probability effect means the
largest movement-option probability difference exceeded the prespecified 0.01
tolerance.

| Fault | Movement probability effects | Selected movement changes | Largest movement probability difference |
| --- | ---: | ---: | ---: |
| Harness | 32/32 | 10/32 | 0.2275 |
| Weights | 32/32 | 30/32 | 0.3575 |

The controller pilot used the clean checkpoint and harness. Its first request
and model-selected action matched the clean pilot; the recorded clean
controller action was `move_x=-1, move_y=0`, while the fault executed
`move_x=0, move_y=-1`. All 100 recorded triggers passed replay and the
selected-versus-executed-action check. An initial controller pilot counted
changes in the default `bomb` field as swap triggers; that attempt is excluded.
The corrected pilot compares normalized actions and retains the initial attempt
in the private run store.

The weights fault is a real checkpoint trained from the clean selected
checkpoint, with an independent SHA-256 of
`feea1950fecde6e702a2fd95eaffa83f5917544d88b62ac4ae05727f256287d6`.
Training used only the reversed-movement examples (249 train, 53 validation),
one epoch, and one GPU on UCSD. The held-out reverse-label accuracy was
0.0377; validation loss moved from 2.1262 to 2.2042. It therefore passes the
local component-change check but does not establish effective learning of the
intended wrong policy. The clean and fault checkpoints were each served on
one GPU at different times. A separate 18 GiB project reservation remained
on a second GPU, within this task's three-GPU cap. The fault model service
was stopped after validation.
Weights & Biases was not configured for this run; the trainer's saved report
contains the example counts, held-out split, losses, accuracy, source checkpoint
hash, training dataset hash, and trained checkpoint hash listed here.

The probe JSONL SHA-256 values are:

| Probe | SHA-256 |
| --- | --- |
| Clean | `781621109b943f4879a8997ac1177b32788f1498db315a133c985e2c161950c6` |
| Harness | `5f873b842bfc9be9fe431aa22103c891e636ef31ada8a790cc541a11d9e6abd1` |
| Weights | `9958d8f77852b6752e09fc160638e24405d1b74e0f108b3610ff6b06858bc858` |

The exact source and fault-training dataset hashes, the excluded Linux
regeneration, portable training commands, and audit procedure are in
[`FAULT_PROTOCOL.md`](FAULT_PROTOCOL.md). Raw probes, pilot recordings,
training logs, and checkpoints remain outside Git. Independent re-execution
currently requires obtaining the clean checkpoint; its public download
location is still pending in PR #221. The fault checkpoint also needs a
distribution location before external reproduction is possible.

These results prove local component effects, not a reliable reduction in
whole-episode score. In the one short paired pilot, the controller fault
changed room clears and kills; the other two faults did not. No 40-episode
sample, diagnostic-model run, or cross-game diagnostic accuracy result has
been collected here. The shared decision contract and a structurally
different game are covered by the focused tests, while all game-specific
fault rendering and action changes stay in this adapter.
