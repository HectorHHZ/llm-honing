# REA-Issac #184 single-component fault candidates

The clean agent is frozen at Git commit
`ed5c9642bb5a1894084965ddc8be4c11e25a68f0` (draft PR #221). It uses
the `model-assisted-combat-v2` controller, harness spec
`h-c2d1d7d325930098`, three ticks per decision, and selected NanoJev weight
SHA-256 `a60db30eb906b46e89499e284a40f31f7e1ea1a790f4a2ac744253b9f90a1c51`.
The existing game, public observation, and action files remain fixed by
`agent_frozen.json`. Fault-building code is on a separate branch based on
this commit. This is a candidate-validation stage, not the 40-episode v0
dataset.

| Component | Single change | Required local evidence |
| --- | --- | --- |
| Harness | Replace only the model-facing `move` instruction with a false reversed-control explanation. | Saved raw observation stays equal; rendered request has a different move instruction and the same options, shooting, and bomb questions. A same-state model query must confirm a probability or decision effect before collection. |
| Weights | Train from the clean selected checkpoint on the v3 demonstration states with every nonzero movement label reversed. Retain only the `move` question, so shooting and bomb labels cannot supply a second training fault. | Independently hashed checkpoint differs from clean weights; same request and action mapping; selected-action or probability changes at a matched state. |
| Controller | Swap `move_x` and `move_y` after the clean v2 controller selects its executed action. | Agent sidecar records model choice, clean controller action, final executed action, trigger flag, and engine replay. Only decisions with a real axis change count as triggers. |

The v3 source demonstration file has SHA-256
`4016873759763a7b70b0ae3daed7dc4ffeef401db3ae96fe53d25e49941d8dab`.
The generated reverse-movement training file has SHA-256
`c1f9c5bc148a6c0ddb92653d39307b8275537a0f548b12822a89bd00c607fb93`:
302 decisions, including 53 validation decisions; 58 no-movement labels were
excluded because reversing them would not change the target. The dataset
builder rejects a changed source manifest, and the trainer checks split
groups. The data and trained checkpoint stay outside Git.

The UCSD Linux regeneration produced the same 360 seed/tick pairs and all
360 movement labels, but its source file SHA-256 was
`9f465df760e5200e4f7755bf8101c5b21437f681c12f88b0f8f0256a7556cce4`.
In 56 requests, engine floating-point text differed by about `1e-14`; the
other 304 requests were equal. That regeneration is an excluded attempt,
not a replacement for the pinned training input. The exact pinned v3 and
fault datasets were transferred to UCSD and verified there by SHA-256 and
`load_batch` before training.

Generate and inspect the weights-fault data from the repository root:

```sh
python -m honing.backends.rea_issac_agent.make_agent_demos \
  --out work/rea_issac_agent/demos-v3 \
  --per-seed 60 --max-state-chars 12000 --selection balanced_rooms
python -m honing.backends.rea_issac_agent.faults \
  --source-dataset work/rea_issac_agent/demos-v3 \
  --out work/rea_issac_agent/fault-demos-reversed-move
```

For training, pass the **directory** of the selected clean checkpoint as
`--base-checkpoint`, the pinned NanoJev checkout as `--nanojev-source`, and a
free GPU index as `--gpu` to `train_agent`. The clean checkpoint must be
downloaded separately; its distribution URL is pending in PR #221. Record
every attempted training run, including rejected checkpoints and reason.

Use `model_preflight --harness-fault harness-reverse-move-v1` or
`--controller-fault controller-swap-move-axes-v1` for a bounded real-model
pilot. The preflight rejects simultaneous harness and controller faults.
Supply a clean or trained decision service separately; never reuse a faulted
service for the harness or controller-only comparison. Preserve clean and
faulted matched-state requests and probabilities before deciding whether a
candidate is valid. A component-change label does not by itself establish
that the fault caused an episode loss. No reference action is asserted from
the scripted teacher.

`fault_probe.py` preselects the first two saved requests in each seed/room
group, stopping after 32, before it sees model replies. Record the clean
selected checkpoint once, then record the same checkpoint with
`--harness-fault harness-reverse-move-v1`. For weights, record the same
requests with the independently hashed fault checkpoint. `compare` requires
identical source requests, confirms only the declared component changed,
and reports selected-action changes and probability deltas above the initial
0.01 tolerance. These probes validate a candidate; they do not establish a
whole-episode score drop. Logs and incomplete/rejected attempts remain in
the experiment run store, outside Git.
