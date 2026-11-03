# Final Space Invaders harness

`final-harness.json` identifies the one controller configuration presented as
the final hand-tuned Jev system. The runner is `lane_pilot.py`. NanoJev chooses
between high-level target lanes or single-alien intercept plans; the harness
handles aiming, lead, shields, movement, firing, dodging, and landing urgency.
The exact runner flags and expected NanoJev weight checksum are in that JSON
file. `docs/space-invaders-final-harness.md` gives the commands to run it.

The runtime files are `lane_pilot.py`, `lane_controller.py`,
`solo_intercept.py`, and `lane_io.py` (shared visible-state and HTTP helpers).
`random_ab_server.py` supports the high-level random control in the
optimization pipeline; it is not a second controller. `render_lane_replay.py`
can render a locally generated trace. The installed
`honing.harness.space_invaders` module provides a separate, controller-free
ALE environment and replay validator.

Install `requirements.txt` in a dedicated Python environment for the OCAtari
runner. ALE needs an external Space Invaders ROM. Download the model checkpoint
from external storage and pass its local `checkpoint.json` to the Honing
NanoJev server. The manifest's `path` must point to the local checkpoint
directory. No ROM, model weights, experiment data, or rendered video is
included in this PR.

The SFT and optimization scripts remain as supporting tools for later work.
They generate their own local data and do not define another shipped harness.
The pipeline rejects GEPA and Meta-Harness registrations until game-specific
adapters exist; a hand edit is registered as `manual_code`.
