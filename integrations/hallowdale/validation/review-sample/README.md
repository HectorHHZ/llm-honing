# One real decision

This directory holds one complete decision from a short run of [`run.py`](../../run.py) against the frozen Laya
typed-decisions model on CPU, in a new game of the pinned hallowdale commit. It was made on 2026-11-18 PST with the
bridge modules in this integration, and [receipt.json](receipt.json) records their hashes, the model revision and the
choice of every decision in that run.

| File | Content |
| --- | --- |
| `request.json` | The complete request sent to `POST /v1/decide` |
| `response.raw` | The exact bytes of the response |
| `transition.json` | The native command, the bridge's acknowledgement, the visible observation before and after, and the debug state before and after |
| `receipt.json` | What produced the run, and one line per decision |
| `check.py` | Offline checks |

Run `python3 -S check.py` from any directory. It rebuilds the request from the visible observation with the current
adapter, validates the raw response with the shared receiver, confirms that the native command is the model's argmax
and that the bridge acknowledged exactly that window, and checks the player's movement in the debug state. It calls
no model and starts no game.

The debug state is included for this offline check. It was not sent to the model.

This is one short author-side run. It shows that requests, responses and native actions connect. It does not measure
how well the model plays, and the demo video comes from a different run. It adds no trace format: the files follow the
review samples of the other game integrations.
