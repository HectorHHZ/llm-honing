# One real decision from the harness v8 run

This sample is decision 1 of 47 from `eval_laya_v8_L1L2`, the run described in
[the experiment README](../../experiments/harness-v8/README.md). `SOURCE.json` says which harness, runtime and model
produced it.

| File | Content |
| --- | --- |
| `request.json` | The complete context and menu the harness sent to Laya |
| `response.raw` | The exact bytes Laya returned (no final newline) |
| `transition.json` | The visible observation the harness read, the 43 native windows it then played, the game's evaluation state before and after, and the skill result |
| `check.py` | Offline check; it makes no HTTP call and launches no game |

From the repository root:

```sh
python3 -S integrations/hollow-knight/validation/review-sample/check.py
```

The check validates the raw response with the shared receiver in `integrations/_shared/decision.py`, confirms that the
recorded choice is the validated argmax, that request, observations and native windows belong to the same moment, that
the windows are contiguous and attributed to this decision, and that the game state after them follows from their frame
counts.

Laya chose the option. Harness code chose every native input after that, which is how this experiment works. The sample
is one decision, so it says nothing about how often the harness succeeds.
