# Inspectable historical decision sample

This is the first accepted real Laya/native decision from the author-side 2026-11-08 PST run. `request.json` contains all submitted context and offered options; `response.raw` is the exact response byte sequence; `transition.json` preserves this game's existing transition record; `original-author-receipt.json` retains the historical source/build/model claims and limitations. `SOURCE.json` records immutable original artifact hashes and the extracted files' hashes.

Run `python3 check.py` from this directory, or pass this file's full relative path from the repository root. It validates the historical response with the current shared adapter check and checks its original native acknowledgement and action effect. It makes no HTTP calls and launches no game.

**Historical evidence:** the source and prompt have since changed locally. This sample and the old video do not prove post-fix behavior, independent reproduction, task completion or training. Post-fix technical regressions are reported separately. This fixture package does not select a new production trace or observation schema. All original receipt hashes are retained.
