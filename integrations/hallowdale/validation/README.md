# Validation files

Everything here is author-side evidence from one Linux machine. A reviewer has not rerun the game.

| File | What it is |
| --- | --- |
| `native-checks.json` | Output of `verify_native.py`: the keys of every offered action, repeated presses, refused commands, the idle watchdog and a rejected model response |
| `action-probes.json` | Output of `verify_native.py`: what each action did in the running game, with the game state before and after every step |
| `replay-regression.json` | Summary of the replay that compares this bridge with a run recorded by the earlier bridge |
| `review-sample/` | One complete real Laya decision, with an offline `check.py` |
| `debug-saves/` | Hand-made save files for the probes of abilities that are normally found later in the game |
| `demo-harness-laya-edited.mp4` | The demo video, described below |
| `demo-cut-list.txt` | Every decision shown in the video, with its speed, and what was cut |

## The demo video

The video is an edited cut of one run in Dean's Pass. The frozen Laya typed-decisions model chose among skills offered
by our human-in-the-loop harness. That harness is not part of this pull request and comes in a later one, so the video
cannot be reproduced with `run.py`, and it shows that harness and the model together.

The run made 63 decisions with no deaths and did not finish the area. The cut shows decisions 1 to 30 at double speed,
with repeated stretches at eight times speed, and leaves out decisions 31 to 63. It was re-encoded to fit the
repository's 1 MB file limit: H.264, 960 x 602, 30 fps, 891 frames, 29.7 s.

The video shows hallowdale's artwork by mega-dean, under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). See "Licensing and provenance" in the
[README](../README.md).

## Debug saves

The three `.save` files use the game's own save format. `abilities.save` starts in the room `forgotten_a` with the
movement abilities, the three spells and the dream nail. `lore.save` and `spikes.save` start a new character in Dean's
Pass, next to a readable sign and next to spikes. They exist only for the probes. `run.py` always starts a new game.
