# Paired checkpoint comparison

Three seat/seed pairs were fixed before each batch: seed 1 red, seed 1 blue,
and seed 2 red. The map, engine, v22 context, candidates, decision cadence,
initial resources and command budgets match within each pair. Only the Laya
checkpoint differs. All six matches reached the 18,000-tick cap with both
outcomes zero; none ended in a win or defeat.

| Checkpoint | Seed / Laya seat | Mining income | Produced infantry / vehicles | Enemy buildings / infantry / vehicles killed | Final credits |
| --- | --- | ---: | ---: | ---: | ---: |
| english | 1 / red | 0 | 0 / 0 | 0 / 0 / 0 | 400 |
| english | 1 / blue | 0 | 0 / 0 | 0 / 0 / 0 | 400 |
| english | 2 / red | 0 | 0 / 0 | 0 / 0 / 0 | 400 |
| typed-decisions | 1 / red | 0 | 0 / 5 | 1 / 12 / 2 | 0 |
| typed-decisions | 1 / blue | 0 | 0 / 0 | 0 / 0 / 0 | 2900 |
| typed-decisions | 2 / red | 0 | 0 / 5 | 1 / 18 / 3 | 0 |

The English checkpoint built 16 power plants in each match, no refinery and
no army. Its last commands were around tick 6,080. In all three matches,
5,957 of 6,000 decisions submitted no command (99.28%). The refinery was an
available choice and the context recommended it while the model continued
to select power plants.

The typed checkpoint produced five Rhino tanks in each red-seat match and
attacked, but lost all five tanks. Its blue-seat match produced no army.
Mining income remained zero in all three. This supports a difference in
behavior for these paired inputs, not a demonstrated win-rate improvement.
The selected seeds and positions were used during development; the three
pairs are not a held-out, balanced multi-map evaluation.

Both policies used normal observations. Supalosa was called every 3 ticks
with up to 16 commands; Laya chose at most one command at the same interval.
Every match passed the context/action audit and independent replay hash check.
The exact settings, counts, hashes and source fingerprints are in
[checkpoint-comparison.json](checkpoint-comparison.json). The original large
logs, proprietary engine and game assets are kept outside this repository.

These matches used the frozen experimental source fingerprints recorded in
the JSON. The example subsequently replaced machine paths with configuration,
formatted the source, extracted token-budget handling and hardened process
startup. Its engine fixtures and frozen-input checks verify that port; the
six matches are not claimed to have run from the final PR checkout.
