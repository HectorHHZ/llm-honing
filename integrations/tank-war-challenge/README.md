# Tank War public challenge

The public Tank War site, where people play against NanoJev at `/play/`, as it runs at play.honinglab.com, with the player features added on top of it. It is a fork of the tank duel game from #90, kept apart from [`integrations/tank-war`](../tank-war/README.md) so that its behavior stays exactly what players see today; the consolidation of #120 and its changed question text never reached it.

## What a player gets

- A match queue with round tickets and a lobby, so many people share the model servers in turn, plus touch controls on phones.
- Two control schemes, chosen in the start dialog and switchable at any time from the left panel: Mouse + keyboard (the mouse turns the hull toward the cursor, W drives forward, S backward, a left click or Space fires) and the original Keyboard scheme. Esc pauses.
- "Too Hard? (Click Me)", which asks for confirmation and then opens the modifiers. They change only the player's tank: no fire cooldown, magazine size (shells in flight at once), shell size, shell range up to infinite, invincibility, hit points, and Super Tank, which crushes interior walls (the maze links the opened cells, so routes use them) and runs the opponent over. Turning them on starts from 5 HP and Super Tank, without invincibility.
- Sound effects synthesized in the browser (shots, bounces, armor hits, explosions, collapsing walls, a click when the magazine is empty, round starts and results) and a fighting-game announcer: ROUND, FIGHT, K.O., PERFECT and the result in the middle of the arena, short calls such as CLOSE ONE in its top-right corner, each spoken with the browser's speech synthesis. Both can be turned off in the left panel.
- A name, set in the bottom-right corner and remembered for the browser, and two leaderboards in the top right: rounds played fair, and rounds played with modifiers. A round counts as modified if modifiers were on at any moment of it.

## Running it

Three processes, each bound to loopback. The decision servers are Honing's, one per GPU:

```bash
python -m honing.backends.jev.server --family nanojev --pins integrations/ra2/pins.json \
    --nanojev-source "$NANOJEV_SOURCE_DIR" --nanojev-inference tree-compiled --max-len 8192 --gpu 4 --port 38844
```

`compat-gateway.mjs` puts each one behind the game's `/health` and `/decision` routes. It asks NanoJev with the question text and option wording the site used before #120 (`legacy-questions.json`, `compat-bridge.mjs`), because on 32 logged game states the stock bridge's wording took the model from 18 fire commands to 4 and changed 14 of its 32 motions; with this wording it fires 18 times and keeps 31 motions. It still sends a decision's motion question and weapon branches in one request. `HONING_RELEASE` names the Honing checkout whose `integrations/tank-war` it reads, by default this one.

```bash
WORKERS_JSON='[{"port":28944,"upstream":"http://127.0.0.1:38844"}]' node compat-gateway.mjs
```

The game server serves the pages and runs the queue:

```bash
PUBLIC_CHALLENGE=1 CLASSIC_NANOJEV_URLS=http://127.0.0.1:28944 PUBLIC_MAX_MATCHES=8 \
    HOST=127.0.0.1 PORT=4177 PUBLIC_ORIGIN=https://example.org node server.mjs
```

With `PUBLIC_CHALLENGE=1` every other page redirects to `/play/`. `CLASSIC_NANOJEV_URLS` takes up to eight gateways, and `PUBLIC_REQUESTS_PER_MODEL` (1 to 8) lets each serve that many requests at once, which tree inference uses to encode one request while the GPU runs another. `PUBLIC_MAX_MATCHES`, `PUBLIC_MAX_WAITING`, `PUBLIC_MATCH_LEASE_MS` and `PUBLIC_DECISION_QUEUE_MS` size the queue. `TANK_METRICS_PORT` adds a loopback metrics route; the gateway has one at `/metrics`.

The leaderboards and chosen names are kept in `data/leaderboard.json` beside `server.mjs`, keyed by the browser's `tank_user` cookie. The game runs in the browser, which reports each round's result, so the boards are for fun rather than proof.

## Tests

```bash
node --test tests/*.test.mjs
```

They run the engine headless to check every modifier, Super Tank's walls and run-overs, the empty-magazine click and near misses, and check the leaderboards, names and the queue's once-per-round results on the server side.
