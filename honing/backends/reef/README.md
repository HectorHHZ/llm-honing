# Reef backend

Honing records its decisions and episode grades in Reef and trains on Reef's
job data (#137). Reef runs as its own service in its own Python 3.12
environment. Honing reaches it only from this package, through
`reef-client` (`pip install -e ".[reef]"`), and never imports `reef`.

## Records

`records.py` defines what Honing writes. Every writer and reader uses it: the
decision server's recorder (#223), the grader's reports (#225), the recipe in
`integrations/reef/` (#226) and the data provider (#227).

**A decision record** is one request the decision server answered. It is an
inference record whose payload is:

```json
{
  "honing": {
    "format": "honing-decision",
    "version": 1,
    "request_id": "blue-3114-1:motion",
    "branch": null,
    "parts": [["essential", ["..."]], ["details", []], ["memory", []], ["logs", []]],
    "questions": [["motion", {"type": "...", "instructions": "...", "criteria": [["...", "..."]]}]],
    "answers": {"motion": {"choice": "...", "probabilities": {"...": 0.5}}}
  },
  "metadata": {"tags": {"episode": "...", "task": "...", "release": "...", "harness": "...", "policy": "..."}}
}
```

`parts`, `questions` and `answers` are the request's and the reply's objects
as sent, except that Reef stores a payload with its object keys sorted. So
whatever has an order (the parts, the questions, each question's options) is a
list of `[name, value]` pairs, and a sample's options keep that order.
`branch` names the branch the request was in, or is `null`.

**An episode report** is the grade of one episode, posted to `/reef/report`:

```json
{
  "score": 1.0,
  "feedback": "...",
  "references": ["<record id>", "..."],
  "metadata": {
    "task": "...",
    "honing": {"format": "honing-episode", "version": 1, "episode": "...", "chosen": [{"motion": "..."}, "..."]}
  }
}
```

`references` are the episode's decisions in play order, and `chosen[i]` is
the policy's choice for `references[i]`. The choice is the training label and
only the game knows it: a rollout draws it from the probabilities, and safety
can change a motion after the draw. In the tank fixtures a fifth of the rows
chose something other than the server's `choice`. So the choice travels in the
report, not in the record. A request whose answer the game did not use is
recorded but not referenced, and never trains.

**Ids.** A decision record's id is the first 32 hex digits of a SHA-256 over
the format, its version, the `episode` tag and the request's `id`. A report's
id hashes the episode alone. A resend is then a retry that Reef stores once,
and a report names its records without asking Reef for receipts. The
`episode` tag must be unique to one attempt of one episode, because Reef
refuses the same id with other content. A request needs a string `id`, unique
within its episode, branches included: the id leaves the branch out because a
`decisions.jsonl` row does not name its branch, and the report finds a row's
record by the row's id alone.

**Tags.** `episode`, `task`, `release`, `harness` and `policy` (the policy
checksum) for a campaign's episodes. A game outside a campaign tags `game`,
and its Reef scenario names the rest. Records carry their tags in
`metadata.tags`, where Reef stores an inference's `x-reef-tag-*` headers, so
one import batch can hold several episodes. A chat call that Reef serves
itself sends them as headers (`tag_headers`).

**References.** Decision records hold no chat messages, so Reef links none of
them into a trajectory, and each is the last call of its own. A report may
therefore reference every decision of its episode, as Reef's rule for
references allows. A tank round has a few hundred.

**Batches.** Reef's import route takes 1 to 1000 records and at most 10 MiB a
request. A tank decision's parts are tens of KB, so `batches` splits by bytes
too, under 9 MiB.

`RecordedDecision.sample` turns a record and its choice into the
`DecisionSample` that `decision_from_row` builds from the `decisions.jsonl`
row, and `tests/test_reef_records.py` checks this on the tank fixtures.

## The Reef code each module follows

Reef [`bc8c150`](https://github.com/Human-Agent-Society/reef/tree/bc8c15081bfdbd2ea0a4ceacc045545f020da668),
`reef-client` [`91af039`](https://github.com/Human-Agent-Society/reef-client/tree/91af039292daa7392142b20a693a8dbf3e31cad3) (0.4.0 on PyPI).
A pull request that pins a later Reef updates these links.

| Honing | Reef | Differs |
| --- | --- | --- |
| `records.py`, `RecordedDecision` | [`AgentRecord`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/core/records_types.py#L49), [`import_records`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/request_handling.py#L106) | The payload is a decision, not a chat exchange, and the id is derived from the episode and the request, not drawn |
| `records.py`, `EpisodeReport` | [`ReportPayload`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/wire.py#L65), [`validate_record`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/control/ingest.py#L82) | `metadata.honing.chosen` carries the policy's choice for each reference |
| `records.py`, tags | [`with_request_tags`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/wire.py#L119) | Tags are written into `metadata.tags` directly, so one batch can carry several episodes |
| `records.py`, `batches` | [`POST /reef/records/batch`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/routes/records.py#L15) | Splits by bytes as well as by count |
