# Reef backend

Honing records its decisions and episode grades in Reef and trains on Reef's
job data (#137). Its chat calls can go through Reef, which records them too.
Reef runs as its own service in its own Python 3.12 environment. Honing
reaches it only from this package, through `reef-client`
(`pip install -e ".[reef]"`) or Reef's OpenAI-compatible routes, and never
imports `reef`.

## Records

`records.py` defines what Honing writes. Every writer and reader uses it: the
decision server's recorder (#223), the grader's reports (#225), the recipe in
`integrations/reef/` (#226) and the data provider (#227, below).

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

## Recording decisions

`recorder.py` (#223) records the decision server's answered requests. Start
the server with `--reef-url <Reef> --reef-scenario <scenario>` and the
`reef` extra installed. The Bearer token comes from `REEF_TOKEN`. The
scenario is Reef's `x-reef-scenario`: one workload with its own records, which
the first import creates if the deployment allows it.

- **What is recorded.** After the reply is sent, the server hands the request,
  the reply and the request's `x-reef-tag-*` headers to the recorder's thread.
  The request makes one record, and each of its `branches` another, with
  `branch` set to the branch's name (`decision_records`). An integer `id` is
  recorded as its decimal string, as RA2 sends a counter. A request without an
  `episode` tag, or one whose `id` is neither a string nor an integer, is
  answered and counted but not recorded: its record could not have a stable
  id. Tags other than `TAG_NAMES` are ignored and counted.
- **Who tags.** Tank's match process and RA2's run take `--reef-tags`, send
  the tags on every decision, and keep them in the run's `manifest.json` under
  `reef_tags`, where a grader finds the episode. Honing's tank and RA2 runners set `episode` (new
  on every attempt), `task`, `harness` and `policy`. They set no `release`,
  since a runner is given the policy's weights, not the release.
- **Reef down.** The recorder resends the same batch, with the same ids, after
  a backoff of 0.5 s doubling to 30 s, until Reef takes it. A record Reef
  refuses (400, or 409 for a known id with other content) is isolated by
  splitting its batch, then logged and counted. The rest of its batch is stored.
- **A wrong token or URL.** HTTP 401 or 403 (Reef rejected the token) and 404
  (no such route, or a scenario Reef will not create) are retried the same
  way, but logged as errors, once and then every 30 s, and counted apart, so a
  deployment mistake does not look like an outage.
- **Memory.** A queued request is charged its request and reply bytes until
  all of its records are stored or refused. Past `--reef-max-pending-mib`
  (256 by default) a new request is not queued but counted and logged. At exit
  the recorder keeps sending for `--reef-flush-seconds` (10). A restart while
  Reef is down loses what was queued.
- **Counts.** `GET /v1/health` reports them under `reef`: `requests`, `sent`,
  `refused`, `invalid`, `skipped_no_episode`, `skipped_no_id`,
  `dropped_requests`, `ignored_tags`, `rejected_token`, `no_route`, `unsent`,
  and what is pending.

## Reporting grades

`reporter.py` (#225) posts one report per graded episode. `ReefReportingGrader`
wraps a `TaskGrader`: it returns the grade as the wrapped grader gave it, then
queues the episode for `ReefReporter`'s thread, which posts to `/reef/report`.
The tank grader factory turns it on with `reef_url` and `reef_scenario` in its
`args` (the decision server's `--reef-url` and `--reef-scenario`); the token is
`REEF_TOKEN`. Without them the factory returns `TankGrader` as before.

- **What is reported.** `EpisodeReport.from_rows` over the run's
  `decisions.jsonl`: the grade's score and feedback, `metadata.task`, one
  reference per row in play order, and each row's `chosen`. The episode is the
  `episode` tag in `manifest.json`'s `reef_tags`. The report id is
  `episode_report_id(episode)`, so grading the same episode again with the same
  grade is a retry, and with another grade is refused (409) and logged.
- **What is skipped.** A run with no `reef_tags` episode (`skipped_no_tags`), no
  decision rows (`skipped_no_decisions`), or files #229's checks refuse or tagged
  with another task (`invalid`). The grade is never affected.
- **Records first.** Reef refuses a report that references a record it has not
  stored, with a 400 that says `must identify an existing inference`. The
  recorder sends on its own thread, so such a report is sent again, whole, for
  `reference_wait_seconds` (300), and then logged with its episode and counted
  as `unknown_reference`. Missing references are never dropped to make it fit.
- **Reef down, a wrong token or URL, memory, exit.** As the recorder:
  unlimited retries with backoff, 401/403/404 logged as errors and counted
  apart, past `max_pending_reports` (10000) an episode is counted as
  `dropped`, and `close()` sends for `flush_seconds` (10). The tank campaign's
  `main` closes the grader when the campaign ends.
- **The report type.** `integrations/reef/report.py` declares
  `HoningEpisodeReport`, a `ScoredRolloutReport` with a required `task`. Reef
  checks it at admission, because `HoningDecisionRecipe` returns it from
  `report_type`. A report type can declare only `score` on top and flat
  `metadata` fields, so `feedback` and `metadata.honing` pass through
  unchecked, and their reader checks them with `EpisodeReport.from_payload`.

`RecordedDecision.sample` turns a record and its choice into the
`DecisionSample` that `decision_from_row` builds from the `decisions.jsonl`
row, and `tests/test_reef_records.py` checks this on the tank fixtures.

## Job data

`provider.py` reads what Reef hands back. Honing's recipe
(`integrations/reef/`) has Reef spool each training job's data to
`<spool>/<job id>.json` and keep the job running until Honing writes
`<job id>.consumed`. `ReefDataProvider` is the campaign's `DataProvider` over
that spool. At a train move it takes the oldest job no move consumed and Reef
did not cancel, turns it into one `typed_decisions` batch, and marks it
consumed. Reef then ends the job with nothing published.

- Each decision is the `DecisionSample` that `decision_from_row` builds from
  its row. The episode's score is its reward, its task the group, its
  `release` and `harness` tags its sources, and it keeps behavior
  probabilities when the contract asks for them.
- The batch id is `reef:<scenario>/<job id>`, and the commit record of the
  move names it (`candidate.batch_id`).
- Reef consumed the job's records when it submitted the job, so data drawn for
  a move the gate rejects never trains again, even after a restart.
- How many episodes a batch holds is Reef's `pipeline.sample2job.job_samples`.

## Chat calls through Reef

`chat.py` (#224) sends chat calls through Reef, which records each one
itself. A `ReefChatRoute` names Reef's root URL, a scenario and Honing's tags.
Its `base_url`, `<url>/reef/scenarios/<scenario>/v1`, is Reef's
OpenAI-compatible route for that scenario, and the key a client sends there is
Reef's token, read from `REEF_TOKEN`. Reef forwards the call to the model its
deployment names, with the provider key that only Reef holds, stores the
exchange as an inference record of the scenario, and answers with the
provider's reply and the record's id in `x-reef-agent-record-id`. A call's
`x-reef-tag-*` headers become the record's `metadata.tags`.

Nothing goes through Reef unless a config names a route:

- **Harness methods' `ChatModel`.** The factory
  `honing.backends.reef.chat:chat_model` builds a `LiteLLMChatModel` from its
  fields. With `reef`, the client calls the route with Reef's token and sends
  the route's tags as headers (`extra_headers`), so it names no `api_base` or
  `api_key_env` of its own. `model` only picks litellm's provider
  (`hosted_vllm/<name>`): Reef asks its upstream for
  `inference.upstream-model`, whatever name the client sends.
- **Policies behind `llm_gateway`.** `honing.backends.llm_gateway:endpoints`
  takes the same `reef`. A policy's endpoint is then the route's URL and its
  key is Reef's token, while its snapshot and checksum stay the model's. A
  Reef deployment forwards to one model, so the endpoints hold one. The agents
  that call a policy (OpenHands, pi) add no headers, so these records carry no
  tags, the route takes none, and the scenario names the calls.
- **SGLang.** A model on an SGLang server goes through Reef as a gateway model
  whose `base_url` is the server's `/v1`, or as a `ChatModel` with the server as
  Reef's upstream. `SGLangPolicyEndpoints` starts a server for each checkpoint
  on a port of its own, which one Reef upstream cannot follow, so it has no
  route.

```bash
# Reef in front of an OpenAI-compatible server, then the scenario the calls go to.
REEF_TOKEN=reef-local reef serve --inference.upstream-url http://gpu-node:30000 --inference.upstream-model honed
curl -X POST http://127.0.0.1:8900/reef/scenarios -H "Authorization: Bearer reef-local" \
  -H "Content-Type: application/json" -d '{"name": "reflection"}'
```

A pipeline config's `chat` component that sends its calls to that scenario:

```json
{
  "factory": "honing.backends.reef.chat:chat_model",
  "args": {
    "model": "hosted_vllm/honed",
    "reef": {"url": "http://127.0.0.1:8900", "scenario": "reflection", "tags": {"game": "tank"}}
  }
}
```

The scenario must exist first: Reef answers 404 for an unknown one, which the
client does not retry. A provider key goes in Reef's
`inference.upstream-api-key` (or `REEF_UPSTREAM_API_KEY`), never in Honing's
config. `tests/test_reef_chat.py` sends a `ChatModel` call and a gateway
policy's call through a real Reef to the stand-in server in
`tests/fakes/chat_upstream.py`, and reads each record back, the
`ChatModel`'s with its tags.

## The Reef code each module follows

Reef [`bc8c150`](https://github.com/Human-Agent-Society/reef/tree/bc8c15081bfdbd2ea0a4ceacc045545f020da668),
`reef-client` [`91af039`](https://github.com/Human-Agent-Society/reef-client/tree/91af039292daa7392142b20a693a8dbf3e31cad3) (0.4.0 on PyPI).
A pull request that pins a later Reef updates these links.

| Honing | Reef | Differs |
| --- | --- | --- |
| `records.py`, `RecordedDecision` | [`AgentRecord`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/core/records_types.py#L49), [`import_records`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/request_handling.py#L106) | The payload is a decision, not a chat exchange, and the id is derived from the episode and the request, not drawn |
| `records.py`, `EpisodeReport` | [`ReportPayload`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/wire.py#L65), [`validate_record`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/control/ingest.py#L82) | `metadata.honing.chosen` carries the policy's choice for each reference |
| `records.py`, tags | [`with_request_tags`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/wire.py#L119) | Tags are written into `metadata.tags` directly, so one batch can carry several episodes |
| `provider.py`, `ReefDataProvider` | [`JobStatus`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/train/component.py#L249), [`JobLoop.poll`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/train/jobs.py#L285) | Honing, not a Reef trainer, consumes the job data, and the job ends with no checkpoint |
| `records.py`, `batches` | [`POST /reef/records/batch`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/routes/records.py#L15) | Splits by bytes as well as by count |
| `recorder.py`, `ReefRecorder` | reef-client's [`upload_batch`](https://github.com/Human-Agent-Society/reef-client/blob/91af039292daa7392142b20a693a8dbf3e31cad3/reef_client/record_import.py#L66), through [`ReefClient.post`](https://github.com/Human-Agent-Society/reef-client/blob/91af039292daa7392142b20a693a8dbf3e31cad3/reef_client/client.py#L111) | Waits without a retry limit, tells a refusal from an outage, and isolates refused records by splitting the batch instead of stopping |
| `recorder.py`, `request_tags`; the runners' `--reef-tags` | [`parse_request_headers`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/wire.py#L47), [`with_request_tags`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/wire.py#L119) | Keeps only Honing's tags, and writes them into each record's `metadata.tags`; the batch request carries no tag headers |
| `reporter.py`, `ReefReporter`, `ReefReportingGrader` | reef-client's [`ReefClient.report`](https://github.com/Human-Agent-Society/reef-client/blob/91af039292daa7392142b20a693a8dbf3e31cad3/reef_client/client.py#L77), Reef's [`POST /reef/report`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/routes/records.py#L96) and [`validate_record`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/control/ingest.py#L82) | Retries without a limit, and waits for missing references before it counts them |
| `tests/fakes/reef_recipe.py`, `EpisodeReportRecipe` | [`Recipe.report_type`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/recipe/base.py#L118) | Test only: records, trains nothing |
| `tests/fakes/reef.py`, `FakeReef` | [`import_records`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/request_handling.py#L106), [`accept_records`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/control/dispatcher.py#L1022), [`RecordStore`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/storage/scenario/records.py#L54) | The import and report routes; no report type, recipe, storage or training |
| `chat.py`, `ReefChatRoute` | [`scenario_inference`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/inference_service.py#L876), [`InferenceReplica.answer`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/inference_service.py#L640), [`parse_request_headers`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/api/wire.py#L47) | Sends only Honing's tags (`TAG_NAMES`), checked as a decision record's are |
| `chat.py`, `chat_model`, and `GatewayEndpoints` with `reef` in `honing/backends/llm_gateway/models.py` | The scenario URL of an installed harness's model binding ([`scenario_url`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/control/operations/harness.py#L99)), which a client calls with its own token, and Reef's [`InferenceProxyRuntime`](https://github.com/Human-Agent-Society/reef/blob/bc8c15081bfdbd2ea0a4ceacc045545f020da668/reef/inference/http.py#L245), which forwards the call | The URL ends in `/v1`, as an OpenAI client expects, and the `ChatModel` also sends its tags |
