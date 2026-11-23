# REA-Issac Exp 0 v0 collection and offline scoring

The collection uses the two NanoJev checkpoints and three interventions in
`FAULT_VALIDATION.md`. The clean checkpoint and the reverse-movement checkpoint
need separately served, accessible `/v1/health` and `/v1/decide` endpoints.
The collector checks each service's revision and weight SHA-256. Both services
can be run by the checkpoint owner on their authorized Linux GPU host. The
person who verifies and scores the exported collection needs only Python and
the pinned checkout.

Run from a committed checkout of the collection code. Start the two services
and set `REA_ISSAC_CLEAN_SERVER` and `REA_ISSAC_FAULT_SERVER` to the respective
service URLs. If both endpoints use Honing bearer authentication, supply
`HONING_TOKEN` through the environment. Keep endpoint credentials out of files.

```sh
python -m honing.backends.rea_issac_agent.collect_exp0 collect \
  --clean-server "$REA_ISSAC_CLEAN_SERVER" \
  --fault-server "$REA_ISSAC_FAULT_SERVER" \
  --out work/rea-issac-exp0/v0
```

The collector first requires the clean model to clear a room on the documented
preflight seed 8001. It then runs the four arms on each of the 12 fixed seeds
9101 through 9112, with 1,800 ticks and three ticks per decision. Each arm gets
one attempt per seed. Audit replay and the selected/executed action sidecar,
repeat the first request to check identity at the 0.01 tolerance, and capture
matched-state evidence on up to 16 fixed, evenly spaced decision indices for
the harness and weights arms. The controller arm must have a verified axis-swap
trigger. The four arms within a seed form a single reset group.

Keep the ten eligible groups with the most injected-arm non-success outcomes,
breaking ties by ascending seed. Record the outcome of each selected control
and use the selected clean episode as the paired outcome for that group. These
pairs document outcomes without independently establishing whole-episode loss
causality. The other attempts and exclusion reasons remain in the collection.
Run batches serially. To continue an interrupted batch, use the same command
with `--resume`. A recorded interrupted episode is excluded and retained.

The completed directory includes the pinned code and model identities, native
logs, agent replies, component evidence, episode index, split manifest,
attempt ledger, exclusions, and file hashes. Before sharing, review the ledger
and machine metadata for your dataset destination. Keep the episode index,
matching metadata, fault comparisons, and evaluator-only evidence out of the
diagnoser context. A no-injection episode is a control with a null verified
component label.

Verify the finished directory using its pinned checkout. Verification checks
file coverage, hashes, counts, split groups, and required evidence without
network calls. Exact native replay is performed on the capture host and can
depend on its platform and engine runtime.

```sh
python -m honing.backends.rea_issac_agent.collect_exp0 verify \
  --out work/rea-issac-exp0/v0
```

The offline scorer consumes one normalized JSONL reply per selected episode
and mode. Each row has `episode_id`, `mode` (`read` or `probe`), `label`
(`harness`, `weights`, `controller`, or `abstain`), and optional `cost_usd`.
Keep raw API replies, probe transcripts, failures, and usage with the separate
evaluation run record. The scorer rejects incomplete or unpaired inputs and
reports component accuracy, macro-F1, coverage, the control false-positive
rate, costs, and paired uncertainty grouped by reset seed. The interval is
conditional on the existing fault instances.

```sh
python -m honing.backends.rea_issac_agent.score_exp0 \
  --collection work/rea-issac-exp0/v0 \
  --replies work/rea-issac-exp0/normalized-replies.jsonl
```

Release the completed experiment under `rea-issac/v0/` in
`honinglab/exp0-data`. Record the Hugging Face commit, Git source revision,
real-model evaluation receipt, and fresh-download verification and scoring
receipt for #184 and #174 once the recorded experiment is available.
