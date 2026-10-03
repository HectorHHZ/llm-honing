# OpenHands in Honing

OpenHands is the first language-model agent behind Honing's `EpisodeRunner` (#69). Each episode is one OpenHands conversation on one task, under one harness, against one policy, and the gate, methods, and release chain see it exactly as they see the RA2 controller or the reference workspace runner.

## Setup

OpenHands needs Python 3.12 or newer and pins its own litellm, so it runs in its own environment and process. Honing imports nothing from it.

```bash
uv venv --python 3.12 integrations/openhands/.venv
uv pip install --python integrations/openhands/.venv/bin/python -r integrations/openhands/requirements.txt
```

`requirements.txt` pins `openhands-sdk` and `openhands-tools` 1.50.1.

## Run it

Plumbing, on a CPU with the stand-in policy server (no model, no key):

```bash
python scripts/e2e/check_runner.py --config configs/e2e/openhands.json
python scripts/e2e/honing_pipeline.py --config configs/e2e/openhands.json
```

Three OfficeQA questions with a cheap hosted model. Needs `HF_TOKEN` (the dataset is gated, accepted automatically) and `OPENROUTER_API_KEY`:

```bash
python -m honing.backends.officeqa.fetch --out integrations/officeqa/.cache --uids UID0002,UID0006,UID0016
python scripts/e2e/check_runner.py --config configs/e2e/openhands_officeqa.json
python scripts/e2e/honing_pipeline.py --config configs/e2e/openhands_officeqa.json
```

The fetch takes the three questions' source bulletins plus two issues on each side, 11 files and 6.6 MB instead of the 460 MB corpus. The model is DeepSeek V4 Flash through OpenRouter, at $0.028 and $0.056 per million input and output tokens. One episode took about a minute and cost about half a cent.

## How an episode runs

```text
OpenHandsEpisodeRunner (Honing, Python 3.10+)          worker.py (OpenHands environment, Python 3.12)
  check_spec: kinds, tool allowlist, MCP registry
  write episode dir:  episode.json, harness/, work/ ──▶  build Agent from harness/ and episode.json
  start worker in its own process group                 LocalConversation(workspace=work/).run()
  wait, kill the group on timeout (exit -9)        ◀──  write result.json, transcript.jsonl, delivery.json,
  return EpisodeOutcome                                 openhands/ (OpenHands' own event store)
TaskGrader reads final_output and the episode dir
```

The two processes share only the episode directory. The policy's API key goes to the worker through the environment and is never written to disk; `episode.json` names the variable, not the value.

## Tool management: what the harness controls

A harness is files. The runner declares three kinds, and each reaches OpenHands one way:

| Kind | Path | Reaches the agent as |
| --- | --- | --- |
| `instructions` | any, e.g. `AGENTS.md` | appended, in path order, to the system message |
| `skill` | `<dir>/<name>/SKILL.md`, or any other path | an OpenHands skill; frontmatter `triggers` make it keyword-activated, otherwise always on |
| `settings` | `openhands/agent.json` | `{"tools": [...], "max_iterations": n}`: which tools the agent has |
| `settings` | `openhands/mcp.json` | `{"enabled": [...]}`: which MCP servers from the operator's registry are attached |
| `settings` | any other path | a file in `work/.harness/` that the agent can read |

Every one of OpenHands' own discovery paths is off: no user skills from `~/.openhands`, no public skill registry, no `AGENTS.md` picked up from the working directory, no persistent memory. What the agent sees is what the harness says and nothing else, so two episodes under one `spec_id` see the same thing.

### What the harness may not control

An evolution method writes harness files, and a method proposes whatever its model suggests. So the parts that can run code belong to the operator, not the harness:

- **Tools.** `ToolPolicy.allowed` lists the tools a harness may enable. A harness naming any other tool is refused with `harness_tools_allowed` before its first episode, and the gate never sees it.
- **MCP servers.** A harness enables servers by name. Their commands and arguments come only from `ToolPolicy.mcp_registry`, which the operator writes in the config. A harness `mcp.json` that tries to define a server itself is refused.
- **Limits.** `max_iterations` from the harness can lower the runner's limit, never raise it. `max_budget_usd` and the timeout are the runner's alone.

That split lets a method search over toolsets and server selections without being able to make the agent run an arbitrary command.

### What the runner reports back

`delivery.json` records, for each harness file, the route it took and the SHA-256 of what was loaded, plus the tools, MCP servers, and limits in effect, and which skills actually activated during the conversation. `OpenHandsDeliveryProbe` counts a file as delivered only when the hash matches. Delivered is not the same as used: a keyword-triggered skill can be delivered and never fire, and `activated_skills` is how to tell.

`transcript.jsonl` has every OpenHands event, one per line: each tool call and its observation, every message, every activated skill. This is the trace a reflective method should read. GEPA's current reflection step reads only `final_output` and the grade's feedback (`honing/harness/methods/gepa/method.py`, `_reflect`); pointing it at this transcript is the obvious next change for any LLM agent.

## OfficeQA

`honing.backends.officeqa` makes OfficeQA a task suite. The prompt is the question plus the protocol the grader needs: the documents are in `corpus/`, and the answer goes in `<FINAL_ANSWER>` tags. Search advice belongs in the harness, as a skill. `OfficeQAGrader` scores with OfficeQA's own `reward.py`, vendored unchanged at a pinned commit, at tolerance 0. Its feedback says what the agent answered and which bulletin holds the figure, and by default not the reference answer, so a reflective method cannot write answers into a skill.

The scripted method in `configs/e2e/openhands_officeqa.json` makes two moves: add a Treasury Bulletin search skill, then add the `grep` and `glob` tools. Each is gated on the same three questions. With three questions and a stochastic agent, a gate decides nothing statistically; the run shows the loop working, not that either edit helps.

## Known gaps

- **A hosted model has no proper home in core.** A release's master must be safetensors. `honing.backends.hosted` gives an API model a stand-in snapshot (location `hosted://...`, checksum from the endpoint and model name) so harness moves can be gated against it. Training, pruning, or quantizing it is meaningless, and nothing stops a config from scheduling one. This needs its own core issue.
- **`PolicyEndpoint.runtime` has no OpenAI-compatible value.** The runner uses `sglang`, which already means "speaks OpenAI chat completions" for the stand-in policy server.
- **The agent runs on the host.** The worker uses OpenHands' local workspace: the terminal tool runs commands as the current user in `work/`. Mounts are symlinks and are not write-protected. There is no Docker on the cluster; a sandbox (#16) should replace this before a method is let loose on the toolset.
- **`command`, `hook`, and `program` kinds are not supported.** OpenHands has hooks and programmable agents; mapping them waits for a method that needs them.
- **Stochastic episodes.** The gate compares one episode per task per side. With an LLM agent, a real comparison needs repeats per task.
