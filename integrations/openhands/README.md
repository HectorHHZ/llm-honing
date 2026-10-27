# OpenHands in Honing

OpenHands is the first language-model agent behind Honing's `EpisodeRunner` (#69). Each episode is one OpenHands conversation on one task, under one harness, against one policy.

## Setup

OpenHands needs Python 3.12 and pins its own litellm, so it runs in its own environment. Honing imports nothing from it.

```bash
uv venv --python 3.12 integrations/openhands/.venv
uv pip install --python integrations/openhands/.venv/bin/python -r integrations/openhands/requirements.txt
```

`requirements.txt` is a lock compiled from `requirements.in`; recompile it with the command at its top after changing the inputs. The lock matters: the SDK accepts any litellm from 1.93.0, and some later versions fail on the first model call.

## Run it

With the stand-in policy server, no model or key:

```bash
python scripts/e2e/check_runner.py --config configs/e2e/openhands.json
python scripts/e2e/honing_pipeline.py --config configs/e2e/openhands.json
```

Three OfficeQA questions with DeepSeek V4 Flash on OpenRouter. Needs `HF_TOKEN` (the dataset is gated, approved automatically) and `OPENROUTER_API_KEY`:

```bash
python -m integrations.officeqa.fetch --out integrations/officeqa/.cache --uids UID0002,UID0006,UID0016
python scripts/e2e/check_runner.py --config configs/e2e/openhands_officeqa.json
python scripts/e2e/honing_pipeline.py --config configs/e2e/openhands_officeqa.json
```

An episode takes from half a minute to a few minutes and costs half a cent to four cents.

## How an episode runs

```text
Honing (Python 3.10+)                                  worker.py (OpenHands environment)
  plan_agent(spec, ToolPolicy)  ── refuse here, before any episode
  write episode dir: episode.json (task, endpoint, plan), harness/, work/
  start the worker in its own process group  ────────▶  build the agent the plan describes
                                                         run one LocalConversation in work/
  kill the group on timeout or cancellation  ◀────────  write result.json, transcript.jsonl,
  return EpisodeOutcome                                  delivery.json, openhands/
```

What a harness file means is decided on Honing's side, in `honing/backends/openhands/harness.py`: `plan_agent` turns a `HarnessSpec` and the operator's `ToolPolicy` into an `AgentPlan`, and the worker applies the plan without interpreting the files again. The rules are therefore checked before an episode and tested without OpenHands.

| Kind | Path | Reaches the agent as |
| --- | --- | --- |
| `instructions` | any, e.g. `AGENTS.md` | appended, in path order, to the system message |
| `skill` | `<dir>/<name>/SKILL.md` | an AgentSkills skill named `<name>`; every other skill file under `<dir>/<name>/` is its resource (scripts, references) |
| `skill` | any other path | a skill of its own, named after its path |
| `settings` | `openhands/agent.json` | `{"tools": [...], "max_iterations": n}` |
| `settings` | `openhands/mcp.json` | `{"enabled": [...]}`, server names from the operator's registry |
| `settings` | any other path | a file in `work/.harness/` |

OpenHands' own discovery is off: no user skills from `~/.openhands`, no public skill registry, no `AGENTS.md` from the working directory, no memory. The agent sees what the harness says.

The worker does not inherit Honing's environment. It gets `PATH`, `HOME`, `LANG`, `TMPDIR`, `TERM`, `LC_*`, `OPENHANDS_*`, the policy's API key, and whatever the runner's `environment` adds, so the agent's terminal cannot print `HF_TOKEN` or cloud credentials. `episode.json` names the key's variable, never its value, and a missing key is refused before the agent starts.

## Bringing your own skills, MCP servers, and tools

What the harness may change and what the operator controls are split on purpose, because an evolution method writes harness files and proposes whatever its model suggests.

**Skills belong to the harness.** Bring a local skills directory into a seed harness, keeping each skill's layout:

```json
"harness_seed": {"factory": "honing.backends.openhands.e2e:seed_harness",
                 "args": {"directory": "~/my-harness", "skills_dirs": ["~/.claude/skills"]}}
```

`directory` is a harness on disk (top-level `*.md` as instructions, `skills/`, `openhands/*.json`); `skills_dirs` adds directories of `<name>/SKILL.md` skills under `skills/`. From Python, `harness_from_directory` and `skill_files` do the same.

**MCP servers belong to the operator; the harness picks among them.** Point the runner at an existing `{"mcpServers": {...}}` file (the format Claude Code, Cursor, and OpenHands write), or define servers inline:

```json
"runner": {"factory": "honing.backends.openhands.e2e:runner",
           "args": {"mcp_registry_file": "~/.cursor/mcp.json"}}
```

A harness then enables servers by name, for example with `"mcp_servers": ["fetch"]` in `seed_harness`. A harness that tries to define a server's command itself is refused.

**Tools belong to the operator; the harness picks among them.** `tools` is the allowlist and `default_tools` what an agent gets when its harness names none. A custom OpenHands tool goes in a module that calls `openhands.sdk.register_tool`; list it under `tool_modules` (importable in the OpenHands environment) and its name under `tools`.

A harness asking for a tool or server outside the policy is refused with `harness_tools_allowed`; a malformed settings file or two skills with one name, with `harness_well_formed`. Either way no agent starts: the episode's outcome has exit code 2 and the reason as its output, so the grader scores it 0 and the gate rejects the candidate like any other, and the campaign goes on.

## What the runner reports

`delivery.json` lists each harness file the worker loaded, with its route and the SHA-256 of the content read, plus the tools and MCP servers in effect and which skills activated. `OpenHandsDeliveryProbe` counts a file as delivered only when that hash matches. A keyword-triggered skill can be delivered and never fire; `activated_skills` tells them apart.

`transcript.jsonl` has every OpenHands event: tool calls, observations, messages. It is the trace a reflective method should read; GEPA's reflection step currently reads only `final_output` and the grade's feedback.

## OfficeQA

`integrations/officeqa` is OfficeQA as an example task suite, kept outside the `honing` package. The prompt carries what the grader needs (the documents are in `corpus/`, the answer goes in `<FINAL_ANSWER>` tags); search advice belongs in the harness. `OfficeQAGrader` scores with OfficeQA's own `reward.py`, which the fetcher downloads at a pinned commit and checks by hash, at tolerance 0, and its feedback gives the agent's answer, the reference answer, and the source bulletin, so a reflective method can see exactly what went wrong. A method can then memorize answers into a skill and pass the next gate on the same questions; set `reveal_answer` to false, or gate on held-out questions, when that matters. `fetch.py` downloads the chosen questions, only their bulletins, and the scorer.

The scripted method (`integrations/officeqa/scripted.py`) in `configs/e2e/openhands_officeqa.json` proposes two edits: the skill in `integrations/officeqa/skills/treasury-bulletins/`, then the `grep` and `glob` tools. With three questions and one episode each, a gate mostly measures sampling noise; the config demonstrates the loop, not that either edit helps.

## Known gaps

- Core cannot represent a model it only reaches through a URL. `honing.backends.llm_gateway` gives one a stand-in snapshot so harness moves can be gated; nothing stops a config from scheduling a train move on it.
- The agent runs on the host, as the current user, in `work/`. Mounts are symlinks and not write-protected. A sandbox (#16) should come before a method searches over toolsets.
- `command`, `hook`, and `program` files are refused.
- A skill's frontmatter is parsed by OpenHands, so a malformed skill fails its episode rather than being refused up front.
