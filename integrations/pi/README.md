# pi in Honing

[pi](https://github.com/earendil-works/pi) is a minimal, MIT-licensed coding agent: a few built-in tools, a system prompt and skills that can be replaced, and an SDK. It is Honing's second language-model agent behind `EpisodeRunner` (#139), after [OpenHands](../openhands/README.md). Each episode is one pi session on one task, under one harness, against one policy.

## Setup

pi needs Node.js 22.19 or newer and runs in its own process. Honing imports nothing from it. Install it from the lock:

```bash
npm ci --ignore-scripts --prefix integrations/pi
```

`package.json` pins `@earendil-works/pi-coding-agent`, and `package-lock.json` pins everything under it. To move to a new pi, change the version in `package.json`, run `npm install --ignore-scripts --prefix integrations/pi`, and commit both files.

## Run it

With the stand-in policy server, no model or key. CI's `pi` job runs both on every pull request:

```bash
python scripts/e2e/check_runner.py --config configs/e2e/pi.json
python scripts/e2e/honing_pipeline.py --config configs/e2e/pi.json
```

Three OfficeQA questions with Qwen3 Coder 30B A3B on Amazon Bedrock, through Bedrock's OpenAI-compatible endpoint. Needs `HF_TOKEN` (the dataset is gated, approved automatically) and `AWS_BEARER_TOKEN_BEDROCK`, a Bedrock API key; a short-term one can be made from AWS credentials with [`aws-bedrock-token-generator`](https://github.com/aws/aws-bedrock-token-generator-python):

```bash
python -m integrations.officeqa.fetch --out integrations/officeqa/.cache --uids UID0002,UID0006,UID0016
python scripts/e2e/check_runner.py --config configs/e2e/pi_officeqa.json
python scripts/e2e/honing_pipeline.py --config configs/e2e/pi_officeqa.json
```

A local model served by SGLang works the same way: point `endpoints` at `honing.backends.sglang.e2e:endpoints` (the `sglang` runtime), and give the runner `model_settings` such as `{"contextWindow": 32768, "maxTokens": 4096}` when the model's context is shorter than pi's defaults (128k context, 16k output).

## How an episode runs

```text
Honing (Python 3.10+)                                  worker.mjs (Node.js, pi's SDK)
  plan_agent(spec, ToolPolicy)  ── refuse here, before any episode
  write episode dir: episode.json (task, model, plan),
    agent/models.json, harness/, work/
  start the worker in its own process group  ────────▶  build the session the plan describes
                                                         check what reached pi: delivery.json
                                                         prompt it once in work/
  kill the group on timeout or cancellation  ◀────────  write result.json, transcript.jsonl
  return EpisodeOutcome
```

What a harness file means is decided on Honing's side, in `honing/backends/pi/harness.py`: `plan_agent` turns a `HarnessSpec` and the operator's `ToolPolicy` into an `AgentPlan`, and the worker applies the plan without interpreting the files again.

| Kind | Path | Reaches the agent as |
| --- | --- | --- |
| `instructions` | `SYSTEM.md` | pi's system prompt, replacing pi's own |
| `instructions` | any other, e.g. `AGENTS.md` | appended, in path order, to the system prompt |
| `skill` | `<dir>/<name>/SKILL.md` | a pi skill named `<name>`; every other skill file under `<dir>/<name>/` is its resource |
| `skill` | any other path | a skill of its own, named after its path |
| `settings` | `pi/settings.json` | `{"defaultTools": [...], "defaultThinkingLevel": "..."}`, tools from the operator's policy |
| `settings` | any other path | a file in `work/.harness/` |

pi lists a skill to the model by its description and loads it only when it has one. A skill file with no frontmatter is described by its first line; one whose frontmatter has no `description` is refused. Skills need the `read` or `bash` tool, which pi uses to open them.

The worker builds the session itself, so pi's own discovery is off: no extensions, MCP servers, prompt templates, `AGENTS.md` or `CLAUDE.md` from the working directory or its parents, skills from `~/.pi` or `~/.agents`, or settings from disk. The agent sees what the harness says.

`agent/models.json` defines one provider, `honing`, for the policy's OpenAI-compatible endpoint (`api: openai-completions`), so a local model served by SGLang (the `sglang` runtime) and a model behind an API (the `openai_chat` runtime, through `honing.backends.llm_gateway`) both work. A Jev decision server (`jev_decision`) is refused. The model's API key is written as `"$NAME"`, which pi reads from the environment, so it is never on disk. With gateway prices, pi reports each episode's cost.

The worker does not inherit Honing's environment. It gets `PATH`, `HOME`, `LANG`, `TMPDIR`, `TERM`, `LC_*`, the policy's API key, pi's own settings (`PI_CODING_AGENT_DIR` for the episode, offline, no version check, no telemetry), and whatever the runner's `environment` adds. A proxy or a custom CA bundle is not inherited either; name the variables in the runner's `inherit_environment`.

## What the runner reports

`delivery.json` lists each harness file that reached pi, with its route and the SHA-256 of the content read. The worker checks pi's own session before the first model call: an instruction file's text must be in pi's system prompt, a skill must be listed there, and the tools from `pi/settings.json` must be the session's. `PiDeliveryProbe` counts a file as delivered only when that hash matches.

`transcript.jsonl` has every pi session event, as `pi --mode json` writes them: the system message pi sent, messages, tool calls and results, retries. `result.json` has the final answer, the status, and the tokens and cost.

An episode that does not finish scores 0 instead of stopping the campaign. A timeout kills the worker and the processes its tools started (exit code -9). A session pi aborted, or one whose provider kept failing after pi's retries, ends with exit code 2. Either way `final_output` starts with the reason, so the grader's feedback says what went wrong. The runner does no scoring; the task's `TaskGrader` does.

## Known gaps

- The agent runs on the host, as the current user, in `work/`. Each episode gets its own copy of every mount; with `mount_mode` set to `link`, mounts are symlinks shared by every episode and not write-protected. A sandbox (#16) should come before a method searches over toolsets.
- pi has no step or cost limit of its own; `timeout_seconds` is the only bound on an episode.
- `command`, `hook`, and `program` files are refused, and so are pi extensions, MCP servers, and prompt templates.
