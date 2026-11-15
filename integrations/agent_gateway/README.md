# Agent gateway

Engineers keep their coding agent (Claude Code, Codex, or anything that speaks an OpenAI or Anthropic API) and point it at a model Honing produced (#188). The agent keeps its own sessions, history and settings; only the model behind it changes.

```text
Claude Code ─ /v1/messages ──┐
Codex ─────── /v1/responses ─┼─▶ LiteLLM proxy ── chat completions ──▶ SGLang serving a release,
other clients /v1/chat/... ──┘   (config from Honing)                  or any OpenAI-compatible API
```

`honing.backends.agent_gateway` writes a LiteLLM proxy config and starts the proxy from this directory's environment. Every request goes to one upstream model, whatever model name the agent asks for, and arrives there as chat completions, which is all SGLang serves.

## Setup

LiteLLM's proxy pulls in a web server and its own dependencies, so it runs in its own environment, installed from a lock. Honing imports nothing from it.

```bash
uv venv --python 3.12 integrations/agent_gateway/.venv
uv pip install --python integrations/agent_gateway/.venv/bin/python -r integrations/agent_gateway/requirements.txt
```

`requirements.txt` is a lock compiled from `requirements.in`; recompile it with the command at its top after changing the inputs.

## Serve a release

1. Serve the release's checkpoint with SGLang, from the environment `scripts/setup-sglang.sh` builds. The checkpoint is the `candidate.master.checkpoint.location` of the release's publish record:

   ```bash
   jq -r 'select(.op == "publish" and .release_id == "r0003") | .candidate.master.checkpoint.location' \
     <campaign>/releases/commits.jsonl
   $SGLANG_PYTHON -m sglang.launch_server --model-path <checkpoint> --served-model-name honed \
     --tool-call-parser <parser> --host 127.0.0.1 --port 30000
   ```

   Coding agents work by calling tools, and SGLang returns tool calls only with the `--tool-call-parser` that matches the model's family (`qwen25`, `llama3`, `deepseekv3`, and so on; see SGLang's docs). Without it, the agent receives the call as plain text and does nothing.

2. Start the gateway in front of it, with the key engineers will present:

   ```bash
   export HONING_GATEWAY_KEY=<a long random string>
   python -m honing.backends.agent_gateway --base-url http://127.0.0.1:30000/v1 --model honed --host 0.0.0.0
   ```

   It prints how to connect each agent. `--api-key-env` names the variable holding the upstream's key when it needs one, such as `OPENROUTER_API_KEY` for a model on OpenRouter. From Python, `AgentGateway.launch` does the same, and `Upstream.from_endpoint` takes a campaign's `PolicyEndpoint`.

To switch to a newer release, serve its checkpoint and restart the gateway. The agents need no change, because every model name they ask for reaches whatever the gateway serves.

## Connect an agent

**Claude Code** reads two variables:

```bash
export ANTHROPIC_BASE_URL=http://<gateway>:4000 ANTHROPIC_AUTH_TOKEN=<the gateway key>
claude
```

Its sessions, `CLAUDE.md`, skills and settings are unchanged; `claude --continue` resumes a session started on Anthropic's models before the switch. Claude Code still names its own models (`claude-sonnet-...`) in requests; the gateway answers each with the served model. While `ANTHROPIC_AUTH_TOKEN` is set, Claude Code does not load claude.ai connectors.

**Codex** takes a model provider, in `~/.codex/config.toml` or on the command line:

```toml
model = "honing"
model_provider = "honing"

[model_providers.honing]
name = "Honing"
base_url = "http://<gateway>:4000/v1"
env_key = "HONING_GATEWAY_KEY"
wire_api = "responses"
```

**Other clients** use `http://<gateway>:4000/v1` as an OpenAI base URL with the gateway key, and the model `honing`.

## Security

- Every request must carry the gateway key (`HONING_GATEWAY_KEY`, or the variable `--key-env` names). A request with no key is a 401; one with a wrong key is a 400 (`no_db_connection`), because LiteLLM looks up any other key in a database of virtual keys that the gateway does not run. Neither reaches the upstream.
- Keys stay in the environment. The config names the variables that hold them, and LiteLLM reads them itself, so neither the gateway's key nor the upstream's is written to disk.
- The gateway speaks plain HTTP. Serving other machines over an untrusted network needs TLS in front of it, from a reverse proxy.
- `--host 127.0.0.1`, the default, serves only this machine.

## Known gaps

- One model at a time. The gateway does not follow a campaign's head; restart it to serve a new release. Following the head needs a chain rebuilt from its `commits.jsonl`, which Honing does not do yet.
- The harness stays behind. A release's evolved instructions and skills are not exported into Claude Code's or Codex's own files.
- Usage and cost are not tracked per engineer: there are no virtual keys without LiteLLM's database.
- Tested with Claude Code 2.1.323 and Codex 0.167.0, against DeepSeek V4 Flash on OpenRouter and against a loopback upstream that serves chat completions only, as SGLang does. Not yet against an SGLang server on a GPU.
