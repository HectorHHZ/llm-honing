"""A stand-in for the ``codex`` CLI, for testing ``diagnose.py`` end to end with no model and no network.

Supports ``--version``, ``login status``, ``features list`` and ``... exec ... --json --output-schema S -``:
it reads the prompt from stdin and prints Codex-style JSONL events (one reasoning item, one agent message
holding a schema-valid diagnosis, one ``turn.completed`` with token usage). The diagnosis is a fixed
heuristic over the supplied evidence, so scores are reproducible and non-trivial:

  - "selected and executed differ" or "requested and sent input differ" in the trace -> controller;
  - probe results with a changed standard re-render -> harness;
  - probe results whose reference and alternate probabilities differ by more than 0.01 -> weights
    (this over-attributes on controls, as Griddly's real probe run did);
  - otherwise uncertain.

``STUB_CODEX_MODE`` selects failure modes: ``invalid_once`` (first call per episode returns a reply that
breaks the schema), ``invalid`` (always), ``tool`` (emits a command execution item), ``fail`` (exit 1).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def diagnosis(prompt: str) -> dict:
    payload = json.loads(prompt[prompt.index("\n\n{") + 2 :])
    trace = payload["trace"]
    if "selected and executed differ" in trace or "requested and sent input differ" in trace:
        return {
            "cause": "controller",
            "confidence": 0.9,
            "explanation": "The trace records a selected option or requested input that differs from what was executed.",
        }
    if "probe_results" in payload:
        rows = json.loads(payload["probe_results"])
        if any(row["standard_rerender_request_changed"] for row in rows):
            return {
                "cause": "harness",
                "confidence": 0.8,
                "explanation": "The standard re-render of a saved state differs from the logged request.",
            }
        for row in rows:
            ref, alt = row["same_request_reference_checkpoint"], row["same_request_alternate_checkpoint"]
            if ref and isinstance(next(iter(ref.values())), dict):  # several questions (contract 0.2.0)
                ref = {(q, o): p for q, values in ref.items() for o, p in values.items()}
                alt = {(q, o): p for q, values in alt.items() for o, p in values.items()}
            if max(abs(ref[key] - alt[key]) for key in ref) > 0.01:
                return {
                    "cause": "weights",
                    "confidence": 0.6,
                    "explanation": "The same request receives different probabilities under the two checkpoints.",
                }
    return {"cause": "uncertain", "confidence": 0.3, "explanation": "The trace alone shows no component evidence."}


def main() -> None:
    args = sys.argv[1:]
    if args == ["--version"]:
        print("codex-cli 0.0.0-stub")
        return
    if args[:2] == ["login", "status"]:
        print("Logged in (stub, no account)")
        return
    if args[:2] == ["features", "list"]:
        for name in (
            "shell_tool",
            "apps",
            "plugins",
            "hooks",
            "multi_agent",
            "multi_agent_v2",
            "skill_search",
            "unified_exec",
            "view_image",
        ):
            print(f"{name:40} stable             true")
        return
    if "exec" not in args or "--json" not in args or args[-1] != "-":
        print(f"stub codex: unsupported arguments {args}", file=sys.stderr)
        sys.exit(2)
    schema = Path(args[args.index("--output-schema") + 1])
    json.loads(schema.read_text(encoding="utf-8"))
    prompt = sys.stdin.read()
    mode = os.environ.get("STUB_CODEX_MODE", "")
    if mode == "fail":
        print("stub failure", file=sys.stderr)
        sys.exit(1)
    result = diagnosis(prompt)
    episode = json.loads(prompt[prompt.index("\n\n{") + 2 :])["episode_id"]
    marker = Path(os.environ.get("STUB_CODEX_STATE", "/nonexistent")) / f"{episode}.seen"
    if mode == "invalid" or (mode == "invalid_once" and marker.parent.is_dir() and not marker.exists()):
        if mode == "invalid_once":
            marker.write_text("1")
        result = {"cause": "everything", "confidence": 2}
    events = [
        {"type": "thread.started", "thread_id": "stub"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"id": "i0", "type": "reasoning", "text": "stub reasoning"}},
    ]
    if mode == "tool":
        events.append({"type": "item.completed", "item": {"id": "i1", "type": "command_execution", "command": "ls"}})
    events.append(
        {"type": "item.completed", "item": {"id": "i2", "type": "agent_message", "text": json.dumps(result)}}
    )
    events.append(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": len(prompt.split()),
                "cached_input_tokens": 0,
                "output_tokens": len(json.dumps(result).split()),
            },
        }
    )
    for event in events:
        print(json.dumps(event))


if __name__ == "__main__":
    main()
