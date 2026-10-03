"""Run one OpenHands episode from an episode directory that Honing's OpenHandsEpisodeRunner prepared.

This script runs inside the OpenHands environment (Python 3.12+, the SDK pinned in
requirements.txt) and imports nothing from Honing. Its whole interface is files:

    <episode>/episode.json     task, policy endpoint, limits, and the operator's tool and MCP allowlists
    <episode>/harness/<path>   the HarnessSpec's files, at their spec paths
    <episode>/work/            the agent's working directory (read-only data is linked in here)

and it writes

    <episode>/result.json      final_output, execution status, iterations, cost and tokens
    <episode>/transcript.jsonl one OpenHands event per line, the trace a reflective method reads
    <episode>/delivery.json    how each harness file reached the agent, and which skills fired
    <episode>/openhands/       OpenHands' own persisted conversation (base_state.json, events/)

How harness files map onto OpenHands, by kind:

- instructions: appended, in path order, to the agent's system message (AgentContext.system_message_suffix);
- skill: an OpenHands skill. ``.../<name>/SKILL.md`` is AgentSkills format; any other path is loaded the
  same way and named after its path. Frontmatter ``triggers`` make it keyword-activated, otherwise it is
  always on;
- settings at ``openhands/agent.json``: ``{"tools": [...], "max_iterations": n}``, the agent's tools,
  each of which must be on the operator's allowlist;
- settings at ``openhands/mcp.json``: ``{"enabled": [...]}``, names of MCP servers from the operator's
  registry. The harness chooses which servers to enable; their commands come only from the registry;
- any other settings file: copied into ``work/.harness/<path>`` for the agent to read.

Exit code: 0 when the conversation finished, 2 when it stopped for another reason (error, stuck,
iteration or budget limit), 1 when the episode could not run at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")

AGENT_SETTINGS = "openhands/agent.json"
MCP_SETTINGS = "openhands/mcp.json"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _skill(path: str, file: Path, root: Path) -> Any:
    from openhands.sdk.skills.skill import Skill

    skill = Skill.load(file, skill_base_dir=root, strict=False)
    if file.name.lower() != "skill.md":
        # Name a loose skill file after its path, so two files in one directory never share a name.
        skill = skill.model_copy(update={"name": path.removesuffix(".md").replace("/", "-")})
    return skill


def _tools(names: list[str]) -> list[Any]:
    """OpenHands registers a bundled tool when its module is imported, and only then."""
    import importlib
    import importlib.util

    from openhands.sdk import Tool

    unknown = [name for name in names if importlib.util.find_spec(f"openhands.tools.{name}") is None]
    if unknown:
        raise ValueError(f"tools {unknown} are not bundled OpenHands tools")
    for name in names:
        importlib.import_module(f"openhands.tools.{name}")
    return [Tool(name=name) for name in names]


def build(episode: dict[str, Any], root: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """The Agent, the conversation settings, and the delivery report for this episode's harness."""
    from openhands.sdk import LLM, Agent, AgentContext

    harness_dir, work = root / "harness", root / "work"
    files = sorted(episode["harness"], key=lambda f: f["path"])
    allowed_tools = list(episode["tools"]["allowed"])
    tools = list(episode["tools"]["default"])
    max_iterations = int(episode["limits"]["max_iterations"])
    registry = episode.get("mcp_registry", {})
    instructions: list[str] = []
    skills: list[Any] = []
    mcp_config: dict[str, Any] = {}
    delivery: dict[str, Any] = {}

    for entry in files:
        path, kind = entry["path"], entry["kind"]
        content = (harness_dir / path).read_text(encoding="utf-8")
        report = {"kind": kind, "sha256": _sha256(content)}
        if kind == "instructions":
            instructions.append(content.strip())
            report["route"] = "system_message"
        elif kind == "skill":
            skill = _skill(path, harness_dir / path, harness_dir)
            skills.append(skill)
            report["route"] = f"skill:{skill.name}"
            report["trigger"] = None if skill.trigger is None else skill.get_triggers()
        elif kind == "settings" and path == AGENT_SETTINGS:
            settings = json.loads(content)
            tools = list(settings.get("tools", tools))
            max_iterations = min(max_iterations, int(settings.get("max_iterations", max_iterations)))
            report["route"] = "agent_settings"
        elif kind == "settings" and path == MCP_SETTINGS:
            for name in json.loads(content).get("enabled", []):
                mcp_config[name] = registry[name]
            report["route"] = "mcp"
        else:
            target = work / ".harness" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            report["route"] = "workspace_file"
        delivery[path] = report

    # The runner checks the harness against the allowlists before starting this process; checking again
    # here keeps the worker safe to run on its own.
    refused = sorted(set(tools) - set(allowed_tools))
    if refused:
        raise ValueError(f"tools {refused} are not on the operator's allowlist {allowed_tools}")

    policy = episode["policy"]
    # Prices are passed explicitly: litellm has no price for most models behind an OpenAI-compatible
    # endpoint, and without one the cost reads 0 and max_budget_usd never stops a run.
    prices = {key: float(policy[key]) for key in ("input_cost_per_token", "output_cost_per_token") if policy.get(key)}
    llm = LLM(
        model=f"openai/{policy['model']}",
        base_url=policy["base_url"],
        api_key=os.environ.get(policy.get("api_key_env") or "", "") or "unused",
        usage_id="policy",
        num_retries=2,
        timeout=int(episode["limits"].get("llm_timeout_seconds", 120)),
        **prices,
    )
    context = AgentContext(
        skills=skills,
        system_message_suffix="\n\n".join(instructions) or None,
        load_user_skills=False,
        load_public_skills=False,
        load_project_skills=False,
        load_memory=False,
    )
    agent = Agent(llm=llm, tools=_tools(tools), mcp_config=mcp_config, agent_context=context)
    settings = {
        "tools": tools,
        "mcp_servers": sorted(mcp_config),
        "max_iterations": max_iterations,
        "max_budget_usd": episode["limits"].get("max_budget_usd"),
    }
    return agent, settings, delivery


def _event_line(event: Any) -> str:
    return event.model_dump_json(exclude_none=True)


def run(root: Path) -> int:
    from openhands.sdk import LocalConversation, MessageEvent
    from openhands.sdk.conversation.response_utils import get_agent_final_response

    episode = json.loads((root / "episode.json").read_text())
    agent, settings, delivery = build(episode, root)
    # LocalConversation directly: the Conversation factory does not forward max_budget_per_run.
    conversation = LocalConversation(
        agent=agent,
        workspace=str(root / "work"),
        persistence_dir=str(root / "openhands"),
        delete_on_close=False,
        visualizer=None,
        max_iteration_per_run=settings["max_iterations"],
        max_budget_per_run=settings["max_budget_usd"],
    )
    status = "error"
    try:
        conversation.send_message(episode["task"]["prompt"])
        conversation.run()
        status = str(conversation.state.execution_status.value)
    finally:
        events = list(conversation.state.events)
        with (root / "transcript.jsonl").open("w", encoding="utf-8") as transcript:
            for event in events:
                transcript.write(_event_line(event) + "\n")
        activated = sorted(
            {name for event in events if isinstance(event, MessageEvent) for name in (event.activated_skills or [])}
        )
        (root / "delivery.json").write_text(
            json.dumps({"files": delivery, "activated_skills": activated, **settings}, indent=2) + "\n"
        )
        metrics = conversation.conversation_stats.get_combined_metrics()
        usage = metrics.accumulated_token_usage
        result = {
            "final_output": get_agent_final_response(events),
            "status": status,
            "events": len(events),
            "cost_usd": metrics.accumulated_cost,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        }
        (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        conversation.close()
    return 0 if status == "finished" else 2


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else os.environ["HONING_EPISODE"]).resolve()
    try:
        return run(root)
    except Exception as error:
        if not (root / "result.json").exists():
            failure = {"final_output": "", "status": "failed", "error": f"{type(error).__name__}: {error}"}
            (root / "result.json").write_text(json.dumps(failure, indent=2) + "\n")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
