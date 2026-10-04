"""Run one OpenHands episode from a directory Honing's OpenHandsEpisodeRunner prepared.

Runs in the OpenHands environment and imports nothing from Honing. It applies
the agent plan in ``episode.json`` as given; what each harness file means was
decided on Honing's side (``honing/backends/openhands/harness.py``).

Exit code: 0 when the conversation finished, 2 when it stopped for another
reason (error, stuck, iteration or budget limit), 1 when it could not run.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")


def build_agent(episode: dict[str, Any], root: Path) -> tuple[Any, dict[str, dict[str, str]]]:
    """Builds the OpenHands agent an episode's plan describes.

    Args:
        episode: The parsed ``episode.json``.
        root: The episode directory.

    Returns:
        The agent, and each harness path it loaded with its route and the
        SHA-256 of the content read.
    """
    from openhands.sdk import LLM, Agent, AgentContext, Tool
    from openhands.sdk.skills.skill import Skill

    plan, policy, harness = episode["agent"], episode["policy"], root / "harness"
    loaded: dict[str, dict[str, str]] = {}

    def read(path: str) -> str:
        content = (harness / path).read_text(encoding="utf-8")
        loaded[path] = {"route": plan["routes"][path], "sha256": hashlib.sha256(content.encode()).hexdigest()}
        return content

    instructions = [read(path).strip() for path in plan["instructions"]]
    skills = []
    for skill in plan["skills"]:
        read(skill["path"])
        for resource in skill["resources"]:
            read(resource)
        loaded_skill = Skill.load(harness / skill["path"], skill_base_dir=harness, strict=False)
        skills.append(loaded_skill.model_copy(update={"name": skill["name"]}))
    for path in plan["files"]:
        target = root / "work" / ".harness" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(read(path), encoding="utf-8")
    for path, route in plan["routes"].items():
        if route in ("tools", "mcp"):
            read(path)

    # OpenHands registers a tool only when its module is imported. A name that is not a bundled tool must be
    # registered by one of the operator's tool modules.
    for module in plan["tool_modules"]:
        importlib.import_module(module)
    for name in plan["tools"]:
        if importlib.util.find_spec(f"openhands.tools.{name}") is not None:
            importlib.import_module(f"openhands.tools.{name}")

    prices = {key: float(policy[key]) for key in ("input_cost_per_token", "output_cost_per_token") if key in policy}
    llm = LLM(
        model=f"openai/{policy['model']}",
        base_url=policy["base_url"],
        api_key=os.environ.get(policy.get("api_key_env") or "") or "unused",
        usage_id="policy",
        num_retries=2,
        timeout=int(episode["limits"]["llm_timeout_seconds"]),
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
    tools = [Tool(name=name) for name in plan["tools"]]
    return Agent(llm=llm, tools=tools, mcp_config=plan["mcp_servers"], agent_context=context), loaded


def run(root: Path) -> int:
    """Runs one conversation and writes ``result.json``, ``transcript.jsonl``, and ``delivery.json``.

    Args:
        root: The episode directory.

    Returns:
        The process exit code.
    """
    from openhands.sdk import LocalConversation, MessageEvent
    from openhands.sdk.conversation.response_utils import get_agent_final_response

    episode = json.loads((root / "episode.json").read_text())
    agent, loaded = build_agent(episode, root)
    plan = episode["agent"]
    # LocalConversation directly: the Conversation factory does not forward max_budget_per_run.
    conversation = LocalConversation(
        agent=agent,
        workspace=str(root / "work"),
        persistence_dir=str(root / "openhands"),
        delete_on_close=False,
        visualizer=None,
        max_iteration_per_run=plan["max_iterations"],
        max_budget_per_run=episode["limits"]["max_budget_usd"],
    )
    status = "error"
    try:
        conversation.send_message(episode["task"]["prompt"])
        conversation.run()
        status = str(conversation.state.execution_status.value)
    finally:
        events = list(conversation.state.events)
        with (root / "transcript.jsonl").open("w", encoding="utf-8") as transcript:
            transcript.writelines(event.model_dump_json(exclude_none=True) + "\n" for event in events)
        activated = sorted(
            {name for event in events if isinstance(event, MessageEvent) for name in (event.activated_skills or [])}
        )
        delivery = {
            "files": loaded,
            "activated_skills": activated,
            "tools": plan["tools"],
            "mcp_servers": sorted(plan["mcp_servers"]),
        }
        (root / "delivery.json").write_text(json.dumps(delivery, indent=2) + "\n")
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
    """Runs the episode in ``argv[1]``, recording a failure in ``result.json`` if it cannot run.

    Args:
        argv: The command line.

    Returns:
        The process exit code.
    """
    root = Path(argv[1]).resolve()
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
