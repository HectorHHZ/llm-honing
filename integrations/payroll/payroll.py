"""The attendance-payroll-audit task family from slm-harness-adaptation, as a Honing runner, grader, and tasks.

The agent is upstream's: an OpenHands software-agent-sdk ``Agent`` with the
terminal and file-editor tools, run in a ``DockerWorkspace`` from the task image
(see ``Dockerfile``), as ``src/runner.py:run_single_instance_agentic`` builds it.
The harness's ``instructions`` files are the agent's system prompt file and its
``skill`` files are the agent context's skills, mounted where upstream mounts them.

Workspace setup and scoring are upstream's own modules
(``src/task_setups/attendance_payroll_audit_s2l.py`` and
``src/task_evals/attendance_payroll_audit_s2l.py``), loaded from a checkout of
https://github.com/malusamayo/slm-harness-adaptation without running its
package ``__init__``. Ground truth is written beside the workspace
(``<workspace>_logs/groundtruth``), outside what the container mounts.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import platform
import sys
import time
import traceback
import types
import uuid
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from openhands.sdk import LLM, Agent, AgentContext, Conversation, Tool
from openhands.sdk.skills import Skill
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool
from openhands.workspace import DockerWorkspace

from honing.core import EpisodeOutcome, Grade, HarnessFile, HarnessSpec, PolicyRef, RunnerSupport, Task

AGENT = "openhands_sdk"
INSTRUCTIONS_PATH = "prompt.md"
DATA_FILE = "data/attendance_payroll_audit_s2l_extra_high.json"
SEED_PROMPT_FILE = "tasks/attendance_payroll_audit_s2l/prompts/default.md"
SERVER_IMAGE = "attendance_payroll_audit_s2l:latest"

# The task workspace sits where upstream's construct_docker_workspace mounts it. Upstream mounts the prompt file and
# each skill folder on their own; here the whole harness folder is one read-only mount, because Docker Desktop's file
# sharing fails under load with many single-file mounts ("mounts denied: EOF", then hung `docker run`s).
CONTAINER_WORKSPACE = "/workspace/project"
CONTAINER_HARNESS = "/workspace/harness"
# Where boto3 looks for credentials by default: the image's HOME is /workspace.
CONTAINER_AWS = "/workspace/.aws"
# How many times an episode tries to start its container before it is graded as failed.
CONTAINER_START_ATTEMPTS = 3


def load_upstream(repo: Path) -> tuple[types.ModuleType, types.ModuleType]:
    """Upstream's setup and eval modules, imported with ``src`` and its subpackages as bare namespaces."""
    for name, path in (
        ("src", repo / "src"),
        ("src.task_setups", repo / "src/task_setups"),
        ("src.task_evals", repo / "src/task_evals"),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module
    setup = importlib.import_module("src.task_setups.attendance_payroll_audit_s2l")
    evaluate = importlib.import_module("src.task_evals.attendance_payroll_audit_s2l")
    return setup, evaluate


def load_examples(repo: Path) -> dict[str, dict[str, Any]]:
    examples = json.loads((repo / DATA_FILE).read_text(encoding="utf-8"))
    return {example["id"]: example for example in examples}


def pick_tasks(examples: Mapping[str, Mapping[str, Any]], per_template: int, offset: int = 0) -> tuple[Task, ...]:
    """``per_template`` examples of every template, starting at the ``offset``-th one of each, in data order."""
    seen: dict[str, int] = {}
    chosen = []
    for example in examples.values():
        index = seen.get(example["template_id"], 0)
        seen[example["template_id"]] = index + 1
        if offset <= index < offset + per_template:
            chosen.append(Task(task_id=example["id"], prompt=example["prompt"]))
    return tuple(chosen)


def seed_harness(repo: Path) -> HarnessSpec:
    """Upstream's ``default`` system prompt."""
    text = (repo / SEED_PROMPT_FILE).read_text(encoding="utf-8")
    return HarnessSpec(agent=AGENT, files=((INSTRUCTIONS_PATH, HarnessFile(kind="instructions", content=text)),))


class OpenHandsEpisodeRunner:
    """Satisfies ``EpisodeRunner`` by running upstream's OpenHands agent on one task in its own container.

    ``model`` is the litellm model name the agent calls from inside the
    container. ``aws_credentials`` is a folder holding a shared-credentials
    file named ``credentials``, mounted read-only where boto3 finds it, so
    secrets never appear on a ``docker run`` command line, where any local
    process or log could read them;
    ``forward_env`` names the non-secret host variables the container gets,
    such as the region. A conversation still running after ``timeout_seconds``
    is stopped; the workspace is graded as it stands.
    """

    def __init__(
        self,
        *,
        examples: Mapping[str, Mapping[str, Any]],
        setup: types.ModuleType,
        workspaces: Path,
        model: str,
        forward_env: Sequence[str],
        aws_credentials: Path | None = None,
        server_image: str = SERVER_IMAGE,
        concurrency: int = 8,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.support = RunnerSupport(
            agents=frozenset({AGENT}),
            file_kinds=frozenset({"instructions", "skill"}),
            programmable_loop=False,
            local_policy_only=False,
        )
        self._examples = examples
        self._setup = setup
        self._workspaces = workspaces
        self._model = model
        self._forward_env = list(forward_env)
        self._aws_credentials = aws_credentials
        self._image = server_image
        self._concurrency = concurrency
        self._timeout = timeout_seconds
        self._slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        # One semaphore per event loop: a campaign runs each gate in its own asyncio.run.
        slots = self._slots.setdefault(asyncio.get_running_loop(), asyncio.Semaphore(self._concurrency))
        async with slots:
            return await asyncio.to_thread(self._run, spec, task, policy)

    def _run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        name = f"{task.task_id}--{spec.spec_id[:12]}--{uuid.uuid4().hex[:8]}"
        workspace = self._workspaces / name
        log_dir = self._workspaces / f"{name}_logs"
        harness_dir = self._workspaces / f"{name}_harness"
        workspace.mkdir(parents=True)
        self._setup.setup_workspace(str(workspace), str(log_dir), dict(self._examples[task.task_id]))

        skills = self._render(spec, harness_dir)
        volumes = [f"{workspace.resolve()}:{CONTAINER_WORKSPACE}", f"{harness_dir.resolve()}:{CONTAINER_HARNESS}:ro"]
        if self._aws_credentials is not None:
            volumes.append(f"{self._aws_credentials.resolve()}:{CONTAINER_AWS}:ro")
        agent = Agent(
            llm=LLM(model=self._model, usage_id="agent"),
            tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
            system_prompt_filename=f"{CONTAINER_HARNESS}/prompt.md",
            agent_context=AgentContext(skills=skills),
        )
        exit_code, error, events = 0, None, []
        # Docker Desktop occasionally refuses a container's mounts ("mounts denied: EOF"). A container that never
        # started has not run the agent, so it is retried rather than graded; a failure after it started is graded.
        for attempt in range(1, CONTAINER_START_ATTEMPTS + 1):
            started = False
            try:
                with DockerWorkspace(
                    working_dir=CONTAINER_WORKSPACE,
                    server_image=self._image,
                    platform="linux/arm64" if platform.machine().lower() in ("arm64", "aarch64") else "linux/amd64",
                    volumes=volumes,
                    forward_env=self._forward_env,
                ) as docker_workspace:
                    started = True
                    conversation = Conversation(agent=agent, workspace=docker_workspace)
                    try:
                        conversation.send_message(task.prompt)
                        conversation.run(timeout=self._timeout)
                    except Exception as caught:  # a timeout or API failure ends the episode; the workspace is graded
                        exit_code, error = 1, f"{type(caught).__name__}: {caught}"
                        traceback.print_exc()
                    finally:
                        events = [event.model_dump(mode="json") for event in conversation.state.events]
                        conversation.close()
                break
            except Exception as caught:  # the container did not start, or stopping it failed
                exit_code, error = 1, f"{type(caught).__name__}: {caught}"
                traceback.print_exc()
                if started or attempt == CONTAINER_START_ATTEMPTS:
                    break
                time.sleep(10 * attempt)

        events = [event for event in events if event.get("kind") != "ConversationStateUpdateEvent"]
        trace = log_dir / "trace.json"
        trace.write_text(json.dumps({"events": events, "error": error}, indent=2), encoding="utf-8")
        return EpisodeOutcome(
            task_id=task.task_id,
            spec_id=spec.spec_id,
            policy_checksum=policy.checkpoint.checksum,
            exit_code=exit_code,
            final_output=error or _final_message(events),
            transcript_location=str(trace),
            workspace_location=str(workspace),
        )

    @staticmethod
    def _render(spec: HarnessSpec, harness_dir: Path) -> list[Skill]:
        """Write the system prompt and skill files into the harness folder; return the skills, sourced in-container."""
        harness_dir.mkdir(parents=True)
        (harness_dir / "prompt.md").write_text(
            "\n\n".join(file.content for _, file in spec.files if file.kind == "instructions"), encoding="utf-8"
        )
        skills = []
        for path, file in spec.files:
            if file.kind != "skill":
                continue
            source = harness_dir / "skills" / path
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(file.content, encoding="utf-8")
            skill = Skill.load(source, strict=False)
            skill.source = f"{CONTAINER_HARNESS}/{source.relative_to(harness_dir).as_posix()}"
            skills.append(skill)
        return skills


def _final_message(events: Sequence[Mapping[str, Any]]) -> str:
    """The last message or observation text, as upstream's save_conversation_trace takes it."""
    if not events:
        return ""
    last = events[-1]
    try:
        if last["kind"] == "ObservationEvent":
            return str(last["observation"]["content"][0]["text"])
        if last["kind"] == "MessageEvent":
            return str(last["llm_message"]["content"][0]["text"])
    except (KeyError, IndexError, TypeError):
        pass
    return ""


class PayrollGrader:
    """Satisfies ``TaskGrader`` with upstream's checkpoint eval: the weighted share of deliverables that are right."""

    def __init__(self, examples: Mapping[str, Mapping[str, Any]], evaluate: types.ModuleType) -> None:
        self._examples = examples
        self._evaluate = evaluate

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        if outcome.workspace_location is None:
            return Grade(score=0.0, feedback="the episode left no workspace")
        result = await asyncio.to_thread(
            self._evaluate.run_single_instance_eval, outcome.workspace_location, dict(self._examples[task.task_id])
        )
        feedback = str(result["feedback"])
        if outcome.exit_code != 0:
            feedback += f" (the agent stopped with an error: {outcome.final_output[:300]})"
        return Grade(score=float(result["score"]), feedback=feedback)


def runner(
    *,
    repo: Path,
    examples: Mapping[str, Mapping[str, Any]],
    workspaces: Path,
    model: str,
    forward_env: Sequence[str],
    aws_credentials: Path | None,
    concurrency: int,
    timeout_seconds: float,
) -> OpenHandsEpisodeRunner:
    setup, _ = load_upstream(repo)
    return OpenHandsEpisodeRunner(
        examples=examples,
        setup=setup,
        workspaces=workspaces,
        model=model,
        forward_env=forward_env,
        aws_credentials=aws_credentials,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
    )


def grader(repo: Path, examples: Mapping[str, Mapping[str, Any]]) -> PayrollGrader:
    _, evaluate = load_upstream(repo)
    return PayrollGrader(examples, evaluate)


__all__: Sequence[str] = [
    "AGENT",
    "OpenHandsEpisodeRunner",
    "PayrollGrader",
    "grader",
    "load_examples",
    "load_upstream",
    "pick_tasks",
    "runner",
    "seed_harness",
]
