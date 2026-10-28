"""Run GEPA end to end against real model APIs, and check what the run did from what it wrote (#100).

``configs/e2e/gepa_live.json`` names every component by import path, as
:mod:`honing.e2e.pipeline` does: a manifest suite graded by
:class:`~honing.harness.graders.CommandGrader`, the minimal agent in
``scripts/e2e/minimal_agent.py`` run by :class:`ManifestWorkspaceRunner`
against an OpenAI-compatible policy, and :class:`~honing.harness.methods.gepa.GEPAMethod`
reflecting through :class:`~honing.harness.models.LiteLLMChatModel`. The model,
data, training and compression planes are the walking skeleton's fakes, because
a campaign needs them and a harness move never touches them.

The run writes, under its work directory, the release chain, ``summary.json``,
``run.log``, and three logs the checks read: ``episodes.jsonl`` (one row per
agent process), ``reflections.jsonl`` (one row per reflection call) and
``faults.jsonl`` (one row per abandoned move). Each check in
:func:`check_run` passes or fails with the rows that show why, and the run exits 0 only
if every one passes. A ``logging`` block in the config also logs the run to
W&B, one step per move with its API calls and tokens (:mod:`honing.e2e.tracking`).

An unavailable policy (the agent exits 75 after its own retries) raises
:class:`PolicyUnavailableError`. The move is abandoned: the campaign has not
recorded or published anything for it, so GEPA's state and ``metric_calls`` are
as they were, and the episodes it ran are in ``faults.jsonl`` instead. A failed
reflection call is GEPA's own concern: no proposal, a log line, and only the
parent's episodes charged.
"""

from __future__ import annotations

import argparse
import contextvars
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from honing.core import (
    ChatMessage,
    ChatReply,
    ContractError,
    EpisodeOutcome,
    EvaluationResult,
    EvolveContext,
    Grade,
    HarnessFile,
    HarnessProposal,
    HarnessSpec,
    HoningError,
    PolicyEndpoint,
    PolicyRef,
    ProposeResult,
    ReleaseError,
    RunnerSupport,
    SelectionDecision,
    Task,
)
from honing.e2e.components import PipelineContext
from honing.e2e.pipeline import PipelineRun, _move, _step, build
from honing.e2e.tracking import (
    WANDB_KEY_ENV,
    ChainRecords,
    NullTracker,
    RunTracker,
    git_commit,
    run_config,
    start_tracker,
    step_values,
)
from honing.e2e.workspace_runner import WorkspaceEpisodeRunner
from honing.harness.graders import CommandGrader
from honing.harness.methods.gepa import Archive, GEPAConfig, GEPAMethod, extract_text
from honing.harness.models import ChatModelError, LiteLLMChatModel, completion_cost_usd
from honing.harness.tasks import TaskSuite, load_task_suite

_LOGGER = logging.getLogger(__name__)

POLICY_UNAVAILABLE = 75  # the minimal agent's exit code, EX_TEMPFAIL
RUN_LOG = "run_log"
# Loggers that could print a request; kept quiet whatever the host configures.
_QUIET_LOGGERS = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy", "litellm", "httpx", "openai", "wandb")


class PolicyUnavailableError(HoningError):
    """The agent could not reach its policy after its own retries: an infrastructure fault, not the agent's.

    Not a :class:`~honing.core.CampaignRuleError`, which GEPA would read as a
    runner refusing the harness, and not graded as a 0, which would put the
    outage into the gate's scores and the reflection prompt.
    """

    def __init__(self, task_id: str, kind: str, status: int | None) -> None:
        self.task_id = task_id
        self.kind = kind
        self.status = status
        super().__init__(f"the policy was unavailable for task {task_id!r}: {kind} (status {status})")


@dataclass
class RunLog:
    """What the run's components saw, row by row, for the checks and the cost report."""

    work_dir: Path
    move: int = 0
    phase: str = "parent"
    started: int = 0
    episodes: list[dict[str, Any]] = field(default_factory=list)
    reflections: list[dict[str, Any]] = field(default_factory=list)
    faults: list[dict[str, Any]] = field(default_factory=list)
    starts: dict[int, Mapping[str, object]] = field(default_factory=dict)
    ends: list[Mapping[str, object]] = field(default_factory=list)
    prompts: dict[str, str] = field(default_factory=dict)

    def write(self) -> None:
        for name, rows in (("episodes", self.episodes), ("reflections", self.reflections), ("faults", self.faults)):
            text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            (self.work_dir / f"{name}.jsonl").write_text(text, encoding="utf-8")


def _canonical(state: object) -> str:
    """A method state as comparable text; a published state is a read-only mapping, not a dict."""
    return json.dumps(state, sort_keys=True, default=dict)


def _run_log(context: PipelineContext) -> RunLog:
    run_log = context.shared.setdefault(RUN_LOG, RunLog(context.work_dir))
    if not isinstance(run_log, RunLog):
        raise TypeError(f"context.shared[{RUN_LOG!r}] is not a RunLog")
    return run_log


def _suite(context: PipelineContext, manifest: str) -> TaskSuite:
    suites = context.shared.setdefault("suites", {})
    assert isinstance(suites, dict)
    if manifest not in suites:
        suites[manifest] = load_task_suite(context.root / manifest)
    suite = suites[manifest]
    assert isinstance(suite, TaskSuite)
    return suite


# ---------------------------------------------------------------- the episode runner and grader

_WORKSPACE: contextvars.ContextVar[Path | None] = contextvars.ContextVar("workspace", default=None)


class ManifestWorkspaceRunner(WorkspaceEpisodeRunner):
    """The reference workspace runner, recording every agent process and raising on an unavailable policy.

    Each ``run`` is one agent process: its row in the run log holds the move,
    the phase (the GEPA parent's minibatch, the child's, or the gate), the
    agent's final output, and the policy calls it made, read from the
    ``usage.jsonl`` it appends to after every call, so an agent that is killed
    still reports what it spent.
    """

    def __init__(self, run_log: RunLog, **options: Any) -> None:
        super().__init__(**options)
        self._run_log = run_log

    def _prepare(self, spec: HarnessSpec, task: Task) -> Path:
        workspace = super()._prepare(spec, task)
        _WORKSPACE.set(workspace)
        return workspace

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        run_log = self._run_log
        run_log.started += 1
        row: dict[str, Any] = {
            "move": run_log.move,
            "phase": run_log.phase,
            "task_id": task.task_id,
            "spec_id": spec.spec_id,
            "exit_code": None,
            "final_output": "",
            "score": None,
            "feedback": None,
            "fault": None,
            "workspace": None,
        }
        run_log.episodes.append(row)
        run_log.prompts[task.task_id] = task.prompt
        token = _WORKSPACE.set(None)
        try:
            outcome = await super().run(spec, task, policy)
        finally:
            workspace = _WORKSPACE.get()
            _WORKSPACE.reset(token)
            row["workspace"] = None if workspace is None else str(workspace)
            row["usage"] = _usage(workspace)
        row["exit_code"] = outcome.exit_code
        row["final_output"] = outcome.final_output
        if outcome.exit_code == POLICY_UNAVAILABLE:
            error = _policy_error(Path(str(outcome.workspace_location)))
            if error is not None:
                row["fault"] = error
                raise PolicyUnavailableError(task.task_id, str(error.get("kind")), error.get("status"))
        return outcome


def _usage(workspace: Path | None) -> dict[str, int]:
    usage = {"calls": 0, "failed_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    path = None if workspace is None else workspace / "usage.jsonl"
    if path is None or not path.is_file():
        return usage
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        try:
            call = json.loads(line)
        except ValueError:
            continue
        usage["calls"] += 1
        usage["failed_calls"] += 0 if call.get("ok") else 1
        usage["prompt_tokens"] += int(call.get("prompt_tokens") or 0)
        usage["completion_tokens"] += int(call.get("completion_tokens") or 0)
    return usage


def _policy_error(workspace: Path) -> dict[str, Any] | None:
    try:
        error = json.loads((workspace / "result.json").read_text(encoding="utf-8")).get("policy_error")
    except (OSError, ValueError, AttributeError):
        return None
    return error if isinstance(error, dict) else None


class RecordingGrader:
    """A :class:`~honing.harness.graders.CommandGrader` that writes each grade into its episode's row."""

    def __init__(self, inner: CommandGrader, run_log: RunLog) -> None:
        self._inner = inner
        self._run_log = run_log

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        grade = await self._inner.grade(task, outcome)
        for row in reversed(self._run_log.episodes):
            if row["workspace"] == outcome.workspace_location:
                row["score"], row["feedback"] = grade.score, grade.feedback
                break
        return grade


# ---------------------------------------------------------------- the reflection model and the method


class RecordingChatModel:
    """Records every reflection call: the prompt, the reply or the error, tokens and cost.

    A call marks the end of the parent's minibatch: the episodes after it are the child's.
    """

    def __init__(self, inner: LiteLLMChatModel, run_log: RunLog) -> None:
        self._inner = inner
        self._run_log = run_log

    def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float) -> ChatReply:
        run_log = self._run_log
        row: dict[str, Any] = {"move": run_log.move, "prompt": messages[-1].content}
        run_log.reflections.append(row)
        run_log.phase = "child"
        try:
            reply = self._inner.complete(messages, max_tokens=max_tokens, temperature=temperature)
        except ChatModelError as error:
            row.update(ok=False, error_type=error.error_type, status=error.status_code, attempts=error.attempts)
            raise
        row.update(
            ok=True,
            reply=reply.text,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            cost_usd=reply.cost_usd,
        )
        return reply


class LiveGEPAMethod(GEPAMethod):
    """GEPA that remembers the state each harness move started from and the state it settled on."""

    def __init__(self, run_log: RunLog, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._run_log = run_log

    def propose(self, context: EvolveContext) -> ProposeResult:
        self._run_log.starts[self._run_log.move] = context.state
        result = super().propose(context)
        self._run_log.phase = "gate"
        if result.proposal is None:
            self._run_log.ends.append(result.state)
        return result

    def record(
        self,
        state: Mapping[str, object],
        proposal: HarnessProposal,
        evaluation: EvaluationResult,
        decision: SelectionDecision,
    ) -> Mapping[str, object]:
        settled = super().record(state, proposal, evaluation, decision)
        self._run_log.ends.append(settled)
        return settled


# ---------------------------------------------------------------- the policy's endpoint


class OpenAIEndpoints:
    """Satisfies :class:`~honing.core.PolicyEndpoints` with one hosted OpenAI-compatible API for every policy.

    The API's model stands in for the campaign's (fake) checkpoint, so every
    checkpoint resolves to it: records name the checkpoint, and the config and
    summary name the model.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self._endpoint = PolicyEndpoint(
            base_url=base_url, tokenizer_location="remote", served_model=model, runtime="openai_chat"
        )

    @property
    def base_url(self) -> str:
        return self._endpoint.base_url

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        return self._endpoint


# Each provider's own API, for the providers the e2e configs use. litellm's price map prices
# these; a policy served anywhere else (a local server, a stub) has no known cost.
_PROVIDER_API_BASES = {"openai": "https://api.openai.com/v1"}


def _url_key(url: str) -> tuple[str, str, str, str, str]:
    parts = urlsplit(url.strip())
    return parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, parts.fragment


def pricing_api_base(model: str, base_url: str) -> str | None:
    """``base_url`` as :func:`~honing.harness.models.completion_cost_usd` should see it.

    ``None`` when it is ``model``'s provider's own API, which the price map prices;
    the URL itself otherwise, so the cost is reported as unknown, as the chat
    client does for an ``api_base``. A provider not listed is never its default.
    """
    default = _PROVIDER_API_BASES.get(model.split("/", 1)[0])
    if default is not None and _url_key(base_url) == _url_key(default):
        return None
    return base_url


# ---------------------------------------------------------------- factories named in the config


def suite_tasks(context: PipelineContext, manifest: str, split: str) -> tuple[Task, ...]:
    return _suite(context, manifest).tasks(split)  # type: ignore[arg-type]


def harness_seed(
    context: PipelineContext, directory: str, files: Mapping[str, str], agent: str = "minimal"
) -> HarnessSpec:
    """The harness in ``directory``: ``files`` maps each path in it to its kind."""
    root = context.root / directory
    return HarnessSpec(
        agent=agent,
        files=tuple(
            (path, HarnessFile(kind=kind, content=(root / path).read_text(encoding="utf-8")))  # type: ignore[arg-type]
            for path, kind in sorted(files.items())
        ),
    )


def model_name(model: str | Mapping[str, str]) -> str:
    """The provider's own id for a model pin ``{"provider": ..., "name": ...}``, or the id itself."""
    return model if isinstance(model, str) else str(model["name"])


def litellm_model(model: str | Mapping[str, str]) -> str:
    """litellm's ``provider/name`` string for a model pin, or the string itself."""
    return model if isinstance(model, str) else f"{model['provider']}/{model['name']}"


def openai_endpoints(context: PipelineContext, base_url: str, model: str | Mapping[str, str]) -> OpenAIEndpoints:
    return OpenAIEndpoints(base_url, model_name(model))


def manifest_runner(
    context: PipelineContext,
    command: Sequence[str],
    manifest: str,
    api_key_env: str,
    agents: Sequence[str] = ("minimal",),
    file_kinds: Sequence[str] = ("instructions", "skill"),
    concurrency: int = 4,
    timeout_seconds: float = 180.0,
    agent_timeout_seconds: float = 60.0,
    agent_max_attempts: int = 3,
    agent_max_tokens: int = 2048,
) -> ManifestWorkspaceRunner:
    """The workspace runner around ``command``, starting each task from its manifest workspace."""
    suite = _suite(context, manifest)
    names = {"python": sys.executable, "root": str(context.root)}
    return ManifestWorkspaceRunner(
        _run_log(context),
        command=[part.format(**names) for part in command],
        workspaces=context.work_dir / "workspaces",
        endpoints=context.component("endpoints"),
        support=RunnerSupport(
            agents=frozenset(agents),
            file_kinds=frozenset(file_kinds),  # type: ignore[arg-type]
            programmable_loop=False,
        ),
        environment={
            "HONING_POLICY_API_KEY_ENV": api_key_env,
            "HONING_AGENT_TIMEOUT_SECONDS": str(agent_timeout_seconds),
            "HONING_AGENT_MAX_ATTEMPTS": str(agent_max_attempts),
            "HONING_AGENT_MAX_TOKENS": str(agent_max_tokens),
        },
        forward_environment=(api_key_env,),
        seed_workspace=lambda task: suite.entry(task.task_id).workspace,
        concurrency=concurrency,
        timeout_seconds=timeout_seconds,
    )


def command_grader(context: PipelineContext, manifest: str, timeout_seconds: float = 120.0) -> RecordingGrader:
    return RecordingGrader(
        CommandGrader(_suite(context, manifest), timeout_seconds=timeout_seconds), _run_log(context)
    )


def gepa_method(
    context: PipelineContext,
    manifest: str,
    reflection: Mapping[str, Any],
    gepa: Mapping[str, Any],
    validation_split: str = "val",
) -> LiveGEPAMethod:
    """GEPA validating on the suite's ``validation_split``, which must be the gate's tasks in order."""
    run_log = _run_log(context)
    validation = [task.task_id for task in _suite(context, manifest).tasks(validation_split)]  # type: ignore[arg-type]
    chat = RecordingChatModel(LiteLLMChatModel(**{**reflection, "model": litellm_model(reflection["model"])}), run_log)
    return LiveGEPAMethod(run_log, chat, GEPAConfig(**gepa), validation)


# ---------------------------------------------------------------- the run


@dataclass(frozen=True)
class LiveOptions:
    max_consecutive_faults: int = 3
    policy_price_model: str | None = None  # litellm's name for the policy, to price its tokens
    policy_api_base: str | None = None  # from pricing_api_base; set means the policy's cost is unknown


def live_options(config: Mapping[str, Any], endpoints: object) -> LiveOptions:
    """The run's options from its config, with the policy priced only behind its provider's own API."""
    live = config.get("live", {})
    price_model = litellm_model(config["model"]) if "model" in config else None
    # Only a policy behind OpenAIEndpoints has a URL to check; any other is not priced.
    if price_model is None or not isinstance(endpoints, OpenAIEndpoints):
        return LiveOptions(max_consecutive_faults=int(live.get("max_consecutive_faults", 3)))
    return LiveOptions(
        max_consecutive_faults=int(live.get("max_consecutive_faults", 3)),
        policy_price_model=price_model,
        policy_api_base=pricing_api_base(price_model, endpoints.base_url),
    )


FIRST_PUBLISH = "first_publish"


def check_schedule(schedule: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a schedule ``run_live`` cannot run, before anything runs.

    Entries: ``{"kind": "harness", "repeat": n}`` (``repeat`` defaults to 1);
    ``{"kind": "harness_until_publish", "max_moves": n}``, harness moves until
    one publishes or ``n`` have run; and ``{"kind": "rollback", "to": id}``,
    where ``to`` is a release id or ``"first_publish"``, the first harness
    release this run published.
    """
    for number, entry in enumerate(schedule, start=1):
        kind = entry.get("kind")
        count = {"harness": entry.get("repeat", 1), "harness_until_publish": entry.get("max_moves")}.get(str(kind))
        if kind == "rollback":
            if not isinstance(entry.get("to"), str):
                raise ContractError(f"schedule step {number}: a rollback names a release id or {FIRST_PUBLISH!r}")
        elif kind in ("harness", "harness_until_publish"):
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                key = "repeat" if kind == "harness" else "max_moves"
                raise ContractError(f"schedule step {number}: {key} must be an integer >= 1, got {count!r}")
        else:
            raise ContractError(f"schedule step {number}: a live run has harness moves and rollbacks, not {kind!r}")


class _AbortedError(Exception):
    """Enough moves in a row were abandoned that the run stops."""


def run_live(
    pipeline: PipelineRun,
    schedule: Sequence[Mapping[str, Any]],
    options: LiveOptions,
    tracker: RunTracker | None = None,
) -> dict[str, Any]:
    """Run the schedule; an unavailable policy abandons its move, and enough of them in a row end the run."""
    campaign, run_log = pipeline.campaign, _run_log(pipeline.context)
    tracker = tracker or NullTracker()
    records = ChainRecords(campaign.chain.record_path)
    steps: list[dict[str, Any]] = []
    consecutive = 0
    first_publish: str | None = None

    def log_step(index: int) -> None:
        usage = move_usage(run_log, index, options.policy_price_model, options.policy_api_base)
        tracker.log_step(index, step_values(steps[-1], records.new(), usage))

    def harness_move(index: int, entry: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal consecutive
        run_log.move, run_log.phase = index, "parent"
        move = _move({"kind": "harness", "label": entry.get("label")})
        try:
            outcome = campaign.advance(move)
        except PolicyUnavailableError as error:
            episodes = [row for row in run_log.episodes if row["move"] == index]
            _LOGGER.error("move %d abandoned: policy unavailable (%s, status %s)", index, error.kind, error.status)
            run_log.faults.append(
                {
                    "move": index,
                    "label": entry.get("label"),
                    "kind": error.kind,
                    "status": error.status,
                    "task_id": error.task_id,
                    "phase": run_log.phase,
                    "episodes_started": len(episodes),
                    "usage": _sum_usage(episodes),
                }
            )
            steps.append({"index": index, "move": move.kind, "label": entry.get("label"), "result": "abandoned"})
            log_step(index)
            consecutive += 1
            if consecutive >= options.max_consecutive_faults:
                raise _AbortedError(f"{consecutive} moves in a row abandoned: the policy is unavailable") from None
            return steps[-1]
        consecutive = 0
        step = _step(outcome)
        # The pipeline calls a gated candidate that was not published "rolled back"; here a
        # rollback is its own step, so the gate's verdict is named for what it was.
        if step["result"] == "rolled back":
            step["result"] = "rejected"
        # GEPA decides against the served candidate's stored mean, not the gate's fresh run of
        # the incumbent (GEPASelector.decide), so the summary shows both.
        if outcome.decision is not None and "gepa_served_mean" in outcome.decision.metrics:
            step["served_mean"] = outcome.decision.metrics["gepa_served_mean"]
        steps.append({"index": index, "label": entry.get("label"), **step})
        log_step(index)
        return steps[-1]

    index = 0
    aborted = None
    try:
        for entry in schedule:
            kind = entry["kind"]
            if kind == "rollback":
                index += 1
                run_log.move, run_log.phase = index, "parent"
                to = first_publish if entry["to"] == FIRST_PUBLISH else str(entry["to"])
                steps.append(_rollback(campaign, run_log, index, to))
                log_step(index)
                continue
            until_publish = kind == "harness_until_publish"
            for _ in range(int(entry["max_moves"] if until_publish else entry.get("repeat", 1))):
                index += 1
                step = harness_move(index, entry)
                if step["result"] == "published" and first_publish is None:
                    first_publish = str(step["release"])
                if until_publish and step["result"] == "published":
                    break
    except _AbortedError as stop:
        aborted = str(stop)
        _LOGGER.error("stopping the run: %s", aborted)
    run_log.write()
    return {"steps": steps, "aborted": aborted}


def _rollback(campaign: Any, run_log: RunLog, index: int, to: str | None) -> dict[str, Any]:
    head = campaign.chain.head()
    current = run_log.ends[-1] if run_log.ends else head.method_state
    if to is None:
        reason = "no harness release was published before the rollback"
        _LOGGER.error("rollback to the first publish cannot run: %s", reason)
        return {"index": index, "move": "rollback", "result": "not runnable", "to": FIRST_PUBLISH, "reason": reason}
    try:
        target = campaign.chain.get(to)
    except ReleaseError as error:
        _LOGGER.error("rollback to %s cannot run: %s", to, error)
        return {"index": index, "move": "rollback", "result": "not runnable", "to": to, "reason": str(error)}
    restored = campaign.rollback(to)
    run_log.ends.append(restored.method_state)
    archive = Archive.from_state(restored.method_state)
    served = dict(archive.candidates[archive.served].components)
    return {
        "index": index,
        "move": "rollback",
        "result": "restored",
        "to": to,
        "from": head.release_id,
        "exercised": _canonical(current) != _canonical(target.method_state),
        "harness_restored": restored.harness == target.harness,
        "state_restored": _canonical(restored.method_state) == _canonical(target.method_state),
        "served_matches_harness": all(
            restored.harness is not None
            and restored.harness.get(path) is not None
            and restored.harness.get(path).content == text
            for path, text in served.items()
        ),
    }


def _sum_usage(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total = {"calls": 0, "failed_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for row in rows:
        for key in total:
            total[key] += int(row.get("usage", {}).get(key, 0))
    return total


def move_usage(
    run_log: RunLog, index: int, policy_price_model: str | None, policy_api_base: str | None = None
) -> dict[str, Any]:
    """One move's agent processes, policy calls and reflection calls, with their tokens and cost."""
    episodes = [row for row in run_log.episodes if row["move"] == index]
    reflections = [row for row in run_log.reflections if row["move"] == index]
    ok = [row for row in reflections if row.get("ok")]
    priced = [row["cost_usd"] for row in ok if row.get("cost_usd") is not None]
    policy = _sum_usage(episodes)
    return {
        "episodes": len(episodes),
        "policy_calls": policy["calls"],
        "policy_failed_calls": policy["failed_calls"],
        "policy_prompt_tokens": policy["prompt_tokens"],
        "policy_completion_tokens": policy["completion_tokens"],
        "policy_cost_usd": (
            completion_cost_usd(
                policy_price_model, policy["prompt_tokens"], policy["completion_tokens"], api_base=policy_api_base
            )
            if policy_price_model
            else None
        ),
        "reflection_calls": len(reflections),
        "reflection_failed_calls": len(reflections) - len(ok),
        "reflection_input_tokens": sum(int(row["input_tokens"]) for row in ok),
        "reflection_output_tokens": sum(int(row["output_tokens"]) for row in ok),
        "reflection_cost_usd": sum(priced) if len(priced) == len(ok) else None,
    }


def usage_report(
    run_log: RunLog, policy_price_model: str | None, policy_api_base: str | None = None
) -> dict[str, Any]:
    """API calls, tokens and cost for the reflection model and the policy, abandoned moves included."""
    ok = [row for row in run_log.reflections if row.get("ok")]
    priced = [row["cost_usd"] for row in ok if row.get("cost_usd") is not None]
    reflection = {
        "calls": len(run_log.reflections),
        "failed_calls": len(run_log.reflections) - len(ok),
        "attempts_in_failed_calls": sum(int(r.get("attempts", 0)) for r in run_log.reflections if not r.get("ok")),
        "input_tokens": sum(int(row["input_tokens"]) for row in ok),
        "output_tokens": sum(int(row["output_tokens"]) for row in ok),
        "cost_usd": sum(priced) if len(priced) == len(ok) else None,
    }
    policy: dict[str, Any] = {"episodes": len(run_log.episodes), **_sum_usage(run_log.episodes)}
    policy["cost_usd"] = (
        completion_cost_usd(
            policy_price_model, policy["prompt_tokens"], policy["completion_tokens"], api_base=policy_api_base
        )
        if policy_price_model
        else None
    )
    in_faults = _sum_usage(run_log.faults)
    known = [cost for cost in (reflection["cost_usd"], policy["cost_usd"]) if cost is not None]
    return {
        "reflection": reflection,
        "policy": policy,
        "abandoned_moves": {"moves": len(run_log.faults), **in_faults},
        "total_cost_usd": sum(known) if len(known) == 2 else None,
    }


# ---------------------------------------------------------------- the checks


@dataclass(frozen=True)
class Check:
    """One check's verdict. ``not_exercised`` means the run never produced what it checks: no check failed."""

    name: str
    passed: bool
    detail: str
    not_exercised: bool = False


# A real model publishes rarely, so a run can end without a rejection or a publish to roll back to.
# Those two checks then say so; every other failure is a real one.
RERUN_CHECKS = frozenset({"4 rejected stays", "6 rollback"})


def only_not_exercised(checks: Sequence[Mapping[str, Any]], names: frozenset[str] = RERUN_CHECKS) -> bool:
    """Whether some checks failed, and each of them only because the run did not exercise it and is in ``names``."""
    failed = [check for check in checks if not check["passed"]]
    return bool(failed) and all(check.get("not_exercised") and check["name"] in names for check in failed)


def _records(work_dir: Path) -> list[dict[str, Any]]:
    path = work_dir / "releases" / "commits.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def _spec(harness: Mapping[str, Any]) -> HarnessSpec:
    return HarnessSpec(
        agent=str(harness["agent"]),
        files=tuple(
            (str(path), HarnessFile(kind=file["kind"], content=file["content"])) for path, file in harness["files"]
        ),
    )


def _state(record: Mapping[str, Any]) -> Mapping[str, object]:
    return record["method_state"] if record["op"] == "skip" else record["candidate"]["method_state"]


def _method_episodes(record: Mapping[str, Any]) -> int:
    if record["op"] == "skip":
        return int(record["episodes_spent"])
    return int(record["candidate"]["metrics"]["method_episodes"])


@dataclass(frozen=True)
class _Move:
    index: int
    step: Mapping[str, Any]
    record: Mapping[str, Any] | None


def _moves(steps: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]) -> list[_Move]:
    """Pair each step with the record it wrote: one per settled harness move or rollback, none when abandoned."""
    later = iter(records[1:])  # records[0] publishes release 0
    paired = []
    for step in steps:
        wrote = step["result"] in ("published", "rejected", "skipped", "restored")
        paired.append(_Move(int(step["index"]), step, next(later) if wrote else None))
    return paired


def check_run(
    summary: Mapping[str, Any], run_log: RunLog, work_dir: Path, validation_ids: Sequence[str], max_metric_calls: int
) -> list[Check]:
    records = _records(work_dir)
    moves = _moves(summary["steps"], records)
    chain = {r["release_id"]: r for r in records if r["op"] == "publish"}
    harness_of = {rid: _spec(r["candidate"]["harness"]) for rid, r in chain.items()}
    gated = [m for m in moves if m.record is not None and m.record["op"] in ("publish", "reject")]
    runs: list[tuple[str, Callable[[], Check | list[Check]]]] = [
        ("1 extracted", lambda: _check_extracted(gated, run_log, harness_of)),
        ("2 reflection prompt", lambda: _check_prompts(moves, run_log)),
        ("3 gate on real scores", lambda: _check_gate_scores(gated, run_log, harness_of, validation_ids)),
        ("4 rejected stays", lambda: _check_rejected_stay(moves)),
        ("5 identities", lambda: _check_identities(moves, run_log, validation_ids)),
        ("6 rollback", lambda: _check_rollbacks(moves, run_log)),
        ("7a failed reflections", lambda: _check_failed_reflections(moves, run_log)),
        ("7b abandoned moves", lambda: _check_faults(moves, run_log)),
        ("budget stop", lambda: _check_budget_stop(moves, max_metric_calls)),
    ]
    checks: list[Check] = []
    for name, run in runs:
        # A record the check cannot read is a failed check, not a lost summary.
        try:
            result = run()
        except Exception as error:
            checks.append(Check(name, False, f"the check could not read the run: {type(error).__name__}: {error}"))
            continue
        checks.extend(result if isinstance(result, list) else [result])
    return checks


def _check_extracted(gated: Sequence[_Move], run_log: RunLog, harness_of: Mapping[str, HarnessSpec]) -> Check:
    if not gated:
        return Check("1 extracted", False, "not exercised: no candidate reached the gate", not_exercised=True)
    problems = []
    for move in gated:
        assert move.record is not None
        archive = Archive.from_state(move.record["candidate"]["method_state"])
        child = archive.candidates[-1]
        if child.parent is None:
            problems.append(f"move {move.index}: the archive's newest candidate is the seed, not this move's child")
            continue
        parent = dict(archive.candidates[child.parent].components)
        changed = [path for path, text in child.components if parent.get(path) != text]
        replies = [r for r in run_log.reflections if r["move"] == move.index and r.get("ok")]
        proposed = _spec(move.record["candidate"]["harness"])
        # The parent may be a candidate the gate rejected, so the proposal is compared with
        # the parent's harness (the head's files with the parent's texts), not with the head's.
        parent_spec = harness_of[move.record["candidate"]["parent_id"]]
        for path, text in parent.items():
            kind = parent_spec.get(path)
            assert kind is not None
            parent_spec = parent_spec.with_file(path, HarnessFile(kind=kind.kind, content=text))
        files_changed = sorted(
            {path for path, file in proposed.files if parent_spec.get(path) != file}
            | {path for path, _ in parent_spec.files if proposed.get(path) is None}
        )
        if len(changed) != 1:
            problems.append(f"move {move.index}: {len(changed)} components changed")
        elif not replies or dict(child.components)[changed[0]] != extract_text(replies[-1]["reply"]):
            problems.append(f"move {move.index}: {changed[0]} is not the extracted reflection reply")
        elif files_changed != changed:
            problems.append(f"move {move.index}: the proposal changed {files_changed}, not just {changed}")
    detail = "; ".join(problems) or f"{len(gated)} gated candidates, each its parent with one file rewritten"
    return Check("1 extracted", not problems, detail)


def _check_prompts(moves: Sequence[_Move], run_log: RunLog) -> Check:
    reflected = [r for r in run_log.reflections if r.get("ok")]
    if not reflected:
        return Check("2 reflection prompt", False, "not exercised: no reflection call succeeded", not_exercised=True)
    problems = []
    for row in reflected:
        parents = [e for e in run_log.episodes if e["move"] == row["move"] and e["phase"] == "parent"]
        if not parents:
            problems.append(f"move {row['move']}: no parent episodes")
        for episode in parents:
            # Upstream's renderer writes each value stripped (instruction_proposal.py:85).
            wanted = {
                "task prompt": run_log.prompts[episode["task_id"]].strip(),
                "final output": (episode["final_output"] or "").strip(),
                "feedback": (episode["feedback"] or "").strip(),
            }
            missing = [name for name, text in wanted.items() if not text or text not in row["prompt"]]
            if missing:
                problems.append(f"move {row['move']} task {episode['task_id']}: missing {missing}")
    detail = "; ".join(problems) or f"{len(reflected)} prompts hold every parent episode's prompt, output and feedback"
    return Check("2 reflection prompt", not problems, detail)


def _check_gate_scores(
    gated: Sequence[_Move], run_log: RunLog, harness_of: Mapping[str, HarnessSpec], validation_ids: Sequence[str]
) -> Check:
    if not gated:
        return Check("3 gate on real scores", False, "not exercised: no gate decision", not_exercised=True)
    problems = []
    for move in gated:
        assert move.record is not None
        candidate = move.record["candidate"]
        archive = Archive.from_state(candidate["method_state"])
        incumbent_id = harness_of[candidate["parent_id"]].spec_id
        candidate_id = _spec(candidate["harness"]).spec_id
        rows = [e for e in run_log.episodes if e["move"] == move.index and e["phase"] == "gate"]

        def scores(spec_id: str, rows: list[dict[str, Any]] = rows) -> list[float | None]:
            by_task = {e["task_id"]: e["score"] for e in rows if e["spec_id"] == spec_id}
            return [by_task.get(task_id) for task_id in validation_ids]

        if list(archive.candidates[-1].validation_scores or ()) != scores(candidate_id):
            problems.append(f"move {move.index}: stored scores are not the graded gate episodes")
        start = Archive.from_state(run_log.starts[move.index])
        seed_scores = list(archive.candidates[0].validation_scores or ())
        if start.candidates[0].validation_scores is None and seed_scores != scores(incumbent_id):
            problems.append(f"move {move.index}: the seed's scores are not the incumbent's gate episodes")
        decision_metrics = move.record["decision"]["metrics"]
        mean = sum(archive.candidates[-1].validation_scores or ()) / len(validation_ids)
        if decision_metrics["gepa_candidate_mean"] != mean:
            problems.append(f"move {move.index}: the decision's mean is not the gate episodes' mean")
    outcomes = [str(m.record["op"]) for m in gated if m.record is not None]
    detail = "; ".join(problems) or f"gate decisions {outcomes} on graded episodes"
    return Check("3 gate on real scores", not problems, detail)


def _check_rejected_stay(moves: Sequence[_Move]) -> Check:
    rejects = [m for m in moves if m.record is not None and m.record["op"] == "reject"]
    if not rejects:
        return Check("4 rejected stays", False, "not exercised: the gate rejected nothing", not_exercised=True)
    problems = []
    for reject in rejects:
        assert reject.record is not None
        # The rejected candidate is the harness the gate judged, read from the record, not from the archive.
        judged = _spec(reject.record["candidate"]["harness"])
        paths = Archive.from_state(reject.record["candidate"]["method_state"]).candidates[0].paths
        wanted = {path: judged.get(path).content for path in paths if judged.get(path) is not None}  # type: ignore[union-attr]
        for later in moves:
            if later.index < reject.index or later.record is None:
                continue
            if later.record["op"] == "rollback":
                break
            if wanted not in [dict(c.components) for c in Archive.from_state(_state(later.record)).candidates]:
                problems.append(f"move {reject.index}'s candidate is missing at move {later.index}")
                break
    detail = "; ".join(problems) or f"{len(rejects)} rejected candidates stay in the archive until a rollback"
    return Check("4 rejected stays", not problems, detail)


def _charge(start: Archive, record: Mapping[str, Any], validation: int) -> int:
    """What GEPA charges for one move: its episodes, plus validation when the gate ran (method.py:settle)."""
    if record["op"] == "skip":
        return _method_episodes(record)
    return _method_episodes(record) + validation * (2 if start.candidates[0].validation_scores is None else 1)


def _check_identities(moves: Sequence[_Move], run_log: RunLog, validation_ids: Sequence[str]) -> list[Check]:
    validation = len(validation_ids)
    per_move, path_problems = [], []
    at_release: dict[str, int] = {"r0000": 0}
    charged = 0
    for move in moves:
        record = move.record
        if record is None:
            continue
        if record["op"] == "rollback":
            charged = at_release[str(record["to"])]
            continue
        start = Archive.from_state(run_log.starts[move.index])
        spent = Archive.from_state(_state(record)).metric_calls - start.metric_calls
        if spent != _charge(start, record, validation):
            per_move.append(f"move {move.index}: charged {spent}, expected {_charge(start, record, validation)}")
        charged += _charge(start, record, validation)
        if record["op"] == "publish":
            at_release[str(record["release_id"])] = charged
    final = next((m.record for m in reversed(moves) if m.record is not None and m.record["op"] != "rollback"), None)
    if final is not None and Archive.from_state(_state(final)).metric_calls != charged:
        path_problems.append(f"final metric_calls {Archive.from_state(_state(final)).metric_calls} != {charged}")

    # Every agent process the runner started is in commits.jsonl or faults.jsonl, and each move's own count matches.
    counted, fault_episodes, row_problems = 0, 0, []
    for move in moves:
        rows = [e for e in run_log.episodes if e["move"] == move.index]
        if move.record is None:
            if move.step["result"] == "abandoned":
                fault = next((f for f in run_log.faults if f["move"] == move.index), None)
                if fault is None:
                    row_problems.append(f"move {move.index}: abandoned, but faults.jsonl has no row for it")
                    continue
                fault_episodes += int(fault["episodes_started"])
                if int(fault["episodes_started"]) != len(rows):
                    row_problems.append(f"move {move.index}: faults.jsonl says {fault['episodes_started']}")
            continue
        if move.record["op"] == "rollback":
            continue
        expected = _method_episodes(move.record) + (2 * validation if move.record["op"] != "skip" else 0)
        counted += expected
        if expected != len(rows):
            row_problems.append(f"move {move.index}: commits.jsonl says {expected}, the runner ran {len(rows)}")
    if run_log.started != counted + fault_episodes:
        row_problems.append(f"runner started {run_log.started}, recorded {counted} + {fault_episodes} in faults")

    # An abandoned move charges nothing: the next move starts from the state the abandoned one started from.
    fault_problems = []
    for fault in run_log.faults:
        later = [m for m in moves if m.index > fault["move"]]
        if not later or later[0].step["move"] == "rollback" or later[0].index not in run_log.starts:
            continue
        if _canonical(run_log.starts[later[0].index]) != _canonical(run_log.starts[fault["move"]]):
            fault_problems.append(f"move {fault['move']} changed GEPA's state")
    return [
        Check("5a metric calls per move", not per_move, "; ".join(per_move) or "every move charged as GEPA charges"),
        Check(
            "5b metric calls on the final path",
            not path_problems,
            "; ".join(path_problems) or f"final metric_calls {charged} = charges on the final path",
        ),
        Check(
            "5c every episode accounted for",
            not row_problems,
            "; ".join(row_problems)
            or f"runner started {run_log.started} = {counted} in commits.jsonl + {fault_episodes} in faults.jsonl",
        ),
        Check(
            "5d abandoned moves charge nothing",
            not fault_problems,
            "; ".join(fault_problems) or f"{len(run_log.faults)} abandoned moves left GEPA's state as it was",
        ),
    ]


def _check_rollbacks(moves: Sequence[_Move], run_log: RunLog) -> Check:
    rollbacks = [m for m in moves if m.step["move"] == "rollback"]
    if not rollbacks:
        return Check("6 rollback", False, "the schedule has no rollback")
    problems: list[str] = []
    unexercised: list[str] = []
    for move in rollbacks:
        step = move.step
        if step["result"] != "restored":
            # Nothing published, so there is nothing to restore: the run did not exercise the rollback.
            # A named release that does not exist is a real failure.
            nothing_published = step.get("to") == FIRST_PUBLISH
            (unexercised if nothing_published else problems).append(
                f"move {move.index}: {step['result']} ({step.get('reason')})"
            )
            continue
        if not step["exercised"]:
            unexercised.append(f"move {move.index}: the state already was {step['to']}'s")
        keys = ("harness_restored", "state_restored", "served_matches_harness")
        problems.extend(f"move {move.index}: {key} is false" for key in keys if not step[key])
        after = [m for m in moves if m.index > move.index and m.index in run_log.starts]
        target = next(r for r in _records(run_log.work_dir) if r.get("release_id") == step["to"])
        if after and _canonical(run_log.starts[after[0].index]) != _canonical(target["candidate"]["method_state"]):
            problems.append(f"move {after[0].index} did not start from {step['to']}'s state")
    if problems:
        return Check("6 rollback", False, "; ".join(problems + unexercised))
    if unexercised:
        return Check("6 rollback", False, "not exercised: " + "; ".join(unexercised), not_exercised=True)
    return Check("6 rollback", True, "archive and served harness restored together, and the next move starts there")


def _check_failed_reflections(moves: Sequence[_Move], run_log: RunLog) -> Check:
    failed = [r for r in run_log.reflections if not r.get("ok")]
    problems = []
    for row in failed:
        move = next(m for m in moves if m.index == row["move"])
        parents = [e for e in run_log.episodes if e["move"] == row["move"] and e["phase"] == "parent"]
        children = [e for e in run_log.episodes if e["move"] == row["move"] and e["phase"] != "parent"]
        if move.record is None or move.record["op"] != "skip":
            problems.append(f"move {row['move']}: not a skip")
        elif _method_episodes(move.record) != len(parents) or children:
            problems.append(f"move {row['move']}: charged {_method_episodes(move.record)}, ran {len(parents)} parent")
    detail = "; ".join(problems) or f"{len(failed)} failed reflections, each a skip charging only its parent episodes"
    return Check("7a failed reflections", not problems, detail)


def _check_faults(moves: Sequence[_Move], run_log: RunLog) -> Check:
    problems = []
    for fault in run_log.faults:
        move = next(m for m in moves if m.index == fault["move"])
        if move.record is not None:
            problems.append(f"move {fault['move']}: the campaign recorded it")
    detail = "; ".join(problems) or f"{len(run_log.faults)} abandoned moves, none recorded by the campaign"
    return Check("7b abandoned moves", not problems, detail)


def _check_budget_stop(moves: Sequence[_Move], max_metric_calls: int) -> Check:
    final = next((m for m in reversed(moves) if m.record is not None and m.record["op"] != "rollback"), None)
    if final is None or final.record is None:
        return Check("budget stop", False, "not exercised: no harness record", not_exercised=True)
    archive = Archive.from_state(_state(final.record))
    stopped = final.record["op"] == "skip" and final.record["episodes_spent"] == 0
    return Check(
        "budget stop",
        stopped and archive.metric_calls >= max_metric_calls,
        f"last move {final.index}: {final.record['op']}, metric_calls {archive.metric_calls} of {max_metric_calls}",
    )


def key_scan(work_dir: Path, names: Sequence[str], extra: Sequence[bytes] = ()) -> Check:
    """No value of the named variables in any file under ``work_dir`` or in ``extra``. Never prints a value."""
    values = [os.environ[name].encode() for name in names if len(os.environ.get(name, "")) >= 8]
    if not values:
        return Check("8 no key written", False, "no key to scan for: the named variables are unset or short")
    files, hits = 0, []
    for path in sorted(p for p in work_dir.rglob("*") if p.is_file() and not p.is_symlink()):
        files += 1
        data = path.read_bytes()
        if any(value in data for value in values):
            hits.append(str(path.relative_to(work_dir)))
    hits += [f"output {index}" for index, data in enumerate(extra) if any(value in data for value in values)]
    return Check("8 no key written", not hits, f"{files} files, {len(hits)} hits" + (f": {hits}" if hits else ""))


def _key_names(config: Any) -> list[str]:
    names: list[str] = list(config.get("requires", {}).get("env", ())) if isinstance(config, Mapping) else []

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key == "api_key_env" and isinstance(item, str):
                    names.append(item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(config)
    if isinstance(config, Mapping) and config.get("logging"):
        names.append(WANDB_KEY_ENV)
    return sorted(set(names))


def _configure_logging(work_dir: Path) -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.FileHandler(work_dir / "run.log", encoding="utf-8"), logging.StreamHandler(sys.stderr)):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GEPA end to end against real model APIs, then check the run.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root; config paths are relative")
    parser.add_argument("--work-dir", type=Path, help="output directory (must not exist); overrides the config")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    schedule = config.get("schedule", [])
    try:
        check_schedule(schedule)
        # The live schedule has kinds the pipeline does not run, so the pipeline gets none.
        pipeline = build({**config, "schedule": [], "expect": []}, args.root, args.work_dir)
    except (ContractError, ImportError) as error:
        print(f"cannot start: {error}", file=sys.stderr)
        return 2
    work_dir = pipeline.context.work_dir
    _configure_logging(work_dir)
    try:
        tracker = start_tracker(
            pipeline.tracking,
            name=work_dir.name,
            work_dir=work_dir,
            root=args.root,
            config=run_config(config, args.root, "gepa_live"),
        )
    except Exception as error:  # wandb raises its own errors for a bad key, project or network
        print(f"cannot start logging: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    if tracker.url:
        print(f"wandb: {tracker.url}")
    options = live_options(config, pipeline.context.components.get("endpoints"))
    code = 1
    try:
        summary = _run_and_check(pipeline, schedule, options, tracker, config, args.config, args.root)
        code = 3 if summary["aborted"] else 0 if all(check["passed"] for check in summary["checks"]) else 1
    finally:
        # After the key scan, so a run whose files hold a key uploads none of them.
        problems = tracker.finish(work_dir, _key_names(config), failed=code != 0)

    for step in summary["steps"]:
        scores = (
            f" (candidate {step['candidate_mean']:.3f} vs served {step['served_mean']:.3f};"
            f" fresh incumbent run {step['incumbent_mean']:.3f})"
            if step.get("served_mean") is not None
            else ""
        )
        print(f"{step['index']}. {step['move']}: {step['result']}{scores}: {step.get('reason', '')}")
    usage = summary["usage"]
    print(f"reflection: {json.dumps(usage['reflection'])}")
    print(f"policy: {json.dumps(usage['policy'])}")
    total = usage["total_cost_usd"]
    print(f"total cost (USD): {'unknown' if total is None else total}")
    for check in summary["checks"]:
        verdict = "pass" if check["passed"] else "NOT EXERCISED" if check["not_exercised"] else "FAIL"
        print(f"[{verdict}] {check['name']}: {check['detail']}")
    print(f"in {work_dir}")
    if only_not_exercised(summary["checks"]):
        names = [check["name"] for check in summary["checks"] if not check["passed"]]
        print(
            f"not exercised, rerun: {', '.join(names)}. No check failed; the models did not produce what these need."
        )
    for problem in problems:
        print("logging:", problem, file=sys.stderr)
    if tracker.url:
        print(f"wandb: {tracker.url}")
    return 1 if problems and code == 0 else code


def _run_and_check(
    pipeline: PipelineRun,
    schedule: Sequence[Mapping[str, Any]],
    options: LiveOptions,
    tracker: RunTracker,
    config: Mapping[str, Any],
    config_path: Path,
    root: Path,
) -> dict[str, Any]:
    """Run the schedule, check the run, and write ``summary.json``; returns the summary."""
    work_dir = pipeline.context.work_dir
    result = run_live(pipeline, schedule, options, tracker)
    run_log = _run_log(pipeline.context)
    method = pipeline.context.component("method")
    validation_ids = list(method.validation_task_ids)
    max_metric_calls = int(method.config.max_metric_calls)
    summary: dict[str, Any] = {
        "config": str(config_path),
        "git_commit": git_commit(root),
        "model": config.get("model"),
        "head": pipeline.campaign.chain.head().release_id,
        **result,
        "usage": usage_report(run_log, options.policy_price_model, options.policy_api_base),
        "work_dir": str(work_dir),
    }
    checks = check_run(summary, run_log, work_dir, validation_ids, max_metric_calls)
    summary["checks"] = [
        {"name": c.name, "passed": c.passed, "detail": c.detail, "not_exercised": c.not_exercised} for c in checks
    ]
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    scan = key_scan(work_dir, _key_names(config))
    summary["checks"].append({"name": scan.name, "passed": scan.passed, "detail": scan.detail, "not_exercised": False})
    (work_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "Check",
    "LiveGEPAMethod",
    "ManifestWorkspaceRunner",
    "OpenAIEndpoints",
    "PolicyUnavailableError",
    "RecordingChatModel",
    "RecordingGrader",
    "RunLog",
    "check_run",
    "check_schedule",
    "command_grader",
    "gepa_method",
    "harness_seed",
    "key_scan",
    "live_options",
    "main",
    "manifest_runner",
    "move_usage",
    "only_not_exercised",
    "openai_endpoints",
    "pricing_api_base",
    "run_live",
    "suite_tasks",
    "usage_report",
]
