"""Play one RA2 task: run the integration's match process against a policy's decision server."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from honing.backends.ra2.scenarios import RA2Suite
from honing.core import (
    ContractError,
    EpisodeOutcome,
    HarnessSpec,
    PolicyEndpoints,
    PolicyRef,
    RunnerSupport,
    Task,
)

AGENT = "ra2_decision"
# Harness files the match process reads, by path in the spec.
CONTROLLER_PATH = "ra2/controller.json"
QUESTIONS_PATH = "ra2/questions.json"
RESULT_FIELDS = ("outcome", "tick", "ended", "truncated", "hash")


class RA2EpisodeRunner:
    """Satisfies :class:`~honing.core.EpisodeRunner` for RA2 tasks.

    ``command`` starts the match process and defaults to the integration's
    ``node integration/laya/run.mjs``; tests pass a stand-in. ``concurrency``
    caps how many matches run at once, since a gate starts every task together.
    Each match gets a fresh directory under ``runs_dir``, which the outcome
    points at; nothing there is deleted.
    """

    support = RunnerSupport(
        agents=frozenset({AGENT}), file_kinds=frozenset({"settings", "instructions"}), programmable_loop=False
    )

    def __init__(
        self,
        *,
        suite: RA2Suite,
        endpoints: PolicyEndpoints,
        integration_dir: Path,
        runs_dir: Path,
        command: Sequence[str] | None = None,
        environment: Mapping[str, str] | None = None,
        concurrency: int = 4,
    ) -> None:
        if concurrency <= 0:
            raise ContractError("concurrency must be positive")
        self._suite = suite
        self._endpoints = endpoints
        self._integration_dir = integration_dir
        self._runs_dir = runs_dir
        self._command = list(command or ["node", str(integration_dir / "integration/laya/run.mjs")])
        self._environment = dict(environment or {})
        self._slots = asyncio.Semaphore(concurrency)

    def _harness_arguments(self, spec: HarnessSpec, directory: Path) -> list[str]:
        if spec.agent != AGENT:
            raise ContractError(f"the RA2 runner plays agent {AGENT!r}, not {spec.agent!r}")
        arguments = []
        for path, flag in ((CONTROLLER_PATH, "--controller"), (QUESTIONS_PATH, "--questions")):
            file = spec.get(path)
            if file is not None:
                target = directory / Path(path).name
                target.write_text(file.content, encoding="utf-8")
                arguments.append(f"{flag}={target}")
        return arguments

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        scenario = self._suite.scenario(task.task_id)
        endpoint = self._endpoints.resolve(policy)
        if endpoint.runtime != "jev_decision":
            raise ContractError(f"RA2 matches need a Jev decision server, not a {endpoint.runtime} endpoint")
        name = f"{task.task_id}--{spec.spec_id}--{policy.checkpoint.checksum[:12]}--{uuid.uuid4().hex[:8]}"
        harness_dir = self._runs_dir / f"{name}.harness"
        harness_dir.mkdir(parents=True)
        out = self._runs_dir / name
        arguments = [
            *self._command,
            f"--map={scenario.map}",
            f"--seat={scenario.seat}",
            f"--seed={scenario.seed}",
            f"--ticks={scenario.ticks}",
            f"--cadence={scenario.cadence}",
            f"--decision-server={endpoint.base_url}",
            f"--out={out}",
            *self._harness_arguments(spec, harness_dir),
        ]
        async with self._slots:
            with (self._runs_dir / f"{name}.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(
                    *arguments,
                    cwd=self._integration_dir,
                    env={**os.environ, **self._environment},
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                )
                exit_code = await process.wait()
        result_path = out / "result.json"
        summary: dict[str, object] = {}
        if result_path.exists():
            result = json.loads(result_path.read_text())
            summary = {key: result.get(key) for key in RESULT_FIELDS}
        return EpisodeOutcome(
            task_id=task.task_id,
            spec_id=spec.spec_id,
            policy_checksum=policy.checkpoint.checksum,
            exit_code=exit_code,
            final_output=json.dumps(summary, sort_keys=True),
            transcript_location=str(out / "decisions.jsonl"),
            workspace_location=str(out),
        )
