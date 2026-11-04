"""Play one tank task: run the integration's match process against a policy's decision server."""

from __future__ import annotations

import asyncio
import json
import os
import random
import uuid
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path

from honing.backends.tank.harness import AGENT, check_spec, question_keys
from honing.backends.tank.scenarios import TankScenario, TankSuite
from honing.core import (
    CampaignRuleError,
    ContractError,
    EpisodeOutcome,
    HarnessSpec,
    PolicyEndpoints,
    PolicyRef,
    RunnerSupport,
    Task,
)

RESULT_FIELDS = ("winner", "reason", "outcome", "sim_seconds", "self_hits", "shots")


class TankEpisodeRunner:
    """Satisfies :class:`~honing.core.EpisodeRunner` for tank tasks.

    ``command`` starts the match process and defaults to the integration's
    ``node scripts/match.mjs``; tests pass a stand-in. ``concurrency`` caps how
    many matches run at once, since a gate starts every task together. With
    ``sample`` set, every policy choice is drawn from the model's
    probabilities under a fresh seed, which the run's manifest records; that is
    how :class:`~honing.backends.tank.rollouts.TankRolloutProvider` plays. Each
    match gets a fresh directory under ``runs_dir``, which the outcome points
    at; nothing there is deleted.
    """

    support = RunnerSupport(
        agents=frozenset({AGENT}), file_kinds=frozenset({"settings", "instructions"}), programmable_loop=False
    )

    def __init__(
        self,
        *,
        suite: TankSuite,
        endpoints: PolicyEndpoints,
        integration_dir: Path,
        runs_dir: Path,
        command: Sequence[str] | None = None,
        environment: Mapping[str, str] | None = None,
        concurrency: int = 4,
        sample: bool = False,
    ) -> None:
        if concurrency <= 0:
            raise ContractError("concurrency must be positive")
        self._suite = suite
        self._endpoints = endpoints
        self._integration_dir = integration_dir
        self._runs_dir = runs_dir
        self._command = list(command or ["node", str(integration_dir / "scripts/match.mjs")])
        self._environment = dict(environment or {})
        self._concurrency = concurrency
        self._sample = sample
        self._questions = question_keys(integration_dir)
        # One semaphore per event loop: a campaign calls asyncio.run once per gate, and an
        # asyncio primitive used in one loop cannot be awaited from the next.
        self._slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    @property
    def suite(self) -> TankSuite:
        return self._suite

    @property
    def runs_dir(self) -> Path:
        return self._runs_dir

    def sampling(self) -> TankEpisodeRunner:
        """The same runner, drawing every choice from the model's probabilities."""
        return TankEpisodeRunner(
            suite=self._suite,
            endpoints=self._endpoints,
            integration_dir=self._integration_dir,
            runs_dir=self._runs_dir,
            command=self._command,
            environment=self._environment,
            concurrency=self._concurrency,
            sample=True,
        )

    def check_spec(self, spec: HarnessSpec) -> None:
        """Refuse, before any match, a harness the match cannot read."""
        check_spec(spec, self._questions)

    def write_harness(self, spec: HarnessSpec, directory: Path) -> None:
        """Every harness file at its path under ``directory``, and ``index.json`` listing them with their kinds."""
        directory.mkdir(parents=True)
        for path, file in spec.files:
            target = directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file.content, encoding="utf-8")
        index = [{"path": path, "kind": file.kind} for path, file in spec.files]
        (directory / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    def arguments(self, scenario: TankScenario, out: Path, harness_dir: Path) -> list[str]:
        return [
            *self._command,
            f"--seed={scenario.seed}",
            f"--seat={scenario.seat}",
            f"--opponent={scenario.opponent}",
            f"--seconds={scenario.seconds:g}",
            f"--interval-ms={scenario.interval_ms}",
            f"--opponent-interval-ms={scenario.opponent_interval_ms}",
            f"--harness-dir={harness_dir}",
            f"--out={out}",
        ]

    async def play(self, arguments: Sequence[str], name: str) -> int:
        """Run one match process within the concurrency cap; its output goes to ``<name>.log``."""
        loop = asyncio.get_running_loop()
        slots = self._slots.setdefault(loop, asyncio.Semaphore(self._concurrency))
        async with slots:
            with (self._runs_dir / f"{name}.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(
                    *arguments,
                    cwd=self._integration_dir,
                    env={**os.environ, **self._environment},
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                )
                return await process.wait()

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        self.check_spec(spec)
        scenario = self._suite.scenario(task.task_id)
        endpoint = self._endpoints.resolve(policy)
        if endpoint.runtime != "jev_decision":
            raise ContractError(f"tank matches need a Jev decision server, not a {endpoint.runtime} endpoint")
        name = f"{task.task_id}--{spec.spec_id}--{policy.checkpoint.checksum[:12]}--{uuid.uuid4().hex[:8]}"
        harness_dir = self._runs_dir / f"{name}.harness"
        self.write_harness(spec, harness_dir)
        out = self._runs_dir / name
        arguments = [*self.arguments(scenario, out, harness_dir), f"--decision-server={endpoint.base_url}"]
        if self._sample:
            arguments.append(f"--sample-seed={random.SystemRandom().randrange(2**31)}")
        exit_code = await self.play(arguments, name)
        error = out / "error.json"
        if exit_code != 0 and error.exists() and json.loads(error.read_text()).get("kind") == "harness":
            raise CampaignRuleError("harness_readable", json.loads(error.read_text())["error"])
        summary: dict[str, object] = {}
        if (out / "result.json").exists():
            result = json.loads((out / "result.json").read_text())
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


__all__ = ["RESULT_FIELDS", "TankEpisodeRunner"]
