"""Isolated native game processes behind Honing's existing EpisodeRunner contract."""

from __future__ import annotations

import asyncio
import json
import os
import random
import uuid
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path

from honing.backends.griddly.harness import AGENT, parse_harness, write_harness
from honing.backends.griddly.scenarios import GriddlySuite
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


def runtime_python(integration_dir: Path) -> str:
    return str(integration_dir / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


class GriddlyEpisodeRunner:
    support = RunnerSupport(
        agents=frozenset({AGENT}), file_kinds=frozenset({"settings", "instructions"}), programmable_loop=False
    )

    def __init__(
        self,
        *,
        suite: GriddlySuite,
        endpoints: PolicyEndpoints,
        integration_dir: Path,
        runs_dir: Path,
        command: Sequence[str] | None = None,
        environment: Mapping[str, str] | None = None,
        concurrency: int = 1,
        sample: bool = False,
    ) -> None:
        if type(concurrency) is not int or concurrency < 1:
            raise ContractError("concurrency must be a positive integer")
        self.suite, self.runs_dir = suite, runs_dir.resolve()
        self._endpoints, self._integration = endpoints, integration_dir.resolve()
        self._command = list(
            command or [runtime_python(self._integration), str(self._integration / "scripts/match.py")]
        )
        self._environment, self._concurrency, self._sample = dict(environment or {}), concurrency, sample
        self._slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    def sampling(self) -> GriddlyEpisodeRunner:
        return GriddlyEpisodeRunner(
            suite=self.suite,
            endpoints=self._endpoints,
            integration_dir=self._integration,
            runs_dir=self.runs_dir,
            command=self._command,
            environment=self._environment,
            concurrency=self._concurrency,
            sample=True,
        )

    def check_spec(self, spec: HarnessSpec) -> None:
        parse_harness(spec)

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        self.check_spec(spec)
        scenario = self.suite.scenario(task.task_id)
        endpoint = self._endpoints.resolve(policy)
        if endpoint.runtime != "jev_decision":
            raise ContractError("Griddly needs a typed Jev decision endpoint")
        name = "spiders-" + uuid.uuid4().hex[:16]
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        out, harness = self.runs_dir / name, self.runs_dir / f"{name}.harness"
        write_harness(spec, harness)
        args = [
            *self._command,
            "--policy",
            "model",
            "--decision-server",
            endpoint.base_url,
            "--policy-checksum",
            policy.checkpoint.checksum,
            "--seed",
            str(scenario.seed),
            "--level",
            str(scenario.level_id),
            "--max-steps",
            str(scenario.max_steps),
            "--harness-dir",
            str(harness),
            "--out",
            str(out),
        ]
        if self._sample:
            args.extend(["--sample", "--policy-seed", str(random.SystemRandom().randrange(2**32))])
        loop = asyncio.get_running_loop()
        slots = self._slots.setdefault(loop, asyncio.Semaphore(self._concurrency))
        async with slots:
            with (self.runs_dir / f"{name}.log").open("wb") as log:
                process = await asyncio.create_subprocess_exec(
                    *args,
                    cwd=self._integration,
                    env={**os.environ, **self._environment},
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                )
                try:
                    code = await process.wait()
                except BaseException:
                    if process.returncode is None:
                        process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), 10)
                        except asyncio.TimeoutError:
                            process.kill()
                            await process.wait()
                    raise
        error_path = out / "error.json"
        if code and error_path.exists():
            error = json.loads(error_path.read_text(encoding="utf-8"))
            if error.get("kind") == "harness":
                raise CampaignRuleError("harness_readable", error["error"])
        result_path = out / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        return EpisodeOutcome(
            task.task_id,
            spec.spec_id,
            policy.checkpoint.checksum,
            code,
            json.dumps(result),
            str(out / "decisions.jsonl"),
            str(out),
        )
