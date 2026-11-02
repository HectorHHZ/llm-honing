"""Run a full-game scenario through the isolated CPU simulation process."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path

from honing.backends.rea_issac.harness import AGENT, parse_harness
from honing.backends.rea_issac.scenarios import ReaIssacSuite
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


class ReaIssacEpisodeRunner:
    """Satisfies EpisodeRunner; no game installation or model package is imported."""

    support = RunnerSupport(
        agents=frozenset({AGENT}), file_kinds=frozenset({"settings", "instructions"}), programmable_loop=False
    )

    def __init__(
        self,
        *,
        suite: ReaIssacSuite,
        endpoints: PolicyEndpoints,
        integration_dir: Path,
        runs_dir: Path,
        command: Sequence[str] | None = None,
        environment: Mapping[str, str] | None = None,
        concurrency: int = 1,
    ) -> None:
        if type(concurrency) is not int or concurrency < 1:
            raise ContractError("concurrency must be a positive integer")
        self.suite, self.runs_dir = suite, runs_dir.resolve()
        self._endpoints, self._integration = endpoints, integration_dir.resolve()
        self._command = list(command or [sys.executable, str(self._integration / "scripts/match.py")])
        self._environment, self._concurrency = dict(environment or {}), concurrency
        self._slots: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )

    def check_spec(self, spec: HarnessSpec) -> None:
        parse_harness(spec)

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        self.check_spec(spec)
        scenario = self.suite.scenario(task.task_id)
        endpoint = self._endpoints.resolve(policy)
        if endpoint.runtime != "jev_decision":
            raise ContractError("REA-Issac needs a typed Jev decision endpoint")
        # Do not use externally supplied task/checkpoint names as file paths.
        name = f"rea-issac-s{scenario.seed}-{uuid.uuid4().hex[:12]}"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        out, harness_dir = self.runs_dir / name, self.runs_dir / f"{name}.harness"
        harness_dir.mkdir()
        for path, file in spec.files:
            target = harness_dir / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(file.content)
        (harness_dir / "index.json").write_text(json.dumps([{"path": p, "kind": f.kind} for p, f in spec.files]))
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
            "--max-ticks",
            str(scenario.max_ticks),
            "--decision-frames",
            str(scenario.decision_frames),
            "--harness-dir",
            str(harness_dir),
            "--out",
            str(out),
        ]
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
            error = json.loads(error_path.read_text())
            if error.get("kind") == "harness":
                raise CampaignRuleError("harness_readable", error["error"])
        result = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
        return EpisodeOutcome(
            task.task_id,
            spec.spec_id,
            policy.checkpoint.checksum,
            code,
            json.dumps(result, sort_keys=True),
            str(out / "decisions.jsonl"),
            str(out),
        )
