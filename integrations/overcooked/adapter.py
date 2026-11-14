"""Integration-local adapters for Honing's existing runner and grader protocols."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path

from honing.core import (
    ContractError,
    EpisodeOutcome,
    Grade,
    HarnessSpec,
    PolicyEndpoints,
    PolicyRef,
    RunnerSupport,
    Task,
)
from integrations.overcooked.engine import ROOT, canonical
from integrations.overcooked.harness import AGENT, dump_harness, parse_harness
from integrations.overcooked.scenarios import Suite


def runtime_python() -> str:
    conda = ROOT / ".venv/python.exe"
    return str(
        conda if conda.exists() else ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )


class OvercookedEpisodeRunner:
    support = RunnerSupport(
        agents=frozenset({AGENT}), file_kinds=frozenset({"settings", "instructions"}), programmable_loop=False
    )

    def __init__(self, *, suite: Suite, endpoints: PolicyEndpoints, runs_dir: Path, python: str | None = None) -> None:
        self.suite, self.endpoints, self.runs_dir = suite, endpoints, runs_dir.resolve()
        self.python = python or runtime_python()

    def check_spec(self, spec: HarnessSpec) -> None:
        parse_harness(spec)

    async def run(self, spec: HarnessSpec, task: Task, policy: PolicyRef) -> EpisodeOutcome:
        self.check_spec(spec)
        scenario = self.suite.scenario(task.task_id)
        endpoint = self.endpoints.resolve(policy)
        if endpoint.runtime != "jev_decision":
            raise ContractError("Overcooked requires a typed decision endpoint")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        token = "overcooked-" + uuid.uuid4().hex[:12]
        out, harness_dir = self.runs_dir / token, self.runs_dir / (token + ".harness")
        dump_harness(spec, harness_dir)
        args = [
            self.python,
            "-m",
            "integrations.overcooked.match",
            "--policy",
            "model",
            "--decision-server",
            endpoint.base_url,
            "--policy-checksum",
            policy.checkpoint.checksum,
            "--layout",
            scenario.layout,
            "--max-steps",
            str(scenario.max_steps),
            "--controlled-player",
            str(scenario.controlled_player),
            "--partner",
            scenario.partner,
            "--target-deliveries",
            str(scenario.target_deliveries),
            "--score-scale",
            str(scenario.score_scale),
            "--harness-dir",
            str(harness_dir),
            "--out",
            str(out),
            "--no-images",
        ]
        if scenario.old_dynamics:
            args.append("--old-dynamics")
        with (self.runs_dir / (token + ".log")).open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                *args, cwd=ROOT.parent.parent, stdout=log, stderr=asyncio.subprocess.STDOUT
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
        result_path = out / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        return EpisodeOutcome(
            task.task_id,
            spec.spec_id,
            policy.checkpoint.checksum,
            code,
            canonical(result),
            str(out / "decisions.jsonl"),
            str(out),
        )


class OvercookedGrader:
    def __init__(self, *, suite: Suite, python: str | None = None) -> None:
        self.suite, self.python = suite, python or runtime_python()

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        if outcome.task_id != task.task_id or outcome.exit_code != 0 or not outcome.workspace_location:
            raise ContractError("Overcooked episode did not complete successfully")
        process = await asyncio.create_subprocess_exec(
            self.python,
            "-m",
            "integrations.overcooked.verify",
            outcome.workspace_location,
            cwd=ROOT.parent.parent,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate()
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        if process.returncode:
            raise ContractError("native replay rejected the run: " + stderr.decode(errors="replace")[-1200:])
        audit = json.loads(stdout.decode().splitlines()[-1])
        manifest, result = audit["manifest"], audit["result"]
        if (
            manifest["scenario"] != self.suite.scenario(task.task_id).record()
            or manifest["controller"] != "model"
            or manifest["kind"] != "evaluation"
            or manifest["spec_id"] != outcome.spec_id
            or manifest["policy_checksum"] != outcome.policy_checksum
            or not result["complete"]
        ):
            raise ContractError("recording does not match the requested model/task/harness")
        return Grade(
            result["grade"],
            f"Delivered {result['deliveries']} soups in {result['steps']} ticks; "
            f"native reward {result['score']}. Grade is native score / (native score + fixed task scale).",
        )
