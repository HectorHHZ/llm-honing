"""Run Meta-Harness on the attendance-payroll-audit tasks with an API task model, logging to wandb.

Every plane other than Harness is the walking skeleton's fakes; only harness
moves are scheduled, so the weights never change. The task agent is upstream's
OpenHands software-agent-sdk agent in the task image (``Dockerfile``), calling
the task model through litellm (default: Qwen3 Coder 30B A3B on Bedrock); the
proposer is Claude Code.

The data is split as upstream's GEPA config splits it (``data[:train + val]``,
the first ``train`` for training): the train tasks are Meta-Harness's search
set, which the gate scores and the proposer reads; the val tasks are held out,
and the seed and every proposed candidate are scored on them once, for logging
only.

The proposer works in a directory outside the repository, and a settings file
denies its tools every path under the user's home and the session scratch
space and turns on Claude Code's Bash sandbox, so it cannot read ground truth
or other runs.

The host environment is pinned to the SDK fork's lock, as the task image is;
with ``SDK=<checkout>/software-agent-sdk``:

    uv venv --python 3.13 .payroll-venv && VIRTUAL_ENV=.payroll-venv uv pip install -c $SDK/constraints.txt \\
        -e $SDK/openhands-sdk -e $SDK/openhands-tools -e $SDK/openhands-workspace -e $SDK/openhands-agent-server \\
        dspy pandas openpyxl python-dotenv wandb
    .payroll-venv/bin/python integrations/payroll/run.py --slm-repo <checkout>

Credentials come from ``.env``. Its AWS keys are written to a ``0600``
shared-credentials file in the work directory, whose folder every task
container mounts read-only; only the region is passed as a variable. ``WANDB_API_KEY`` is
the one value loaded into this process, so the proposer keeps the shell's own
Claude Code login.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import wandb
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

import payroll  # noqa: E402

from honing.campaign import Campaign, CampaignDefaults, CompressionComponents, HarnessComponents  # noqa: E402
from honing.core import (  # noqa: E402
    CompressionBudget,
    EpisodeOutcome,
    EpisodeRecord,
    EvaluationResult,
    Grade,
    HarnessMoveConfig,
    HarnessSpec,
    HoneRelease,
    Move,
    PolicyRef,
    QuantScheme,
    Task,
    TrainJobSpec,
    UpdateCandidate,
    run_graded,
)
from honing.e2e.components import PipelineContext  # noqa: E402
from honing.harness.methods.meta_harness import Archive, MetaHarnessMethod  # noqa: E402
from honing.harness.proposers import ClaudeCodeProposer  # noqa: E402
from honing.release import EpisodeEvaluator, Gate, ReleaseChain, WinLossSelector  # noqa: E402
from tests.fakes import pipeline as fakes  # noqa: E402

#: Paths the proposer's tools may not touch, besides the folder holding the upstream checkout (every example's
#: data): the user's home, with the repository, its runs and their ground truth, and credentials.
DENIED = ("//Users/**", "~/**")


def parse_args() -> argparse.Namespace:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slm-repo", type=Path, required=True, help="checkout of slm-harness-adaptation")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "runs" / f"payroll-meta-harness-{stamp}")
    parser.add_argument("--proposer-dir", type=Path, help="the proposer's directory, outside the repository")
    parser.add_argument("--model", default="bedrock/qwen.qwen3-coder-30b-a3b-v1:0")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--train", type=int, default=10)
    parser.add_argument("--val", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--episode-timeout", type=float, default=600.0)
    parser.add_argument("--proposer-model", default="opus")
    parser.add_argument("--proposer-effort", default="high")
    parser.add_argument("--proposer-timeout", type=float, default=2400.0)
    parser.add_argument("--wandb-project", default="llm-honing-meta-harness")
    return parser.parse_args()


def split(examples: dict[str, dict[str, Any]], train: int, val: int) -> tuple[tuple[Task, ...], tuple[Task, ...]]:
    """Upstream GEPA's split: the first ``train`` examples, then the next ``val``, in data order."""
    tasks = [Task(task_id=example["id"], prompt=example["prompt"]) for example in examples.values()]
    return tuple(tasks[:train]), tuple(tasks[train : train + val])


def proposer_settings(path: Path, upstream: Path) -> Path:
    """Deny the proposer's tools the user's files and the upstream checkout, and sandbox its Bash.

    ``Read`` rules also bind the sandboxed Bash's reads, and ``Edit`` rules cover every file-editing tool.
    The checkout's parent is denied rather than all of ``/private/tmp/claude-<uid>``, which holds
    Claude Code's own working files.
    """
    patterns = (*DENIED, f"/{upstream.parent}/**")
    rules = [f"{tool}({pattern})" for pattern in patterns for tool in ("Read", "Edit")]
    settings = {
        "permissions": {"deny": rules},
        "sandbox": {"enabled": True, "autoAllowBashIfSandboxed": True, "allowUnsandboxedCommands": False},
    }
    path.write_text(json.dumps(settings, indent=2))
    return path


class CandidateOnlyEvaluator:
    """Satisfies ``Evaluator`` by running only the candidate, as upstream Meta-Harness scores each candidate once.

    The incumbent's side of the result is the served candidate's recorded
    train scores, which the Meta-Harness selector compares against anyway.
    When those scores are missing or were taken under other weights, it runs
    the paired evaluation instead, so a decision never rests on stale scores.
    """

    def __init__(self, runner: Any, grader: Any, tasks: Sequence[Task]) -> None:
        self._runner = runner
        self._grader = grader
        self._tasks = tuple(tasks)
        self._paired = EpisodeEvaluator(runner, grader, tasks)

    def evaluate(self, candidate: UpdateCandidate, incumbent: HoneRelease) -> EvaluationResult:
        served = Archive.from_state(candidate.method_state).served_candidate
        if served.scores is None or served.policy != incumbent.policy().checkpoint.checksum:
            return self._paired.evaluate(candidate, incumbent)
        assert candidate.harness is not None

        async def run_all() -> list[EpisodeRecord]:
            policy = candidate.policy()
            runs = (run_graded(self._runner, self._grader, candidate.harness, task, policy) for task in self._tasks)
            return list(await asyncio.gather(*runs))

        records = asyncio.run(run_all())
        return EvaluationResult(
            task_ids=tuple(task.task_id for task in self._tasks),
            candidate_scores=tuple(record.score for record in records),
            incumbent_scores=served.scores,
            evaluated_variant=candidate.evaluated_variant,
            records=tuple(records),
        )


class LiveGrader:
    """Grades through upstream's eval and logs every episode to wandb as soon as it is graded."""

    def __init__(self, inner: payroll.PayrollGrader) -> None:
        self.inner = inner
        self.phase = "train"
        self.iteration = 0
        self.incumbent: str | None = None
        self.count = 0

    async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
        grade = await self.inner.grade(task, outcome)
        self.count += 1
        side = self.phase
        if self.phase == "train":
            side = "train_incumbent" if outcome.spec_id == self.incumbent else "train_candidate"
        wandb.log(
            {
                "episode/index": self.count,
                "episode/iteration": self.iteration,
                f"episode/score_{side}": grade.score,
                "episode/exit_code": outcome.exit_code,
            }
        )
        print(f"  [{side}] iter {self.iteration} {task.task_id}: {grade.score:.2f} {grade.feedback}", flush=True)
        return grade


#: The ``.env`` keys that go into the mounted credentials file, and the non-secret ones forwarded as variables.
AWS_SECRETS = {
    "AWS_ACCESS_KEY_ID": "aws_access_key_id",
    "AWS_SECRET_ACCESS_KEY": "aws_secret_access_key",
    "AWS_SESSION_TOKEN": "aws_session_token",
}
AWS_REGIONS = ("AWS_REGION_NAME", "AWS_REGION", "AWS_DEFAULT_REGION")


def aws_credentials(values: dict[str, str], folder: Path) -> Path:
    """Write the AWS keys from ``.env`` to ``folder/credentials``, readable only by this user."""
    folder.mkdir(mode=0o700)
    lines = ["[default]", *(f"{name} = {values[key]}" for key, name in AWS_SECRETS.items() if key in values)]
    (folder / "credentials").touch(mode=0o600)
    (folder / "credentials").write_text("\n".join(lines) + "\n")
    return folder


def build(args: argparse.Namespace, values: dict[str, str]) -> dict[str, Any]:
    work = args.work_dir.resolve()
    work.mkdir(parents=True)
    proposer_dir = (args.proposer_dir or Path(f"/private/tmp/mh-proposer-{work.name}")).resolve()
    proposer_dir.mkdir(parents=True)
    repo = args.slm_repo.resolve()
    examples = payroll.load_examples(repo)
    train, val = split(examples, args.train, args.val)
    runner = payroll.runner(
        repo=repo,
        examples=examples,
        workspaces=work / "runs",
        model=args.model,
        forward_env=[key for key in AWS_REGIONS if key in values],
        aws_credentials=aws_credentials(values, work / "aws"),
        concurrency=args.concurrency,
        timeout_seconds=args.episode_timeout,
    )
    grader = LiveGrader(payroll.grader(repo, examples))
    proposer = ClaudeCodeProposer(
        args.proposer_model,
        work / "proposer-logs",
        effort=args.proposer_effort,
        timeout_seconds=args.proposer_timeout,
        settings=proposer_settings(work / "proposer-settings.json", repo),
    )
    method = MetaHarnessMethod(proposer, proposer_dir, [task.task_id for task in train])
    context = PipelineContext(root=ROOT, work_dir=work)
    campaign = Campaign(
        chain=ReleaseChain(work / "releases"),
        gate=Gate(CandidateOnlyEvaluator(runner, grader, train), default_selector=WinLossSelector()),
        trainer=fakes.trainer(context),
        data=fakes.data(context),
        compression=CompressionComponents(
            scorer=fakes.scorer(context),
            selector=fakes.selector(context),
            editor=fakes.editor(context),
            quantizer=fakes.quantizer(context),
            telemetry=fakes.telemetry(context),
        ),
        # Room for the served candidate's baseline on every train task, then the smoke test.
        harness=HarnessComponents(
            runner=runner, grader=grader, method=method, tasks=train, episode_budget=len(train) + 1
        ),
        defaults=CampaignDefaults(
            train=TrainJobSpec(algorithm="sft", steps=1),
            budget=CompressionBudget(keep_ratio=0.75),
            recovery=None,
            scheme=QuantScheme(weight_format="int4", granularity="group", group_size=128),
        ),
    )
    campaign.start(fakes.model_source(context), work, harness_seed=payroll.seed_harness(repo))
    return {
        "campaign": campaign,
        "runner": runner,
        "grader": grader,
        "examples": examples,
        "train": train,
        "val": val,
        "work": work,
        "proposer_dir": proposer_dir,
    }


def score_val(
    runner: Any, grader: LiveGrader, spec: HarnessSpec, val: Sequence[Task], policy: PolicyRef
) -> list[EpisodeRecord]:
    async def run_all() -> list[EpisodeRecord]:
        return list(await asyncio.gather(*(run_graded(runner, grader, spec, task, policy) for task in val)))

    grader.phase = "val"
    try:
        return asyncio.run(run_all())
    finally:
        grader.phase = "train"


def mean(scores: Sequence[float]) -> float:
    return sum(scores) / len(scores)


def harness_text(files: Any) -> str:
    return "\n\n".join(f"===== {path} ({file.kind}) =====\n{file.content}" for path, file in files)


def proposer_session(work: Path) -> dict[str, float]:
    """Cost and duration of the newest proposer session, from its stream-json result event."""
    logs = sorted((work / "proposer-logs").glob("session-*.jsonl"), key=lambda p: int(p.stem.split("-")[1]))
    if not logs:
        return {}
    for line in reversed(logs[-1].read_text().splitlines()):
        event = json.loads(line) if line.startswith("{") else {}
        if event.get("type") == "result":
            return {
                "proposer/cost_usd": float(event.get("total_cost_usd") or 0.0),
                "proposer/minutes": float(event.get("duration_ms") or 0.0) / 60000,
                "proposer/turns": float(event.get("num_turns") or 0),
            }
    return {}


EPISODE_COLUMNS = ["iteration", "set", "side", "task_id", "template", "score", "feedback", "exit_code", "final_output"]
CANDIDATE_COLUMNS = [
    "iteration",
    "name",
    "parent",
    "selected",
    "train_mean",
    "val_mean",
    "hypothesis",
    "changes",
    "harness",
]


def rows(iteration: int, records: Sequence[EpisodeRecord], name: str, side: str, examples: dict) -> list[list[Any]]:
    return [
        [
            iteration,
            name,
            side,
            record.task_id,
            examples[record.task_id]["template_id"],
            record.score,
            record.feedback or "",
            record.outcome.exit_code,
            (record.outcome.final_output or "")[:1000],
        ]
        for record in records
    ]


def main() -> int:
    args = parse_args()
    values = {key: value for key, value in dotenv_values(ROOT / ".env").items() if value is not None}
    # Only the region and wandb's key enter this process; the region is forwarded to the task containers.
    for key in [*AWS_REGIONS, "WANDB_API_KEY"]:
        if key in values:
            os.environ[key] = values[key]
    # Claude Code refuses to start inside another session; the proposer is a separate session on purpose.
    os.environ.pop("CLAUDECODE", None)

    built = build(args, values)
    campaign, runner, grader = built["campaign"], built["runner"], built["grader"]
    examples, train, val, work = built["examples"], built["train"], built["val"], built["work"]
    run = wandb.init(
        project=args.wandb_project,
        name=work.name,
        dir=str(work),
        config={
            "task_model": args.model,
            "agent": "openhands software-agent-sdk (malusamayo fork)",
            "proposer_model": args.proposer_model,
            "proposer_effort": args.proposer_effort,
            "iterations": args.iterations,
            "train_tasks": [task.task_id for task in train],
            "val_tasks": [task.task_id for task in val],
            "dataset": payroll.DATA_FILE,
            "method": "meta_harness",
        },
    )
    run.define_metric("episode/index")
    run.define_metric("episode/*", step_metric="episode/index")
    run.define_metric("iteration")
    for prefix in ("train", "val", "proposer", "archive"):
        run.define_metric(f"{prefix}/*", step_metric="iteration")
    episodes: list[list[Any]] = []
    candidates: list[list[Any]] = []
    print(f"work dir: {work}; proposer dir: {built['proposer_dir']}; wandb: {run.url}", flush=True)

    head = campaign.chain.head()
    seed_val = score_val(runner, grader, head.harness, val, head.policy())
    val_means = {"seed": mean([r.score for r in seed_val])}
    episodes += rows(0, seed_val, "val", "seed", examples)
    wandb.log({"iteration": 0, "val/served_mean": val_means["seed"], "val/seed_mean": val_means["seed"]})
    print(f"seed val mean {val_means['seed']:.3f}", flush=True)

    for iteration in range(1, args.iterations + 1):
        started = time.time()
        grader.iteration = iteration
        grader.incumbent = campaign.chain.head().harness.spec_id
        outcome = campaign.advance(Move(kind="harness", harness=HarnessMoveConfig(label=f"iter{iteration}")))
        archive = Archive.from_state(campaign._method_state(campaign.chain.head()))
        metrics: dict[str, Any] = {
            "iteration": iteration,
            "archive/size": len(archive.candidates),
            "archive/proposed": int(outcome.decision is not None),
            "archive/minutes": (time.time() - started) / 60,
            **proposer_session(work),
        }
        if outcome.decision is None or outcome.evaluation is None:
            print(f"iter {iteration}: no proposal ({outcome.skipped_reason})", flush=True)
        else:
            evaluation, decision = outcome.evaluation, outcome.decision
            newest = archive.candidates[-1]
            spec = newest.spec(payroll.AGENT)
            grader.iteration = iteration
            candidate_val = score_val(runner, grader, spec, val, campaign.chain.head().policy())
            val_means[newest.name] = mean([r.score for r in candidate_val])
            candidate_records = [r for r in evaluation.records if r.spec_id == spec.spec_id]
            incumbent_records = [r for r in evaluation.records if r.spec_id != spec.spec_id]
            episodes += rows(iteration, candidate_records, "train", "candidate", examples)
            episodes += rows(iteration, incumbent_records, "train", "incumbent", examples)
            episodes += rows(iteration, candidate_val, "val", "candidate", examples)
            candidates.append(
                [
                    iteration,
                    newest.name,
                    newest.parent,
                    decision.selected,
                    newest.mean,
                    val_means[newest.name],
                    newest.hypothesis,
                    newest.changes,
                    harness_text(newest.files),
                ]
            )
            metrics |= {
                "train/selected": int(decision.selected),
                "train/candidate_mean": decision.metrics["meta_harness_candidate_mean"],
                "train/bar_mean": decision.metrics["meta_harness_served_mean"],
                "val/candidate_mean": val_means[newest.name],
            }
            print(
                f"iter {iteration}: {newest.name} (parent {newest.parent}) train {newest.mean:.3f} vs bar"
                f" {decision.metrics['meta_harness_served_mean']:.3f} -> {decision.outcome};"
                f" val {val_means[newest.name]:.3f}",
                flush=True,
            )
        served = archive.served_candidate
        metrics |= {"train/served_mean": served.mean, "val/served_mean": val_means.get(served.name)}
        wandb.log(
            {
                **metrics,
                "episodes": wandb.Table(columns=EPISODE_COLUMNS, data=episodes),
                "candidates": wandb.Table(columns=CANDIDATE_COLUMNS, data=candidates),
            }
        )
        artifact = wandb.Artifact(f"{work.name}-iter{iteration}", type="meta-harness-iteration")
        artifact.add_dir(str(built["proposer_dir"] / "archive"), name="archive")
        artifact.add_dir(str(work / "proposer-logs"), name="proposer-logs")
        (work / "state.json").write_text(json.dumps(archive.to_state(), indent=2))
        artifact.add_file(str(work / "state.json"), name="state.json")
        run.log_artifact(artifact)

    final = Archive.from_state(campaign._method_state(campaign.chain.head()))
    served = final.served_candidate
    run.summary["served_candidate"] = served.name
    run.summary["served_train_mean"] = served.mean
    run.summary["served_val_mean"] = val_means.get(served.name)
    run.summary["seed_train_mean"] = final.get("seed").mean
    run.summary["seed_val_mean"] = val_means["seed"]
    run.summary["served_harness"] = harness_text(served.files)
    print(
        f"served {served.name}: train {served.mean} val {val_means.get(served.name)};"
        f" seed train {final.get('seed').mean} val {val_means['seed']}",
        flush=True,
    )
    run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
