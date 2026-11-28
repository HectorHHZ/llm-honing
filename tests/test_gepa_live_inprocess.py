"""The live GEPA run (#100) in this process, with a scripted reflection model in place of litellm.

``tests/test_gepa_live_offline.py`` runs the script against the loopback stub
in the CI job that installs litellm. These tests run ``gepa_live.main`` in the
test process, so the main job measures it too: the policy is still the stub
(``tests/fakes/openai_stub.py``), reached by the real minimal agent, and the
reflection model is ``tests/fakes/litellm_stub.py`` answering from a script.
The checks are then fed changed records, one fault at a time, to show that
each fails for its own reason.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import io
import json
import logging
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gepa", reason="requires the optional 'gepa' extra")

from honing.core import ChatMessage, ChatReply, ContractError, EpisodeOutcome, Grade, ReleaseError, Task
from honing.e2e import gepa_live
from honing.e2e.components import PipelineContext
from honing.e2e.gepa_live import (
    RUN_LOG,
    Check,
    LiveOptions,
    OpenAIEndpoints,
    RecordingChatModel,
    RecordingGrader,
    RunLog,
    check_run,
    check_schedule,
    key_scan,
    live_options,
    move_usage,
    only_not_exercised,
    pricing_api_base,
    usage_report,
)
from honing.harness.models import ChatModelError
from tests.fakes.litellm_stub import APIError, FakeLiteLLM, install
from tests.fakes.openai_stub import REFLECTION_LINES, OpenAIStub

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/e2e/gepa_live.json"
KEY_ENV = "HONING_STUB_KEY"
KEY = "sk-stub-0123456789abcdef"
STUB_MODEL = {"provider": "openai", "name": "stub-model"}
MAX_METRIC_CALLS = 50
# Moves until the first publish, one more, a rollback to that publish, then moves until the budget is spent.
SCHEDULE = [
    {"kind": "harness_until_publish", "label": "pre_training", "max_moves": 4},
    {"kind": "harness", "label": "pre_training"},
    {"kind": "rollback", "to": "first_publish"},
    {"kind": "harness", "label": "pre_training", "repeat": 10},
]


def scripted_reflection() -> Callable[[Mapping[str, Any]], str]:
    """The first reply adds a convention only a training task needs, so the gate rejects it; later replies add
    every convention, so the next candidate that reaches the gate is published."""
    replies = 0

    def reflect(call: Mapping[str, Any]) -> str:
        nonlocal replies
        current = str(call["messages"][-1]["content"]).split("```\n", 1)[1].split("\n```", 1)[0]
        lines = [line for line in current.split("\n") if line.strip()]
        lines += [line for line in (REFLECTION_LINES[:1] if replies == 0 else REFLECTION_LINES) if line not in lines]
        replies += 1
        return "Here is the new instruction.\n```\n" + "\n".join(lines) + "\n```"

    return reflect


@dataclass
class LiveRun:
    code: int
    stdout: str
    stderr: str
    work_dir: Path
    run_log: RunLog | None
    validation_ids: list[str]
    litellm: FakeLiteLLM

    @property
    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = json.loads((self.work_dir / "summary.json").read_text())
        return summary

    @property
    def checks(self) -> dict[str, dict[str, Any]]:
        return {check["name"]: check for check in self.summary["checks"]}

    def rows(self, name: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (self.work_dir / f"{name}.jsonl").read_text().splitlines()]


@contextlib.contextmanager
def restored_logging() -> Iterator[None]:
    """``main`` configures the root logger for a run; put it back as it was."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    quiet = {name: logging.getLogger(name).level for name in gepa_live._QUIET_LOGGERS}
    try:
        yield
    finally:
        for handler in [h for h in root.handlers if h not in handlers]:
            root.removeHandler(handler)
            handler.close()
        root.setLevel(level)
        for name, value in quiet.items():
            logging.getLogger(name).setLevel(value)


def live_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub: OpenAIStub,
    *,
    policy: str = "ok",
    reflection: Any = None,
    schedule: list[dict[str, Any]] | None = None,
    max_consecutive_faults: int = 3,
) -> LiveRun:
    """Run ``gepa_live.main`` on the live config with both models replaced, and return what it left."""
    monkeypatch.setenv(KEY_ENV, KEY)
    # The graders run "python3" from PATH: make it this interpreter, which reads the tasks' annotations.
    shims = tmp_path / "bin"
    shims.mkdir()
    (shims / "python3").symlink_to(sys.executable)
    monkeypatch.setenv("PATH", f"{shims}:{os.environ['PATH']}")
    fake = install(monkeypatch, FakeLiteLLM([reflection or scripted_reflection()]))
    config = json.loads(CONFIG.read_text())
    config["requires"]["env"] = [KEY_ENV]
    config["model"] = STUB_MODEL
    components = config["components"]
    components["endpoints"]["args"]["base_url"] = stub.url(policy)
    components["runner"]["args"].update({"api_key_env": KEY_ENV, "agent_timeout_seconds": 10, "agent_max_attempts": 1})
    method = components["method"]["args"]
    method["reflection"].update({"api_base": stub.url("ok"), "api_key_env": KEY_ENV, "max_attempts": 1})
    method["gepa"]["max_metric_calls"] = MAX_METRIC_CALLS
    config["live"]["max_consecutive_faults"] = max_consecutive_faults
    config["schedule"] = SCHEDULE if schedule is None else schedule
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    built: list[Any] = []
    build = gepa_live.build
    monkeypatch.setattr(gepa_live, "build", lambda *args: built.append(build(*args)) or built[-1])
    stdout, stderr = io.StringIO(), io.StringIO()
    work_dir = tmp_path / "run"
    with restored_logging(), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = gepa_live.main(["--config", str(path), "--root", str(REPO), "--work-dir", str(work_dir)])
    run_log = built[0].context.shared[RUN_LOG] if built else None
    validation_ids = list(built[0].context.component("method").validation_task_ids) if built else []
    return LiveRun(code, stdout.getvalue(), stderr.getvalue(), work_dir, run_log, validation_ids, fake)


@pytest.fixture(scope="module")
def passing(tmp_path_factory: pytest.TempPathFactory) -> LiveRun:
    """One run that publishes, rejects, rolls back and spends its budget, for the tests that read or change it.

    Its patches end with the run, so the tests that read it see this process as it was.
    """
    with pytest.MonkeyPatch.context() as monkeypatch, OpenAIStub() as stub:
        return live_run(monkeypatch, tmp_path_factory.mktemp("live"), stub)


@pytest.fixture
def stub() -> Iterator[OpenAIStub]:
    with OpenAIStub() as server:
        yield server


# ---------------------------------------------------------------- whole runs


@pytest.mark.functional
def test_a_run_that_publishes_rejects_and_rolls_back_passes_every_check(passing: LiveRun) -> None:
    run = passing

    assert run.code == 0, run.stdout + run.stderr
    assert not [check for check in run.summary["checks"] if not check["passed"]]
    results = [(step["move"], step["result"]) for step in run.summary["steps"]]
    assert results[:4] == [
        ("harness", "rejected"),
        ("harness", "published"),
        ("harness", "skipped"),
        ("rollback", "restored"),
    ]
    rollback = run.summary["steps"][3]
    assert rollback["to"] == run.summary["steps"][1]["release"] and rollback["exercised"]
    assert run.summary["steps"][1]["served_mean"] is not None
    usage = run.summary["usage"]
    assert usage["policy"]["episodes"] == usage["policy"]["calls"] == len(run.rows("episodes"))
    assert usage["policy"]["prompt_tokens"] == 100 * usage["policy"]["calls"]  # the stub's usage, summed
    assert usage["reflection"]["calls"] == len(run.rows("reflections")) == len(run.litellm.calls)
    # Both models are on the stub, which nobody prices.
    assert (usage["policy"]["cost_usd"], usage["reflection"]["cost_usd"], usage["total_cost_usd"]) == (
        None,
        None,
        None,
    )
    assert "total cost (USD): unknown" in run.stdout
    assert "[pass] budget stop: last move" in run.stdout and "FAIL" not in run.stdout
    assert re.search(
        r"^2\. harness: published \(candidate [01]\.\d{3} vs served [01]\.\d{3}; fresh incumbent", run.stdout, re.M
    )
    assert run.rows("faults") == []
    assert all(KEY not in path.read_text(errors="ignore") for path in run.work_dir.rglob("*") if path.is_file())


@pytest.mark.functional
def test_an_outage_inside_a_move_abandons_only_that_move(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub: OpenAIStub
) -> None:
    # Policy calls 3 and 4 fail with 429: the first move's child episodes, after its parent's three.
    schedule = [{"kind": "harness", "label": "pre_training", "repeat": 3}]
    run = live_run(monkeypatch, tmp_path, stub, policy="window-3-5", schedule=schedule)

    assert run.summary["steps"][0]["result"] == "abandoned"
    (fault,) = run.rows("faults")
    assert (fault["move"], fault["kind"], fault["status"], fault["phase"]) == (1, "rate_limit", 429, "child")
    assert fault["episodes_started"] == len([row for row in run.rows("episodes") if row["move"] == 1])
    for name in ("5c every episode accounted for", "5d abandoned moves charge nothing", "7b abandoned moves"):
        assert run.checks[name]["passed"], run.checks[name]
    assert run.summary["usage"]["abandoned_moves"]["moves"] == 1
    assert run.summary["aborted"] is None


@pytest.mark.functional
def test_a_policy_that_stays_unavailable_ends_the_run_after_the_allowed_faults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub: OpenAIStub
) -> None:
    run = live_run(monkeypatch, tmp_path, stub, policy="401", max_consecutive_faults=2)

    assert run.code == 3
    assert run.summary["aborted"] == "2 moves in a row abandoned: the policy is unavailable"
    assert [step["result"] for step in run.summary["steps"]] == ["abandoned", "abandoned"]
    assert [(f["move"], f["kind"], f["status"]) for f in run.rows("faults")] == [(1, "auth", 401), (2, "auth", 401)]
    assert run.litellm.calls == []  # no parent minibatch finished, so nothing was reflected on


@pytest.mark.functional
def test_a_failed_reflection_is_a_skip_that_charges_only_the_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub: OpenAIStub
) -> None:
    schedule = [{"kind": "harness", "label": "pre_training", "repeat": 2}]
    run = live_run(monkeypatch, tmp_path, stub, reflection=APIError(429, "slow down"), schedule=schedule)

    assert [step["result"] for step in run.summary["steps"]] == ["skipped", "skipped"]
    rows = run.rows("reflections")
    assert [(r["ok"], r["error_type"], r["status"], r["attempts"]) for r in rows] == [(False, "APIError", 429, 1)] * 2
    assert run.checks["7a failed reflections"]["passed"], run.checks["7a failed reflections"]
    assert run.summary["usage"]["reflection"]["failed_calls"] == 2
    assert all(row["phase"] == "parent" for row in run.rows("episodes"))


@pytest.mark.functional
@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        (
            [{"kind": "train"}],
            "cannot start: schedule step 1: a live run has harness moves and rollbacks, not 'train'",
        ),
        ([{"kind": "rollback"}], "cannot start: schedule step 1: a rollback names a release id or 'first_publish'"),
        ([{"kind": "harness"}, {"kind": "harness", "repeat": 0}], "cannot start: schedule step 2: repeat must be"),
        ([{"kind": "harness_until_publish"}], "cannot start: schedule step 1: max_moves must be an integer >= 1"),
    ],
)
def test_a_schedule_the_run_cannot_run_is_refused_before_anything_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub: OpenAIStub, schedule: list[dict[str, Any]], message: str
) -> None:
    run = live_run(monkeypatch, tmp_path, stub, schedule=schedule)

    assert run.code == 2 and message in run.stderr
    assert not run.work_dir.exists() and run.run_log is None


@pytest.mark.functional
def test_a_run_whose_logging_cannot_start_stops_before_any_move(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub: OpenAIStub
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise RuntimeError("bad W&B key")

    monkeypatch.setattr(gepa_live, "start_tracker", refuse)

    run = live_run(monkeypatch, tmp_path, stub)

    assert run.code == 2
    assert "cannot start logging: RuntimeError: bad W&B key" in run.stderr
    assert not (run.work_dir / "episodes.jsonl").exists()


class Tracker:
    url = "https://wandb.example/run/1"

    def __init__(self) -> None:
        self.steps: list[tuple[int, Mapping[str, Any]]] = []
        self.finished: list[tuple[Path, list[str], bool]] = []

    def log_step(self, index: int, values: Mapping[str, Any]) -> None:
        self.steps.append((index, values))

    def finish(self, work_dir: Path, key_names: list[str], *, failed: bool) -> list[str]:
        self.finished.append((work_dir, list(key_names), failed))
        return ["the upload failed"]


@pytest.mark.functional
def test_a_logged_run_logs_every_step_and_a_logging_problem_fails_a_passing_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub: OpenAIStub
) -> None:
    tracker = Tracker()
    monkeypatch.setattr(gepa_live, "start_tracker", lambda *args, **kwargs: tracker)
    schedule = [{"kind": "harness", "label": "pre_training"}]

    run = live_run(monkeypatch, tmp_path, stub, schedule=schedule)

    assert [index for index, _ in tracker.steps] == [1]
    assert tracker.steps[0][1]["usage/episodes"] == len(run.rows("episodes"))
    assert tracker.finished == [(run.work_dir, [KEY_ENV], True)]
    assert run.stdout.count(f"wandb: {Tracker.url}") == 2
    assert "logging: the upload failed" in run.stderr
    assert run.code == 1


# ---------------------------------------------------------------- each check fails for its own reason


Change = Callable[[dict[str, Any], RunLog, list[dict[str, Any]]], None]


def checked(run: LiveRun, tmp_path: Path, change: Change) -> dict[str, Check]:
    """The checks of ``run`` after ``change(summary, run_log, records)``, on copies of all three."""
    assert run.run_log is not None
    work_dir = tmp_path / "copy"
    (work_dir / "releases").mkdir(parents=True)
    summary = copy.deepcopy(run.summary)
    run_log = dataclasses.replace(copy.deepcopy(run.run_log), work_dir=work_dir)
    records = [json.loads(line) for line in (run.work_dir / "releases/commits.jsonl").read_text().splitlines()]
    change(summary, run_log, records)
    (work_dir / "releases" / "commits.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return {check.name: check for check in check_run(summary, run_log, work_dir, run.validation_ids, MAX_METRIC_CALLS)}


def gated_move(run: LiveRun, op: str) -> int:
    records = [json.loads(line) for line in (run.work_dir / "releases/commits.jsonl").read_text().splitlines()]
    steps = run.summary["steps"]
    written = [s for s in steps if s["result"] in ("published", "rejected", "skipped", "restored")]
    return next(int(step["index"]) for step, record in zip(written, records[1:], strict=True) if record["op"] == op)


@pytest.mark.functional
def test_the_unchanged_run_passes_every_check_it_is_fed(passing: LiveRun, tmp_path: Path) -> None:
    checks = checked(passing, tmp_path, lambda summary, run_log, records: None)

    assert all(check.passed for check in checks.values()), [c for c in checks.values() if not c.passed]


@pytest.mark.functional
@pytest.mark.parametrize(
    ("name", "change", "detail"),
    [
        (
            "2 reflection prompt",
            lambda s, log, r: next(e for e in log.episodes if e["phase"] == "parent").update(final_output="invented"),
            "missing ['final output']",
        ),
        (
            "3 gate on real scores",
            lambda s, log, r: next(e for e in log.episodes if e["phase"] == "gate").update(score=0.5),
            "stored scores are not the graded gate episodes",
        ),
        (
            "5c every episode accounted for",
            lambda s, log, r: setattr(log, "started", log.started + 1),
            "recorded",
        ),
        (
            "7a failed reflections",
            lambda s, log, r: log.reflections.append({"move": 1, "ok": False}),
            "move 1: not a skip",
        ),
        (
            "7b abandoned moves",
            lambda s, log, r: log.faults.append({"move": 1, "episodes_started": 0, "usage": {}}),
            "move 1: the campaign recorded it",
        ),
        (
            "6 rollback",
            lambda s, log, r: next(st for st in s["steps"] if st["move"] == "rollback").update(state_restored=False),
            "state_restored is false",
        ),
    ],
    ids=["prompt", "gate-score", "episode-count", "failed-reflection", "fault", "rollback"],
)
def test_a_changed_record_fails_its_check_with_the_reason(
    passing: LiveRun, tmp_path: Path, name: str, change: Any, detail: str
) -> None:
    checks = checked(passing, tmp_path, change)

    assert not checks[name].passed and detail in checks[name].detail
    assert not checks[name].not_exercised


@pytest.mark.functional
def test_a_child_that_is_not_the_extracted_reply_fails_the_extraction_check(passing: LiveRun, tmp_path: Path) -> None:
    move = gated_move(passing, "publish")

    def change(summary: dict[str, Any], run_log: RunLog, records: list[dict[str, Any]]) -> None:
        row = next(r for r in reversed(run_log.reflections) if r["move"] == move and r.get("ok"))
        row["reply"] = "```\nsomething else\n```"

    checks = checked(passing, tmp_path, change)

    assert not checks["1 extracted"].passed
    assert checks["1 extracted"].detail.startswith(f"move {move}: ")
    assert checks["1 extracted"].detail.endswith(" is not the extracted reflection reply")


@pytest.mark.functional
def test_a_move_that_started_from_another_state_fails_the_metric_call_identities(
    passing: LiveRun, tmp_path: Path
) -> None:
    published = gated_move(passing, "publish")

    def change(summary: dict[str, Any], run_log: RunLog, records: list[dict[str, Any]]) -> None:
        run_log.starts[published] = run_log.starts[1]

    checks = checked(passing, tmp_path, change)

    assert not checks["5a metric calls per move"].passed
    assert f"move {published}: charged" in checks["5a metric calls per move"].detail


@pytest.mark.functional
def test_a_rejected_candidate_missing_from_a_later_archive_fails_its_check(passing: LiveRun, tmp_path: Path) -> None:
    rejected, published = gated_move(passing, "reject"), gated_move(passing, "publish")

    def change(summary: dict[str, Any], run_log: RunLog, records: list[dict[str, Any]]) -> None:
        # Record 0 is release 0; the next two are the rejected move's and the published one's.
        later = records[2]["candidate"]["method_state"]["candidates"][1]
        later["components"] = dict.fromkeys(later["components"], "a different text")

    checks = checked(passing, tmp_path, change)

    assert (rejected, published) == (1, 2)
    assert not checks["4 rejected stays"].passed
    assert checks["4 rejected stays"].detail == "move 1's candidate is missing at move 2"


@pytest.mark.functional
def test_a_check_that_cannot_read_the_records_fails_instead_of_losing_the_summary(
    passing: LiveRun, tmp_path: Path
) -> None:
    def change(summary: dict[str, Any], run_log: RunLog, records: list[dict[str, Any]]) -> None:
        for record in records[1:]:
            if record["op"] == "reject":
                del record["candidate"]["method_state"]

    checks = checked(passing, tmp_path, change)

    for name in ("1 extracted", "3 gate on real scores", "4 rejected stays"):
        assert not checks[name].passed
        assert checks[name].detail == "the check could not read the run: KeyError: 'method_state'"
    assert checks["2 reflection prompt"].passed and checks["budget stop"].passed


@pytest.mark.functional
def test_an_unreadable_published_record_fails_the_checks_instead_of_losing_the_summary(
    passing: LiveRun, tmp_path: Path
) -> None:
    def change(summary: dict[str, Any], run_log: RunLog, records: list[dict[str, Any]]) -> None:
        del next(record for record in records[1:] if record["op"] == "publish")["candidate"]["harness"]

    checks = checked(passing, tmp_path, change)

    for name in ("1 extracted", "3 gate on real scores"):
        assert not checks[name].passed
        assert checks[name].detail == "the check could not read the run: KeyError: 'harness'"
    assert checks["2 reflection prompt"].passed and checks["budget stop"].passed


@pytest.mark.functional
def test_a_run_short_of_its_budget_fails_the_budget_stop(passing: LiveRun, tmp_path: Path) -> None:
    assert passing.run_log is not None
    checks = check_run(passing.summary, passing.run_log, passing.work_dir, passing.validation_ids, 10_000)

    budget = next(check for check in checks if check.name == "budget stop")
    assert not budget.passed and budget.detail.endswith("of 10000")


# ---------------------------------------------------------------- pieces


@pytest.mark.unit
@pytest.mark.parametrize(
    "schedule",
    [
        [],
        [{"kind": "harness"}, {"kind": "harness", "repeat": 3}],
        [{"kind": "harness_until_publish", "max_moves": 2}, {"kind": "rollback", "to": "first_publish"}],
        [{"kind": "rollback", "to": "r0003"}],
    ],
)
def test_a_schedule_of_harness_moves_and_rollbacks_is_accepted(schedule: list[dict[str, Any]]) -> None:
    check_schedule(schedule)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"kind": "harness", "repeat": True}, "repeat must be an integer >= 1, got True"),
        ({"kind": "harness", "repeat": "2"}, "repeat must be an integer >= 1, got '2'"),
        ({"kind": "harness_until_publish", "max_moves": 0}, "max_moves must be an integer >= 1, got 0"),
        ({"kind": "rollback", "to": 3}, "a rollback names a release id"),
        ({"kind": "stop"}, "a live run has harness moves and rollbacks, not 'stop'"),
    ],
)
def test_a_schedule_entry_the_run_cannot_run_is_refused_with_its_number(entry: dict[str, Any], message: str) -> None:
    with pytest.raises(ContractError, match=f"schedule step 2: {message}"):
        check_schedule([{"kind": "harness"}, entry])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "base_url", "expected"),
    [
        ("openai/gpt-4.1-mini", "https://api.openai.com/v1", None),
        ("openai/gpt-4.1-mini", " HTTPS://API.OPENAI.COM/v1/ ", None),
        ("openai/gpt-4.1-mini", "https://api.openai.com/v1?x=1", "https://api.openai.com/v1?x=1"),
        ("openai/gpt-4.1-mini", "http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1"),
        ("anthropic/claude", "https://api.openai.com/v1", "https://api.openai.com/v1"),
    ],
)
def test_only_the_provider_s_own_api_is_priced_whatever_the_url_s_case_or_trailing_slash(
    model: str, base_url: str, expected: str | None
) -> None:
    assert pricing_api_base(model, base_url) == expected


@pytest.mark.unit
def test_the_options_price_the_policy_only_behind_its_provider_s_own_api() -> None:
    hosted = OpenAIEndpoints("https://api.openai.com/v1", "gpt-4.1-mini")
    config = {"model": {"provider": "openai", "name": "gpt-4.1-mini"}, "live": {"max_consecutive_faults": 5}}

    assert live_options(config, hosted) == LiveOptions(5, "openai/gpt-4.1-mini", None)
    assert live_options(config, OpenAIEndpoints("http://stub/v1", "m")) == LiveOptions(
        5, "openai/gpt-4.1-mini", "http://stub/v1"
    )
    assert live_options(config, object()) == LiveOptions(5)
    assert live_options({}, hosted) == LiveOptions(3)


@pytest.mark.unit
def test_every_checkpoint_resolves_to_the_one_hosted_model() -> None:
    endpoints = gepa_live.openai_endpoints(None, "https://api.openai.com/v1", {"provider": "openai", "name": "m"})  # type: ignore[arg-type]

    endpoint = endpoints.resolve(None)  # type: ignore[arg-type]
    assert (endpoint.base_url, endpoint.served_model, endpoint.runtime) == (
        "https://api.openai.com/v1",
        "m",
        "openai_chat",
    )
    assert gepa_live.model_name("plain") == "plain" and gepa_live.litellm_model("openai/m") == "openai/m"
    assert gepa_live.litellm_model({"provider": "openai", "name": "m"}) == "openai/m"


class Chat:
    def __init__(self, outcome: ChatReply | Exception) -> None:
        self.outcome = outcome

    def complete(self, messages: Any, *, max_tokens: int, temperature: float) -> ChatReply:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.mark.unit
def test_a_reflection_call_is_recorded_and_ends_the_parent_s_minibatch(tmp_path: Path) -> None:
    run_log = RunLog(tmp_path, move=4)
    chat = RecordingChatModel(Chat(ChatReply("```\nnew\n```", 11, 3, 0.002)), run_log)  # type: ignore[arg-type]

    reply = chat.complete([ChatMessage("user", "prompt text")], max_tokens=10, temperature=1.0)

    assert reply.text == "```\nnew\n```" and run_log.phase == "child"
    assert run_log.reflections == [
        {
            "move": 4,
            "prompt": "prompt text",
            "ok": True,
            "reply": "```\nnew\n```",
            "input_tokens": 11,
            "output_tokens": 3,
            "cost_usd": 0.002,
        }
    ]


@pytest.mark.unit
def test_a_failed_reflection_call_is_recorded_and_raised(tmp_path: Path) -> None:
    run_log = RunLog(tmp_path, move=2)
    chat = RecordingChatModel(Chat(ChatModelError("openai/m", "APIError", 503, 3)), run_log)  # type: ignore[arg-type]

    with pytest.raises(ChatModelError):
        chat.complete([ChatMessage("user", "p")], max_tokens=10, temperature=1.0)

    assert run_log.reflections == [
        {"move": 2, "prompt": "p", "ok": False, "error_type": "APIError", "status": 503, "attempts": 3}
    ]


@pytest.mark.unit
def test_a_grade_is_written_into_the_latest_row_of_its_episode(tmp_path: Path) -> None:
    class Inner:
        async def grade(self, task: Task, outcome: EpisodeOutcome) -> Grade:
            return Grade(0.75, "3 of 4 tests pass.")

    run_log = RunLog(tmp_path)
    run_log.episodes += [{"workspace": "w1", "score": None}, {"workspace": "w2", "score": None}]
    run_log.episodes += [{"workspace": "w1", "score": None}]
    grader = RecordingGrader(Inner(), run_log)  # type: ignore[arg-type]

    asyncio.run(grader.grade(Task("t", "p"), EpisodeOutcome("t", "s", "c", 0, workspace_location="w1")))

    assert [row["score"] for row in run_log.episodes] == [None, None, 0.75]
    assert run_log.episodes[2]["feedback"] == "3 of 4 tests pass."


@pytest.mark.unit
def test_an_agent_s_usage_file_is_summed_skipping_lines_that_are_not_json(tmp_path: Path) -> None:
    (tmp_path / "usage.jsonl").write_text(
        '{"ok": true, "prompt_tokens": 10, "completion_tokens": 2}\n\nnot json\n'
        '{"ok": false, "status": 429}\n{"ok": true, "prompt_tokens": null, "completion_tokens": 5}\n'
    )

    assert gepa_live._usage(tmp_path) == {"calls": 3, "failed_calls": 1, "prompt_tokens": 10, "completion_tokens": 7}
    assert (
        gepa_live._usage(None)
        == gepa_live._usage(tmp_path / "missing")
        == {
            "calls": 0,
            "failed_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("result", "error"),
    [
        (None, None),
        ("not json", None),
        ("[1, 2]", None),
        ('{"policy_error": "auth"}', None),
        ('{"policy_error": {"kind": "auth", "status": 401}}', {"kind": "auth", "status": 401}),
    ],
    ids=["missing", "not-json", "not-an-object", "not-a-mapping", "error"],
)
def test_only_a_well_formed_policy_error_counts_as_an_unavailable_policy(
    tmp_path: Path, result: str | None, error: dict[str, Any] | None
) -> None:
    if result is not None:
        (tmp_path / "result.json").write_text(result)

    assert gepa_live._policy_error(tmp_path) == error


def _usage_log(tmp_path: Path) -> RunLog:
    run_log = RunLog(tmp_path)
    usage = {"calls": 2, "failed_calls": 1, "prompt_tokens": 100, "completion_tokens": 10}
    run_log.episodes += [{"move": 1, "usage": usage}, {"move": 2, "usage": usage}]
    run_log.reflections += [
        {"move": 1, "ok": True, "input_tokens": 30, "output_tokens": 5, "cost_usd": 0.01},
        {"move": 2, "ok": False, "attempts": 3},
        {"move": 2, "ok": True, "input_tokens": 20, "output_tokens": 4, "cost_usd": None},
    ]
    run_log.faults.append({"move": 3, "usage": {"calls": 1, "failed_calls": 1}})
    return run_log


@pytest.mark.unit
def test_a_move_s_usage_counts_its_own_calls_and_prices_the_policy_s_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, FakeLiteLLM(prices={"openai/m": (0.001, 0.002)}))
    run_log = _usage_log(tmp_path)

    first = move_usage(run_log, 1, "openai/m")
    second = move_usage(run_log, 2, "openai/m", "http://127.0.0.1:1/v1")

    assert first == {
        "episodes": 1,
        "policy_calls": 2,
        "policy_failed_calls": 1,
        "policy_prompt_tokens": 100,
        "policy_completion_tokens": 10,
        "policy_cost_usd": pytest.approx(100 * 0.001 + 10 * 0.002),
        "reflection_calls": 1,
        "reflection_failed_calls": 0,
        "reflection_input_tokens": 30,
        "reflection_output_tokens": 5,
        "reflection_cost_usd": 0.01,
    }
    # A policy behind another server has no known cost, nor does a move with an unpriced reflection.
    assert (second["policy_cost_usd"], second["reflection_cost_usd"]) == (None, None)
    assert (second["reflection_calls"], second["reflection_failed_calls"]) == (2, 1)
    assert move_usage(run_log, 1, None)["policy_cost_usd"] is None


@pytest.mark.unit
def test_the_run_s_cost_is_known_only_when_every_call_s_cost_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, FakeLiteLLM(prices={"openai/m": (0.001, 0.002)}))
    run_log = _usage_log(tmp_path)

    report = usage_report(run_log, "openai/m")

    assert report["reflection"] == {
        "calls": 3,
        "failed_calls": 1,
        "attempts_in_failed_calls": 3,
        "input_tokens": 50,
        "output_tokens": 9,
        "cost_usd": None,
    }
    assert report["policy"]["episodes"] == 2 and report["policy"]["cost_usd"] == pytest.approx(0.24)
    assert report["abandoned_moves"] == {
        "moves": 1,
        "calls": 1,
        "failed_calls": 1,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    assert report["total_cost_usd"] is None
    run_log.reflections[2]["cost_usd"] = 0.02
    assert usage_report(run_log, "openai/m")["total_cost_usd"] == pytest.approx(0.24 + 0.03)


@pytest.mark.unit
def test_a_rerun_is_suggested_only_when_every_failure_is_an_unexercised_rerun_check() -> None:
    def check(name: str, passed: bool, not_exercised: bool = False) -> dict[str, Any]:
        return {"name": name, "passed": passed, "not_exercised": not_exercised}

    assert only_not_exercised([check("1 extracted", True), check("6 rollback", False, True)])
    assert not only_not_exercised([check("1 extracted", True)])
    assert not only_not_exercised([check("6 rollback", False, True), check("budget stop", False)])
    assert not only_not_exercised([check("1 extracted", False, True)])


@pytest.mark.unit
def test_the_key_scan_finds_a_key_in_any_file_or_output_and_never_prints_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SCAN_KEY", "sk-0123456789")
    monkeypatch.setenv("SHORT_KEY", "short")
    (tmp_path / "deep" / "er").mkdir(parents=True)
    (tmp_path / "deep" / "er" / "log.txt").write_text("Authorization: Bearer sk-0123456789")
    (tmp_path / "clean.txt").write_text("nothing here, not even short")
    (tmp_path / "link").symlink_to(tmp_path / "deep" / "er" / "log.txt")

    check = key_scan(tmp_path, ["SCAN_KEY", "SHORT_KEY", "UNSET_KEY"], extra=[b"ok", b"echo sk-0123456789"])

    assert not check.passed
    assert check.detail == "2 files, 2 hits: ['deep/er/log.txt', 'output 1']"
    assert "sk-0123456789" not in check.detail
    assert key_scan(tmp_path, ["SHORT_KEY"]).detail == "no key to scan for: the named variables are unset or short"


@pytest.mark.unit
def test_the_key_names_are_the_required_ones_every_api_key_env_and_w_and_b_s_when_logging() -> None:
    config = {
        "requires": {"env": ["A_KEY"]},
        "components": {"runner": {"args": {"api_key_env": "B_KEY"}}, "list": [{"api_key_env": "C_KEY"}, "x"]},
        "logging": {"wandb": {}},
    }

    assert gepa_live._key_names(config) == sorted({"A_KEY", "B_KEY", "C_KEY", gepa_live.WANDB_KEY_ENV})
    assert gepa_live._key_names({"components": {"api_key_env": 3}}) == []
    assert gepa_live._key_names(["not", "a", "mapping"]) == []


@pytest.mark.unit
def test_a_shared_run_log_slot_holding_something_else_is_refused(tmp_path: Path) -> None:
    context = PipelineContext(root=tmp_path, work_dir=tmp_path)
    context.shared[RUN_LOG] = {"not": "a run log"}

    with pytest.raises(TypeError, match="is not a RunLog"):
        gepa_live._run_log(context)


class Chain:
    def __init__(self, releases: Mapping[str, Any]) -> None:
        self.releases = releases

    def head(self) -> Any:
        return type("Head", (), {"release_id": "r0002", "method_state": {"seed": 1}})()

    def get(self, release_id: str) -> Any:
        if release_id not in self.releases:
            raise ReleaseError(f"no release {release_id!r}")
        return self.releases[release_id]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("to", "reason"),
    [(None, "no harness release was published before the rollback"), ("r0099", "no release 'r0099'")],
)
def test_a_rollback_with_nothing_to_restore_is_not_runnable_and_restores_nothing(
    tmp_path: Path, to: str | None, reason: str
) -> None:
    campaign = type("Campaign", (), {"chain": Chain({})})()
    run_log = RunLog(tmp_path)

    step = gepa_live._rollback(campaign, run_log, 5, to)

    assert step == {
        "index": 5,
        "move": "rollback",
        "result": "not runnable",
        "to": to or gepa_live.FIRST_PUBLISH,
        "reason": reason,
    }
    assert run_log.ends == []
