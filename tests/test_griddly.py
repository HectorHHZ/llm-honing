"""Contracts need no game installation; optional tests execute the real Spiders rules."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from honing.backends.griddly import (
    GriddlyEpisodeRunner,
    GriddlyGrader,
    GriddlyScenario,
    GriddlySuite,
    build_suite,
    load_suite,
    seed_harness,
)
from honing.backends.griddly.decisions import decision_samples
from honing.backends.griddly.demonstrations import GriddlyDemonstrationProvider
from honing.backends.griddly.engine import GriddlyEngine, canonical, digest
from honing.backends.griddly.harness import CONTROLLER_PATH, load_harness, parse_harness, write_harness
from honing.backends.griddly.match import run_match
from honing.backends.griddly.model import DecisionClient, baseline, make_request, validate_reply
from honing.backends.griddly.replay import audit_run, verify_replay
from honing.backends.griddly.rollouts import GriddlyRolloutProvider
from honing.backends.jev.server import DecisionServer
from honing.core import (
    CampaignRuleError,
    ContractError,
    DataContract,
    EpisodeRunner,
    HarnessFile,
    PolicyRef,
    TaskGrader,
    run_graded,
)
from tests.fakes.griddly import PublicStateModel, StandInEndpoints
from tests.fakes.model import FakeModelSource, FakeWorld

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations/griddly"
SEED = seed_harness(INTEGRATION)
REAL = importlib.util.find_spec("griddly") is not None
native = pytest.mark.skipif(not REAL, reason="optional pinned Griddly runtime is not installed")


def public_frame() -> dict[str, Any]:
    return {
        "step": 0,
        "observation": {
            "grid": {"Width": 4, "Height": 4},
            "tick": 0,
            "objects": [
                {"Name": "gnome", "Location": [1, 1], "Orientation": "NONE"},
                {"Name": "gem", "Location": [2, 2], "Orientation": "NONE"},
                {"Name": "spider", "Location": [2, 1], "Orientation": "LEFT"},
            ],
        },
        "legal_actions": ["WAIT", "TURN_LEFT", "TURN_RIGHT", "BACKWARD"],
        "native_state": {"DelayedActions": ["private future action"], "Hash": 123},
    }


def reply_for(request: dict[str, Any]) -> dict[str, Any]:
    return {"id": request["id"], **PublicStateModel().predict(request["parts"], request["questions"])}


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [
        {"seed": -1},
        {"seed": True},
        {"level_id": 5},
        {"level_id": True},
        {"max_steps": 0},
        {"max_steps": True},
        {"observation": "local"},
    ],
)
def test_invalid_scenarios_fail_before_native_import(config: dict[str, Any]) -> None:
    with pytest.raises(ContractError):
        GriddlyScenario(**config)


@pytest.mark.unit
def test_suite_is_committed_and_layouts_are_disjoint() -> None:
    suite = load_suite(INTEGRATION / "tasks/suite.jsonl")
    assert list(suite.records) == build_suite()
    assert [len(suite.tasks(split)) for split in ("train", "val", "heldout")] == [8, 8, 4]
    records = build_suite()
    records[8]["griddly"] = records[0]["griddly"]
    with pytest.raises(ContractError, match="share levels or seeds"):
        GriddlySuite(tuple(records))


@pytest.mark.unit
def test_harness_delivery_and_task_settings_are_separate(tmp_path: Path) -> None:
    spec = SEED.with_file("notes/route.md", HarnessFile("instructions", "Observe enemy positions."))
    write_harness(spec, tmp_path / "harness")
    assert load_harness(tmp_path / "harness") == spec
    assert "Observe enemy" in parse_harness(spec).notes
    bad = SEED.with_file(CONTROLLER_PATH, HarnessFile("settings", '{"history_steps": 3, "max_steps": 10000}'))
    with pytest.raises(CampaignRuleError):
        parse_harness(bad)
    (tmp_path / "outside").write_text("private")
    (tmp_path / "harness/index.json").write_text('[{"path":"../outside","kind":"instructions"}]')
    with pytest.raises(CampaignRuleError, match="leaves"):
        load_harness(tmp_path / "harness")


@pytest.mark.unit
def test_request_contains_only_public_state_and_exact_candidates() -> None:
    request = make_request(public_frame(), GriddlyScenario(), parse_harness(SEED), [])
    assert set(request["questions"]["action"]["criteria"]) == set(public_frame()["legal_actions"])
    assert "private future action" not in canonical(request) and "DelayedActions" not in canonical(request)
    assert "NONE facing" in request["parts"]["essential"][0]
    assert baseline(public_frame()) in public_frame()["legal_actions"]
    frame = public_frame()
    frame["legal_actions"] = ["WAIT"]
    with pytest.raises(ValueError, match="forced"):
        make_request(frame, GriddlyScenario(), parse_harness(SEED), [])


@pytest.mark.unit
@pytest.mark.parametrize(
    "fault",
    ["id", "state", "trim", "nan", "inf", "negative", "over_one", "bool", "zero", "missing", "sum", "choice", "audit"],
)
def test_model_replies_cannot_change_context_or_distribution(fault: str) -> None:
    request = make_request(public_frame(), GriddlyScenario(), parse_harness(SEED), [])
    reply = reply_for(request)
    assert validate_reply(request, reply)
    answer = reply["answers"]["action"]
    if fault == "id":
        reply["id"] = "other"
    elif fault == "state":
        reply["stateText"] += "extra"
    elif fault == "trim":
        reply["trimmed"] = ["essential"]
    elif fault == "nan":
        answer["probabilities"]["WAIT"] = float("nan")
    elif fault == "inf":
        answer["probabilities"]["WAIT"] = float("inf")
    elif fault == "negative":
        answer["probabilities"]["WAIT"] = -0.01
    elif fault == "over_one":
        answer["probabilities"]["WAIT"] = 1.01
    elif fault == "bool":
        answer["probabilities"]["WAIT"] = True
    elif fault == "zero":
        answer["probabilities"] = dict.fromkeys(answer["probabilities"], 0.0)
    elif fault == "missing":
        del answer["probabilities"]["WAIT"]
    elif fault == "sum":
        answer["probabilities"]["WAIT"] = 0.99
    elif fault == "choice":
        answer["choice"] = min(answer["probabilities"], key=answer["probabilities"].get)
    else:
        reply["tokenAudit"]["action"]["stateTruncated"] = True
    with pytest.raises(ValueError):
        validate_reply(request, reply)


@pytest.mark.unit
@pytest.mark.parametrize(
    "values",
    [
        [0.5001, 0.5],
        [0.3333, 0.3333, 0.3333],
        [0.3334, 0.2222, 0.2222, 0.2223],
        [0.3333, 0.3333, 0.1111, 0.1111, 0.1111],
    ],
)
def test_rounded_probabilities_are_normalized_without_changing_the_reply(values: list[float]) -> None:
    frame = public_frame()
    frame["legal_actions"] = ["WAIT", "TURN_LEFT", "FORWARD", "TURN_RIGHT", "BACKWARD"][: len(values)]
    request = make_request(frame, GriddlyScenario(), parse_harness(SEED), [])
    probabilities = dict(zip(frame["legal_actions"], values, strict=True))
    reply = reply_for(request)
    reply["answers"]["action"].update(probabilities=probabilities, choice=max(probabilities, key=probabilities.get))
    original = canonical(reply)
    normalized = validate_reply(request, reply)
    assert math.fsum(normalized.values()) == pytest.approx(1.0, abs=1e-15)
    assert normalized == {name: p / math.fsum(values) for name, p in probabilities.items()}
    assert canonical(reply) == original


@pytest.mark.unit
@pytest.mark.parametrize(
    "values",
    [
        [0.5001] * 2,
        [0.4999] * 2,
        [0.3334] * 3,
        [0.3332] * 3,
        [0.2501] * 4,
        [0.2499] * 4,
        [0.2001] * 5,
        [0.1999] * 5,
    ],
)
def test_sum_error_beyond_four_decimal_rounding_is_refused(values: list[float]) -> None:
    frame = public_frame()
    frame["legal_actions"] = ["WAIT", "TURN_LEFT", "FORWARD", "TURN_RIGHT", "BACKWARD"][: len(values)]
    request = make_request(frame, GriddlyScenario(), parse_harness(SEED), [])
    reply = reply_for(request)
    probabilities = dict(zip(frame["legal_actions"], values, strict=True))
    reply["answers"]["action"].update(probabilities=probabilities, choice=max(probabilities, key=probabilities.get))
    with pytest.raises(ValueError, match="sum to one"):
        validate_reply(request, reply)


@pytest.fixture
def rounded_server() -> Any:
    class RoundedModel(PublicStateModel):
        def predict(self, parts: Mapping[str, object], questions: Mapping[str, object]) -> Mapping[str, object]:
            reply: Any = super().predict(parts, questions)
            answer = reply["answers"]["action"]
            answer["probabilities"] = {name: round(p, 4) for name, p in answer["probabilities"].items()}
            return reply

    server = DecisionServer(("127.0.0.1", 0), RoundedModel())
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.stop()
        thread.join(timeout=5)


@pytest.mark.integration
def test_http_rounded_response_preserves_raw_values_and_normalizes_training(rounded_server: DecisionServer) -> None:
    request = make_request(public_frame(), GriddlyScenario(), parse_harness(SEED), [])
    reply = DecisionClient(rounded_server.url).decide(request)
    raw = reply["answers"]["action"]["probabilities"]
    assert math.fsum(raw.values()) == pytest.approx(1.0001)
    original = canonical(reply)
    expected = {name: value / math.fsum(raw.values()) for name, value in raw.items()}
    row = {"request": request, "response": reply, "chosen": {"action": "WAIT"}, "forced_action": False}
    sample = decision_samples([row], group="train", reward=1.0, behavior=True)[0]
    assert dict(dict(sample.behavior_probs)["action"]) == expected
    assert dict(sample.chosen)["action"] == "WAIT"
    assert canonical(reply) == original


@pytest.mark.unit
def test_training_label_is_sampled_choice_not_server_argmax() -> None:
    request = make_request(public_frame(), GriddlyScenario(), parse_harness(SEED), [])
    reply = reply_for(request)
    sampled = next(
        name for name in request["questions"]["action"]["criteria"] if name != reply["answers"]["action"]["choice"]
    )
    row = {"request": request, "response": reply, "chosen": {"action": sampled}, "forced_action": False}
    samples = decision_samples(
        [row, {"forced_action": True}], group="train", reward=1.0, release_id="r1", harness_id="h1", behavior=True
    )
    assert len(samples) == 1 and dict(samples[0].chosen)["action"] == sampled
    assert dict(dict(samples[0].behavior_probs)["action"]) == reply["answers"]["action"]["probabilities"]
    assert samples[0].source_release == "r1" and samples[0].source_harness == "h1"


@pytest.mark.unit
def test_base_import_and_help_do_not_load_native_packages() -> None:
    code = (
        "import sys; sys.modules['griddly']=None; sys.modules['numpy']=None; "
        "import honing.backends.griddly; from honing.backends.griddly.match import main; main(['--help'])"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--decision-server" in result.stdout


@pytest.mark.unit
def test_components_fit_existing_protocols(tmp_path: Path) -> None:
    from honing.backends.griddly import e2e
    from honing.backends.jev import JevEndpoints
    from honing.e2e.components import PipelineContext, build_component

    context = PipelineContext(ROOT, tmp_path, components={"endpoints": JevEndpoints({})})
    assert e2e.harness_seed(context).agent == SEED.agent
    assert len(e2e.tasks(context, limit=1)) == 1
    for name, protocol in (("episode_runner", EpisodeRunner), ("grader", TaskGrader)):
        assert isinstance(
            build_component({"factory": f"honing.backends.griddly.e2e:{name}"}, context, name, protocol), protocol
        )


@pytest.fixture
def endpoints() -> Any:
    value = StandInEndpoints()
    try:
        yield value
    finally:
        value.close()


@pytest.fixture(scope="module")
def successful_scenario() -> GriddlyScenario:
    # Native random-number distributions may differ across platforms.
    for seed in [7, *range(1, 7), *range(8, 21)]:
        scenario = GriddlyScenario(seed=seed, max_steps=15)
        with GriddlyEngine(scenario) as engine:
            frame = engine.reset()
            while frame["status"] == "running":
                frame = engine.step(baseline(frame))
            if frame["status"] == "won":
                return scenario
    raise AssertionError("no successful native baseline trajectory found")


@native
@pytest.mark.integration
def test_native_state_legality_noop_reset_and_terminal() -> None:
    engine = GriddlyEngine(GriddlyScenario(seed=7, max_steps=5))
    first = engine.reset()
    assert first["observation"]["grid"] == {"Width": 6, "Height": 6}
    assert first["legal_actions"] == ["WAIT", "TURN_LEFT", "TURN_RIGHT", "BACKWARD"]
    assert engine.observe() == first
    with pytest.raises(ValueError):
        engine.step("FORWARD")
    engine.step("WAIT")
    assert engine.reset() == first
    for _ in range(5):
        last = engine.step("WAIT")
    assert last["status"] == "step_limit" and last["truncated"] and not last["terminated"]
    assert last["legal_actions"] == []
    with pytest.raises(RuntimeError):
        engine.step("WAIT")
    engine.close()
    engine.close()
    with pytest.raises(RuntimeError):
        engine.reset()


@native
@pytest.mark.integration
@pytest.mark.parametrize("policy", ["baseline", "random", "teacher", "actions"])
def test_native_runs_replay_in_a_fresh_process(
    tmp_path: Path, policy: str, successful_scenario: GriddlyScenario
) -> None:
    out = tmp_path / policy
    budget = 15
    result = run_match(
        successful_scenario,
        out,
        SEED,
        policy=policy,
        actions=("WAIT",) * budget if policy == "actions" else (),
    )
    assert result["steps"] <= budget
    assert verify_replay(out)["verified"]
    checked = subprocess.run(
        [sys.executable, str(INTEGRATION / "scripts/verify_replay.py"), str(out)], capture_output=True, text=True
    )
    assert checked.returncode == 0, checked.stderr
    assert json.loads(checked.stdout.splitlines()[-1])["final_frame_hash"] == result["final_frame_hash"]
    if policy in ("baseline", "teacher"):
        assert result["success"]
    with pytest.raises(FileExistsError):
        run_match(GriddlyScenario(), out, SEED)


@native
@pytest.mark.integration
def test_real_engine_http_runner_grader_and_sampling(
    tmp_path: Path, endpoints: StandInEndpoints, successful_scenario: GriddlyScenario
) -> None:
    suite = GriddlySuite(
        (
            {
                "task_id": "native-val",
                "split": "val",
                "prompt": "Reach the gem",
                "griddly": successful_scenario.record(),
            },
        )
    )
    snapshot = FakeModelSource(FakeWorld()).materialize(tmp_path)
    policy = PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")
    runner = GriddlyEpisodeRunner(
        suite=suite,
        endpoints=endpoints,
        integration_dir=INTEGRATION,
        runs_dir=tmp_path / "runs",
        command=[sys.executable, str(INTEGRATION / "scripts/match.py")],
    )
    grader = GriddlyGrader(
        suite=suite,
        integration_dir=INTEGRATION,
        command=[sys.executable, str(INTEGRATION / "scripts/verify_replay.py")],
    )
    task = suite.tasks("val")[0]
    record = asyncio.run(run_graded(runner, grader, SEED, task, policy))
    assert record.score == 1
    run = Path(record.outcome.workspace_location)
    manifest, _rows, _result = audit_run(run)
    assert manifest["controller"] == "model" and manifest["model"]["metadata"]["family"] == "stand-in"
    sampled = asyncio.run(run_graded(runner.sampling(), grader, SEED, task, policy))
    assert sampled.score in (0, 1)
    assert audit_run(Path(sampled.outcome.workspace_location))[0]["kind"] == "rollout"
    forged = replace(record.outcome, policy_checksum="different")
    with pytest.raises(ContractError, match="does not match"):
        asyncio.run(grader.grade(task, forged))


@native
@pytest.mark.integration
def test_rounded_http_sampling_audit_replay_and_training_agree(tmp_path: Path, rounded_server: DecisionServer) -> None:
    out = tmp_path / "rounded"
    result = run_match(
        GriddlyScenario(seed=7, max_steps=15),
        out,
        SEED,
        policy="model",
        decision_server=rounded_server.url,
        sample=True,
        policy_seed=93,
    )
    assert verify_replay(out)["steps"] == result["steps"]
    rows = [json.loads(line) for line in (out / "decisions.jsonl").read_text().splitlines()]
    first = rows[0]["response"]["answers"]["action"]["probabilities"]
    assert math.fsum(first.values()) == pytest.approx(1.0001)
    samples = decision_samples(rows, group="train", reward=1.0, behavior=True)
    for row, sample in zip((row for row in rows if not row["forced_action"]), samples, strict=True):
        raw = row["response"]["answers"]["action"]["probabilities"]
        assert dict(dict(sample.behavior_probs)["action"]) == {
            name: p / math.fsum(raw.values()) for name, p in raw.items()
        }
        assert dict(sample.chosen) == row["chosen"]


@native
@pytest.mark.integration
@pytest.mark.parametrize("fault", ["request", "action", "reward", "seed", "result", "harness"])
def test_audit_or_native_replay_detects_changed_records(
    tmp_path: Path, endpoints: StandInEndpoints, fault: str
) -> None:
    original = tmp_path / "original"
    run_match(
        GriddlyScenario(seed=7, max_steps=15), original, SEED, policy="model", decision_server=endpoints.server.url
    )
    run = tmp_path / "changed"
    shutil.copytree(original, run)
    if fault in ("request", "action"):
        path = run / "decisions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == "request":
            rows[0]["request"]["parts"]["essential"][0] += "invented state"
        else:
            rows[0]["applied"] = "WAIT"
        path.write_text("\n".join(canonical(row) for row in rows) + "\n")
    elif fault == "reward":
        path = run / "steps.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["frame"]["reward"] += 1
        frame = rows[1]["frame"]
        frame["frame_hash"] = digest({k: v for k, v in frame.items() if k != "frame_hash"})
        path.write_text("\n".join(canonical(row) for row in rows) + "\n")
    elif fault == "harness":
        (run / "harness/griddly/questions.json").write_text('{"action":"Different instructions"}')
    else:
        path = run / ("manifest.json" if fault == "seed" else "result.json")
        value = json.loads(path.read_text())
        if fault == "seed":
            value["scenario"]["seed"] += 1
        else:
            value["success"] = not value["success"]
        path.write_text(canonical(value))
    with pytest.raises((ValueError, CampaignRuleError)):
        verify_replay(run)


@native
@pytest.mark.integration
def test_failed_model_request_keeps_error_and_no_result(
    tmp_path: Path, endpoints: StandInEndpoints, monkeypatch: pytest.MonkeyPatch
) -> None:
    from honing.backends.griddly.model import DecisionClient

    def fail(self: Any, request: Any) -> Any:
        raise TimeoutError("stand-in timeout")

    monkeypatch.setattr(DecisionClient, "decide", fail)
    out = tmp_path / "failed"
    with pytest.raises(TimeoutError):
        run_match(GriddlyScenario(seed=7), out, SEED, policy="model", decision_server=endpoints.server.url)
    assert (out / "error.json").exists() and not (out / "result.json").exists()
    assert len((out / "steps.jsonl").read_text().splitlines()) == 1


@native
@pytest.mark.integration
def test_training_providers_accept_only_the_right_splits_and_run_kinds(
    tmp_path: Path, endpoints: StandInEndpoints
) -> None:
    suite = load_suite(INTEGRATION / "tasks/suite.jsonl")
    teacher = tmp_path / "teacher"
    task = suite.tasks("train")[0]
    run_match(suite.scenario(task.task_id), teacher, SEED, policy="teacher")
    command = [sys.executable, str(INTEGRATION / "scripts/verify_replay.py")]
    provider = GriddlyDemonstrationProvider([teacher], suite=suite, integration_dir=INTEGRATION, command=command)
    batch = provider.next_batch(DataContract("typed_decisions"))
    assert batch and batch.decisions and provider.next_batch(DataContract("typed_decisions")) is None
    assert all(d.behavior_probs is None and d.reward is None for d in batch.decisions)
    heldout = tmp_path / "heldout"
    run_match(suite.scenario(suite.tasks("heldout")[0].task_id), heldout, SEED, policy="teacher")
    with pytest.raises(ContractError, match="train"):
        GriddlyDemonstrationProvider([heldout], suite=suite, integration_dir=INTEGRATION, command=command).next_batch(
            DataContract("typed_decisions")
        )
    snapshot = FakeModelSource(FakeWorld()).materialize(tmp_path)
    policy = PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")
    runner = GriddlyEpisodeRunner(
        suite=suite,
        endpoints=endpoints,
        integration_dir=INTEGRATION,
        runs_dir=tmp_path / "rollouts",
        command=[sys.executable, str(INTEGRATION / "scripts/match.py")],
    )
    grader = GriddlyGrader(suite=suite, integration_dir=INTEGRATION, command=command)
    rollouts = GriddlyRolloutProvider(runner, grader)
    with pytest.raises(ContractError, match="train"):
        rollouts.collect(suite.tasks("val")[:1], policy, SEED, "r1", 2)
    with pytest.raises(ContractError, match="two repeats"):
        rollouts.collect([task], policy, SEED, "r1", 1)
    collected = rollouts.collect([task], policy, SEED, "r1", 2)
    assert collected.decisions and all(
        d.behavior_probs is not None and d.reward in (0, 1) for d in collected.decisions
    )
    assert collected.source_releases == frozenset({"r1"}) and collected.source_harnesses == frozenset({SEED.spec_id})


@native
@pytest.mark.integration
@pytest.mark.skipif(
    importlib.util.find_spec("imageio_ffmpeg") is None, reason="optional video encoder is not installed"
)
def test_upstream_vector_video_finishes_and_replays(tmp_path: Path, successful_scenario: GriddlyScenario) -> None:
    import imageio.v2 as imageio

    out = tmp_path / "video"
    result = subprocess.run(
        [
            sys.executable,
            str(INTEGRATION / "scripts/match.py"),
            "--policy",
            "baseline",
            "--seed",
            str(successful_scenario.seed),
            "--max-steps",
            "15",
            "--record-video",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.splitlines()[-1])
    with imageio.get_reader(out / "replay.mp4") as video:
        assert video.count_frames() == summary["steps"] + 1
        frame = video.get_data(0)
        assert frame.shape[-1] == 3 and frame.std() > 0
    assert verify_replay(out)["verified"]
