"""Exercise the actual CPU game through Honing contracts and local HTTP decisions."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from honing.backends.jev import JevEndpoints
from honing.backends.rea_issac import (
    Engine,
    ReaIssacEpisodeRunner,
    ReaIssacGrader,
    ReaIssacScenario,
    ReaIssacSuite,
    build_suite,
    load_suite,
    match,
    replay,
    seed_harness,
)
from honing.backends.rea_issac.catalog import OPTIONS
from honing.backends.rea_issac.engine import replay as replay_game
from honing.backends.rea_issac.harness import CONTROLLER_PATH, QUESTIONS_PATH, parse_harness
from honing.backends.rea_issac.model import DecisionClient, make_request, validate_reply
from honing.core import (
    CampaignRuleError,
    ContractError,
    EpisodeRunner,
    HarnessFile,
    PolicyEndpoint,
    PolicyRef,
    TaskGrader,
    run_graded,
)
from tests.fakes.model import FakeModelSource, FakeWorld

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations/rea-issac"
SEED = seed_harness(INTEGRATION)


def reply_for(request: dict[str, Any]) -> dict[str, Any]:
    choices = {"move": "E", "shoot": "N", "bomb": "NO"}
    return {
        "id": request["id"],
        "stateText": request["parts"]["essential"][0],
        "trimmed": [],
        "tokenAudit": {},
        "answers": {
            name: {"choice": choices[name], "probabilities": {a: float(a == choices[name]) for a in options}}
            for name, options in OPTIONS.items()
        },
    }


@pytest.fixture
def decision_server() -> Any:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/v1/health"
            self.send_json({"ready": True, "metadata": {"model": "local-protocol-stand-in"}})

        def do_POST(self) -> None:
            assert self.path == "/v1/decide"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            self.send_json(reply_for(request))

        def send_json(self, value: dict[str, Any]) -> None:
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": True},
        {"seed": 2**32},
        {"max_ticks": 0},
        {"max_ticks": 108001},
        {"decision_frames": True},
        {"decision_frames": 31},
    ],
)
def test_scenario_rejects_invalid_budgets(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ContractError):
        ReaIssacScenario(**kwargs)


@pytest.mark.unit
def test_suite_is_disjoint_and_matches_builder() -> None:
    suite = load_suite(INTEGRATION / "tasks/suite.jsonl")
    assert list(suite.records) == build_suite()
    assert len(suite.tasks("val", limit=1)) == 1
    records = build_suite()
    records[2]["rea_issac"] = records[0]["rea_issac"]
    with pytest.raises(ContractError, match="share seeds"):
        ReaIssacSuite(tuple(records))
    with pytest.raises(ContractError):
        suite.scenario("missing")


@pytest.mark.unit
@pytest.mark.parametrize(
    "path,kind,content",
    [
        (CONTROLLER_PATH, "settings", '{"max_ticks": 999}'),
        (CONTROLLER_PATH, "settings", '{"history_steps": true}'),
        (CONTROLLER_PATH, "settings", '{"history_steps": 33}'),
        (QUESTIONS_PATH, "instructions", '{"other": "x"}'),
        (QUESTIONS_PATH, "settings", '{"move":"x","shoot":"y","bomb":"z"}'),
    ],
)
def test_harness_cannot_change_rules(path: str, kind: str, content: str) -> None:
    with pytest.raises(CampaignRuleError):
        parse_harness(SEED.with_file(path, HarnessFile(kind, content)))  # type: ignore[arg-type]


@pytest.mark.unit
def test_context_contains_only_public_state_and_bounded_history() -> None:
    game = Engine(7, 20)
    scenario = ReaIssacScenario(7, 20)
    spec = SEED.with_file("notes/survive.md", HarnessFile("instructions", "Keep a safe escape route."))
    before = game.digest()
    request = make_request(game.observe(), scenario, parse_harness(spec), [{"test_history": "entry"}] * 8)
    state = request["parts"]["essential"][0]
    assert game.digest() == before
    assert "Keep a safe escape route." in state
    assert state.count('"test_history":"entry"') == 4
    assert '"rng"' not in state and '"state_hash"' not in state
    assert len(request["questions"]["move"]["criteria"]) == 9
    assert validate_reply(request, reply_for(request)) == {
        "move_x": 1,
        "move_y": 0,
        "shoot_x": 0,
        "shoot_y": -1,
        "bomb": False,
    }


@pytest.mark.unit
@pytest.mark.parametrize("corruption", ["id", "state", "trimmed", "truncated", "missing", "nan", "sum", "choice"])
def test_strict_model_protocol(corruption: str) -> None:
    request = make_request(Engine(7).observe(), ReaIssacScenario(), parse_harness(SEED), [])
    reply = reply_for(request)
    if corruption == "id":
        reply["id"] = "other"
    elif corruption == "state":
        reply["stateText"] = "shortened"
    elif corruption == "trimmed":
        reply["trimmed"] = ["essential"]
    elif corruption == "truncated":
        reply["tokenAudit"] = {"move": {"stateTruncated": True}}
    elif corruption == "missing":
        del reply["answers"]["move"]["probabilities"]["NONE"]
    elif corruption == "nan":
        reply["answers"]["move"]["probabilities"]["NONE"] = float("nan")
    elif corruption == "sum":
        reply["answers"]["move"]["probabilities"]["E"] = 0.5
    else:
        reply["answers"]["move"]["choice"] = "W"
    with pytest.raises(ValueError):
        validate_reply(request, reply)


@pytest.mark.unit
@pytest.mark.parametrize("url", ["file:///tmp/key", "http://user:secret@host", "https://host?token=secret"])
def test_model_urls_do_not_record_credentials(url: str) -> None:
    with pytest.raises(ValueError):
        DecisionClient(url)


@pytest.mark.integration
def test_actual_game_with_http_model_and_verified_ui_replay(tmp_path: Path, decision_server: Any) -> None:
    url, requests = decision_server
    out = tmp_path / "http-model"
    result = match.run_match(ReaIssacScenario(7, 20, 3), out, SEED, policy="model", decision_server=url)
    assert result["tick"] == 20 and result["status"] == "truncated"
    assert result["inference_calls"] == len(requests) == 7
    assert replay.verify_replay(out)["verified"]
    assert replay_game(json.loads((out / "replay.json").read_text())).digest() == result["final_hash"]
    decisions = [json.loads(s) for s in (out / "decisions.jsonl").read_text().splitlines()]
    assert [d["tick"] for d in decisions] == [0, 3, 6, 9, 12, 15, 18]
    assert decisions[-1]["frames"] == 2
    assert json.loads((out / "manifest.json").read_text())["model"]["metadata"]["model"] == "local-protocol-stand-in"
    with pytest.raises(FileExistsError):
        match.run_match(ReaIssacScenario(), out, SEED)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["state", "result", "action", "request", "harness", "replay"])
def test_real_replay_rejects_tampering(tmp_path: Path, decision_server: Any, field: str) -> None:
    out = tmp_path / "tampered"
    match.run_match(ReaIssacScenario(7, 6), out, SEED, policy="model", decision_server=decision_server[0])
    if field == "harness":
        (out / "harness" / CONTROLLER_PATH).write_text('{"history_steps": 3}')
    elif field in ("result", "replay"):
        path = out / f"{field}.json"
        value = json.loads(path.read_text())
        value["final_hash"] = "invented"
        path.write_text(json.dumps(value))
    else:
        path = out / ("steps.jsonl" if field == "state" else "decisions.jsonl")
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        if field == "state":
            rows[1]["observation"]["player"]["health"] = 99
        elif field == "action":
            rows[0]["action"]["move_x"] = -1
        else:
            rows[0]["request"]["parts"]["essential"][0] = "A different game state"
            # Even a matching fake reply cannot replace the real public observation.
            rows[0]["response"] = reply_for(rows[0]["request"])
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        replay.audit_run(out)


@pytest.mark.unit
@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_failed_model_preserves_partial_run_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    monkeypatch.setattr(DecisionClient, "health", lambda self: {"ready": True, "metadata": {}})

    def fail(*args: Any) -> Any:
        raise failure("stand-in error")

    monkeypatch.setattr(DecisionClient, "decide", fail)
    out = tmp_path / "failed"
    with pytest.raises(failure):
        match.run_match(ReaIssacScenario(), out, SEED, policy="model", decision_server="http://127.0.0.1:1")
    assert not (out / "result.json").exists()
    assert json.loads((out / "replay.json").read_text())["actions"] == []
    assert json.loads((out / "error.json").read_text())["status"] == (
        "interrupted" if failure is KeyboardInterrupt else "error"
    )
    assert len((out / "steps.jsonl").read_text().splitlines()) == 1


@pytest.mark.unit
def test_component_factories_use_existing_protocols(tmp_path: Path) -> None:
    from honing.backends.rea_issac import e2e
    from honing.e2e.components import PipelineContext, build_component

    context = PipelineContext(ROOT, tmp_path, components={"endpoints": JevEndpoints({})})
    assert e2e.harness_seed(context).agent == SEED.agent
    assert len(e2e.tasks(context, limit=1)) == 1
    for name, protocol in (("episode_runner", EpisodeRunner), ("grader", TaskGrader)):
        component = build_component({"factory": f"honing.backends.rea_issac.e2e:{name}"}, context, name, protocol)
        assert isinstance(component, protocol)


@pytest.mark.integration
def test_real_subprocess_runner_and_grader(tmp_path: Path, decision_server: Any) -> None:
    snapshot = FakeModelSource(FakeWorld()).materialize(tmp_path)
    policy = PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")
    suite = ReaIssacSuite(
        (
            {
                "task_id": "tiny",
                "split": "val",
                "prompt": "Defeat the Warden.",
                "rea_issac": ReaIssacScenario(7, 6).record(),
            },
        )
    )
    endpoints = JevEndpoints(
        {
            policy.checkpoint.checksum: PolicyEndpoint(
                decision_server[0], "tokenizer", policy.checkpoint.checksum, runtime="jev_decision"
            )
        }
    )
    runner = ReaIssacEpisodeRunner(
        suite=suite, endpoints=endpoints, integration_dir=INTEGRATION, runs_dir=tmp_path / "runs"
    )
    grader = ReaIssacGrader(suite=suite)
    record = asyncio.run(run_graded(runner, grader, SEED, suite.tasks("val")[0], policy))
    assert record.score == 0 and record.outcome.exit_code == 0
    assert record.outcome.workspace_location is not None
    out = Path(record.outcome.workspace_location)
    assert replay.verify_replay(out)["tick"] == 6
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["policy_checksum"] == policy.checkpoint.checksum
    manifest["kind"] = "scripted"
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ContractError):
        asyncio.run(grader.grade(suite.tasks("val")[0], record.outcome))


@pytest.mark.unit
def test_base_import_and_cli_help_need_no_model_packages() -> None:
    code = (
        "import sys; sys.modules['torch']=None; sys.modules['numpy']=None; "
        "from honing.backends.rea_issac.match import main; main(['--help'])"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--decision-server" in result.stdout


@pytest.mark.unit
def test_interrupt_inside_engine_step_exports_only_completed_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Engine.step
    calls = 0

    def interrupted(self: Engine, action: Any, frames: int = 1) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            original(self, action, 1)
            raise KeyboardInterrupt()
        return original(self, action, frames)

    monkeypatch.setattr(Engine, "step", interrupted)
    out = tmp_path / "interrupted-step"
    with pytest.raises(KeyboardInterrupt):
        match.run_match(ReaIssacScenario(7, 20), out, SEED)
    saved = json.loads((out / "replay.json").read_text())
    assert len(saved["actions"]) == 1 and saved["result"]["tick"] == 3
    assert replay_game(saved).digest() == saved["final_hash"]
    assert not (out / "result.json").exists()
