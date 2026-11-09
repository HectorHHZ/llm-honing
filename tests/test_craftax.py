"""Craftax contracts run without game packages; native tests opt in when installed."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from honing.backends.craftax import (
    CraftaxEpisodeRunner,
    CraftaxGrader,
    CraftaxScenario,
    CraftaxSuite,
    build_suite,
    load_suite,
    match,
    replay,
    seed_harness,
)
from honing.backends.craftax.harness import CONTROLLER_PATH, QUESTIONS_PATH, parse_harness
from honing.backends.craftax.model import DecisionClient, make_request, validate_reply
from honing.backends.jev import JevEndpoints
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
from integrations.craftax.catalog import ACTIONS
from tests.fakes.craftax import FakeCraftaxEngine, fake_versions
from tests.fakes.model import FakeModelSource, FakeWorld

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations/craftax"
SEED = seed_harness(INTEGRATION)
REAL = all(importlib.util.find_spec(p) is not None for p in ("craftax", "jax", "imageio"))


def reply_for(request: dict[str, Any], choice: str = "DO") -> dict[str, Any]:
    return {
        "id": request["id"],
        "stateText": request["parts"]["essential"][0],
        "trimmed": [],
        "tokenAudit": {},
        "answers": {"action": {"choice": choice, "probabilities": {a: float(a == choice) for a in ACTIONS}}},
    }


@pytest.fixture
def fake_game(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(match, "CraftaxEngine", FakeCraftaxEngine)
    monkeypatch.setattr(replay, "CraftaxEngine", FakeCraftaxEngine)
    monkeypatch.setattr(match, "runtime_versions", fake_versions)
    monkeypatch.setattr(replay, "runtime_versions", fake_versions)


@pytest.fixture
def decision_server() -> Any:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/v1/health"
            self.send_json({"ready": True, "metadata": {"model": "test-stand-in"}})

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
        {"max_steps": 0},
        {"max_steps": True},
        {"goals": ()},
        {"goals": ("FAKE",)},
        {"goals": ("COLLECT_WOOD", "COLLECT_WOOD")},
    ],
)
def test_bad_scenarios_are_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ContractError):
        CraftaxScenario(**kwargs)


@pytest.mark.unit
def test_suite_matches_source_and_disallows_shared_split_seeds() -> None:
    suite = load_suite(INTEGRATION / "tasks/suite.jsonl")
    assert list(suite.records) == build_suite()
    assert len(suite.tasks("val", limit=1)) == 1
    records = build_suite()
    records[2]["craftax"] = records[0]["craftax"]
    with pytest.raises(ContractError, match="share seeds"):
        CraftaxSuite(tuple(records))
    with pytest.raises(ContractError):
        suite.scenario("missing")


@pytest.mark.unit
@pytest.mark.parametrize(
    "path,kind,content",
    [
        (CONTROLLER_PATH, "settings", '{"max_steps": 999}'),
        (CONTROLLER_PATH, "settings", '{"history_steps": true}'),
        (CONTROLLER_PATH, "settings", '{"history_steps": 33}'),
        (QUESTIONS_PATH, "instructions", '{"action": ""}'),
        (QUESTIONS_PATH, "instructions", '{"other": "x"}'),
        (QUESTIONS_PATH, "settings", '{"action": "x"}'),
    ],
)
def test_harness_cannot_change_task_rules(path: str, kind: str, content: str) -> None:
    with pytest.raises(CampaignRuleError):
        parse_harness(SEED.with_file(path, HarnessFile(kind, content)))  # type: ignore[arg-type]


@pytest.mark.unit
def test_context_is_public_and_contains_all_native_options() -> None:
    game = FakeCraftaxEngine(CraftaxScenario())
    spec = SEED.with_file("notes/survive.md", HarnessFile("instructions", "Avoid danger."))
    harness = parse_harness(spec)
    request = make_request(game.reset(), game.scenario, harness, [{"action": "UP"}] * 8)
    state = request["parts"]["essential"][0]
    assert "Avoid danger" in state and "null" in state and "native_state_hash" not in state
    assert len(request["questions"]["action"]["criteria"]) == 43
    assert state.count('"action":"UP"') == 4
    assert validate_reply(request, reply_for(request)) == "DO"


@pytest.mark.unit
@pytest.mark.parametrize("corruption", ["id", "state", "trimmed", "truncated", "missing", "nan", "sum", "choice"])
def test_model_answers_are_strictly_validated(corruption: str) -> None:
    game = FakeCraftaxEngine(CraftaxScenario())
    request = make_request(game.reset(), game.scenario, parse_harness(SEED), [])
    reply = reply_for(request)
    if corruption == "id":
        reply["id"] = "other"
    elif corruption == "state":
        reply["stateText"] = "shortened"
    elif corruption == "trimmed":
        reply["trimmed"] = ["essential"]
    elif corruption == "truncated":
        reply["tokenAudit"] = {"action": {"stateTruncated": True}}
    elif corruption == "missing":
        del reply["answers"]["action"]["probabilities"]["NOOP"]
    elif corruption == "nan":
        reply["answers"]["action"]["probabilities"]["NOOP"] = float("nan")
    elif corruption == "sum":
        reply["answers"]["action"]["probabilities"]["DO"] = 0.5
    else:
        reply["answers"]["action"]["choice"] = "NOOP"
    with pytest.raises(ValueError):
        validate_reply(request, reply)


@pytest.mark.unit
@pytest.mark.parametrize("url", ["file:///tmp/key", "http://name:secret@host", "https://host?token=secret"])
def test_credentials_cannot_be_embedded_in_recorded_urls(url: str) -> None:
    with pytest.raises(ValueError):
        DecisionClient(url)


@pytest.mark.integration
def test_real_http_client_runs_and_replays_a_stand_in_game(
    tmp_path: Path, fake_game: None, decision_server: Any
) -> None:
    url, requests = decision_server
    out = tmp_path / "http"
    result = match.run_match(
        CraftaxScenario(max_steps=3), out, SEED, policy="model", decision_server=url, record_video=False
    )
    assert result["success"] and len(requests) == 3
    assert replay.verify_replay(out)["verified"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["model"]["metadata"]["model"] == "test-stand-in"


@pytest.mark.unit
def test_prescribed_actions_and_output_preservation(tmp_path: Path, fake_game: None) -> None:
    out = tmp_path / "actions"
    result = match.run_match(
        CraftaxScenario(max_steps=2), out, SEED, policy="actions", actions=("UP", "DO"), record_video=False
    )
    assert result["status"] == "step_limit" and result["achievements"] == ["COLLECT_WOOD"]
    assert replay.verify_replay(out)["steps"] == 2
    with pytest.raises(FileExistsError):
        match.run_match(CraftaxScenario(), out, SEED, record_video=False)


@pytest.mark.unit
@pytest.mark.parametrize("field", ["frame", "result", "decision"])
def test_tampered_recordings_are_refused(tmp_path: Path, fake_game: None, field: str) -> None:
    out = tmp_path / "run"
    match.run_match(CraftaxScenario(max_steps=2), out, SEED, record_video=False)
    if field == "frame":
        path = out / "steps.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        rows[1]["frame"]["observation"]["player"]["health"] = 99
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif field == "decision":
        path = out / "decisions.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        rows[0]["chosen"]["action"] = "NOOP"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    else:
        path = out / "result.json"
        result = json.loads(path.read_text())
        result["steps"] = 500
        path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        replay.audit_run(out)


@pytest.mark.unit
@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_failed_model_does_not_turn_into_random_play(
    tmp_path: Path, fake_game: None, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]
) -> None:
    monkeypatch.setattr(DecisionClient, "health", lambda self: {"ready": True, "metadata": {}})

    def fail(*args: Any) -> Any:
        raise error("stand-in failure")

    monkeypatch.setattr(DecisionClient, "decide", fail)
    out = tmp_path / "failed"
    with pytest.raises(error):
        match.run_match(
            CraftaxScenario(), out, SEED, policy="model", decision_server="http://127.0.0.1:1", record_video=False
        )
    assert not (out / "result.json").exists()
    assert len((out / "steps.jsonl").read_text().splitlines()) == 1
    assert json.loads((out / "error.json").read_text())["status"] == (
        "interrupted" if error is KeyboardInterrupt else "error"
    )


@pytest.mark.unit
def test_import_and_help_need_no_game_packages() -> None:
    code = (
        "import sys; sys.modules['craftax']=None; sys.modules['jax']=None; sys.modules['numpy']=None; "
        "import honing.backends.craftax; from honing.backends.craftax.match import main; main(['--help'])"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--decision-server" in result.stdout


@pytest.mark.unit
def test_component_factories_fit_the_existing_loader(tmp_path: Path) -> None:
    from honing.backends.craftax import e2e
    from honing.e2e.components import PipelineContext, build_component

    context = PipelineContext(ROOT, tmp_path, components={"endpoints": JevEndpoints({})})
    assert e2e.harness_seed(context).agent == SEED.agent
    assert len(e2e.tasks(context, limit=1)) == 1
    for name, protocol in (("episode_runner", EpisodeRunner), ("grader", TaskGrader)):
        component = build_component({"factory": f"honing.backends.craftax.e2e:{name}"}, context, name, protocol)
        assert isinstance(component, protocol)


@pytest.mark.integration
@pytest.mark.parametrize("quality,score", [("strong", 1.0), ("partial", 1 / 3), ("weak", 0.0)])
def test_runner_and_grader_fit_existing_honing_contracts(tmp_path: Path, quality: str, score: float) -> None:
    snapshot = FakeModelSource(FakeWorld()).materialize(tmp_path)
    policy = PolicyRef(replace(snapshot.checkpoint, checksum=quality), snapshot.architecture, "master")
    suite = load_suite(INTEGRATION / "tasks/suite.jsonl")
    config = suite.records[2]
    suite = CraftaxSuite(({**config, "craftax": {**config["craftax"], "max_steps": 3}},))  # type: ignore[dict-item]
    endpoints = JevEndpoints(
        {quality: PolicyEndpoint(f"http://{quality}:8080", "tokenizer", quality, runtime="jev_decision")}
    )
    runner = CraftaxEpisodeRunner(
        suite=suite,
        endpoints=endpoints,
        integration_dir=INTEGRATION,
        runs_dir=tmp_path / "runs",
        command=[sys.executable, str(ROOT / "tests/fakes/craftax_match.py")],
    )
    grader = CraftaxGrader(suite=suite)
    assert isinstance(runner, EpisodeRunner) and isinstance(grader, TaskGrader)
    spec = SEED.with_file("notes/test.md", HarnessFile("instructions", "Keep moving."))
    record = asyncio.run(run_graded(runner, grader, spec, suite.tasks("val")[0], policy))
    assert record.score == score and record.spec_id == spec.spec_id
    out = Path(record.outcome.workspace_location)  # type: ignore[arg-type]
    assert json.loads((out / "manifest.json").read_text())["policy_checksum"] == quality
    assert json.loads((out / "manifest.json").read_text())["harness"][-1]["path"] == "notes/test.md"
    if quality == "strong":
        manifest = json.loads((out / "manifest.json").read_text())
        manifest["kind"] = "scripted"
        (out / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ContractError):
            asyncio.run(grader.grade(suite.tasks("val")[0], record.outcome))


@pytest.mark.integration
@pytest.mark.skipif(not REAL, reason="optional Craftax runtime is not installed")
def test_native_actions_video_and_deterministic_replay(tmp_path: Path) -> None:
    out = tmp_path / "native"
    result = match.run_match(
        CraftaxScenario(seed=1, max_steps=2), out, SEED, policy="actions", actions=("UP", "DO"), record_video=True
    )
    assert result["steps"] == 2
    assert replay.verify_replay(out)["verified"]
    import imageio.v2 as imageio

    with imageio.get_reader(out / "replay.mp4") as video:
        assert video.count_frames() == 3
        assert video.get_data(0).shape == (832, 704, 3)


@pytest.mark.integration
@pytest.mark.skipif(not REAL, reason="optional Craftax runtime is not installed")
def test_native_engine_with_http_decisions(tmp_path: Path, decision_server: Any) -> None:
    url, requests = decision_server
    out = tmp_path / "native-http"
    result = match.run_match(
        CraftaxScenario(seed=2, max_steps=2), out, SEED, policy="model", decision_server=url, record_video=False
    )
    assert result["steps"] == len(requests) == 2
    assert replay.verify_replay(out)["verified"]
