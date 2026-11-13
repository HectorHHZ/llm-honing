"""Native rules, graphics, exact replay, HTTP decisions and Honing seams."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
import urllib.error
import urllib.request
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from integrations.overcooked.decisions import make_request, validate_reply
from integrations.overcooked.engine import ACTIONS, LAYOUTS, Engine, Scenario
from integrations.overcooked.harness import parse_harness, seed_harness
from integrations.overcooked.match import run_match
from integrations.overcooked.policy import joint_baseline
from integrations.overcooked.recording import replay, result_for
from integrations.overcooked.scenarios import Suite, load_suite
from integrations.overcooked.ui import Application, make_server


def reply_for(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": request["id"],
        "stateText": request["parts"]["essential"][0],
        "trimmed": [],
        "tokenAudit": {
            "action": {
                "stateTruncated": False,
                "options": 6,
                "stateTokens": 100,
                "stateRoom": 1000,
                "inputTokens": 200,
            }
        },
        "answers": {"action": {"choice": "STAY", "probabilities": dict.fromkeys(ACTIONS, 1 / 6)}},
    }


@pytest.fixture
def decision_server() -> Any:
    requests = []
    invalid = [False]

    class Handler(BaseHTTPRequestHandler):
        def output(self, value: object) -> None:
            data = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self.output({"ready": True, "metadata": {"family": "protocol-stand-in", "real_model": False}})

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            reply = reply_for(request)
            if invalid[0]:
                reply["trimmed"] = ["required-state"]
            self.output(reply)

        def log_message(self, *_args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, invalid
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_observation_is_detached_and_reads_do_not_step() -> None:
    with Engine(Scenario(max_steps=2)) as engine:
        frame = engine.observe()
        frame["map"]["terrain"][0][0] = "changed"
        assert engine.observe()["map"]["terrain"][0][0] == "X"
        initial = engine.observe()
        assert initial == engine.observe()
        with pytest.raises(ValueError):
            engine.step(["EAST", "not-an-action"])
        assert initial == engine.observe()
        engine.step(["STAY", "STAY"])
        terminal = engine.step(["STAY", "STAY"])
        assert terminal["truncated"] and not terminal["terminated"]
        with pytest.raises(RuntimeError):
            engine.step(["STAY", "STAY"])
        assert engine.reset()["step"] == 0


@pytest.mark.parametrize("old", [False, True])
def test_native_cooking_rules_and_delivery_events(old: bool) -> None:
    from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedState, PlayerState, SoupState

    with Engine(Scenario(old_dynamics=old)) as engine:
        soup = SoupState.get_soup((2, 0), num_onions=2)
        engine.env.state = OvercookedState(
            [PlayerState((2, 1), (0, -1), ObjectState("onion", (2, 1))), PlayerState((3, 2), (0, 1))],
            {(2, 0): soup},
            all_orders=[{"ingredients": ["onion"] * 3}],
        )
        frame = engine.step(["INTERACT", "STAY"])
        assert frame["state"]["objects"][0]["is_cooking"] is old
        if not old:
            frame = engine.step(["INTERACT", "STAY"])
            assert frame["state"]["objects"][0]["is_cooking"]
        for _ in range(20):
            frame = engine.step(["STAY", "STAY"])
        assert frame["state"]["objects"][0]["is_ready"]
        engine.env.state.players[0].position = (3, 1)
        engine.env.state.players[0].orientation = (1, 0)
        engine.env.state.players[0].held_object = SoupState.get_soup((3, 1), num_onions=3, finished=True)
        # Put the soup-carrying chef next to the serving counter at (3,3).
        engine.env.state.players[1].position = (1, 2)
        engine.env.state.players[0].position = (3, 2)
        engine.env.state.players[0].orientation = (0, 1)
        engine.env.state.players[0].held_object.position = (3, 2)
        frame = engine.step(["INTERACT", "STAY"])
        assert frame["score"] == 20 and frame["deliveries"] == 1
        assert frame["events"]["soup_delivery"] == [True, False]


def test_native_graphics_are_png_and_rendering_does_not_step(tmp_path: Path) -> None:
    with Engine() as engine:
        before = engine.observe()
        picture = engine.png()
        assert picture.startswith(b"\x89PNG\r\n\x1a\n") and len(picture) > 5000
        (tmp_path / "frame.png").write_bytes(picture)
        assert engine.observe() == before


def test_disallowed_recipe_delivery_does_not_earn_task_credit() -> None:
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, PlayerState, SoupState

    with Engine() as engine:
        engine.env.state = OvercookedState(
            [
                PlayerState((3, 2), (0, 1), SoupState.get_soup((3, 2), num_onions=1, finished=True)),
                PlayerState((1, 2), (0, 1)),
            ],
            {},
            all_orders=[{"ingredients": ["onion"] * 3}],
        )
        frame = engine.step(["INTERACT", "STAY"])
        assert frame["score"] == 0 and frame["delivery_attempts"] == 1
        assert frame["deliveries"] == 0 and frame["events"]["soup_delivery"] == [True, False]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("old", [False, True])
def test_every_selectable_kitchen_runs_and_renders(layout: str, old: bool) -> None:
    with Engine(Scenario(layout=layout, old_dynamics=old, max_steps=3)) as engine:
        frame = engine.observe()
        for _ in range(3):
            frame = engine.step(joint_baseline(frame))
        assert frame["map"]["layout"] == layout
        assert engine.png().startswith(b"\x89PNG")


def test_baseline_delivers_and_replays_every_frame(tmp_path: Path) -> None:
    out = tmp_path / "native-baseline"
    result = run_match(Scenario(max_steps=150), out)
    assert result["deliveries"] >= 2
    assert replay(out)["result"]["deliveries"] == result["deliveries"]
    rows = (out / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[3])
    row["frame"]["state"]["players"][0]["position"] = [99, 99]
    rows[3] = json.dumps(row)
    (out / "steps.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="native replay differs"):
        replay(out)


@pytest.mark.parametrize("sample", [False, True])
def test_real_engine_http_decisions_and_native_replay(tmp_path: Path, decision_server: Any, sample: bool) -> None:
    endpoint, requests, _ = decision_server
    out = tmp_path / "model"
    result = run_match(
        Scenario(max_steps=8),
        out,
        policy="model",
        decision_server=endpoint,
        policy_checksum="stand-in-checkpoint",
        sample=sample,
        policy_seed=17,
        record_images=False,
    )
    assert result["complete"] and len(requests) == 8
    audit = replay(out)
    assert audit["manifest"]["kind"] == ("rollout" if sample else "evaluation")
    rows = [json.loads(line) for line in (out / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["response"]["tokenAudit"]["action"]["stateTruncated"] = True
    (out / "decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="audit"):
        replay(out)


def test_model_error_stops_without_a_fallback(tmp_path: Path, decision_server: Any) -> None:
    endpoint, _, invalid = decision_server
    invalid[0] = True
    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="context"):
        run_match(Scenario(max_steps=3), out, policy="model", decision_server=endpoint, record_images=False)
    assert not (out / "result.json").exists()
    assert (out / "error.json").exists()
    assert len((out / "steps.jsonl").read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.parametrize("mutation", ["id", "sum", "choice", "truncation", "audit"])
def test_invalid_probabilities_or_context_are_refused(mutation: str) -> None:
    with Engine() as engine:
        request = make_request(engine.observe(), 0)
    reply = reply_for(request)
    if mutation == "id":
        reply["id"] = "stale"
    elif mutation == "sum":
        reply["answers"]["action"]["probabilities"]["NORTH"] = 0.9
    elif mutation == "choice":
        reply["answers"]["action"]["probabilities"] = {a: float(a == "NORTH") for a in ACTIONS}
    elif mutation == "truncation":
        reply["trimmed"] = ["essential"]
    else:
        reply["tokenAudit"]["action"]["stateTokens"] = 1001
    with pytest.raises(ValueError):
        validate_reply(request, reply)


def test_layout_disjoint_tasks_and_harness_boundaries() -> None:
    suite = load_suite()
    assert suite.tasks("train") and suite.tasks("val") and suite.tasks("heldout")
    rows = copy.deepcopy(suite.records)
    rows[2]["overcooked"]["layout"] = rows[0]["overcooked"]["layout"]
    with pytest.raises(ValueError, match="overlap"):
        Suite(rows)
    from honing.core import HarnessFile

    spec = seed_harness().with_file("overcooked/controller.json", HarnessFile("settings", '{"max_steps":1}'))
    with pytest.raises(ValueError, match="history_steps"):
        parse_harness(spec)


def test_grading_preserves_native_score_order_after_the_delivery_target() -> None:
    scenario = Scenario()
    base = {"step": 400, "deliveries": 2, "status": "step_limit", "frame_hash": "test"}
    grades = [result_for({**base, "score": score}, scenario)["grade"] for score in (0, 20, 40, 200, 1000)]
    assert grades == sorted(set(grades))
    assert grades[0] == 0 and grades[-1] < 1
    assert grades[2] == 0.5
    assert grades[3] == pytest.approx(5 / 6)
    for bad in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="score_scale"):
            Scenario(score_scale=bad)


def test_teacher_records_exact_inputs_and_replays(tmp_path: Path) -> None:
    out = tmp_path / "teacher"
    run_match(Scenario(max_steps=5), out, policy="teacher", record_images=False)
    audit = replay(out)
    assert audit["manifest"]["kind"] == "teacher-demonstration"
    rows = [json.loads(line) for line in (out / "decisions.jsonl").read_text().splitlines()]
    assert all(row["request"] and row["response"] is None for row in rows)


def test_frozen_benchmark_has_disjoint_maps_and_unseen_slow_partner() -> None:
    from integrations.overcooked.engine import ROOT

    suite = load_suite(ROOT / "tasks/benchmark-v1.jsonl")
    for row in suite.records:
        scenario = suite.scenario(row["task_id"])
        assert scenario.max_steps == 400 and scenario.score_scale == 40
        if scenario.partner == "solo_slow":
            assert row["split"] == "heldout"
    with Engine(Scenario(partner="solo_slow", max_steps=3)) as engine:
        frame = engine.step(["STAY", "STAY"])
        from integrations.overcooked.policy import baseline

        assert baseline(frame, 1, "solo_slow") == "STAY"


def test_sft_labels_require_explicit_train_teacher_and_preserve_requests(tmp_path: Path) -> None:
    from integrations.overcooked.data import balanced, build_dataset, teacher_rows, training_batch

    train = Scenario(max_steps=5)
    validation = Scenario(layout="asymmetric_advantages", max_steps=5)
    suite = Suite(
        tuple(
            {"task_id": split, "split": split, "prompt": "Deliver soup.", "overcooked": scenario.record()}
            for split, scenario in (("train", train), ("val", validation))
        )
    )
    train_run, val_run, baseline_run = (tmp_path / name for name in ("train", "val", "baseline"))
    run_match(train, train_run, policy="teacher", record_images=False)
    run_match(validation, val_run, policy="teacher", record_images=False)
    run_match(train, baseline_run, policy="baseline", record_images=False)
    rows = teacher_rows(train_run, suite, "fit:0")
    batch = training_batch(balanced(rows, 1), "audited-test")
    assert batch.contract.kind == "typed_decisions"
    assert len(batch.decisions) == len({row["chosen"] for row in rows})
    assert dict(batch.decisions[0].parts) == {k: tuple(v) for k, v in rows[0]["request"]["parts"].items()}
    assert dict(batch.decisions[0].chosen)["action"] == rows[0]["chosen"]
    with pytest.raises(ValueError, match="val/heldout"):
        teacher_rows(val_run, suite, "fit:0")
    with pytest.raises(ValueError, match="explicitly recorded"):
        teacher_rows(baseline_run, suite, "fit:0")
    with pytest.raises(ValueError, match="separate teacher episodes"):
        build_dataset([train_run], [train_run], tmp_path / "leaked.json")


def test_checkpoint_retry_clears_readonly_only_inside_training_output(tmp_path: Path) -> None:
    import stat
    from integrations.overcooked.train import writable_checkpoint

    work = tmp_path / "training"
    checkpoint = work / "train-1/checkpoint/tokenizer"
    checkpoint.mkdir(parents=True)
    copied = checkpoint / "tokenizer.json"
    copied.write_text("first")
    copied.chmod(stat.S_IREAD)
    writable_checkpoint(work / "train-1/checkpoint", work)
    copied.write_text("second")
    assert copied.read_text() == "second"
    with pytest.raises(ValueError):
        writable_checkpoint(tmp_path, work)
    with pytest.raises(ValueError, match="must be a child"):
        writable_checkpoint(work, work)


@pytest.mark.parametrize(
    "base_scores,new_scores,expected",
    [((0, 0), (0, 0), "reject"), ((0, 0), (0.5, 0), "select"), ((0.5, 0), (0, 0.5), "reject")],
)
def test_pilot_uses_paired_win_loss_and_refuses_heldout_selection(
    tmp_path: Path, base_scores: tuple[float, ...], new_scores: tuple[float, ...], expected: str
) -> None:
    from dataclasses import asdict
    from integrations.overcooked.engine import ROOT, digest
    from integrations.overcooked.select_candidate import compare
    from tests.fakes.model import FakeModelSource, FakeWorld

    snapshot = FakeModelSource(FakeWorld()).materialize(tmp_path / "checkpoint")
    protocol = json.loads((ROOT / "experiments/pilot-v1.json").read_text())
    common = {
        "protocol_sha256": digest({"protocol": protocol, "suite": (ROOT / protocol["suite"]).read_text()}),
        "complete": True,
        "split": "val",
        "controller": "model",
        "spec_id": "same-harness",
    }
    summaries = []
    for checksum, scores in (("parent", base_scores), (snapshot.checkpoint.checksum, new_scores)):
        summaries.append(
            {
                **common,
                "model": {"weightsSha256": checksum},
                "episodes": [
                    {"task_id": key, "verified": True, "complete": True, "grade": grade, "score": grade * 100}
                    for key, grade in zip(protocol["pilot_tasks"]["val"], scores, strict=True)
                ],
            }
        )
    training = {"snapshot": asdict(snapshot), "base": {"checkpoint": {"checksum": "parent"}}}
    assert compare(*summaries, training)["decision"]["outcome"] == expected
    summaries[1]["split"] = "heldout"
    with pytest.raises(ValueError, match="validation episodes"):
        compare(*summaries, training)


def test_gui_api_state_graphics_stale_actions_and_export(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr("integrations.overcooked.ui.new_run", lambda _prefix: tmp_path / "live")
    app = Application(Scenario(max_steps=3))
    server = make_server(app, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"

    def get(path: str) -> bytes:
        with urllib.request.urlopen(url + path) as response:
            return response.read()

    def post(path: str, data: object, origin: str | None = None) -> Any:
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        req = urllib.request.Request(url + path, data=json.dumps(data).encode(), headers=headers)
        with urllib.request.urlopen(req) as response:
            return json.load(response)

    try:
        state = json.loads(get("/api/state"))
        assert get("/frame.png").startswith(b"\x89PNG")
        assert json.loads(get("/api/map"))["layout"] == "cramped_room"
        assert json.loads(get("/api/rules"))["old_dynamics"] is False
        assert b"Overcooked" in get("/")
        payload = {"policy": "human", "action": "NORTH", "frame_hash": state["frame"]["frame_hash"]}
        after = post("/api/step", payload)
        assert after["frame"]["step"] == 1
        with pytest.raises(urllib.error.HTTPError) as stale:
            post("/api/step", payload)
        assert stale.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as foreign:
            post("/api/reset", {}, "https://example.org")
        assert foreign.value.code == 403
        monkeypatch.setattr("integrations.overcooked.recording.new_run", lambda _prefix: tmp_path / "export")
        exported = post("/api/export", {})
        audit = replay(Path(exported["out"]))
        assert audit["verified"] and not audit["result"]["complete"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        app.close()


def test_honing_runner_and_grader_with_native_replay(tmp_path: Path, decision_server: Any) -> None:
    from honing.core import PolicyEndpoint, PolicyRef, run_graded
    from integrations.overcooked.adapter import OvercookedEpisodeRunner, OvercookedGrader
    from tests.fakes.model import FakeModelSource, FakeWorld

    endpoint, _, _ = decision_server
    snapshot = FakeModelSource(FakeWorld()).materialize(tmp_path / "weights")
    policy = PolicyRef(snapshot.checkpoint, snapshot.architecture, "master")

    class Endpoints:
        def resolve(self, _policy: Any) -> PolicyEndpoint:
            return PolicyEndpoint(endpoint, "stand-in", "stand-in", "jev_decision")

    scenario = Scenario(max_steps=3)
    suite = Suite(
        ({"task_id": "native-model", "split": "train", "prompt": "Deliver soup.", "overcooked": scenario.record()},)
    )
    runner = OvercookedEpisodeRunner(suite=suite, endpoints=Endpoints(), runs_dir=tmp_path / "runs")
    grader = OvercookedGrader(suite=suite)
    record = asyncio.run(run_graded(runner, grader, seed_harness(), suite.tasks("train")[0], policy))
    assert record.outcome.exit_code == 0 and record.grade.score == 0
    changed = replace(record.outcome, policy_checksum="wrong-checkpoint")
    from honing.core import ContractError

    with pytest.raises(ContractError):
        asyncio.run(grader.grade(suite.tasks("train")[0], changed))


def test_gui_replay_seeks_exact_recorded_states(tmp_path: Path) -> None:
    out = tmp_path / "recorded"
    run_match(Scenario(max_steps=5), out, record_images=False)
    audit = replay(out)
    app = Application(Scenario(), replay_dir=out)
    try:
        assert app.state()["replay_length"] == 6
        assert app.advance({"index": 4})["frame"] == audit["frames"][4]
        assert app.advance({})["frame"] == audit["frames"][5]
        with pytest.raises(ValueError, match="index"):
            app.advance({})
        assert app.reset({})["frame"] == audit["frames"][0]
    finally:
        app.close()
