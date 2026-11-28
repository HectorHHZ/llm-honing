"""Spiders matches, audits and replays on a stand-in engine (tests/fakes/griddly_engine.py), so CI needs no Griddly.

``tests/test_griddly.py`` runs the same code on the native engine where it is
installed. Here the engine is a fixed 4x4 map with Griddly's frame layout, so
the tests pin what the match and replay code decide on their own: which
controller picks each action and what it records, which refusals come before
anything is written, and that the audit refuses every kind of changed record
with its own reason.
"""

from __future__ import annotations

import json
import shutil
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from honing.backends.griddly import match, replay
from honing.backends.griddly.harness import CONTROLLER_PATH, load_harness, seed_harness, write_harness
from honing.backends.griddly.match import run_match
from honing.backends.griddly.replay import audit_run, verify_replay
from honing.backends.griddly.scenarios import GriddlyScenario
from honing.backends.griddly.source import adapter_sha256, source_files
from honing.core import CampaignRuleError, HarnessFile
from integrations.griddly.catalog import ACTIONS
from integrations.griddly.engine import canonical, digest
from tests.fakes.griddly import StandInEndpoints
from tests.fakes.griddly_engine import FakeSpidersEngine, fake_versions

ROOT = Path(__file__).resolve().parents[1]
SEED = seed_harness(ROOT / "integrations/griddly")
# The baseline's path on the stand-in's map: up the left edge, then right along the top to the gem.
BASELINE_PATH = ["FORWARD", "FORWARD", "FORWARD", "TURN_RIGHT", "FORWARD", "FORWARD", "FORWARD"]
INTO_THE_SPIDER = ("TURN_RIGHT", "FORWARD", "FORWARD", "FORWARD", "WAIT", "WAIT")


@pytest.fixture(autouse=True)
def spiders(monkeypatch: pytest.MonkeyPatch) -> list[FakeSpidersEngine]:
    """The stand-in engine and runtime in both modules; returns the engines opened, oldest first."""
    for module in (match, replay):
        monkeypatch.setattr(module, "GriddlyEngine", FakeSpidersEngine)
        monkeypatch.setattr(module, "runtime_versions", fake_versions)
    monkeypatch.setattr(FakeSpidersEngine, "opened", [])
    return FakeSpidersEngine.opened


@pytest.fixture
def endpoints() -> Iterator[StandInEndpoints]:
    value = StandInEndpoints()
    try:
        yield value
    finally:
        value.close()
        value.server.server_close()  # close() stops serving but leaves the socket open


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")


def rehash(frame: dict[str, Any]) -> None:
    frame["frame_hash"] = digest({key: value for key, value in frame.items() if key != "frame_hash"})


def edit_json(path: Path, change: Callable[[dict[str, Any]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    change(value)
    path.write_text(json.dumps(value), encoding="utf-8")


# ---------------------------------------------------------------- matches


@pytest.mark.functional
def test_a_scripted_match_writes_a_record_that_audits_and_replays(
    tmp_path: Path, spiders: list[FakeSpidersEngine]
) -> None:
    out = tmp_path / "run"

    result = run_match(GriddlyScenario(seed=3, max_steps=12), out, SEED, policy="baseline", policy_seed=5)

    assert {k: result[k] for k in ("status", "success", "steps", "reward")} == {
        "status": "won",
        "success": True,
        "steps": 7,
        "reward": 1.0,
    }
    manifest = json.loads((out / "manifest.json").read_text())
    assert (manifest["format"], manifest["controller"], manifest["kind"]) == (
        "griddly-match-v1",
        "baseline",
        "scripted",
    )
    assert (manifest["sample"], manifest["policy_seed"], manifest["model"]) == (False, 5, None)
    assert manifest["scenario"] == GriddlyScenario(seed=3, max_steps=12).record()
    assert (manifest["runtime"], manifest["adapter_sha256"], manifest["actions"]) == (
        fake_versions(),
        adapter_sha256(),
        ACTIONS,
    )
    assert load_harness(out / "harness").spec_id == SEED.spec_id == manifest["spec_id"]
    for name, path in source_files().items():
        assert (out / "source" / name).read_bytes() == path.read_bytes()
    decisions = jsonl(out / "decisions.jsonl")
    assert [row["applied"] for row in decisions] == BASELINE_PATH
    assert all(row["request"] is None and row["response"] is None for row in decisions)
    steps = jsonl(out / "steps.jsonl")
    assert [row["event"] for row in steps] == ["reset", *["step"] * 7]
    assert json.loads((out / "result.json").read_text()) == result
    assert all(engine.closed for engine in spiders)

    assert audit_run(out)[2] == result
    assert verify_replay(out) == {"verified": True, "steps": 7, "final_frame_hash": result["final_frame_hash"]}


@pytest.mark.functional
@pytest.mark.parametrize(
    ("policy", "kind"), [("random", "scripted"), ("teacher", "teacher-demonstration"), ("baseline", "scripted")]
)
def test_every_scripted_controller_records_what_the_audit_can_rebuild_including_forced_steps(
    tmp_path: Path, policy: str, kind: str
) -> None:
    out = tmp_path / policy
    # On level 1 the gnome is stunned every third tick: WAIT is the only legal action.
    run_match(GriddlyScenario(seed=3, level_id=1, max_steps=12), out, SEED, policy=policy, policy_seed=11)

    decisions = jsonl(out / "decisions.jsonl")
    forced = [row for row in decisions if row["forced_action"]]
    assert forced and all(row["applied"] == "WAIT" and row["request"] is None for row in forced)
    # Only the teacher asks a question, and never at a forced step.
    asked = [row["request"] is not None for row in decisions if not row["forced_action"]]
    assert set(asked) == {policy == "teacher"}
    assert audit_run(out)[0]["kind"] == kind
    assert verify_replay(out)["steps"] == len(decisions)


@pytest.mark.functional
def test_prescribed_actions_are_played_as_given_and_a_spider_ends_the_episode(tmp_path: Path) -> None:
    out = tmp_path / "actions"

    result = run_match(GriddlyScenario(max_steps=6), out, SEED, policy="actions", actions=INTO_THE_SPIDER)

    assert {k: result[k] for k in ("status", "success", "steps", "reward")} == {
        "status": "lost",
        "success": False,
        "steps": 4,
        "reward": -1.0,
    }
    assert [row["applied"] for row in jsonl(out / "decisions.jsonl")] == list(INTO_THE_SPIDER[:4])
    assert audit_run(out)[2] == result


@pytest.mark.functional
def test_a_match_that_runs_out_of_ticks_ends_at_the_step_limit(tmp_path: Path) -> None:
    result = run_match(GriddlyScenario(max_steps=3), tmp_path / "short", SEED, policy="baseline")

    assert (result["status"], result["success"], result["steps"]) == ("step_limit", False, 3)
    assert verify_replay(tmp_path / "short")["steps"] == 3


@pytest.mark.functional
def test_model_play_records_the_exact_request_and_reply_and_acts_on_the_choice(
    tmp_path: Path, endpoints: StandInEndpoints
) -> None:
    out = tmp_path / "model"

    result = run_match(
        GriddlyScenario(seed=3, level_id=1, max_steps=12),
        out,
        SEED,
        policy="model",
        decision_server=endpoints.server.url,
        policy_checksum="sha256:policy",
    )

    manifest = json.loads((out / "manifest.json").read_text())
    assert (manifest["kind"], manifest["policy_checksum"]) == ("evaluation", "sha256:policy")
    assert manifest["model"]["metadata"]["model"] == "spiders-public-state-rule"
    asked = [row for row in jsonl(out / "decisions.jsonl") if not row["forced_action"]]
    assert asked and all(row["request"]["questions"]["action"]["criteria"] for row in asked)
    assert all(row["applied"] == row["response"]["answers"]["action"]["choice"] for row in asked)
    assert result["success"] is True
    assert audit_run(out)[0]["kind"] == "evaluation"


@pytest.mark.functional
def test_sampled_play_draws_from_the_reply_with_the_policy_seed_and_the_audit_redraws_it(
    tmp_path: Path, endpoints: StandInEndpoints
) -> None:
    def play(name: str, seed: int) -> list[str]:
        run_match(
            GriddlyScenario(seed=3, max_steps=10),
            tmp_path / name,
            SEED,
            policy="model",
            decision_server=endpoints.server.url,
            sample=True,
            policy_seed=seed,
        )
        assert audit_run(tmp_path / name)[0]["kind"] == "rollout"
        return [row["applied"] for row in jsonl(tmp_path / name / "decisions.jsonl")]

    first = play("first", 17)

    assert play("again", 17) == first
    # The stand-in puts 0.8 on the baseline's action, so a sampled run strays from the baseline's path.
    assert any(play(f"seed-{seed}", seed) != BASELINE_PATH for seed in range(3))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"policy": "greedy"}, "unknown policy"),
        ({"policy": "model"}, "only model play takes a server and can sample"),
        ({"policy": "random", "decision_server": "http://127.0.0.1:1"}, "only model play takes a server"),
        ({"policy": "random", "sample": True}, "only model play takes a server and can sample"),
        ({"policy_seed": -1}, "policy_seed must be uint32"),
        ({"policy_seed": 2**32}, "policy_seed must be uint32"),
        ({"policy_seed": True}, "policy_seed must be uint32"),
        ({"policy": "actions", "actions": ("WAIT",)}, "prescribed actions must cover the budget"),
        ({"policy": "actions", "actions": ("WAIT", "JUMP", "WAIT")}, "use native names"),
        ({"policy": "random", "actions": ("WAIT", "WAIT", "WAIT")}, "prescribed actions require policy=actions"),
    ],
)
def test_a_match_that_cannot_be_played_as_asked_is_refused_before_anything_is_written(
    tmp_path: Path, settings: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        run_match(GriddlyScenario(max_steps=3), tmp_path / "out", SEED, **settings)

    assert not (tmp_path / "out").exists()


@pytest.mark.functional
def test_a_match_never_writes_into_an_existing_run(tmp_path: Path) -> None:
    run_match(GriddlyScenario(max_steps=2), tmp_path / "run", SEED)

    with pytest.raises(FileExistsError):
        run_match(GriddlyScenario(max_steps=2), tmp_path / "run", SEED)
    assert not (tmp_path / "run" / "error.json").exists()


@pytest.mark.functional
def test_a_harness_the_adapter_cannot_read_is_recorded_as_a_harness_error(tmp_path: Path) -> None:
    spec = SEED.with_file(CONTROLLER_PATH, HarnessFile("settings", '{"history_steps": 99}'))

    with pytest.raises(CampaignRuleError, match=r"history_steps must be in 0\.\.8"):
        run_match(GriddlyScenario(max_steps=2), tmp_path / "run", spec)

    error = json.loads((tmp_path / "run" / "error.json").read_text())
    assert (error["kind"], error["status"]) == ("harness", "error")
    assert error["error"].startswith("CampaignRuleError: ")
    assert not (tmp_path / "run" / "result.json").exists()
    with pytest.raises(ValueError, match="an incomplete or failed run cannot be audited"):
        audit_run(tmp_path / "run")


@pytest.mark.functional
def test_an_interrupted_match_is_recorded_as_interrupted_and_leaves_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spiders: list[FakeSpidersEngine]
) -> None:
    def interrupt(self: FakeSpidersEngine, action: str) -> dict[str, Any]:
        raise KeyboardInterrupt

    monkeypatch.setattr(FakeSpidersEngine, "step", interrupt)

    with pytest.raises(KeyboardInterrupt):
        run_match(GriddlyScenario(max_steps=2), tmp_path / "run", SEED)

    error = json.loads((tmp_path / "run" / "error.json").read_text())
    assert (error["kind"], error["status"], error["error"]) == ("match", "interrupted", "KeyboardInterrupt: ")
    assert not (tmp_path / "run" / "result.json").exists()
    assert spiders[-1].closed


class Writer:
    def __init__(self, path: Path, **options: Any) -> None:
        self.path, self.options, self.frames, self.closes = path, options, [], 0

    def append_data(self, frame: object) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closes += 1


@pytest.fixture
def imageio(monkeypatch: pytest.MonkeyPatch) -> list[Writer]:
    """``imageio.v2`` with a writer that keeps the frames; returns the writers opened."""
    writers: list[Writer] = []

    def get_writer(path: Path, **options: Any) -> Writer:
        writers.append(Writer(path, **options))
        return writers[-1]

    v2 = types.ModuleType("imageio.v2")
    v2.get_writer = get_writer  # type: ignore[attr-defined]
    package = types.ModuleType("imageio")
    package.v2 = v2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "imageio", package)
    monkeypatch.setitem(sys.modules, "imageio.v2", v2)
    return writers


@pytest.mark.functional
def test_a_recorded_video_holds_the_reset_and_every_step_and_is_closed_once(
    tmp_path: Path, imageio: list[Writer], spiders: list[FakeSpidersEngine]
) -> None:
    result = run_match(GriddlyScenario(max_steps=12), tmp_path / "run", SEED, policy="baseline", record_video=True)

    (writer,) = imageio
    assert writer.path == tmp_path / "run" / "replay.mp4"
    assert writer.options == {"fps": 10, "codec": "libx264", "macro_block_size": 1}
    assert len(writer.frames) == result["steps"] + 1 and writer.closes == 1
    assert spiders[0].render
    assert json.loads((tmp_path / "run" / "manifest.json").read_text())["record_video"] is True


@pytest.mark.functional
def test_a_match_that_fails_mid_video_still_closes_the_video(
    tmp_path: Path, imageio: list[Writer], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(self: FakeSpidersEngine, action: str) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(FakeSpidersEngine, "step", fail)

    with pytest.raises(RuntimeError, match="boom"):
        run_match(GriddlyScenario(max_steps=12), tmp_path / "run", SEED, record_video=True)

    (writer,) = imageio
    assert (len(writer.frames), writer.closes) == (1, 1)
    assert json.loads((tmp_path / "run" / "error.json").read_text())["error"] == "RuntimeError: boom"


@pytest.mark.functional
def test_the_match_command_plays_prescribed_actions_from_a_file_with_a_delivered_harness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    actions = tmp_path / "actions.json"
    actions.write_text(json.dumps(list(INTO_THE_SPIDER)))
    harness = tmp_path / "harness"
    spec = SEED.with_file("notes/tips.md", HarnessFile("instructions", "Spiders bite."))
    write_harness(spec, harness)

    played = ["--policy", "actions", "--actions", str(actions), "--max-steps", "6", "--harness-dir", str(harness)]
    match.main([*played, "--seed", "4", "--level", "2", "--out", str(tmp_path / "run")])

    printed = json.loads(capsys.readouterr().out)
    assert (printed["out"], printed["status"], printed["steps"]) == (str(tmp_path / "run"), "lost", 4)
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["scenario"] == GriddlyScenario(seed=4, level_id=2, max_steps=6).record()
    assert manifest["spec_id"] == spec.spec_id


@pytest.mark.functional
def test_the_match_command_writes_under_runs_by_default_with_the_seed_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)

    match.main(["--policy", "baseline", "--max-steps", "3"])

    out = Path(json.loads(capsys.readouterr().out)["out"])
    assert out.parent == tmp_path / "runs" / "griddly"
    assert out.name.startswith("spiders-baseline-") and len(out.name) == len("spiders-baseline-") + 12
    assert json.loads((out / "manifest.json").read_text())["spec_id"] == SEED.spec_id


@pytest.mark.functional
@pytest.mark.parametrize("content", ['{"action": "WAIT"}', '["WAIT", 3]'], ids=["object", "not-names"])
def test_the_match_command_refuses_actions_that_are_not_a_list_of_names(
    tmp_path: Path, content: str, capsys: pytest.CaptureFixture[str]
) -> None:
    actions = tmp_path / "actions.json"
    actions.write_text(content)

    with pytest.raises(SystemExit) as raised:
        match.main(["--policy", "actions", "--actions", str(actions), "--out", str(tmp_path / "run")])

    assert raised.value.code == 2
    assert "actions must be a JSON list of native names" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


# ---------------------------------------------------------------- audits and replays


@pytest.fixture
def recorded(tmp_path: Path) -> Callable[..., Path]:
    """Play a match on the stand-in and return its directory."""

    def play(policy: str = "teacher", level: int = 0, **settings: Any) -> Path:
        out = tmp_path / f"{policy}-{level}-{len(list(tmp_path.iterdir()))}"
        run_match(GriddlyScenario(seed=3, level_id=level, max_steps=12), out, SEED, policy=policy, **settings)
        return out

    return play


def change_step(index: int, change: Callable[[dict[str, Any]], None], hashed: bool = True) -> Callable[[Path], None]:
    def apply(run: Path) -> None:
        rows = jsonl(run / "steps.jsonl")
        change(rows[index]["frame"])
        if hashed:
            rehash(rows[index]["frame"])
        write_jsonl(run / "steps.jsonl", rows)

    return apply


def change_decision(index: int, change: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    def apply(run: Path) -> None:
        rows = jsonl(run / "decisions.jsonl")
        change(rows[index])
        write_jsonl(run / "decisions.jsonl", rows)

    return apply


def change_manifest(change: Callable[[dict[str, Any]], None]) -> Callable[[Path], None]:
    return lambda run: edit_json(run / "manifest.json", change)


def drop_last(run: Path) -> None:
    for name in ("steps.jsonl", "decisions.jsonl"):
        write_jsonl(run / name, jsonl(run / name)[:-1])


AUDIT_FAULTS = [
    pytest.param(lambda run: (run / "error.json").write_text("{}"), "an incomplete or failed run", id="error-file"),
    pytest.param(
        change_manifest(lambda m: m.update(adapter_sha256="0" * 64)),
        "requires the original Griddly adapter source",
        id="adapter",
    ),
    pytest.param(
        lambda run: (run / "source" / "model.py").write_text("# changed\n"),
        "recorded source snapshot differs",
        id="source",
    ),
    pytest.param(change_manifest(lambda m: m.update(format="griddly-match-v0")), "unsupported", id="format"),
    pytest.param(change_manifest(lambda m: m["actions"].pop("WAIT")), "unsupported", id="actions"),
    pytest.param(
        lambda run: (run / "harness" / "griddly" / "questions.json").write_text('{"action": "Other words."}'),
        "recorded harness differs from delivered files",
        id="harness",
    ),
    pytest.param(change_manifest(lambda m: m.update(controller="human")), "inconsistent controller", id="controller"),
    pytest.param(change_manifest(lambda m: m.update(sample=True)), "inconsistent controller", id="sample-no-model"),
    pytest.param(change_manifest(lambda m: m.update(sample=1)), "inconsistent controller", id="sample-not-bool"),
    pytest.param(change_manifest(lambda m: m.update(kind="scripted")), "inconsistent controller", id="kind"),
    pytest.param(change_manifest(lambda m: m.update(policy_seed=-1)), "invalid policy seed", id="seed-negative"),
    pytest.param(change_manifest(lambda m: m.update(policy_seed="1")), "invalid policy seed", id="seed-text"),
    pytest.param(drop_last, "result differs from completed trajectory", id="unfinished"),
    pytest.param(
        lambda run: write_jsonl(run / "decisions.jsonl", jsonl(run / "decisions.jsonl")[:-1]),
        "incomplete or over-budget trajectory",
        id="missing-decision",
    ),
    pytest.param(
        change_manifest(lambda m: m["scenario"].update(max_steps=3)), "incomplete or over-budget", id="over-budget"
    ),
    pytest.param(
        change_step(2, lambda f: f.update(reward=0.5), hashed=False), "invalid frame hash or step at 2", id="hash"
    ),
    pytest.param(change_step(2, lambda f: f.update(step=5)), "invalid frame hash or step at 2", id="step-number"),
    pytest.param(change_step(2, lambda f: f.update(reward="0")), "nonfinite frame reward", id="reward-type"),
    pytest.param(change_step(0, lambda f: f.update(reward=1.0)), "invalid reset frame", id="reset-reward"),
    pytest.param(change_step(0, lambda f: f.update(status="won")), "invalid reset frame", id="reset-status"),
    pytest.param(
        change_decision(1, lambda d: d.update(applied="WAIT")), "invalid action or decision at step 2", id="applied"
    ),
    pytest.param(change_decision(1, lambda d: d.update(step=7)), "invalid action or decision at step 2", id="d-step"),
    pytest.param(
        change_decision(1, lambda d: d.update(before_frame_hash="0" * 64)), "invalid action or decision", id="before"
    ),
    pytest.param(change_decision(1, lambda d: d.update(forced_action=True)), "invalid action or decision", id="force"),
    pytest.param(change_decision(1, lambda d: d.update(decision_ms=-1)), "invalid decision latency", id="latency"),
    pytest.param(
        change_decision(1, lambda d: d.update(decision_ms="1")), "invalid decision latency", id="latency-txt"
    ),
    pytest.param(
        change_decision(1, lambda d: d["request"]["parts"]["essential"].append("invented")),
        "recorded request differs from public state",
        id="request",
    ),
    pytest.param(
        change_decision(1, lambda d: d.update(response={"answers": {}})),
        "a non-model or forced decision contains a model reply",
        id="reply",
    ),
    pytest.param(
        lambda run: edit_json(run / "result.json", lambda r: r.update(reward=2.0)),
        "result differs from completed trajectory",
        id="result",
    ),
]


@pytest.mark.functional
@pytest.mark.parametrize(("fault", "message"), AUDIT_FAULTS)
def test_the_audit_refuses_each_kind_of_changed_record_with_its_reason(
    recorded: Callable[..., Path], fault: Callable[[Path], None], message: str
) -> None:
    run = recorded("teacher")
    audit_run(run)
    fault(run)

    with pytest.raises(ValueError, match=message):
        audit_run(run)


@pytest.mark.functional
def test_the_audit_refuses_an_action_executed_where_it_was_not_legal(recorded: Callable[..., Path]) -> None:
    run = recorded("baseline")

    # The gnome starts at the bottom edge facing up, so BACKWARD would leave the map.
    def illegal(row: dict[str, Any]) -> None:
        row["applied"] = row["chosen"]["action"] = "BACKWARD"

    change_decision(0, illegal)(run)
    rows = jsonl(run / "steps.jsonl")
    rows[1]["action"] = "BACKWARD"
    write_jsonl(run / "steps.jsonl", rows)

    with pytest.raises(ValueError, match="invalid action or decision at step 1"):
        audit_run(run)


@pytest.mark.functional
def test_the_audit_refuses_a_random_run_whose_seed_would_have_drawn_other_actions(
    recorded: Callable[..., Path],
) -> None:
    run = recorded("random", policy_seed=5)
    change_manifest(lambda m: m.update(policy_seed=6))(run)

    with pytest.raises(ValueError, match="executed action differs from the recorded policy distribution"):
        audit_run(run)


@pytest.mark.functional
def test_the_audit_refuses_a_model_run_that_did_not_play_the_reply_s_choice(
    recorded: Callable[..., Path], endpoints: StandInEndpoints
) -> None:
    run = recorded("model", decision_server=endpoints.server.url)
    rows = jsonl(run / "decisions.jsonl")
    reply = rows[0]["response"]["answers"]["action"]
    other = next(name for name in reply["probabilities"] if name != reply["choice"])
    probabilities = reply["probabilities"]
    probabilities[other], probabilities[reply["choice"]] = probabilities[reply["choice"]], probabilities[other]
    reply["choice"] = other
    write_jsonl(run / "decisions.jsonl", rows)

    # A valid reply, but its choice is not the action the record says was executed.
    with pytest.raises(ValueError, match="executed action differs from the recorded policy distribution"):
        audit_run(run)


@pytest.mark.functional
def test_the_audit_refuses_a_model_reply_whose_choice_is_not_its_most_likely_option(
    recorded: Callable[..., Path], endpoints: StandInEndpoints
) -> None:
    run = recorded("model", decision_server=endpoints.server.url)
    rows = jsonl(run / "decisions.jsonl")
    reply = rows[0]["response"]["answers"]["action"]
    other = next(name for name in reply["probabilities"] if name != reply["choice"])
    reply["probabilities"] = {name: float(name == other) for name in reply["probabilities"]}
    write_jsonl(run / "decisions.jsonl", rows)

    with pytest.raises(ValueError, match="server choice must maximize its probabilities"):
        audit_run(run)


@pytest.mark.functional
def test_a_replay_on_another_runtime_is_refused(recorded: Callable[..., Path]) -> None:
    run = recorded("baseline")
    change_manifest(lambda m: m["runtime"].update(griddly="1.6.6"))(run)

    assert audit_run(run)[0]["runtime"]["griddly"] == "1.6.6"
    with pytest.raises(ValueError, match="native replay requires the recorded runtime versions"):
        verify_replay(run)


@pytest.mark.functional
def test_a_replay_whose_engine_starts_elsewhere_is_refused(
    recorded: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = recorded("baseline")
    reset = FakeSpidersEngine.reset

    def moved_reset(self: FakeSpidersEngine) -> dict[str, Any]:
        frame = reset(self)
        frame["observation"]["tick"] = 1
        return frame

    monkeypatch.setattr(FakeSpidersEngine, "reset", moved_reset)

    with pytest.raises(ValueError, match="native reset differs from recording"):
        verify_replay(run)


@pytest.mark.functional
def test_a_replay_whose_engine_diverges_names_the_first_step_that_differs(
    recorded: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = recorded("baseline")
    step = FakeSpidersEngine.step

    def drifting_step(self: FakeSpidersEngine, action: str) -> dict[str, Any]:
        frame = step(self, action)
        if frame["step"] == 3:
            frame["reward"] = 0.25
        return frame

    monkeypatch.setattr(FakeSpidersEngine, "step", drifting_step)

    with pytest.raises(ValueError, match="native replay differs at step 3"):
        verify_replay(run)


@pytest.mark.functional
@pytest.mark.parametrize(
    ("flags", "expected"),
    [(["--audit-only"], {"audited": True, "steps": 7}), ([], {"verified": True, "steps": 7})],
    ids=["audit-only", "replay"],
)
def test_the_replay_command_prints_what_it_checked(
    recorded: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: list[str],
    expected: dict[str, object],
) -> None:
    run = recorded("baseline")
    monkeypatch.setattr(sys, "argv", ["verify_replay.py", str(run), *flags])

    replay.main()

    printed = json.loads(capsys.readouterr().out)
    assert {key: printed[key] for key in expected} == expected
    if "verified" in expected:
        assert printed["final_frame_hash"] == json.loads((run / "result.json").read_text())["final_frame_hash"]


@pytest.mark.functional
def test_a_copied_run_audits_like_the_original(recorded: Callable[..., Path], tmp_path: Path) -> None:
    run = recorded("teacher", level=1)
    copy = tmp_path / "copy"
    shutil.copytree(run, copy)

    assert audit_run(copy) == audit_run(run)
