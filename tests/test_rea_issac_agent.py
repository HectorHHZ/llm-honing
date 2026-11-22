"""The model-backed agent may navigate, while frozen game and observation code stays fixed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from honing.backends.jev.nanojev import NanoJevSettings, resolve_checkpoint
from honing.backends.rea_issac.scenarios import ReaIssacScenario
from honing.backends.rea_issac_agent.checkpoint import CHECKPOINT_PIN, PINS, prepare_checkpoint
from honing.backends.rea_issac_agent.controller import (
    MODEL_ASSISTED_COMBAT,
    MODEL_ASSISTED_COMBAT_V2,
    MODEL_ASSISTED_COMBAT_V3,
    NAVIGATE_EMPTY_ROOMS,
    AgentController,
)
from honing.backends.rea_issac_agent.make_agent_demos import build_dataset, choose_across_rooms
from honing.backends.rea_issac_agent.train_agent import load_batch
from integrations.rea_issac.engine import Engine
from integrations.rea_issac.policy import Baseline

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_frozen_game_observation_and_tasks() -> None:
    frozen = json.loads((ROOT / "honing/backends/rea_issac_agent/agent_frozen.json").read_text())
    for name, checksum in frozen["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == checksum, name


@pytest.mark.unit
def test_agent_harness_does_not_change_default_game_harness() -> None:
    default = json.loads((ROOT / "integrations/rea_issac/config/controller.json").read_text())
    agent = json.loads((ROOT / "honing/backends/rea_issac_agent/config/controller.json").read_text())
    assert default == {"history_steps": 4}
    assert agent == {"history_steps": 0}


@pytest.mark.unit
def test_downloaded_checkpoint_can_be_verified_and_relocated(tmp_path: Path) -> None:
    checkpoint = tmp_path / "downloaded" / "checkpoint"
    checkpoint.mkdir(parents=True)
    pins = json.loads(PINS.read_text())["nanojevModels"]["nanojev"]
    files = {
        "best.safetensors": b"test weights",
        "config.json": json.dumps({"model": pins["base"], "resolved_model_revision": pins["baseRevision"]}).encode(),
        "backbone_config/config.json": b"{}",
        "tokenizer/chat_template.jinja": b"template",
        "tokenizer/tokenizer.json": b"{}",
        "tokenizer/tokenizer_config.json": b"{}",
    }
    for name, content in files.items():
        path = checkpoint / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    pin = json.loads(CHECKPOINT_PIN.read_text())
    pin["files"] = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    pin["weights_sha256"] = pin["files"]["best.safetensors"]
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin))

    manifest_path = prepare_checkpoint(checkpoint, checkpoint_pin=pin_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["path"] == str(checkpoint.resolve())
    assert manifest["weightsSha256"] == pin["weights_sha256"]
    all_pins = json.loads(PINS.read_text())
    resolved = resolve_checkpoint(
        NanoJevSettings(PINS, tmp_path, checkpoint_manifest=manifest_path),
        all_pins,
        all_pins["nanojevSource"]["revision"],
    )
    assert resolved["path"] == str(checkpoint.resolve())
    assert resolved["weightsSha256"] == pin["weights_sha256"]
    assert manifest_path == prepare_checkpoint(checkpoint, checkpoint_pin=pin_path)
    (checkpoint / "tokenizer/tokenizer.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match=r"tokenizer/tokenizer\.json"):
        prepare_checkpoint(checkpoint, checkpoint_pin=pin_path)


@pytest.mark.unit
def test_controller_keeps_model_action_in_combat() -> None:
    controller = AgentController(NAVIGATE_EMPTY_ROOMS)
    selected = {"move_x": 1, "move_y": 1, "shoot_x": 1, "shoot_y": 1, "bomb": False}
    empty = Engine(1002).observe()
    executed, reason = controller.apply(empty, selected)
    assert reason == "empty_room_navigation"
    assert executed != selected
    combat = {**empty, "room": {**empty["room"], "enemies": [{"id": "visible-enemy"}]}}
    assert controller.apply(combat, selected) == (selected, "model_combat")


@pytest.mark.unit
def test_assisted_controller_uses_model_only_for_near_rule_directions() -> None:
    combat = Engine(1002).observe()
    combat["room"]["enemies"] = [{"x": 500.0, "y": 300.0, "radius": 12.0}]
    rule = Baseline().act(combat)
    controller = AgentController(MODEL_ASSISTED_COMBAT)
    rejected = {
        "move_x": -rule["move_x"],
        "move_y": -rule["move_y"],
        "shoot_x": 0,
        "shoot_y": 0,
        "bomb": False,
    }
    action, reason = controller.apply(combat, rejected)
    assert action == rule
    assert reason == "assisted_combat_move_0_shoot_0"
    action, reason = controller.apply(combat, {**rule, "bomb": True})
    assert action == rule
    assert reason == "assisted_combat_move_1_shoot_1"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("position", "move", "destination"),
    [
        ((593.9269119345827, 209.67422074112307), (1, 0), "1,0"),
        ((46.0730880654173, 209.67422074112307), (-1, 0), "-1,0"),
        ((329.67422074112307, 54.0730880654173), (0, -1), "0,-1"),
        ((329.67422074112307, 361.9269119345827), (0, 1), "0,1"),
    ],
)
def test_controller_nudges_across_a_door_trigger_skipped_by_cardinal_steps(
    position: tuple[float, float], move: tuple[int, int], destination: str
) -> None:
    engine = Engine(4001, 100)
    engine.player["x"], engine.player["y"] = position
    cardinal = {"move_x": move[0], "move_y": move[1], "shoot_x": 0, "shoot_y": 0}
    engine.step(cardinal, 1)
    assert engine.observe()["room"]["id"] == "0,0"
    assert (engine.player["x"], engine.player["y"]) == position
    previous = AgentController(MODEL_ASSISTED_COMBAT)
    previous.navigator.act = lambda _observation: cardinal  # type: ignore[method-assign]
    assert previous.apply(engine.observe(), {**cardinal, "bomb": False}) == (cardinal, "empty_room_navigation")
    controller = AgentController(MODEL_ASSISTED_COMBAT_V2)
    controller.navigator.act = lambda _observation: cardinal  # type: ignore[method-assign]
    action, reason = controller.apply(engine.observe(), {**cardinal, "bomb": False})
    assert reason == "door_entry_nudge"
    assert (action["move_x"], action["move_y"]) != move
    engine.step(action, 1)
    assert engine.observe()["room"]["id"] == destination


@pytest.mark.unit
def test_cleanup_alignment_prevents_moving_away_from_a_one_shot_enemy() -> None:
    observation = Engine(7004).observe()
    observation["room"]["kind"] = "combat"
    observation["room"]["enemies"] = [{"x": 165.0, "y": 261.0, "health": 3.0}]
    observation["player"].update(x=55.0, y=314.0)
    rule = {"move_x": 0, "move_y": 1, "shoot_x": 1, "shoot_y": 0, "bomb": False}
    controller = AgentController(MODEL_ASSISTED_COMBAT_V3)
    controller.navigator.act = lambda _observation: rule  # type: ignore[method-assign]
    action, reason = controller.apply(observation, rule)
    assert reason == "cleanup_alignment"
    assert action == {**rule, "move_y": -1}
    observation["player"]["y"] = 273.0
    assert controller.apply(observation, rule) == ({**rule, "move_y": -1}, "cleanup_alignment")
    observation["room"]["enemies"][0]["health"] = 20.0
    assert controller.apply(observation, rule) == (rule, "assisted_combat_move_1_shoot_1")


@pytest.mark.integration
@pytest.mark.parametrize("seed", [1002, 2001])
def test_controller_reaches_a_combat_room_using_public_observations(seed: int) -> None:
    scenario = ReaIssacScenario(seed, 300, 3)
    engine = Engine(scenario.seed, scenario.max_ticks)
    controller = AgentController()
    selected = {"move_x": 1, "move_y": 1, "shoot_x": 1, "shoot_y": 1, "bomb": False}
    first_combat = None
    while engine.status == "running":
        observation = engine.observe()
        executed, reason = controller.apply(observation, selected)
        if reason == "model_combat":
            first_combat = (engine.tick, observation["room"]["id"])
            break
        engine.step(executed, scenario.decision_frames)
    assert first_combat == (63, "-1,0")


@pytest.mark.unit
def test_demonstrations_keep_seed_splits_and_frozen_inputs(tmp_path: Path) -> None:
    dataset = tmp_path / "demos"
    summary = build_dataset(dataset, max_ticks=120, per_seed=2)
    batch, loaded = load_batch(dataset)
    assert loaded == summary
    assert len(batch.decisions) == summary["decisions"]
    assert all(set(dict(sample.chosen)) == {"move", "shoot", "bomb"} for sample in batch.decisions)
    assert all(
        sample.group and "3001" not in sample.group and "3002" not in sample.group for sample in batch.decisions
    )
    rows = [json.loads(line) for line in (dataset / "decisions.jsonl").read_text().splitlines()]
    rows[0]["split"] = "validation"
    (dataset / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="dataset manifest"):
        load_batch(dataset)


@pytest.mark.unit
def test_balanced_demonstration_selection_reaches_later_rooms() -> None:
    rows = [{"room": "first", "tick": tick} for tick in range(12)] + [
        {"room": "boss", "tick": tick} for tick in range(12, 18)
    ]
    selected = choose_across_rooms(rows, 6)
    assert len(selected) == 6
    assert [row["room"] for row in selected] == ["first", "boss"] * 3
    assert [row["tick"] for row in selected if row["room"] == "first"] == [0, 4, 8]
