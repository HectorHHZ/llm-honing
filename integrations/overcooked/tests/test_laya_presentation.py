"""Native action grounding and exact replay for the experiment-only presentation files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from honing.core import HarnessFile
from integrations.overcooked.decisions import make_request
from integrations.overcooked.engine import ACTIONS, LAYOUTS, Engine, Scenario
from integrations.overcooked.harness import parse_harness, seed_harness
from integrations.overcooked.match import run_match
from integrations.overcooked.presentation import action_descriptions, readable_context
from integrations.overcooked.recording import replay
from integrations.overcooked.tests.test_overcooked import decision_server as decision_server


@pytest.mark.parametrize("layout", LAYOUTS)
def test_readable_map_and_local_choices_are_readonly_and_keep_every_action(layout: str) -> None:
    with Engine(Scenario(layout=layout)) as game:
        before = game.observe()
        text = readable_context(before, 0, [], "")
        for y, row in enumerate(before["map"]["terrain"]):
            assert f"row y={y}: {''.join(row).replace(' ', '.')}" in text
        assert "Chef 0 (YOU)" in text and "Chef 1 (teammate)" in text
        choices = action_descriptions(before, 0)
        assert list(choices) == list(ACTIONS)
        assert game.observe() == before


@pytest.mark.parametrize("case", ["take_onion", "add_onion", "start_pot", "collect_soup", "serve_soup"])
def test_local_interaction_wording_agrees_with_the_native_engine(case: str) -> None:
    from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedState, PlayerState, SoupState

    with Engine() as game:
        position, facing, held, objects = (2, 1), (0, -1), None, {}
        if case == "take_onion":
            position, facing = (1, 1), (-1, 0)
        elif case == "add_onion":
            held = ObjectState("onion", position)
        elif case == "start_pot":
            objects[(2, 0)] = SoupState.get_soup((2, 0), num_onions=3)
        elif case == "collect_soup":
            held = ObjectState("dish", position)
            objects[(2, 0)] = SoupState.get_soup((2, 0), num_onions=3, finished=True)
        else:
            position, facing = (3, 2), (0, 1)
            held = SoupState.get_soup(position, num_onions=3, finished=True)
        game.env.state = OvercookedState(
            [PlayerState(position, facing, held), PlayerState((1, 2), (0, 1))],
            objects,
            all_orders=[{"ingredients": ["onion"] * 3}],
        )
        text = action_descriptions(game.observe(), 0)["INTERACT"]
        after = game.step(["INTERACT", "STAY"])
        if case == "take_onion":
            assert "take an object" in text and after["state"]["players"][0]["held_object"]["name"] == "onion"
        elif case == "add_onion":
            assert "add held onion" in text and len(after["state"]["objects"][0]["_ingredients"]) == 1
        elif case == "start_pot":
            assert "start cooking" in text and after["state"]["objects"][0]["is_cooking"]
        elif case == "collect_soup":
            assert "collect ready soup" in text and after["state"]["players"][0]["held_object"]["name"] == "soup"
        else:
            assert "deliver the held soup" in text and after["score"] == 20


@pytest.mark.parametrize("choices", ["static", "local_effects"])
def test_custom_harness_http_run_rebuilds_exact_context_and_choices_for_replay(
    tmp_path: Path, decision_server: Any, choices: str
) -> None:
    spec = seed_harness().with_file(
        "overcooked/presentation.json",
        HarnessFile("settings", json.dumps({"context": "readable", "choices": choices})),
    )
    endpoint, requests, _ = decision_server
    out = tmp_path / "custom-harness"
    run_match(Scenario(max_steps=4), out, policy="model", decision_server=endpoint, spec=spec, record_images=False)
    assert len(requests) == 4 and "row y=0" in requests[0]["parts"]["essential"][0]
    assert replay(out)["verified"]
    path = out / "harness/files.json"
    rows = json.loads(path.read_text())
    next(row for row in rows if row["path"] == "overcooked/presentation.json")[
        "content"
    ] = '{"context":"json","choices":"static"}'
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        replay(out)


def test_presentation_cannot_change_game_rules_or_drop_choices() -> None:
    bad = seed_harness().with_file(
        "overcooked/presentation.json",
        HarnessFile("settings", '{"context":"readable","choices":"local_effects","partner":"stay"}'),
    )
    with pytest.raises(ValueError):
        parse_harness(bad)
    with Engine() as game:
        frame = game.observe()
        assert make_request(frame, 0) == make_request(frame, 0, presentation={"context": "json", "choices": "static"})


def test_experiment_initializes_all_candidate_bundles_and_disjoint_tasks(
    tmp_path: Path, decision_server: Any, monkeypatch: Any
) -> None:
    from integrations.overcooked import laya_experiment
    from tests.fakes.model import FakeModelSource, FakeWorld

    target = tmp_path / "experiment"
    snap = FakeModelSource(FakeWorld()).materialize(tmp_path / "weights")
    monkeypatch.setattr(laya_experiment, "DIRECTORY", target)
    monkeypatch.setattr(laya_experiment, "ENDPOINT", decision_server[0])
    monkeypatch.setattr(laya_experiment, "snapshot", lambda: snap)
    laya_experiment.initialize()
    assert len(list((target / "candidates").iterdir())) == 4
    assert len(laya_experiment.suite().records) == 5
    assert len(laya_experiment.suite().tasks("train")) == 2
    for name in ("seed", "readable", "readable_objective", "readable_local_effects"):
        parse_harness(laya_experiment.candidate(name))
