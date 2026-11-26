"""The VVVVVV Exp 0 v0 collector, adapter and release driver, offline: recorded release rows, no game, model or network.

The fixtures are the first model rows of two attempts of the release collection, cut unchanged.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]
INTEGRATION = REPO / "integrations" / "vvvvvv"
V0 = INTEGRATION / "exp0" / "v0"
COMMON = REPO / "integrations" / "_shared" / "exp0"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "exp0_vvvvvv"

# pins.harness and pins.controller of every release manifest (contract.fingerprint_files over the collecting
# workspace's names integration/<file>); the release ran on integrations/vvvvvv at 9d5327f.
RECORDED_HARNESS = "64cfe78b0d147d63ad9dd1c9777c27fe2111448f4c29aa3a854984cdc4f49901"
RECORDED_CONTROLLER = "3f102b42697107794da06a2a287b29172f598086a920c8ec935033bbd36eff5d"


def rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def exact(value: object) -> str:
    """Request bytes with keys in the order they were sent; option order changes Laya's probabilities."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


@pytest.fixture(scope="module")
def modules() -> Iterator[dict[str, ModuleType]]:
    """Load the v0 scripts by path; drop the bare-name modules and paths they added afterwards."""
    before_modules, before_path = set(sys.modules), list(sys.path)
    loaded = {}
    for name in ("exp0_vvvvvv", "adapter", "fault_checks", "release_v0"):
        spec = importlib.util.spec_from_file_location(f"_exp0_vvvvvv_{name}", V0 / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded[name] = module
    yield loaded
    for name in set(sys.modules) - before_modules:
        del sys.modules[name]
    sys.path[:] = before_path


@pytest.mark.unit
def test_recorded_harness_and_controller_are_the_merged_integration_files() -> None:
    def fingerprint(files: dict[str, Path]) -> str:
        digest = hashlib.sha256()
        for name in sorted(files):
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(hashlib.sha256(files[name].read_bytes()).hexdigest().encode("ascii") + b"\n")
        return digest.hexdigest()

    harness = {f"integration/{name}": INTEGRATION / name for name in ("decision.py", "run.py")}
    controller = {f"integration/{name}": INTEGRATION / name for name in ("HoningBridge.inc", "run.py")}
    assert fingerprint(harness) == RECORDED_HARNESS
    assert fingerprint(controller) == RECORDED_CONTROLLER


@pytest.mark.unit
def test_adapter_rerenders_every_logged_clean_request(modules: dict[str, ModuleType]) -> None:
    adapter = modules["adapter"].ADAPTER
    assert adapter.game_id == "vvvvvv"
    for row in rows("clean.jsonl"):
        assert exact(adapter.rerender(row["state"], [])) == exact(row["request"]), row["index"]


@pytest.mark.unit
def test_terrain_omission_changes_only_the_terrain_and_the_receiver_accepts_it(
    modules: dict[str, ModuleType],
) -> None:
    collector, adapter = modules["exp0_vvvvvv"], modules["adapter"].ADAPTER
    harness_rows = rows("harness.jsonl")
    assert harness_rows
    for row in harness_rows:
        clean, faulted = row["clean_request"], row["request"]
        assert exact(adapter.rerender(row["state"], [])) == exact(clean)
        observation, ticks = row["state"]["observation"], row["state"]["ticks"]
        assert exact(collector.request_for(collector.omit_terrain(observation), ticks)) == exact(faulted)
        assert exact(faulted) != exact(clean)
        assert exact(faulted["questions"]) == exact(clean["questions"])
        collector.receiver_check(faulted, row["reply"].encode("utf-8"))


@pytest.mark.unit
def test_controller_swap_mirrors_only_horizontal_inputs(modules: dict[str, ModuleType]) -> None:
    mirror = modules["exp0_vvvvvv"].MIRROR
    assert set(mirror) == {"left", "right", "flip_left", "flip_right"}
    assert all(mirror[mirror[action]] == action != mirror[action] for action in mirror)


@pytest.mark.unit
def test_native_events_follow_the_death_room_and_checkpoint_signals(modules: dict[str, ModuleType]) -> None:
    collector = modules["exp0_vvvvvv"]
    before = {"tick": 30, "deaths": 0, "deathSequence": -1, "room": [113, 105], "checkpoint": [113, 105, 200, 161, 0]}
    dying = {**before, "tick": 45, "deathSequence": 10}
    respawned = {**before, "tick": 90, "deaths": 1, "room": [114, 105], "checkpoint": [114, 105, 40, 161, 0]}
    assert [event["kind"] for event in collector.events_between(before, dying)] == ["death"]
    assert [event["kind"] for event in collector.events_between(dying, respawned)] == ["room", "checkpoint"]
    assert collector.died(before, dying, before) and collector.died(before, respawned, before)
    assert not collector.died(before, before, before)


@pytest.mark.unit
def test_frozen_release_plan_follows_the_contract() -> None:
    spec = importlib.util.spec_from_file_location("_exp0_vvvvvv_schema_lite", COMMON / "schema_lite.py")
    assert spec is not None and spec.loader is not None
    schema_lite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema_lite)
    plan = json.loads((V0 / "plans" / "release_v0_frozen.json").read_text(encoding="utf-8"))
    schema = schema_lite.load_schema(COMMON / "schemas" / "collection.schema.json")
    assert schema_lite.errors(plan, schema) == []
    assert (plan["game_id"], plan["dataset_dir"], plan["per_condition"], plan["initial_groups"]) == (
        "vvvvvv",
        "vvvvvv/v0/",
        10,
        10,
    )
    assert len(plan["groups"]) == 16
    assert (V0 / "collection.json").read_bytes() == (V0 / "plans" / "release_v0_frozen.json").read_bytes()


@pytest.mark.unit
def test_release_driver_snapshots_only_files_that_exist(modules: dict[str, ModuleType]) -> None:
    driver = modules["release_v0"]
    assert sorted(name for name, path in driver.SOURCES.items() if not path.exists()) == []
    assert driver.COMMON == COMMON
    assert sorted(path.name for path in driver.TEMPLATES.glob("*.md")) == [
        "DATASET.md",
        "LABELS.md",
        "PROTOCOL.md",
        "RELEASE_RECEIPT.md",
        "RESULTS.md",
        "RUNBOOK.md",
        "TODO.md",
    ]
