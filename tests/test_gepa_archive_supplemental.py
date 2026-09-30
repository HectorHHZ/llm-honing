"""Additional coverage for archive updates, isolation, and invalid inputs."""

import pytest

from honing.core import ContractError
from honing.harness.methods.gepa import Archive

pytestmark = pytest.mark.unit

COMPONENTS = {"CLAUDE.md": "be careful", "skills/tests.md": "run the tests"}
TASK_IDS = ("t0", "t1")


def seed():
    return Archive.seed(COMPONENTS, TASK_IDS, rng_seed=0)


def test_charge_accumulates_without_changing_old_archive():
    archive = seed()
    updated = archive.charge(2).charge(3)
    assert updated.metric_calls == 5
    assert archive.metric_calls == 0


def test_unvalidated_sampling_returns_updated_served_candidate():
    archive = seed()
    archive, child = archive.add({**COMPONENTS, "CLAUDE.md": "new instructions"}, parent=0)
    updated = archive.with_served(child)
    after, selected = updated.sample_parent()
    assert updated.served == child
    assert archive.served == 0
    assert selected == child
    assert after.rng_state == updated.rng_state


def test_exported_state_does_not_share_mutable_data():
    archive = seed()
    state = archive.to_state()
    state["candidates"][0]["components"]["CLAUDE.md"] = "changed"
    assert dict(archive.candidates[0].components) == COMPONENTS


def test_added_candidate_copies_input_components_and_scores():
    archive = seed()
    components = {**COMPONENTS, "CLAUDE.md": "new instructions"}
    scores = [0.5, 0.75]
    updated, child = archive.add(components, parent=0, minibatch_scores=scores)
    components["CLAUDE.md"] = "changed after add"
    scores[0] = 0.0
    assert dict(updated.candidates[child].components)["CLAUDE.md"] == "new instructions"
    assert updated.candidates[child].minibatch_scores == (0.5, 0.75)
    assert len(archive.candidates) == 1


def test_restored_archive_does_not_share_mutable_input_state():
    archive = seed().validate(0, TASK_IDS, (0.5, 0.75))
    state = archive.to_state()
    restored = Archive.from_state(state)
    state["candidates"][0]["components"]["CLAUDE.md"] = "changed"
    state["candidates"][0]["validation_scores"][0] = 0.0
    state["task_ids"][0] = "different-task"
    state["rng_state"][1][0] = 0
    assert restored == archive


def test_worse_candidate_is_not_sampled_despite_sharing_a_front():
    archive = seed().validate(0, TASK_IDS, (1.0, 0.5))
    archive, child = archive.add({**COMPONENTS, "CLAUDE.md": "better"}, parent=0)
    archive = archive.validate(child, TASK_IDS, (1.0, 1.0))
    assert archive.fronts()["t0"] == {0, child}
    picks = []
    for _ in range(20):
        archive, parent = archive.sample_parent()
        picks.append(parent)
    assert set(picks) == {child}


def test_components_rotate_in_sorted_path_order():
    archive = Archive.seed({"c.md": "", "a.md": "", "b.md": ""}, TASK_IDS, rng_seed=0)
    picks = []
    for _ in range(3):
        archive, path = archive.next_component(0)
        picks.append(path)
    assert picks == ["a.md", "b.md", "c.md"]


def test_validation_rejects_wrong_score_count():
    with pytest.raises(ContractError, match="scores"):
        seed().validate(0, TASK_IDS, (0.5,))


@pytest.mark.parametrize("score", [-0.1, float("nan"), float("inf"), float("-inf")])
def test_validation_rejects_invalid_scores(score):
    with pytest.raises(ContractError, match="scores"):
        seed().validate(0, TASK_IDS, (score, 0.5))


@pytest.mark.parametrize("parent", [-1, 1])
def test_add_rejects_invalid_parent(parent):
    with pytest.raises(ContractError, match="parent"):
        seed().add(COMPONENTS, parent=parent)


def test_charge_rejects_negative_count():
    with pytest.raises(ContractError, match="metric_calls"):
        seed().charge(-1)


@pytest.mark.parametrize("index", [-1, 1])
def test_with_served_rejects_invalid_index(index):
    with pytest.raises(ContractError, match="index"):
        seed().with_served(index)
