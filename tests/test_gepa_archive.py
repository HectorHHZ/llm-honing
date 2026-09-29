"""The GEPA archive (#21): JSON-native state, recomputed Pareto fronts, and parent sampling that matches upstream."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from honing.core import ContractError
from honing.core.state import validate_method_state
from honing.harness.methods.gepa import ARCHIVE_SCHEMA_VERSION, Archive

TASK_IDS = ("t0", "t1", "t2", "t3")
COMPONENTS = {"CLAUDE.md": "be careful", "skills/tests.md": "run the tests"}


def _archive(*validation_scores: tuple[float, ...], rng_seed: int = 0) -> Archive:
    """A seed plus one child per extra score row, each child parented on the seed, all validated."""
    task_ids = TASK_IDS[: len(validation_scores[0])]
    archive = Archive.seed(COMPONENTS, task_ids, rng_seed)
    for index, scores in enumerate(validation_scores):
        if index > 0:
            archive, _ = archive.add({**COMPONENTS, "CLAUDE.md": f"variant {index}"}, parent=0)
        archive = archive.validate(index, task_ids, scores)
    return archive


def _draws(archive: Archive, count: int) -> list[int]:
    picks = []
    for _ in range(count):
        archive, parent = archive.sample_parent()
        picks.append(parent)
    return picks


def _random_archive(rng: random.Random) -> Archive:
    """Random adds, validations, parent draws, cursor moves and charges, in a random order."""
    archive = Archive.seed(COMPONENTS, TASK_IDS, rng.randrange(1000))
    for _ in range(rng.randrange(1, 25)):
        action = rng.choice(["add", "validate", "sample", "cursor", "charge", "serve", "minibatch"])
        size = len(archive.candidates)
        if action == "add":
            archive, _ = archive.add(
                {**COMPONENTS, "CLAUDE.md": f"text {rng.random()}"},
                parent=rng.randrange(size),
                minibatch_scores=[rng.choice([0.0, 0.5, 1.0]) for _ in range(rng.randrange(4))],
            )
        elif action == "validate":
            unvalidated = [i for i, c in enumerate(archive.candidates) if c.validation_scores is None]
            if unvalidated:
                scores = [rng.choice([0.0, 0.25, 0.5, 1.0, rng.random()]) for _ in TASK_IDS]
                archive = archive.validate(rng.choice(unvalidated), TASK_IDS, scores)
        elif action == "sample":
            archive, _ = archive.sample_parent()
        elif action == "cursor":
            archive, _ = archive.next_component(rng.randrange(size))
        elif action == "charge":
            archive = archive.charge(rng.randrange(10))
        elif action == "minibatch":
            train_ids = rng.choice([["a", "b", "c"], ["a", "b", "c", "d", "e"]])
            size = rng.randrange(1, 4)
            archive, batch = archive.next_minibatch(train_ids, size)
            assert len(batch) == size
        else:
            archive = archive.with_served(rng.randrange(size))
    return archive


# Acceptance: state round trip.


@pytest.mark.unit
def test_state_round_trips_after_random_additions_and_validations() -> None:
    pytest.importorskip("gepa", reason="parent draws need the 'gepa' extra")
    rng = random.Random(21)
    for _ in range(200):
        archive = _random_archive(rng)
        assert Archive.from_state(archive.to_state()) == archive
        # Through actual JSON text too, which is what the commit record stores.
        assert Archive.from_state(json.loads(json.dumps(archive.to_state()))) == archive


@pytest.mark.unit
def test_state_dumps_without_a_default_hook_and_passes_the_method_state_rules() -> None:
    archive = _archive((0.5, 1.0), (1.0, 0.0))
    archive, _ = archive.add(COMPONENTS, parent=1, minibatch_scores=[0.5])
    state = archive.to_state()

    json.dumps(state)
    validate_method_state(state)
    assert state["schema_version"] == ARCHIVE_SCHEMA_VERSION
    assert "fronts" not in state


# Acceptance: fronts.


@pytest.mark.unit
def test_a_strictly_better_score_replaces_the_front() -> None:
    archive = _archive((0.5, 0.5), (0.75, 0.25))

    assert archive.fronts()["t0"] == {1}


@pytest.mark.unit
def test_an_equal_score_joins_the_front() -> None:
    archive = _archive((0.5, 0.5), (0.75, 0.5), (0.25, 0.5))

    assert archive.fronts() == {"t0": {1}, "t1": {0, 1, 2}}


@pytest.mark.unit
def test_an_unvalidated_candidate_is_on_no_front() -> None:
    archive = _archive((0.5, 0.5))
    archive, _ = archive.add(COMPONENTS, parent=0, minibatch_scores=[1.0, 1.0])

    assert archive.fronts() == {"t0": {0}, "t1": {0}}


@pytest.mark.unit
def test_a_candidate_on_no_front_is_never_sampled_as_a_parent() -> None:
    pytest.importorskip("gepa", reason="parent draws need the 'gepa' extra")
    archive = _archive((1.0, 0.0, 0.5), (0.0, 1.0, 0.5), (0.25, 0.25, 0.25))
    assert all(2 not in front for front in archive.fronts().values())

    picks = _draws(archive, 2000)

    assert 2 not in picks
    assert set(picks) == {0, 1}


@pytest.mark.unit
def test_a_dominated_candidate_is_never_sampled_as_a_parent() -> None:
    # Identical on every task: each front holding one also holds the other, so
    # upstream keeps exactly one of them.
    pytest.importorskip("gepa", reason="parent draws need the 'gepa' extra")
    archive = _archive((0.5, 0.5), (0.5, 0.5))

    assert len(set(_draws(archive, 500))) == 1


# Acceptance: parity with upstream.


def _upstream_fronts(task_ids: tuple[str, ...], score_rows: list[tuple[float, ...]]) -> dict[str, set[int]]:
    """Build fronts with upstream GEPA's own update method, one program at a time, the way GEPAState does."""
    from gepa.core.state import GEPAState

    state = SimpleNamespace(pareto_front_valset={}, program_at_pareto_front_valset={}, best_outputs_valset=None)
    for program, scores in enumerate(score_rows):
        for task_id, score in zip(task_ids, scores, strict=True):
            GEPAState._update_pareto_front_for_val_id(state, task_id, score, program, None, None, 0)  # type: ignore[arg-type]
    return state.program_at_pareto_front_valset


def _random_rows(rows_seed: int) -> list[tuple[float, ...]]:
    # Twelve candidates on coarse scores, so fronts have several members.
    rows_rng = random.Random(rows_seed)
    return [tuple(rows_rng.choice([0.0, 0.5, 1.0]) for _ in TASK_IDS) for _ in range(12)]


# 0 and 8 share t0's front and each lead a task alone, so neither is dominated.
# They also land in the same slot of a small set, so {0, 8} iterates in
# insertion order: built 0-then-8 it gives [0, 8], built 8-then-0 it gives
# [8, 0], and upstream's draw depends on that order. Random rows rarely
# produce such a pair, so this case pins it.
_COLLIDING_ROWS = [
    (1.0, 1.0, 0.0, 0.0),
    *[(0.5, 0.0, 0.0, 0.0)] * 7,
    (1.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("score_rows", "rng_seed"),
    [
        (_COLLIDING_ROWS, 0),
        (_COLLIDING_ROWS, 5),
        (_random_rows(1000), 0),
        (_random_rows(1001), 1),
        (_random_rows(1007), 7),
        (_random_rows(12345), 12345),
    ],
)
def test_parent_sampling_matches_the_upstream_function(score_rows: list[tuple[float, ...]], rng_seed: int) -> None:
    pytest.importorskip("gepa", reason="parity needs the 'gepa' extra")
    from gepa.gepa_utils import select_program_candidate_from_pareto_front

    archive = _archive(*score_rows, rng_seed=rng_seed)
    expected_fronts = _upstream_fronts(TASK_IDS, score_rows)
    means = [sum(row) / len(row) for row in score_rows]
    upstream_rng = random.Random(rng_seed)
    expected = [select_program_candidate_from_pareto_front(expected_fronts, means, upstream_rng) for _ in range(50)]

    assert archive.fronts() == expected_fronts
    # Set equality ignores order; the draw does not, so compare iteration order too.
    assert [list(front) for front in archive.fronts().values()] == [list(f) for f in expected_fronts.values()]
    assert _draws(archive, 50) == expected


@pytest.mark.unit
def test_the_colliding_fixture_really_depends_on_insertion_order() -> None:
    def inserted(order: tuple[int, ...]) -> set[int]:
        # A set literal would be compiled to a constant and lose the order; add one at a time.
        members: set[int] = set()
        for member in order:
            members.add(member)
        return members

    assert list(inserted((0, 8))) != list(inserted((8, 0)))


# Rollback and the RNG.


@pytest.mark.unit
def test_a_restored_state_repeats_the_same_parent_picks() -> None:
    pytest.importorskip("gepa", reason="parent draws need the 'gepa' extra")
    archive = _archive((1.0, 0.0, 0.5), (0.0, 1.0, 0.5), (0.5, 0.5, 1.0), rng_seed=3)
    archive, _ = archive.sample_parent()
    published = json.loads(json.dumps(archive.to_state()))

    assert _draws(Archive.from_state(published), 20) == _draws(archive, 20)


@pytest.mark.unit
def test_sampling_before_any_validation_returns_the_served_candidate_without_drawing() -> None:
    archive = Archive.seed(COMPONENTS, TASK_IDS, rng_seed=0)

    after, parent = archive.sample_parent()

    assert parent == 0
    assert after.rng_state == archive.rng_state


@pytest.mark.unit
def test_the_rng_state_is_stored_as_plain_json_lists() -> None:
    rng_state = Archive.seed(COMPONENTS, TASK_IDS, rng_seed=0).to_state()["rng_state"]

    assert isinstance(rng_state, list)
    assert rng_state[0] == 3
    assert isinstance(rng_state[1], list)
    assert len(rng_state[1]) == 625
    assert rng_state[2] is None


# Components.


@pytest.mark.unit
def test_components_rotate_round_robin_and_a_child_starts_at_its_parents_cursor() -> None:
    archive = Archive.seed({"a.md": "", "b.md": "", "c.md": ""}, TASK_IDS, rng_seed=0)
    picks = []
    for _ in range(4):
        archive, path = archive.next_component(0)
        picks.append(path)
    archive, child = archive.add({"a.md": "x", "b.md": "", "c.md": ""}, parent=0)
    archive, child_pick = archive.next_component(child)

    assert picks == ["a.md", "b.md", "c.md", "a.md"]
    assert child_pick == "b.md"


@pytest.mark.unit
def test_a_child_must_rewrite_the_seeds_components() -> None:
    archive = Archive.seed(COMPONENTS, TASK_IDS, rng_seed=0)

    with pytest.raises(ContractError, match="seed's components"):
        archive.add({"CLAUDE.md": "only one"}, parent=0)


# Refusals.


@pytest.mark.unit
def test_validation_refuses_other_tasks_out_of_range_scores_and_a_second_validation() -> None:
    archive = Archive.seed(COMPONENTS, ("t0", "t1"), rng_seed=0)

    with pytest.raises(ContractError, match="not the archive's tasks"):
        archive.validate(0, ("t1", "t0"), (0.5, 0.5))
    with pytest.raises(ContractError, match=r"\[0, 1\]"):
        archive.validate(0, ("t0", "t1"), (0.5, 1.5))
    with pytest.raises(ContractError, match=r"\[0, 1\]"):
        archive.validate(0, ("t0", "t1"), (True, 0.5))
    validated = archive.validate(0, ("t0", "t1"), (0.5, 0.5))
    with pytest.raises(ContractError, match="already validated"):
        validated.validate(0, ("t0", "t1"), (0.5, 0.5))


@pytest.mark.unit
def test_from_state_refuses_another_schema_version() -> None:
    state = Archive.seed(COMPONENTS, TASK_IDS, rng_seed=0).to_state()

    with pytest.raises(ContractError, match="schema_version"):
        Archive.from_state({**state, "schema_version": ARCHIVE_SCHEMA_VERSION + 1})
    with pytest.raises(ContractError, match="schema_version"):
        Archive.from_state({key: value for key, value in state.items() if key != "schema_version"})


@pytest.mark.unit
def test_from_state_refuses_a_malformed_rng_state() -> None:
    state = Archive.seed(COMPONENTS, TASK_IDS, rng_seed=0).to_state()
    rng_state = state["rng_state"]
    assert isinstance(rng_state, list)
    version, words, gauss = rng_state
    assert isinstance(words, list)

    with pytest.raises(ContractError, match="625 words"):
        Archive.from_state({**state, "rng_state": [version, words[:-1], gauss]})
    with pytest.raises(ContractError, match="rng_state"):
        Archive.from_state({**state, "rng_state": [version, [*words[:-1], 10_000], gauss]})
    with pytest.raises(ContractError, match="rng_state version"):
        Archive.from_state({**state, "rng_state": [2, words, gauss]})
