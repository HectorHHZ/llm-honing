"""GEPA's search state: every candidate it has proposed, their scores, and the RNG that picks the next parent.

Honing publishes a method's state with each release and hands it back on
rollback, so the archive converts to and from JSON-native values only. The
Pareto fronts are not part of that state: they are recomputed from the
recorded validation scores, so a front can never disagree with the scores it
came from.

Parent sampling is delegated to upstream GEPA
(``gepa.gepa_utils.select_program_candidate_from_pareto_front``) rather than
reimplemented, so a parent pick here is the pick GEPA itself would make. The
training minibatches follow upstream's ``EpochShuffledBatchSampler`` and draw
from the same RNG, as upstream's ``gepa.optimize`` shares one RNG between the
two.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from honing.core import ContractError, require
from honing.core.state import METHOD_STATE_VERSION_KEY, method_state_version

# 2: the training-minibatch sampler's state (iteration, train_ids, train_order, train_epoch,
# train_minibatch_size).
ARCHIVE_SCHEMA_VERSION = 2

# ``random.Random.getstate()`` for the Mersenne Twister: a version, 624 words
# of generator state plus the position within them, and a cached Gaussian.
_RNG_VERSION = 3
_RNG_WORDS = 625

RngState = tuple[int, tuple[int, ...], float | None]


def _require_scores(scores: Sequence[float], name: str) -> None:
    for score in scores:
        require(
            isinstance(score, (int, float)) and not isinstance(score, bool) and 0.0 <= score <= 1.0,
            f"{name} must hold scores in [0, 1], got {score!r}",
        )


@dataclass(frozen=True)
class ArchiveCandidate:
    """One harness GEPA has proposed, with what it has been scored on so far.

    ``components`` pairs each rewritable file path with its text, sorted by
    path, so the same files always compare equal. ``validation_scores`` follows
    the archive's task order and stays ``None`` until the gate has scored the
    candidate on every task at once.
    """

    components: tuple[tuple[str, str], ...]
    parent: int | None
    validation_scores: tuple[float, ...] | None = None
    minibatch_scores: tuple[float, ...] = ()
    component_cursor: int = 0

    def __post_init__(self) -> None:
        paths = [path for path, _ in self.components]
        require(len(paths) > 0, "a GEPA candidate needs at least one component to rewrite")
        require(paths == sorted(set(paths)), "candidate components must be sorted by path and name each path once")
        require(
            all(isinstance(path, str) and isinstance(text, str) for path, text in self.components),
            "candidate components pair a path string with a text string",
        )
        require(self.parent is None or self.parent >= 0, f"parent must be a candidate index, got {self.parent}")
        require(
            0 <= self.component_cursor < len(paths),
            f"component_cursor must index one of {len(paths)} components, got {self.component_cursor}",
        )
        if self.validation_scores is not None:
            _require_scores(self.validation_scores, "validation_scores")
        _require_scores(self.minibatch_scores, "minibatch_scores")

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(path for path, _ in self.components)


@dataclass(frozen=True)
class Archive:
    """Every candidate GEPA has proposed, the one being served, the budget spent, and the parent RNG.

    Each operation returns a new archive and leaves this one unchanged, so a
    state already handed to the campaign cannot shift underneath it.
    ``task_ids`` is the gate's task order; validation scores are stored in it.

    The last five fields are the training-minibatch sampler's state:
    ``iteration`` counts the minibatches drawn (upstream's ``state.i + 1``),
    ``train_ids`` are the training task ids the current epoch was shuffled
    from, ``train_order`` is that epoch's shuffled and padded order,
    ``train_epoch`` is its epoch number (``-1`` before the first draw), and
    ``train_minibatch_size`` is the size ``train_order`` was padded for
    (``0`` before the first draw).
    """

    task_ids: tuple[str, ...]
    candidates: tuple[ArchiveCandidate, ...]
    rng_state: RngState
    served: int = 0
    metric_calls: int = 0
    iteration: int = 0
    train_ids: tuple[str, ...] = ()
    train_order: tuple[str, ...] = ()
    train_epoch: int = -1
    train_minibatch_size: int = 0

    def __post_init__(self) -> None:
        require(len(self.task_ids) > 0, "a GEPA archive needs at least one validation task")
        require(len(set(self.task_ids)) == len(self.task_ids), "archive task ids must be unique")
        require(len(self.candidates) > 0, "a GEPA archive starts from a seed candidate")
        require(self.candidates[0].parent is None, "the seed candidate has no parent")
        for index, candidate in enumerate(self.candidates[1:], start=1):
            require(
                candidate.parent is not None and candidate.parent < index,
                f"candidate {index} must name an earlier candidate as its parent, got {candidate.parent}",
            )
            require(
                candidate.paths == self.candidates[0].paths,
                f"candidate {index} must rewrite the seed's components {list(self.candidates[0].paths)}",
            )
            if candidate.validation_scores is not None:
                require(
                    len(candidate.validation_scores) == len(self.task_ids),
                    f"candidate {index} has {len(candidate.validation_scores)} validation scores"
                    f" for {len(self.task_ids)} tasks",
                )
        seed_scores = self.candidates[0].validation_scores
        require(
            seed_scores is None or len(seed_scores) == len(self.task_ids),
            f"the seed has {0 if seed_scores is None else len(seed_scores)} validation scores"
            f" for {len(self.task_ids)} tasks",
        )
        require(0 <= self.served < len(self.candidates), f"served must index a candidate, got {self.served}")
        require(self.metric_calls >= 0, f"metric_calls must be >= 0, got {self.metric_calls}")
        require(self.iteration >= 0, f"iteration must be >= 0, got {self.iteration}")
        require(self.train_epoch >= -1, f"train_epoch must be >= -1, got {self.train_epoch}")
        require(len(set(self.train_ids)) == len(self.train_ids), "train task ids must be unique")
        require(
            bool(self.train_ids) == bool(self.train_order) and set(self.train_order) <= set(self.train_ids),
            "train_order must be a padded shuffle of train_ids, or both empty",
        )
        require(
            (
                self.train_minibatch_size == 0
                if not self.train_order
                else self.train_minibatch_size >= 1 and len(self.train_order) % self.train_minibatch_size == 0
            ),
            f"train_order holds {len(self.train_order)} ids, which is not padded for minibatches of"
            f" {self.train_minibatch_size}",
        )
        # Rebuilding the generator is the only complete check that the state is one it accepts.
        _rng_from_state(self.rng_state)

    @classmethod
    def seed(cls, components: Mapping[str, str], task_ids: Sequence[str], rng_seed: int) -> Archive:
        """An archive holding only the seed harness, served, unscored, with a freshly seeded RNG."""
        return cls(
            task_ids=tuple(task_ids),
            candidates=(ArchiveCandidate(components=tuple(sorted(components.items())), parent=None),),
            rng_state=_rng_state_of(random.Random(rng_seed)),
        )

    def add(
        self, components: Mapping[str, str], parent: int, minibatch_scores: Sequence[float] = ()
    ) -> tuple[Archive, int]:
        """Append a child of ``parent``. It starts at its parent's component cursor, as upstream GEPA does."""
        require(0 <= parent < len(self.candidates), f"parent must index a candidate, got {parent}")
        _require_scores(minibatch_scores, "minibatch_scores")
        child = ArchiveCandidate(
            components=tuple(sorted(components.items())),
            parent=parent,
            minibatch_scores=tuple(float(s) for s in minibatch_scores),
            component_cursor=self.candidates[parent].component_cursor,
        )
        return replace(self, candidates=(*self.candidates, child)), len(self.candidates)

    def validate(self, index: int, task_ids: Sequence[str], scores: Sequence[float]) -> Archive:
        """Record the gate's per-task scores for one candidate. A candidate is validated once."""
        require(0 <= index < len(self.candidates), f"index must name a candidate, got {index}")
        require(
            tuple(task_ids) == self.task_ids,
            f"validation tasks {list(task_ids)} are not the archive's tasks {list(self.task_ids)}",
        )
        require(len(scores) == len(self.task_ids), f"got {len(scores)} scores for {len(self.task_ids)} tasks")
        require(self.candidates[index].validation_scores is None, f"candidate {index} is already validated")
        _require_scores(scores, "validation_scores")
        validated = replace(self.candidates[index], validation_scores=tuple(float(s) for s in scores))
        return self._with_candidate(index, validated)

    def fronts(self) -> dict[str, set[int]]:
        """Each task's Pareto front: the validated candidates that share the best score on it.

        GEPA's rule, applied in candidate order: a strictly better score
        replaces the front, an equal score joins it. Indices are inserted in
        ascending order, which is also the order upstream builds its sets in.
        That matters for parity: a set's iteration order can depend on
        insertion order, and upstream's parent draw depends on iteration order.
        """
        fronts: dict[str, set[int]] = {}
        for position, task_id in enumerate(self.task_ids):
            best = float("-inf")
            members: list[int] = []
            for index, candidate in enumerate(self.candidates):
                if candidate.validation_scores is None:
                    continue
                score = candidate.validation_scores[position]
                if score > best:
                    best, members = score, [index]
                elif score == best:
                    members.append(index)
            if members:
                fronts[task_id] = set(members)
        return fronts

    def mean_scores(self) -> list[float]:
        """Each candidate's mean validation score, indexed by candidate, as upstream expects.

        An unvalidated candidate gets ``-inf``, which is also what upstream
        assigns a program with no scores. Upstream only reads the scores of
        candidates on some front, so this value is never compared.
        """
        # sum/len rather than statistics.fmean, to match upstream's arithmetic exactly:
        # the mean orders candidates when upstream removes dominated ones.
        return [
            float("-inf") if c.validation_scores is None else sum(c.validation_scores) / len(c.validation_scores)
            for c in self.candidates
        ]

    def sample_parent(self) -> tuple[Archive, int]:
        """Pick the next parent from the Pareto fronts, and return the archive with the RNG advanced.

        Before any candidate is validated there are no fronts; upstream would
        fail an assertion, so the served candidate is returned instead and the
        RNG is left untouched.
        """
        fronts = self.fronts()
        if not fronts:
            return self, self.served
        try:
            from gepa.gepa_utils import select_program_candidate_from_pareto_front
        except ImportError as error:
            raise ImportError("GEPA parent sampling needs the 'gepa' extra: pip install 'llm-honing[gepa]'") from error
        rng = _rng_from_state(self.rng_state)
        parent = select_program_candidate_from_pareto_front(fronts, self.mean_scores(), rng)
        return replace(self, rng_state=_rng_state_of(rng)), parent

    def next_component(self, index: int) -> tuple[Archive, str]:
        """The component to rewrite next for this candidate, round-robin over its paths in sorted order.

        Upstream's ``RoundRobinReflectionComponentSelector`` also cycles
        round-robin, but reads a full ``GEPAState``, which Honing does not
        build. The order differs on purpose: upstream follows the seed dict's
        insertion order, and this follows sorted path order, which does not
        depend on dict order surviving a save and load through JSON.
        """
        require(0 <= index < len(self.candidates), f"index must name a candidate, got {index}")
        candidate = self.candidates[index]
        paths = candidate.paths
        advanced = replace(candidate, component_cursor=(candidate.component_cursor + 1) % len(paths))
        return self._with_candidate(index, advanced), paths[candidate.component_cursor]

    def next_minibatch(self, train_ids: Sequence[str], size: int) -> tuple[Archive, tuple[str, ...]]:
        """The next training minibatch, as upstream's ``EpochShuffledBatchSampler`` draws it.

        Each epoch shuffles the training ids with the archive's RNG and pads the
        order to a multiple of ``size`` with the least used ids; minibatch ``i``
        is slice ``i`` of that order. Two differences from upstream, on purpose:
        upstream reshuffles when the training set's size changes, and this
        reshuffles whenever the ids change, since a move can pass other tasks of
        the same size; and this also reshuffles when ``size`` changes, since the
        order is padded for one size, where upstream fixes the size for a run.
        """
        train_ids = tuple(train_ids)
        require(size >= 1, f"minibatch size must be >= 1, got {size}")
        require(len(train_ids) > 0, "a minibatch needs at least one training task")
        require(len(set(train_ids)) == len(train_ids), "training task ids must be unique")
        # Upstream: base_idx = state.i * minibatch_size, with the epoch measured
        # against the previous epoch's order (batch_sampler.py:86-92).
        base = self.iteration * size
        epoch = 0 if self.train_epoch == -1 else base // max(len(self.train_order), 1)
        archive = self
        if (
            not self.train_order
            or train_ids != self.train_ids
            or size != self.train_minibatch_size
            or epoch > self.train_epoch
        ):
            rng = _rng_from_state(self.rng_state)
            order = list(train_ids)
            rng.shuffle(order)
            uses = Counter(order)
            for _ in range((size - len(order) % size) % size):
                least_used = uses.most_common()[::-1][0][0]
                order.append(least_used)
                uses[least_used] += 1
            archive = replace(
                self,
                rng_state=_rng_state_of(rng),
                train_ids=train_ids,
                train_order=tuple(order),
                train_epoch=epoch,
                train_minibatch_size=size,
            )
        start = base % len(archive.train_order)
        batch = archive.train_order[start : start + size]
        return replace(archive, iteration=self.iteration + 1), batch

    def charge(self, metric_calls: int) -> Archive:
        """Add episodes spent to the budget."""
        require(metric_calls >= 0, f"metric_calls to charge must be >= 0, got {metric_calls}")
        return replace(self, metric_calls=self.metric_calls + metric_calls)

    def with_served(self, index: int) -> Archive:
        """Point the served candidate at ``index``."""
        require(0 <= index < len(self.candidates), f"index must name a candidate, got {index}")
        return replace(self, served=index)

    def to_state(self) -> dict[str, object]:
        """The archive as JSON-native method state. The fronts are left out; they are recomputed."""
        version, words, gauss = self.rng_state
        return {
            METHOD_STATE_VERSION_KEY: ARCHIVE_SCHEMA_VERSION,
            "task_ids": list(self.task_ids),
            "candidates": [
                {
                    "components": dict(candidate.components),
                    "parent": candidate.parent,
                    "validation_scores": (
                        None if candidate.validation_scores is None else list(candidate.validation_scores)
                    ),
                    "minibatch_scores": list(candidate.minibatch_scores),
                    "component_cursor": candidate.component_cursor,
                }
                for candidate in self.candidates
            ],
            "served": self.served,
            "metric_calls": self.metric_calls,
            "rng_state": [version, list(words), gauss],
            "iteration": self.iteration,
            "train_ids": list(self.train_ids),
            "train_order": list(self.train_order),
            "train_epoch": self.train_epoch,
            "train_minibatch_size": self.train_minibatch_size,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> Archive:
        """Rebuild an archive from :meth:`to_state` output, refusing any other schema version."""
        version = method_state_version(state)
        require(
            version == ARCHIVE_SCHEMA_VERSION,
            f"GEPA archive state has schema_version {version}, this code reads {ARCHIVE_SCHEMA_VERSION}",
        )
        raw_candidates = _list(state.get("candidates"), "candidates")
        rng = _list(state.get("rng_state"), "rng_state")
        require(len(rng) == 3, f"rng_state must be [version, words, gauss], got {len(rng)} items")
        gauss = rng[2]
        require(
            gauss is None or (isinstance(gauss, float) and not isinstance(gauss, bool)),
            f"rng_state gauss must be a float or null, got {gauss!r}",
        )
        return cls(
            task_ids=tuple(_str(item, "task_ids[]") for item in _list(state.get("task_ids"), "task_ids")),
            candidates=tuple(_candidate_from_state(item, index) for index, item in enumerate(raw_candidates)),
            rng_state=(
                _int(rng[0], "rng_state version"),
                tuple(_int(word, "rng_state word") for word in _list(rng[1], "rng_state words")),
                gauss if isinstance(gauss, float) else None,
            ),
            served=_int(state.get("served"), "served"),
            metric_calls=_int(state.get("metric_calls"), "metric_calls"),
            iteration=_int(state.get("iteration"), "iteration"),
            train_ids=tuple(_str(item, "train_ids[]") for item in _list(state.get("train_ids"), "train_ids")),
            train_order=tuple(_str(item, "train_order[]") for item in _list(state.get("train_order"), "train_order")),
            train_epoch=_int(state.get("train_epoch"), "train_epoch"),
            train_minibatch_size=_int(state.get("train_minibatch_size"), "train_minibatch_size"),
        )

    def _with_candidate(self, index: int, candidate: ArchiveCandidate) -> Archive:
        candidates = list(self.candidates)
        candidates[index] = candidate
        return replace(self, candidates=tuple(candidates))


def _rng_state_of(rng: random.Random) -> RngState:
    version, words, gauss = rng.getstate()
    return version, tuple(words), gauss


def _rng_from_state(state: RngState) -> random.Random:
    version, words, _ = state
    require(version == _RNG_VERSION, f"rng_state version must be {_RNG_VERSION}, got {version}")
    require(len(words) == _RNG_WORDS, f"rng_state must hold {_RNG_WORDS} words, got {len(words)}")
    rng = random.Random()
    try:
        rng.setstate(state)
    except (TypeError, ValueError, OverflowError) as error:
        raise ContractError(f"rng_state is not a valid generator state: {error}") from error
    return rng


def _candidate_from_state(raw: object, index: int) -> ArchiveCandidate:
    fields = _dict(raw, f"candidates[{index}]")
    components = _dict(fields.get("components"), f"candidates[{index}].components")
    parent = fields.get("parent")
    scores = fields.get("validation_scores")
    return ArchiveCandidate(
        components=tuple(
            sorted((_str(path, "component path"), _str(text, "component text")) for path, text in components.items())
        ),
        parent=None if parent is None else _int(parent, f"candidates[{index}].parent"),
        validation_scores=(
            None
            if scores is None
            else tuple(_float(s, "validation score") for s in _list(scores, f"candidates[{index}].validation_scores"))
        ),
        minibatch_scores=tuple(
            _float(s, "minibatch score")
            for s in _list(fields.get("minibatch_scores"), f"candidates[{index}].minibatch_scores")
        ),
        component_cursor=_int(fields.get("component_cursor"), f"candidates[{index}].component_cursor"),
    )


def _dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object, got {type(value).__name__}")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be a list, got {type(value).__name__}")
    return value


def _int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer, got {value!r}")
    return value


def _float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a number, got {value!r}")
    return float(value)


def _str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{name} must be a string, got {value!r}")
    return value
