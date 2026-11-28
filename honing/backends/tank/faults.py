"""Tank War faults with known causes, each built from one clean release (#171).

A fault is the clean release with one part changed, so whatever it does to the
score has a known cause:

- ``harness``: one ``tank/controller.json`` or ``tank/questions.json`` that
  ``integrations/tank-war/scripts/build-harness-faults.py``, the Exp 0 v0
  builder (#186), writes from the clean harness;
- ``weights``: a checkpoint a Honing trainer makes from the clean weights (a
  direction edit, or sft on the teacher's batch with one question relabeled),
  or the checkpoint the clean release's fire edit was made from;
- ``controller``: a file from ``scripts/build-controller-faults.py`` that
  ``scripts/match.mjs`` reads with ``--controller-fault-file``, a test-only
  switch that breaks how the agent's selected action is executed;
- ``chance``: the clean release itself, whose losses have no injected cause.

:func:`build_library` writes ``library.json``. For the clean release and for
every fault it records the SHA-256 of each harness file and of the controller
switch, and the checksum of the weights; for every fault also its id, cause,
the questions it affects and the symptom it should show. A build from the same
clean release and teacher batch, with deterministic trainers, gives the same
checksums (:func:`checksums`). :func:`gate_library` plays each fault against
the clean release on fresh validation seeds, and keeps it when #166's paired
interval selector would publish the clean release over it.

Two harness faults #171 lists have nothing to build them from. Planned
steering needs the full context, so no setting strips one field, and the shot
guard is already off in the clean harness, which is the only setting planned
steering accepts.

Run: ``python scripts/e2e/tank_faults.py --config configs/e2e/tank_faults_dry_run.json``
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import fmean
from types import ModuleType
from typing import Any, Literal

from honing.backends.tank.e2e import DEFAULT_INTEGRATION, DEFAULT_SUITE, E2EContext
from honing.backends.tank.gate import refuse_pool_overlap
from honing.backends.tank.harness import AGENT, CONTROLLER_PATH, QUESTIONS_PATH, seed_harness
from honing.backends.tank.runner import TankEpisodeRunner
from honing.backends.tank.scenarios import SEED_POOLS, TankSuite, load_suite, pool_tasks
from honing.core import (
    ContractError,
    DataContract,
    DataProvider,
    DirectionEditSpec,
    EpisodeRecord,
    EvaluationResult,
    HarnessFile,
    HarnessSpec,
    ModelSnapshot,
    ModelSource,
    PolicyEndpoints,
    PolicyRef,
    TaskGrader,
    Trainer,
    TrainingBatch,
    TrainJobSpec,
    UpdateCandidate,
    require,
    run_graded,
)
from honing.e2e.components import build_component, build_named_trainers_and_methods, missing_requirements
from honing.release import PairedIntervalSelector

Cause = Literal["harness", "weights", "controller", "chance"]
CLEAN = "clean"
FORMAT, VERSION = "tank-fault-library", 1
#: The direction edit that makes a clean release from the pinned NanoJev (#147): toward fire on the weapon question.
FIRE_EDIT = DirectionEditSpec("weapon", "fire", 1.0)
FLIPPED_WEAPON = (("fire", "hold"), ("hold", "fire"))
MIRRORED_TURNS = tuple(
    (f"{drive}_{turn}", f"{drive}_{other}")
    for drive in ("forward", "reverse", "stop")
    for turn, other in (("left", "right"), ("right", "left"))
)
STOPPED_MOTION = tuple(
    (f"{drive}_{turn}", "stop_none")
    for drive in ("forward", "reverse", "stop")
    for turn in ("left", "none", "right")
    if (drive, turn) != ("stop", "none")
)


@dataclass(frozen=True)
class Fault:
    """One catalog entry: the part it changes, the questions that change touches, and what it should do.

    A harness fault names a kind of ``build-harness-faults.py`` and a
    controller fault a kind of ``build-controller-faults.py``. A weights fault
    is a direction ``edit`` of the clean weights, sft on the teacher's batch
    with one question's labels mapped through ``relabel``, or the ``earlier``
    checkpoint. The chance control names nothing.
    """

    fault_id: str
    cause: Cause
    questions: tuple[str, ...]
    symptom: str
    harness_kind: str | None = None
    controller_kind: str | None = None
    edit: DirectionEditSpec | None = None
    relabel: tuple[str, tuple[tuple[str, str], ...]] | None = None
    earlier: bool = False


FAULTS: tuple[Fault, ...] = (
    Fault(
        "harness-disable-assistance",
        "harness",
        ("motion", "weapon"),
        "Dodge assistance is off and its fields leave the state: nothing steers the tank out of a shell's path,"
        " more deaths to shells.",
        harness_kind="disable_assistance",
    ),
    Fault(
        "harness-swap-fire-hold",
        "harness",
        ("weapon",),
        "The fire option reads as hold and hold as fire: the model holds safe shots, fewer shots and hits.",
        harness_kind="swap_fire_hold_descriptions",
    ),
    Fault(
        "harness-reverse-turn-words",
        "harness",
        ("motion",),
        "The motion question says to turn left for a positive heading error: the tank turns away from its aim.",
        harness_kind="reverse_turn_words",
    ),
    Fault(
        "harness-extreme-standoff",
        "harness",
        ("motion",),
        "The planner's strategy keeps 1800 to 2200 px from the enemy: fewer shots, more timeouts.",
        harness_kind="extreme_standoff",
    ),
    Fault(
        "harness-penalize-safe-shots",
        "harness",
        ("motion",),
        "The planner's strategy scores plans with a verified safe shot lower: fewer safe shots, a lower score.",
        harness_kind="penalize_safe_shots",
    ),
    Fault(
        "harness-reward-shell-danger",
        "harness",
        ("motion",),
        "The planner's strategy prefers paths close to observed shells: more deaths to shells.",
        harness_kind="reward_shell_danger",
    ),
    Fault(
        "harness-point-blank-pursuit",
        "harness",
        ("motion",),
        "The planner's strategy closes to under 30 px of the enemy: standoff shots are skipped, more return fire.",
        harness_kind="point_blank_pursuit",
    ),
    Fault(
        "weights-earlier-checkpoint",
        "weights",
        ("weapon",),
        "The weights the clean release's fire edit was made from: fewer safe shots, a lower score.",
        earlier=True,
    ),
    Fault(
        "weights-reverse-fire-edit",
        "weights",
        ("weapon",),
        "Three units of the fire direction taken off the head: the model holds nearly every shot.",
        edit=DirectionEditSpec("weapon", "fire", -3.0),
    ),
    Fault(
        "weights-sft-flipped-weapon",
        "weights",
        ("weapon",),
        "Trained to answer the weapon question against the teacher: holds the shots the teacher takes.",
        relabel=("weapon", FLIPPED_WEAPON),
    ),
    Fault(
        "weights-sft-mirrored-turns",
        "weights",
        ("motion",),
        "Trained on the teacher's motions with left and right swapped: turns away from its aim.",
        relabel=("motion", MIRRORED_TURNS),
    ),
    Fault(
        "weights-stop-edit",
        "weights",
        ("motion",),
        "Three units of the stop_none direction added to the head: the tank stands still at most decisions.",
        edit=DirectionEditSpec("motion", "stop_none", 3.0),
    ),
    Fault(
        "weights-sft-stopped-motion",
        "weights",
        ("motion",),
        "Trained on the teacher's rounds with every motion relabeled stop_none: the tank stops instead of moving.",
        relabel=("motion", STOPPED_MOTION),
    ),
    Fault(
        "controller-swap-selected-turn",
        "controller",
        ("motion",),
        "The tank executes the opposite turn to the one selected: the log shows turns it did not make.",
        controller_kind="swap_selected_turn",
    ),
    Fault(
        "controller-drop-fire",
        "controller",
        ("weapon",),
        "A selected fire is dropped after selection: the log shows fire, but no shell leaves.",
        controller_kind="drop_fire_after_selection",
    ),
    Fault(
        "controller-wrong-legal-projection",
        "controller",
        ("motion", "weapon"),
        "The legality projection swaps the selected motion for another admissible one and cancels its shot.",
        controller_kind="wrong_legal_projection",
    ),
    Fault("chance-clean", "chance", (), "None: the clean release, whose losses have no injected cause."),
)


@dataclass(frozen=True)
class Release:
    """What a fault's rounds are played with: a harness, weights, and the controller switch if there is one."""

    harness: HarnessSpec
    weights: ModelSnapshot
    controller_fault: Path | None = None

    def policy(self) -> PolicyRef:
        return PolicyRef(self.weights.checkpoint, self.weights.architecture, "master")


@dataclass(frozen=True)
class Library:
    """A built library: ``library.json``, and the clean release and each fault's release, by fault id."""

    path: Path
    manifest: Mapping[str, Any]
    clean: Release
    releases: Mapping[str, Release]


def relabel(batch: TrainingBatch, question: str, mapping: Mapping[str, str]) -> TrainingBatch:
    """``batch`` with ``question``'s label mapped through ``mapping`` on every decision that asks it.

    A batch in which no label changes is refused: sft on it would train on the teacher's own answers.
    """
    decisions, changed = [], 0
    for decision in batch.decisions:
        chosen = tuple(
            (key, mapping.get(option, option) if key == question else option) for key, option in decision.chosen
        )
        changed += chosen != decision.chosen
        decisions.append(replace(decision, chosen=chosen))
    if not changed:
        raise ContractError(
            f"relabeling {question} through {dict(mapping)} changes no label in batch {batch.batch_id}"
        )
    return TrainingBatch(f"{batch.batch_id}-{question}-relabeled", batch.contract, decisions=tuple(decisions))


def _script(integration_dir: Path, name: str) -> ModuleType:
    """A builder script from the integration's ``scripts/``, imported by path: its name is not a module name."""
    path = integration_dir / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_harness(directory: Path) -> HarnessSpec:
    """The harness in a directory the runner or ``build-harness-faults.py`` wrote: ``index.json`` and its files."""
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    files = tuple(
        (entry["path"], HarnessFile(entry["kind"], (directory / entry["path"]).read_text(encoding="utf-8")))
        for entry in index
    )
    return HarnessSpec(AGENT, files)


def _artifacts(release: Release) -> dict[str, str | None]:
    """The checksum of every part of a release: each harness file, the weights, and the controller switch."""
    artifacts: dict[str, str | None] = {path: _sha256(file.content) for path, file in release.harness.files}
    artifacts["weights"] = release.weights.checkpoint.checksum
    switch = release.controller_fault
    artifacts["controller_fault"] = _sha256(switch.read_text(encoding="utf-8")) if switch else None
    return artifacts


def build_library(
    out: Path,
    *,
    integration_dir: Path,
    clean: ModelSnapshot,
    earlier: ModelSnapshot | None,
    batch: TrainingBatch,
    trainer: Trainer,
    edit_trainer: Trainer,
    sft_steps: int = 1,
    faults: Sequence[Fault] = FAULTS,
) -> Library:
    """Build every fault in ``faults`` from the clean release and write ``out/library.json``.

    The clean release is ``clean`` with the integration's seed harness and no
    controller switch. ``earlier`` is the checkpoint ``clean`` was edited from,
    or ``None``, and then the earlier-checkpoint fault is skipped and listed
    under ``skipped``. ``trainer`` runs sft and ``edit_trainer`` the direction
    edits, each from ``clean`` on ``batch``. A fault that differs from the
    clean release in anything but its own cause's part is refused.
    """
    require(sft_steps > 0, f"sft_steps must be positive, got {sft_steps}")
    out.mkdir(parents=True, exist_ok=False)
    _script(integration_dir, "build-harness-faults.py").build(out / "harness-candidates")
    _script(integration_dir, "build-controller-faults.py").build(out / "controller-candidates")
    harness = seed_harness(integration_dir)
    TankEpisodeRunner.write_harness(harness, out / "clean-harness")
    clean_release = Release(harness, clean)
    clean_artifacts = _artifacts(clean_release)
    releases: dict[str, Release] = {}
    entries, skipped = [], []
    for fault in faults:
        recipe: dict[str, object]
        if fault.harness_kind is not None:
            release = Release(_read_harness(out / "harness-candidates" / fault.harness_kind), clean)
            recipe = {"builder": "scripts/build-harness-faults.py", "kind": fault.harness_kind}
        elif fault.controller_kind is not None:
            release = Release(harness, clean, out / "controller-candidates" / f"{fault.controller_kind}.json")
            recipe = {"builder": "scripts/build-controller-faults.py", "kind": fault.controller_kind}
        elif fault.edit is not None:
            job = TrainJobSpec("direction_edit", contract="typed_decisions", direction_edit=fault.edit)
            release = Release(harness, edit_trainer.train(job, clean, batch).snapshot)
            recipe = {"algorithm": "direction_edit", "direction_edit": asdict(fault.edit), "batch": batch.batch_id}
        elif fault.relabel is not None:
            question, mapping = fault.relabel
            relabeled = relabel(batch, question, dict(mapping))
            job = TrainJobSpec("sft", steps=sft_steps, contract="typed_decisions")
            release = Release(harness, trainer.train(job, clean, relabeled).snapshot)
            recipe = {
                "algorithm": "sft",
                "steps": sft_steps,
                "relabel": {question: dict(mapping)},
                "batch": batch.batch_id,
            }
        elif fault.earlier:
            if earlier is None:
                skipped.append(
                    {"id": fault.fault_id, "reason": "the clean release was not edited from an earlier one"}
                )
                continue
            release = Release(harness, earlier)
            recipe = {"weights": "the checkpoint the clean release's fire edit was made from"}
        else:
            release, recipe = clean_release, {}
        artifacts = _artifacts(release)
        changed = sorted(key for key in artifacts if artifacts[key] != clean_artifacts.get(key))
        expected = {
            "harness": ([CONTROLLER_PATH], [QUESTIONS_PATH]),
            "weights": (["weights"],),
            "controller": (["controller_fault"],),
            "chance": ([],),
        }[fault.cause]
        if changed not in expected:
            raise ContractError(f"{fault.fault_id} is a {fault.cause} fault, but it changes {changed or 'nothing'}")
        releases[fault.fault_id] = release
        entries.append(
            {
                "id": fault.fault_id,
                "cause": fault.cause,
                "affected_questions": list(fault.questions),
                "expected_symptom": fault.symptom,
                "artifacts": artifacts,
                "recipe": recipe,
                "harness_dir": f"harness-candidates/{fault.harness_kind}" if fault.harness_kind else "clean-harness",
                "controller_fault_file": release.controller_fault
                and release.controller_fault.relative_to(out).as_posix(),
                "weights_location": release.weights.checkpoint.location,
            }
        )
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        CLEAN: {
            "artifacts": clean_artifacts,
            "harness_dir": "clean-harness",
            "weights_location": clean.checkpoint.location,
        },
        "faults": entries,
        "skipped": skipped,
    }
    path = out / "library.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return Library(path, manifest, clean_release, releases)


def checksums(manifest: Mapping[str, Any]) -> dict[str, dict[str, str | None]]:
    """Every release's artifact checksums by id, the clean release under ``"clean"``: what two builds must share."""
    return {CLEAN: manifest[CLEAN]["artifacts"], **{entry["id"]: entry["artifacts"] for entry in manifest["faults"]}}


# ---------------------------------------------------------------- the gate


def _evaluate(
    runners: tuple[TankEpisodeRunner, TankEpisodeRunner],
    grader: TaskGrader,
    releases: tuple[Release, Release],
    seeds: range,
) -> EvaluationResult:
    """The first release as the candidate and the second as the incumbent, each through its runner, on ``seeds``.

    A seed scores the mean of its two seats, as in :class:`~honing.backends.tank.gate.FreshSeedEvaluator`.
    """
    tasks = pool_tasks("val", seeds)

    async def play(runner: TankEpisodeRunner, release: Release) -> list[EpisodeRecord]:
        return await asyncio.gather(
            *(run_graded(runner, grader, release.harness, task, release.policy()) for task in tasks)
        )

    async def both() -> list[list[EpisodeRecord]]:
        return await asyncio.gather(
            *(play(runner, release) for runner, release in zip(runners, releases, strict=True))
        )

    candidate, incumbent = asyncio.run(both())

    def per_seed(records: Sequence[EpisodeRecord]) -> tuple[float, ...]:
        # pool_tasks lists both seats of a seed together.
        return tuple(fmean(record.score for record in records[i : i + 2]) for i in range(0, len(records), 2))

    return EvaluationResult(
        task_ids=tuple(f"tank-val-s{seed}" for seed in seeds),
        candidate_scores=per_seed(candidate),
        incumbent_scores=per_seed(incumbent),
        evaluated_variant="master",
        records=(*candidate, *incumbent),
    )


def gate_library(
    library: Library,
    runner: TankEpisodeRunner,
    grader: TaskGrader,
    suite: TankSuite,
    *,
    rounds: int = 64,
    confidence: float = 0.95,
    first_seed: int | None = None,
) -> list[dict[str, Any]]:
    """Play every fault against the clean release, and keep the faults the clean release beats.

    Fault ``i`` plays the ``rounds // 2`` val pool seeds after fault
    ``i - 1``'s, from both seats, and the clean release plays the same rounds.
    A seed scores the mean of its two seats, as in
    :class:`~honing.backends.tank.gate.FreshSeedEvaluator`. That evaluator plays
    both sides through one runner, and a controller fault needs two: it is the
    clean release played through ``runner.with_controller_fault``.
    :class:`~honing.release.PairedIntervalSelector` (#166) decides with the
    clean release as the candidate and the fault as the incumbent. It selects
    the clean release when the interval on clean minus fault lies above zero,
    and that keeps the fault. Each row reports the interval the other way
    round, on fault minus clean.
    """
    require(
        rounds > 0 and rounds % 2 == 0, f"each seed is played from both seats, so rounds must be even, got {rounds}"
    )
    refuse_pool_overlap(suite)
    pool, seeds_per_fault = SEED_POOLS["val"], rounds // 2
    first = pool.start if first_seed is None else first_seed
    require(first in pool, f"first_seed must be a seed of the val pool, {pool.start} to {pool.stop - 1}")
    selector = PairedIntervalSelector(confidence=confidence)
    clean = library.clean
    candidate = UpdateCandidate("init", None, clean.weights, harness=clean.harness)
    rows = []
    for index, entry in enumerate(library.manifest["faults"]):
        release = library.releases[entry["id"]]
        seeds = range(first + index * seeds_per_fault, first + (index + 1) * seeds_per_fault)
        require(seeds.stop <= pool.stop, "the val seed pool is used up")
        switched = runner.with_controller_fault(release.controller_fault) if release.controller_fault else runner
        evaluation = _evaluate((runner, switched), grader, (clean, release), seeds)
        decision = selector.decide(candidate, evaluation)
        rows.append(
            {
                "id": entry["id"],
                "cause": entry["cause"],
                "seeds": [seeds.start, seeds.stop - 1],
                "rounds": rounds,
                "clean_mean": evaluation.candidate_mean,
                "fault_mean": evaluation.incumbent_mean,
                "drop_low": -decision.metrics["difference_high"],
                "drop_high": -decision.metrics["difference_low"],
                "kept": decision.selected,
                "reason": decision.reason,
            }
        )
    return rows


def table(rows: Sequence[Mapping[str, Any]]) -> str:
    """The gate's rows as a Markdown table: each fault's mean score against the clean release's, with the interval."""
    lines = [
        "| Fault | Cause | Clean | Fault | Fault minus clean, interval | Kept |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        interval = f"[{row['drop_low']:+.3f}, {row['drop_high']:+.3f}]"
        mean = row["fault_mean"] - row["clean_mean"]
        lines.append(
            f"| {row['id']} | {row['cause']} | {row['clean_mean']:.3f} | {row['fault_mean']:.3f}"
            f" | {mean:+.3f} {interval} | {'yes' if row['kept'] else 'no'} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- config


SLOTS: tuple[tuple[str, type], ...] = (
    ("model_source", ModelSource),
    ("endpoints", PolicyEndpoints),
    ("runner", TankEpisodeRunner),
    ("grader", TaskGrader),
    ("trainer", Trainer),
    ("data", DataProvider),
)
SETTINGS = frozenset(
    {"description", "integration_dir", "suite", "work_dir", "teacher_tasks", "clean_fire_edit", "sft_steps"}
    | {"gate", "requires", "components"}
)
GATE_SETTINGS = frozenset({"rounds", "confidence", "first_seed"})


def build(config: Mapping[str, Any], root: Path, work_dir: Path | None = None) -> tuple[Library, E2EContext]:
    """Build the components the config names, the clean release, and the library under ``work_dir/library``.

    The clean release is the ``model_source`` checkpoint, edited toward fire
    on the weapon question by the ``direction_edit`` trainer when the config
    sets ``clean_fire_edit`` (the strength). The teacher batch is one batch
    from ``data`` (``teacher_tasks`` train tasks), and every trainer reads it.
    A setting the config does not know, here or in ``gate``, is refused
    before anything is built.
    """
    unknown = (set(config) - SETTINGS) | (set(config.get("gate") or {}) - GATE_SETTINGS)
    if unknown:
        raise ContractError(f"unknown settings {sorted(unknown)}")
    missing = missing_requirements(config.get("requires", {}))
    if missing:
        raise ContractError("this config needs " + ", ".join(missing))
    integration_dir = (root / config.get("integration_dir", DEFAULT_INTEGRATION)).resolve()
    suite = load_suite(integration_dir / config.get("suite", DEFAULT_SUITE))
    work = (work_dir or root / config.get("work_dir", "runs/tank-faults")).resolve()
    work.mkdir(parents=True, exist_ok=False)
    context = E2EContext(
        root=root.resolve(),
        integration_dir=integration_dir,
        work_dir=work,
        suite=suite,
        validation_tasks=(),
        harness_tasks=suite.tasks("train", limit=config.get("teacher_tasks")),
    )
    components = config.get("components", {})
    unknown = set(components) - {slot for slot, _ in SLOTS} - {"trainers"}
    if unknown:
        raise ContractError(f"unknown components {sorted(unknown)}")
    for slot, protocol in SLOTS:
        if slot not in components:
            raise ContractError(f"the config names no {slot} component")
        context.components[slot] = build_component(components[slot], context, slot, protocol)
    trainers, _ = build_named_trainers_and_methods(components, context)
    if "direction_edit" not in trainers:
        raise ContractError("the config names no trainers.direction_edit component")
    batch = context.component("data").next_batch(DataContract("typed_decisions"))
    if batch is None:
        raise ContractError("the data component returned no typed_decisions batch")
    clean = context.component("model_source").materialize(work / "model")
    earlier = None
    strength = config.get("clean_fire_edit")
    if strength is not None:
        job = TrainJobSpec(
            "direction_edit", contract="typed_decisions", direction_edit=replace(FIRE_EDIT, strength=float(strength))
        )
        earlier, clean = clean, trainers["direction_edit"].train(job, clean, batch).snapshot
    library = build_library(
        work / "library",
        integration_dir=integration_dir,
        clean=clean,
        earlier=earlier,
        batch=batch,
        trainer=context.component("trainer"),
        edit_trainer=trainers["direction_edit"],
        sft_steps=int(config.get("sft_steps", 1)),
    )
    return library, context


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Tank War fault library and, with a gate, gate it.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, default=Path.cwd(), help="repository root; config paths are relative to it"
    )
    parser.add_argument("--work-dir", type=Path, help="output directory (must not exist); overrides the config")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        library, context = build(config, args.root, args.work_dir)
    except ContractError as error:
        print(f"cannot build: {error}", file=sys.stderr)
        return 2
    built = library.manifest["faults"]
    print(f"{len(built)} faults in {library.path}; skipped {[s['id'] for s in library.manifest['skipped']]}")
    gate = config.get("gate")
    if gate is not None:
        rows = gate_library(library, context.component("runner"), context.component("grader"), context.suite, **gate)
        (library.path.parent / "gate.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(table(rows))
    return 0


__all__ = [
    "FAULTS",
    "FIRE_EDIT",
    "Fault",
    "Library",
    "Release",
    "build",
    "build_library",
    "checksums",
    "gate_library",
    "main",
    "relabel",
    "table",
]
