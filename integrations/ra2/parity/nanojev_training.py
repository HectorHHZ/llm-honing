"""NanoJevTrainer on a GPU: supervised training on labelled matches, serving the result, and policy gradient.

Usage: nanojev_training.py <sft-gpu> <serve-gpu> <grpo-gpu> <work-dir> <demo-match-dir>...

Each step holds its GPU's lock, which refuses a GPU that already runs a process, so the steps use three GPUs.

1. sft: the matches (written by make_demos.py) train NanoJev for one epoch; validation loss is reported before and
   after, and NanoJevDecisionModel loads the trained checkpoint from its manifest and answers a request.
2. grpo: synthetic rollouts as in policy_gradient.py (16 scenario groups of 6 sampled choices, a reward that
   favors option B, one group whose rewards agree). The favored option's raw probability is measured on every
   state before and after.

Needs NANOJEV_SOURCE_DIR (the NanoJev checkout at the pinned commit) and the pinned model in the Hugging Face cache.
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

from honing.backends.jev.nanojev import nanojev_snapshot
from honing.backends.jev.nanojev_trainer import NanoJevTrainer, NanoJevTrainerSettings, release_path
from honing.backends.jev.trainer import laya_question
from honing.backends.ra2 import RA2DemonstrationProvider
from honing.core import DataContract, DecisionQuestion, DecisionSample, TrainingBatch, TrainJobSpec, requirements_for

ROOT = Path(__file__).resolve().parents[1]
sft_gpu, serve_gpu, grpo_gpu = sys.argv[1:4]
work, matches = Path(sys.argv[4]), [Path(p) for p in sys.argv[5:]]
source = Path(os.environ["NANOJEV_SOURCE_DIR"])
pins = json.loads((ROOT / "pins.json").read_text())
release = release_path(pins)
base = nanojev_snapshot(release)
report: dict[str, object] = {"base_checksum": base.checkpoint.checksum}

# ---------------------------------------------------------------- 1. supervised
protocol = json.loads((ROOT / "tasks/protocol.json").read_text())
trainer = NanoJevTrainer(
    NanoJevTrainerSettings(
        pins_path=ROOT / "pins.json",
        source_dir=source,
        work_dir=work / "sft",
        gpu=sft_gpu,
        deterministic=True,
    )
)
job = TrainJobSpec("sft", steps=1, contract="typed_decisions")
batch = RA2DemonstrationProvider(matches, protocol).next_batch(requirements_for(job).data_contract())
result = trainer.train(job, base, batch)
trained = Path(result.snapshot.checkpoint.location)
manifest_path = trained.parent / "checkpoint.json"
report["sft"] = {"metrics": dict(result.metrics), "checksum": result.snapshot.checkpoint.checksum}

decision = batch.decisions[0]
request = {
    "parts": {n: list(v) for n, v in decision.parts},
    "questions": {k: laya_question(q) for k, q in decision.questions},
}
SERVE = """
import json, sys
from pathlib import Path
from honing.backends.jev.nanojev import NanoJevDecisionModel, NanoJevSettings
root, source, gpu, manifest, request = sys.argv[1:6]
request = json.loads(request)
pins = Path(root) / "pins.json"
manifest = Path(manifest)
settings = NanoJevSettings(pins_path=pins, source_dir=Path(source), gpu=gpu, checkpoint_manifest=manifest)
with NanoJevDecisionModel.load(settings) as model:
    answer = model.predict(request["parts"], request["questions"])
    meta = model.metadata
    print(json.dumps({"revision": meta["revision"], "answers": answer["answers"]}))
"""
served = subprocess.run(
    [sys.executable, "-c", SERVE, str(ROOT), str(source), serve_gpu, str(manifest_path), json.dumps(request)],
    check=True,
    capture_output=True,
    text=True,
)
report["serve_trained"] = json.loads(served.stdout.strip().splitlines()[-1])

# ---------------------------------------------------------------- 2. policy gradient
OPTIONS = (("A", "Take no action for this group this turn."), ("B", "Build an ore refinery"), ("C", "Build barracks"))
FAVORED = "B"
rng = random.Random(0)
decisions = []
for group in range(16):
    state = (("essential", (f"Tick {30 * (group + 1)}. Credits {rng.randint(500, 9000)}. Power plant built.",)),)
    question = DecisionQuestion("choice", "What should be built next?", OPTIONS)
    for _ in range(6):
        choice = rng.choice("ABC")
        decisions.append(
            DecisionSample(
                parts=state,
                questions=(("build", question),),
                chosen=(("build", choice),),
                behavior_probs=(("build", (("A", 0.34), ("B", 0.33), ("C", 0.33))),),
                reward=1.0 if choice == FAVORED else 0.0,
                group=f"scenario-{group}",
                source_release="r0000",
            )
        )
decisions += [
    DecisionSample(
        parts=(("essential", ("Tick 9999. Nothing changes.",)),),
        questions=(("build", DecisionQuestion("choice", "What should be built next?", OPTIONS)),),
        chosen=(("build", c),),
        behavior_probs=(("build", (("A", 0.34), ("B", 0.33), ("C", 0.33))),),
        reward=1.0,
        group="flat",
        source_release="r0000",
    )
    for c in "ABC"
]
contract = DataContract("typed_decisions", requires_reward=True, requires_logprobs=True, requires_group=True)
rollouts = TrainingBatch("synthetic-rollouts", contract, decisions=tuple(decisions))
pg = NanoJevTrainer(
    NanoJevTrainerSettings(
        pins_path=ROOT / "pins.json",
        source_dir=source,
        work_dir=work / "grpo",
        gpu=grpo_gpu,
        backbone_lr=5e-5,
        head_lr=5e-4,
        batch_questions=8,
        deterministic=True,
    )
)
pg_result = pg.train(TrainJobSpec("grpo", steps=3, rollout="harness", contract="typed_decisions"), base, rollouts)


def favored_probabilities(checkpoint: Path) -> list[float]:
    """The favored option's probability on each group's state, from NanoJev's own predictor (temperature 1)."""
    import torch

    from honing.backends.jev.nanojev import import_nanojev

    upstream, _ = import_nanojev(source, pins)
    predictor = upstream.DecisionPredictor(str(checkpoint), max_length=2048, device_name="cuda:0", precision="bf16")
    values, seen = [], set()
    for d in decisions:
        if d.group in seen or d.group == "flat":
            continue
        seen.add(d.group)
        question = laya_question(dict(d.questions)["build"])
        state = "\n".join(line for _, lines in d.parts for line in lines)
        request = {"states": [{"id": "s", "state": state, "questions": {"build": question}}]}
        answer = predictor.predict(request, batch_questions=1)["states"][0]["answers"]["build"]
        values.append(float(answer["probabilities"][FAVORED]))
    del predictor
    torch.cuda.empty_cache()
    return values


before = favored_probabilities(release)
after = favored_probabilities(Path(pg_result.snapshot.checkpoint.location))
pg_manifest = json.loads((Path(pg_result.snapshot.checkpoint.location).parent / "checkpoint.json").read_text())
report["grpo"] = {
    "favored": FAVORED,
    "mean_probability_before": sum(before) / len(before),
    "mean_probability_after": sum(after) / len(after),
    "states_where_it_rose": f"{sum(a > b for a, b in zip(after, before, strict=True))}/{len(before)}",
    "decisions_without_signal": pg_manifest["decisionsWithoutSignal"],
    "history": pg_manifest["history"],
    "metrics": dict(pg_result.metrics),
}
print(json.dumps(report, indent=1))
