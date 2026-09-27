"""#57 on a GPU: does LocalJevTrainer's grpo move probability toward what the reward favors?

Synthetic rollouts: 16 scenario groups, 6 decisions each, options sampled uniformly from three.
The reward favors one option ("B" or "C", the first argument). The rewarded option's probability
is measured before and after training with the raw softmax training uses, on every state.
"""

import json
import os
import random
import sys
from pathlib import Path

from honing.backends.jev.laya import laya_snapshot
from honing.backends.jev.trainer import JevTrainerSettings, LocalJevTrainer, laya_question
from honing.core import DataContract, DecisionQuestion, DecisionSample, TrainingBatch, TrainJobSpec

favored, gpu, work = sys.argv[1], sys.argv[2], Path(sys.argv[3])
PR49 = Path(os.environ["PR49_DIR"])
record_path = PR49 / ".cache/models/typed-decisions.json"
parent = Path(json.loads(record_path.read_text())["path"])
OPTIONS = (("A", "Take no action for this group this turn."), ("B", "Build an ore refinery"), ("C", "Build barracks"))
rng = random.Random(0)
decisions = []
for group in range(16):
    state = (("essential", (f"Tick {30 * (group + 1)}. Credits {rng.randint(500, 9000)}. Power plant built.",)),)
    question = DecisionQuestion("choice", "What should be built next?", OPTIONS)
    for _ in range(6):
        choice = rng.choice("ABC")
        reward = 1.0 if choice == favored else 0.0
        decisions.append(
            DecisionSample(
                parts=state,
                questions=(("build", question),),
                chosen=(("build", choice),),
                behavior_probs=(("build", (("A", 0.34), ("B", 0.33), ("C", 0.33))),),
                reward=reward,
                group=f"scenario-{group}",
                source_release="r0000",
            )
        )
# One group whose rewards all agree, which must contribute nothing.
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
batch = TrainingBatch("synthetic-rollouts", contract, decisions=tuple(decisions))
settings = JevTrainerSettings(
    laya_source=PR49 / "deps/laya",
    pins_path=PR49 / "pins.json",
    model_record=record_path,
    work_dir=work,
    gpu=gpu,
    batch_size=8,
    encoder_lr=2e-5,
    deterministic=True,
)
trainer = LocalJevTrainer(settings)
job = TrainJobSpec("grpo", steps=3, rollout="harness", contract="typed_decisions")
result = trainer.train(job, laya_snapshot(parent), batch)


def probabilities(checkpoint: Path) -> list[float]:
    """The favored option's raw softmax probability on each group's state, as training computes it."""
    torch, agent, _qtypes, build_sequence, render_options, tok, model, _, _ = trainer._setup(laya_snapshot(checkpoint))
    from honing.backends.jev.token_budget import prepare_request

    model.eval()
    formatter = type("F", (), {"tok": tok, "_to_internal": staticmethod(agent._to_internal)})
    values = []
    seen = set()
    for d in decisions:
        if d.group in seen or d.group == "flat":
            continue
        seen.add(d.group)
        q = laya_question(dict(d.questions)["build"])
        state, _, _ = prepare_request(
            formatter, {n: list(v) for n, v in d.parts}, {"build": q}, 2048, 512, build_sequence, render_options
        )
        ids, markers = build_sequence(tok, state, agent._to_internal(q), 2048, 512)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x = torch.tensor([ids], device="cuda")
            logits, _ = model(
                x,
                torch.ones_like(x),
                torch.tensor([markers], device="cuda"),
                torch.ones((1, len(markers)), dtype=torch.bool, device="cuda"),
                torch.zeros(1, dtype=torch.long, device="cuda"),
            )
        values.append(float(torch.softmax(logits.float()[0], -1)["ABC".index(favored)]))
    return values


before = probabilities(parent)
after = probabilities(Path(result.snapshot.checkpoint.location))
manifest = json.loads((Path(result.snapshot.checkpoint.location).parent / "checkpoint.json").read_text())
up = sum(a > b for a, b in zip(after, before, strict=True))
print(
    json.dumps(
        {
            "favored": favored,
            "mean_probability_before": sum(before) / len(before),
            "mean_probability_after": sum(after) / len(after),
            "states_where_it_rose": f"{up}/{len(before)}",
            "decisions_without_signal": manifest["decisionsWithoutSignal"],
            "history": manifest["history"],
            "mean_behavior_gap": manifest["meanBehaviorGap"],
        },
        indent=1,
    )
)
