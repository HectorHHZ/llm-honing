"""Sample grouped, rewarded rollouts from a checkpoint with SGLang; run inside Slime's image.

Each prompt is a small arithmetic question that asks for the answer in
``\\boxed{}``. SGLang samples ``--samples-per-prompt`` answers per question and
returns each answer's token ids and log-probabilities, the record an agent
adapter (slime.agent's ``OpenAIAdapter``) keeps per session. The reward is 1
when the boxed answer is right and 0 otherwise. The output is a JSON list of
:class:`~honing.core.PolicySample` fields, which ``slime_grpo_step.py`` turns
into a ``TrainingBatch``; the prompt tokens carry mask 0 and log-probability 0.0.

    python3 scripts/e2e/slime_rollouts.py --model DIR --out rollouts.json --prompts 16 --samples-per-prompt 8
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def questions(count: int, seed: int) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    items = []
    for _ in range(count):
        a, b, c = rng.randint(10, 99), rng.randint(10, 99), rng.randint(2, 9)
        items.append((f"Compute ({a} + {b}) * {c}. Put only the final number in \\boxed{{}}.", (a + b) * c))
    return items


def reward(text: str, answer: int) -> float:
    found = BOXED.findall(text)
    return 1.0 if found and found[-1].strip().replace(",", "") == str(answer) else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--release", default=None)
    args = parser.parse_args()

    import sglang
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine = sglang.Engine(model_path=args.model, mem_fraction_static=0.6, random_seed=args.seed)
    items = questions(args.prompts, args.seed)
    prompts = [
        tokenizer.encode(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
            ),
            add_special_tokens=False,
        )
        for q, _ in items
    ]
    requests = [ids for ids in prompts for _ in range(args.samples_per_prompt)]
    outputs = engine.generate(
        input_ids=requests,
        sampling_params={"temperature": args.temperature, "max_new_tokens": args.max_new_tokens},
        return_logprob=True,
    )
    engine.shutdown()

    samples: list[dict[str, Any]] = []
    for index, output in enumerate(outputs):
        task = index // args.samples_per_prompt
        prompt = requests[index]
        pairs = output["meta_info"]["output_token_logprobs"]  # (logprob, token_id, text) per generated token
        response = [int(token) for _, token, _ in pairs]
        if not response:
            response, logprobs = [tokenizer.eos_token_id], [0.0]
        else:
            logprobs = [float(value) for value, _, _ in pairs]
        samples.append(
            {
                "tokens": list(prompt) + response,
                "loss_mask": [0] * len(prompt) + [1] * len(response),
                "logprobs": [0.0] * len(prompt) + logprobs,
                "reward": reward(output["text"], items[task][1]),
                "group": f"q{task}",
                "source_release": args.release,
            }
        )
    rewards = [sample["reward"] for sample in samples]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"reward_mean": statistics.fmean(rewards), "samples": samples}) + "\n")
    print(json.dumps({"samples": len(samples), "reward_mean": statistics.fmean(rewards)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
