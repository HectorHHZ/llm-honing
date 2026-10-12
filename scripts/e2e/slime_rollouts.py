"""Sample grouped, rewarded rollouts from a checkpoint with SGLang; run inside Slime's image.

Each prompt is a small arithmetic question that asks for the answer in
``\\boxed{}``. SGLang samples ``--samples-per-prompt`` answers per question and
returns each answer's token ids and log-probabilities, the record an agent
adapter (slime.agent's ``OpenAIAdapter``) keeps per session. The reward is 1
when the boxed answer is right and 0 otherwise. The output is a JSON list of
:class:`~honing.core.PolicySample` fields, which ``slime_train_step.py`` turns
into a ``TrainingBatch``; the prompt tokens carry mask 0 and log-probability 0.0.

With ``--teacher-hint`` every answer is also scored by the same model under
privileged context, the question followed by its correct answer, which the
sampling prompt left out. Those are the teacher log-probabilities on-policy
self-distillation trains toward (``PolicySample.teacher_logprobs``).

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


def hinted(question: str, answer: int) -> str:
    """The question with privileged context: its correct answer, which the student never sees."""
    return f"{question}\n(For reference, the correct final answer is {answer}.)"


def score(engine: Any, prompt: list[int], response: list[int]) -> list[float]:
    """The engine's log-probability of each response token after ``prompt``, without sampling anything new."""
    # SGLang gives no log-probability for the first position of the window, so the window starts one token
    # early, on the prompt's last token, and that entry is dropped.
    output = engine.generate(
        input_ids=prompt + response,
        sampling_params={"temperature": 0.0, "max_new_tokens": 1},
        return_logprob=True,
        logprob_start_len=len(prompt) - 1,
    )
    pairs = output["meta_info"]["input_token_logprobs"][1:]  # (logprob, token_id, text) per response token
    if [int(token) for _, token, _ in pairs] != response or any(value is None for value, _, _ in pairs):
        raise RuntimeError(f"the engine scored {len(pairs)} positions for a {len(response)}-token response")
    return [float(value) for value, _, _ in pairs]


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
    parser.add_argument("--teacher-hint", action="store_true", help="also score each answer under privileged context")
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
    if args.teacher_hint:
        privileged = [
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": hinted(q, answer)}], tokenize=False, add_generation_prompt=True
                ),
                add_special_tokens=False,
            )
            for q, answer in items
        ]
        for index, sample in enumerate(samples):
            prompt_length = len(requests[index])
            response = sample["tokens"][prompt_length:]
            teacher = score(engine, privileged[index // args.samples_per_prompt], response)
            sample["teacher_logprobs"] = [0.0] * prompt_length + teacher
    engine.shutdown()

    rewards = [sample["reward"] for sample in samples]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"reward_mean": statistics.fmean(rewards), "samples": samples}) + "\n")
    print(json.dumps({"samples": len(samples), "reward_mean": statistics.fmean(rewards)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
