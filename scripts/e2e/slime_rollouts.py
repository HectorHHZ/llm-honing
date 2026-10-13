"""Sample grouped, rewarded rollouts from a checkpoint with SGLang; run inside Slime's image.

Each prompt is a small arithmetic question that asks for the answer in
``\\boxed{}``. SGLang samples ``--samples-per-prompt`` answers per question and
returns each answer's token ids and log-probabilities, the record an agent
adapter (slime.agent's ``OpenAIAdapter``) keeps per session. The reward is 1
when the boxed answer is right and 0 otherwise. The output is a JSON list of
:class:`~honing.core.PolicySample` fields, which ``slime_train_step.py`` turns
into a ``TrainingBatch``; the prompt tokens carry mask 0 and log-probability 0.0.

With ``--teacher answer`` or ``--teacher solution`` every answer is also scored
by the same model under privileged context, the question followed by its
correct answer or by a worked solution, which the sampling prompt left out.
Those are the teacher log-probabilities on-policy self-distillation trains
toward (``PolicySample.teacher_logprobs``).
``--teacher same`` scores under the sampling prompt itself, a check of the
scoring: the teacher's log-probabilities should then match the sampler's
(``teacher_gap`` in the output), up to the engine's numerics.

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


def questions(count: int, seed: int) -> list[tuple[str, int, str]]:
    """Questions, their answers, and a worked solution of each."""
    rng = random.Random(seed)
    items = []
    for _ in range(count):
        a, b, c = rng.randint(10, 99), rng.randint(10, 99), rng.randint(2, 9)
        total, answer = a + b, (a + b) * c
        solution = f"{a} + {b} = {total}, and {total} * {c} = {answer}."
        items.append((f"Compute ({a} + {b}) * {c}. Put only the final number in \\boxed{{}}.", answer, solution))
    return items


def hinted(question: str, answer: int, solution: str, context: str) -> str:
    """The question with privileged context the student never sees: its answer, or a worked solution.

    With the answer alone, the teacher's likeliest reply is the boxed answer with no working, which the
    student cannot reproduce without the answer; a worked solution makes the working itself likely.
    """
    if context == "solution":
        return f"{question}\n(For reference, a worked solution: {solution} The final answer is {answer}.)"
    return f"{question}\n(For reference, the correct final answer is {answer}.)"


def score(engine: Any, prompts: list[list[int]], responses: list[list[int]]) -> list[list[float]]:
    """The engine's log-probability of each response token after its prompt, without sampling anything new."""
    # SGLang gives no log-probability for the first position of the window, so each window starts one token
    # early, on the prompt's last token, and that entry is dropped.
    outputs = engine.generate(
        input_ids=[prompt + response for prompt, response in zip(prompts, responses, strict=True)],
        sampling_params={"temperature": 0.0, "max_new_tokens": 1},
        return_logprob=True,
        logprob_start_len=[len(prompt) - 1 for prompt in prompts],
    )
    scores = []
    for output, response in zip(outputs, responses, strict=True):
        pairs = output["meta_info"]["input_token_logprobs"][1:]  # (logprob, token_id, text) per response token
        if [int(token) for _, token, _ in pairs] != response or any(value is None for value, _, _ in pairs):
            raise RuntimeError(f"the engine scored {len(pairs)} positions for a {len(response)}-token response")
        scores.append([float(value) for value, _, _ in pairs])
    return scores


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
    parser.add_argument(
        "--teacher",
        choices=("none", "answer", "solution", "same"),
        default="none",
        help="also score each answer: with the answer or a worked solution as privileged context, or under its own"
        " prompt as a check (same)",
    )
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
        for q, *_ in items
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
    teacher_gap = None
    if args.teacher != "none":
        contexts = [q if args.teacher == "same" else hinted(q, a, s, args.teacher) for q, a, s in items]
        teacher_prompts = [
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": context}], tokenize=False, add_generation_prompt=True
                ),
                add_special_tokens=False,
            )
            for context in contexts
        ]
        lengths = [len(request) for request in requests]
        responses = [sample["tokens"][length:] for sample, length in zip(samples, lengths, strict=True)]
        scores = score(engine, [teacher_prompts[i // args.samples_per_prompt] for i in range(len(samples))], responses)
        gaps = []
        for sample, length, teacher in zip(samples, lengths, scores, strict=True):
            sample["teacher_logprobs"] = [0.0] * length + teacher
            gaps += [abs(a - b) for a, b in zip(sample["logprobs"][length:], teacher, strict=True)]
        teacher_gap = statistics.fmean(gaps)
    engine.shutdown()

    rewards = [sample["reward"] for sample in samples]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"reward_mean": statistics.fmean(rewards), "teacher_gap": teacher_gap}
    args.out.write_text(json.dumps({**summary, "samples": samples}) + "\n")
    print(json.dumps({"samples": len(samples), **summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
