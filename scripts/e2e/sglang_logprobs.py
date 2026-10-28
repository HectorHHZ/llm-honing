"""SGLang's next-token log-probabilities against the checkpoint's own transformers forward pass, for any decoder.

Usage, from an environment with torch and transformers that can load the checkpoint, with SGLANG_PYTHON naming the
SGLang environment's interpreter:

    python scripts/e2e/sglang_logprobs.py <reference-gpu> <sglang-gpu> <out.json> <checkpoint> [<checkpoint> ...]

For each checkpoint, a handful of chat prompts (short to a few thousand tokens) run through the checkpoint's own
transformers class in fp32, the ground truth, and in bf16 with the last hidden state read out through the output
head in fp32, the way the decision families read their letters. The same token ids then go to an SGLang server
started by ``honing.backends.sglang.SGLangEngine`` (bf16, fp32 output head). Both bf16 forward passes are compared
with fp32 on the 20 tokens fp32 finds likeliest: whether the likeliest token agrees, and the largest log-probability
and probability differences. SGLang should be as close to fp32 as transformers in bf16 is.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PROMPTS = [
    "What is 17 * 23? Answer with the number only.",
    "Name the capital of Australia in one word.",
    "Summarise the following log in one sentence.\n"
    + "\n".join(f"tick {t}: unit {t % 13} moved to ({t % 50}, {(t * 7) % 50}); credits {t * 3}" for t in range(150)),
    "Which option is best? A) build power B) build refinery C) build barracks. Reply with the letter.\n"
    + "\n".join(f"memory {j}: the enemy was seen at sector {j % 9}" for j in range(400)),
]
TOP = 20


def head_logits(config: dict, hidden: Any, head: Any) -> Any:
    """The next-token logits from the final hidden state in fp32, with what the checkpoint's own head applies after
    the projection: a final softcap (Gemma 2) or a logit scale (Cohere multiplies, Granite divides)."""
    import torch

    text = config.get("text_config", config)
    logits = hidden.float() @ head.T
    if text.get("logit_scale"):
        logits = logits * float(text["logit_scale"])
    if text.get("logits_scaling"):
        logits = logits / float(text["logits_scaling"])
    if text.get("final_logit_softcapping"):
        cap = float(text["final_logit_softcapping"])
        logits = cap * torch.tanh(logits / cap)
    return logits


def render(tokenizer, prompt: str) -> list[int]:
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        return tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.encode(prompt)


def reference(checkpoint: Path, prompts: list[list[int]], dtype: str) -> list[Any]:
    """Each prompt's next-token log-probabilities over the whole vocabulary, from transformers in ``dtype`` with an
    fp32 output head."""
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(checkpoint)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        checkpoint, dtype=getattr(torch, dtype), device_map={"": "cuda"}
    )
    body = getattr(model.model, "language_model", model.model).eval()
    head = model.get_output_embeddings().weight.float()
    out = []
    with torch.inference_mode():
        for ids in prompts:
            hidden = body(input_ids=torch.tensor([ids], device="cuda")).last_hidden_state[0, -1].float()
            out.append(torch.log_softmax(head_logits(config.to_dict(), hidden, head), -1).cpu())
    del model, body, head
    torch.cuda.empty_cache()
    return out


def differences(truth: list[tuple[int, float]], values: list[float]) -> dict[str, Any]:
    return {
        "top1_same": max(range(len(values)), key=values.__getitem__) == 0,
        "max_abs_logprob": max(abs(v - h) for (_, v), h in zip(truth, values, strict=True)),
        "max_abs_prob": max(abs(math.exp(v) - math.exp(h)) for (_, v), h in zip(truth, values, strict=True)),
    }


def main() -> None:
    from transformers import AutoTokenizer

    from honing.backends.sglang import SGLangEngine, SGLangSettings
    from honing.backends.sglang.checkpoint import context_length

    reference_gpu, sglang_gpu, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    report = json.loads(out.read_text()) if out.exists() else {}
    for checkpoint in map(Path, sys.argv[4:]):
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            limit = context_length(checkpoint)
            # A prompt longer than the model reads is left out, not truncated.
            prompts = [ids for ids in (render(tokenizer, p) for p in PROMPTS) if limit is None or len(ids) < limit]
            os.environ["CUDA_VISIBLE_DEVICES"] = reference_gpu
            fp32 = reference(checkpoint, prompts, "float32")
            bf16 = reference(checkpoint, prompts, "bfloat16")
            expected = []
            for full in fp32:
                values, tokens = full.topk(TOP)
                expected.append([(int(t), float(v)) for t, v in zip(tokens.tolist(), values.tolist(), strict=True)])
            settings = SGLangSettings(
                python=Path(os.environ["SGLANG_PYTHON"]),
                gpu=sglang_gpu,
                mem_fraction=0.7,
                context_length=limit,
                log_path=out.with_suffix(f".{checkpoint.parent.parent.name}.log"),
            )
            with SGLangEngine.launch(checkpoint, settings) as engine:
                got = [
                    engine.next_token_logprobs([ids], [t for t, _ in want])[0]
                    for ids, want in zip(prompts, expected, strict=True)
                ]
                versions = engine.versions
            rows = []
            for ids, want, have, half in zip(prompts, expected, got, bf16, strict=True):
                rows.append(
                    {
                        "tokens": len(ids),
                        "sglang": differences(want, have),
                        "transformers_bf16": differences(want, [float(half[t]) for t, _ in want]),
                    }
                )
            config = json.loads((checkpoint / "config.json").read_text())
            name = checkpoint.parent.parent.name.removeprefix("models--").replace("--", "/")
            report[name] = {
                "architecture": config["architectures"][0],
                "model_type": config.get("model_type"),
                "engine": versions,
                "prompts": rows,
            }
            for side in ("sglang", "transformers_bf16"):
                print(
                    f"{name} ({config['architectures'][0]}) {side} vs fp32: top-1 same "
                    f"{sum(r[side]['top1_same'] for r in rows)}/{len(rows)}, "
                    f"max |dlogp| {max(r[side]['max_abs_logprob'] for r in rows):.4f}, "
                    f"max |dp| {max(r[side]['max_abs_prob'] for r in rows):.4f}, "
                    f"prompt tokens {[r['tokens'] for r in rows]}",
                    flush=True,
                )
            out.write_text(json.dumps(report, indent=1))
        except Exception as error:  # noqa: PERF203 - one checkpoint that fails to load does not stop the others
            name = checkpoint.parent.parent.name.removeprefix("models--").replace("--", "/")
            report[name] = {"error": f"{type(error).__name__}: {error}"}
            print(f"{name}: failed: {type(error).__name__}: {error}", flush=True)
            out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
