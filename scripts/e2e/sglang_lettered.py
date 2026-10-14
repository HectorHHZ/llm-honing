"""Honing's lettered decision readout on SGLang against the same readout on the checkpoint's own transformers class.

Usage, from an environment with torch and transformers that can load the checkpoints, with SGLANG_PYTHON naming the
SGLang environment's interpreter:

    python scripts/e2e/sglang_lettered.py <reference-gpu> <sglang-gpu> <out.json> <checkpoint> [<checkpoint> ...]

Every side runs :class:`honing.backends.sglang.lettered.LetteredDecisionModel` on the same requests; only the
forward pass differs: the checkpoint's transformers class in fp32 (the ground truth) and in bf16, each with its
output head in fp32, and ``SGLangEngine`` (bf16, fp32 output head). SGLang and transformers in bf16 are each
compared with fp32, so the difference SGLang makes can be told from bf16's own. The requests are the 24 serving
parity requests (choices over 2 to 8 options, states with up to 400 log entries) and three more: a yes/no question,
a score over five levels, and a choice over 30 options, which runs in two rounds. Reported per checkpoint: same
choice (for choices) and the largest probability difference over all answers, and the median latency on each side.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "integrations/ra2/parity"))

from decision_requests import requests  # noqa: E402

from honing.backends.sglang import SGLangEngine, SGLangSettings  # noqa: E402
from honing.backends.sglang.checkpoint import context_length  # noqa: E402
from honing.backends.sglang.lettered import LetteredDecisionModel, LetteredSettings, load_tokenizer  # noqa: E402


class TransformersScorer:
    """next_token_logprobs from the checkpoint's own transformers class: bf16 forward, fp32 output head."""

    def __init__(self, checkpoint: Path, dtype: str = "bfloat16") -> None:
        import torch
        import transformers

        config = transformers.AutoConfig.from_pretrained(checkpoint)
        model = getattr(transformers, config.architectures[0]).from_pretrained(
            checkpoint, dtype=getattr(torch, dtype), device_map={"": "cuda"}
        )
        self.body = getattr(model.model, "language_model", model.model).eval()
        self.config = config.to_dict()
        self.head = model.get_output_embeddings().weight.float()
        self.torch = torch

    def next_token_logprobs(self, prompts, token_ids):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for ids in prompts:
                hidden = self.body(input_ids=torch.tensor([list(ids)], device="cuda")).last_hidden_state[0, -1]
                logprobs = torch.log_softmax(head_logits(self.config, hidden, self.head), -1)
                out.append(logprobs[list(token_ids)].tolist())
        return out


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


def all_requests():
    reqs = requests()
    state = {"essential": ["Tick 300. Credits 4200. Army 9."], "details": ["Enemy base seen to the east."]}
    reqs.append(
        {
            "id": "noul",
            "parts": state,
            "questions": {
                "attack": {
                    "type": "noul",
                    "instructions": "Should the army attack now?",
                    "criteria": {"true": "attack the base", "false": "keep defending"},
                }
            },
        }
    )
    reqs.append(
        {
            "id": "score",
            "parts": state,
            "questions": {
                "threat": {
                    "type": "score",
                    "instructions": "How threatening is the enemy right now?",
                    "criteria": {"0": "none", "1": "low", "2": "moderate", "3": "high", "4": "critical"},
                }
            },
        }
    )
    reqs.append(
        {
            "id": "wide",
            "parts": state,
            "questions": {
                "target": {
                    "type": "choice",
                    "instructions": "Which cell should the scout explore next?",
                    "criteria": {f"cell{i}": f"cell at ({i % 6}, {i // 6}), {i * 3} tiles away" for i in range(30)},
                }
            },
        }
    )
    return reqs


def run(model, reqs):
    results = {}
    for r in reqs:
        answer = model.predict(r["parts"], r["questions"])
        results[str(r["id"])] = (answer["answers"], answer["latencyMs"])
    return results


def compare(ours, theirs):
    rows = []
    for rid, (answers, _) in ours.items():
        for key, a in answers.items():
            b = theirs[rid][0][key]
            if a["type"] == "noul":
                rows.append({"request": rid, "same": None, "max_abs": abs(a["noul"] - b["noul"])})
            else:
                diff = max(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"])
                same = a.get("choice") == b.get("choice") if a["type"] == "choice" else None
                rows.append({"request": rid, "same": same, "max_abs": diff})
    return rows


def main() -> None:
    reference_gpu, sglang_gpu, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    report = json.loads(out.read_text()) if out.exists() else {}
    reqs = all_requests()
    for checkpoint in map(Path, sys.argv[4:]):
        try:
            name = checkpoint.parent.parent.name.removeprefix("models--").replace("--", "/")
            tokenizer = load_tokenizer(checkpoint)
            # As honing.backends.sglang.lettered.load does: the model's own limit when it reads fewer tokens.
            settings = LetteredSettings(max_length=min(8192, (context_length(checkpoint) or 8200) - 8))
            os.environ["CUDA_VISIBLE_DEVICES"] = reference_gpu
            answers = {}
            for dtype in ("float32", "bfloat16"):
                scorer = TransformersScorer(checkpoint, dtype)
                answers[dtype] = run(LetteredDecisionModel(scorer, tokenizer, settings, {}, concurrent=False), reqs)
                del scorer
                import torch

                torch.cuda.empty_cache()
            theirs = answers["float32"]
            engine_settings = SGLangSettings(
                python=Path(os.environ["SGLANG_PYTHON"]), gpu=sglang_gpu, mem_fraction=0.7,
                context_length=settings.max_length + 8,
                log_path=out.with_suffix(f".{checkpoint.parent.parent.name}.log"),
            )  # fmt: skip
            with SGLangEngine.launch(checkpoint, engine_settings) as engine:
                model = LetteredDecisionModel(engine, tokenizer, settings, {})
                run(model, reqs[:1])  # warm-up
                ours = run(model, reqs)
                letters = model.letters
            rows = compare(ours, theirs)
            bf16_rows = compare(answers["bfloat16"], theirs)
            bf16_choices = [r for r in bf16_rows if r["same"] is not None]
            choices = [r for r in rows if r["same"] is not None]
            report[name] = {
                "transformers_bf16": {
                    "same_choice": sum(r["same"] for r in bf16_choices),
                    "max_abs": max(r["max_abs"] for r in bf16_rows),
                    "mean_abs": statistics.fmean(r["max_abs"] for r in bf16_rows),
                    "rows": bf16_rows,
                },
                "letters": letters[:3],
                "chat_template": bool(tokenizer.chat_template),
                "questions": len(rows),
                "same_choice": sum(r["same"] for r in choices),
                "choices": len(choices),
                "max_abs": max(r["max_abs"] for r in rows),
                "mean_abs": statistics.fmean(r["max_abs"] for r in rows),
                "median_ms": {
                    "sglang": statistics.median(t for _, t in ours.values()),
                    "transformers": statistics.median(t for _, t in theirs.values()),
                },
                "rows": rows,
            }
            r = report[name]
            b = r["transformers_bf16"]
            print(
                f"{name} transformers bf16 vs fp32: same choice {b['same_choice']}/{r['choices']}, "
                f"max |dp| {b['max_abs']:.4f}, mean |dp| {b['mean_abs']:.4f}",
                flush=True,
            )
            print(
                f"{name} sglang vs fp32: {r['questions']} answers, same choice {r['same_choice']}/{r['choices']}, "
                f"max |dp| {r['max_abs']:.4f}, mean |dp| {r['mean_abs']:.4f}, median ms sglang "
                f"{r['median_ms']['sglang']:.1f} / transformers {r['median_ms']['transformers']:.1f}, "
                f"chat template {r['chat_template']}",
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
