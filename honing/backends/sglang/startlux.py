"""StartLux-Decision answered by an SGLang engine: StartLux's own decision code with SGLang's forward pass.

The adapter is StartLux's :class:`startlux_decision.model.StartLuxDecision` at
the pinned commit with one method replaced. ``_logits`` renders each question
with StartLux's own renderer and asks the engine for the log-probabilities of
the option letters after the prompt, instead of running the checkpoint
itself. Everything else is StartLux's unchanged: the question split, the rounds
of a choice over 26 options, the temperature of each question type, the
softmax over the listed options, and the answer format. A log-probability
differs from its logit by one constant per prompt, so the softmax over the
listed letters is the same.

No StartLux weights are loaded in this process. It needs the checkpoint's
tokenizer and ``decision_config.json``, and torch for StartLux's softmax on the
CPU.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from honing.backends.sglang.engine import SGLangEngine


def sglang_startlux(upstream: Any, prompts: Any, engine: SGLangEngine, checkpoint: Path, max_length: int) -> Any:
    """A ``StartLuxDecision`` from StartLux's ``model`` module whose forward pass is ``engine``.

    ``upstream`` and ``prompts`` are the pinned checkout's
    ``startlux_decision.model`` and ``startlux_decision.jevfmt``.
    """
    import torch
    from transformers import AutoTokenizer

    class SGLangStartLuxDecision(upstream.StartLuxDecision):
        def __init__(self) -> None:  # StartLux's __init__ loads the weights; set only what decisions read
            config = json.loads((checkpoint / "decision_config.json").read_text())
            self.tok = AutoTokenizer.from_pretrained(str(checkpoint))
            self.letters = prompts.check_tokenizer(self.tok)
            if self.letters != config["letter_token_ids"]:
                raise ValueError("tokenizer letter ids differ from decision_config.json")
            self.temperature = {k: float(v) for k, v in config["temperature_by_type"].items()}
            wide = config.get("wide_choice", {})
            self.group, self.keep = wide.get("group", 25), wide.get("keep", 3)
            self.residual = wide.get("residual", 1e-3)
            self.max_length = int(max_length)
            self.graphs: dict[Any, Any] = {}
            self.share, self.fast_kernels = False, True
            self._media = None
            self._local = threading.local()
            self.sglang = engine

        # decide() resets these two per request; keep them per thread so concurrent requests stay apart.
        @property
        def _kept(self) -> Any:
            return getattr(self._local, "kept", None)

        @_kept.setter
        def _kept(self, value: Any) -> None:
            self._local.kept = value

        def _logits(self, rows: list[dict[str, Any]]) -> tuple[list[Any], int]:
            """rows: rendered records (<= 26 options) -> (letter log-probabilities per row in option order, tokens)."""
            encoded, counts = [], []
            for row in rows:
                order = [option["id"] for option in row["options"]]
                encoded.append(self._render(row, order))
                counts.append(len(order))
            values = engine.next_token_logprobs(encoded, self.letters)
            out = [torch.tensor(v[:count], dtype=torch.float32) for v, count in zip(values, counts, strict=True)]
            return out, sum(len(ids) for ids in encoded)

    return SGLangStartLuxDecision()
