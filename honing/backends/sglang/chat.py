"""A :class:`~honing.core.ChatModel` over an OpenAI-compatible chat route, such as an SGLang engine started with
``chat``: what an evolution method's reflection or a harness runner's agent calls."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Sequence

from honing.core import ChatMessage, ChatReply


class SGLangChatModel:
    """``complete`` posts to ``<base_url>/chat/completions`` and returns the reply with its token counts.

    ``base_url`` is the OpenAI-style base, ending in ``/v1``
    (:attr:`~honing.backends.sglang.SGLangEngine.openai_url`).
    """

    def __init__(self, base_url: str, model: str, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float) -> ChatReply:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", json.dumps(payload).encode(), {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                answer = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"chat completion answered {error.code}: {error.read()[:500]!r}") from None
        usage = answer.get("usage") or {}
        return ChatReply(
            text=answer["choices"][0]["message"]["content"] or "",
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )
