"""A stand-in for the ``litellm`` module, so the chat client's own logic runs without the 'litellm' extra.

:func:`install` puts a :class:`FakeLiteLLM` in ``sys.modules`` under the name
``litellm``, with a module spec, so ``importlib.util.find_spec("litellm")``
finds it and ``import litellm`` returns it. It has the names
:mod:`honing.harness.models` reads: ``completion``, ``completion_cost``,
``cost_per_token``, ``validate_environment`` and the two exception classes a
retry looks for. It never opens a socket.

``completion`` answers from a script: each entry is an exception to raise, a
reply text, or a function of the call's keyword arguments that returns either.
The last entry repeats. Every call's keyword arguments are kept in ``calls``.
Prices come from ``prices`` (USD per prompt token and per completion token);
a model missing from it raises, as litellm does for a model missing from its
price map.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest


class Timeout(Exception):  # noqa: N818 - litellm's name
    """litellm's timeout, which carries status 408."""

    status_code = 408


class APIConnectionError(Exception):
    """litellm's dropped connection, which carries status 500."""

    status_code = 500


class APIError(Exception):
    """Any other litellm error: a provider's status and its response body in the message."""

    def __init__(self, status_code: int | None, body: str = "") -> None:
        super().__init__(f"provider said: {body}")
        self.status_code = status_code


Outcome = BaseException | str | Callable[[Mapping[str, Any]], "BaseException | str"]


def response(text: object, prompt_tokens: int | None = 7, completion_tokens: int | None = 3, model: str = "m") -> Any:
    """A completion response with one choice, as litellm's ``ModelResponse`` reads."""
    message = types.SimpleNamespace(role="assistant", content=text)
    usage = types.SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=usage, model=model)


class FakeLiteLLM(types.ModuleType):
    Timeout = Timeout
    APIConnectionError = APIConnectionError

    def __init__(
        self,
        script: Sequence[Outcome] = ("hello",),
        *,
        prices: Mapping[str, tuple[float, float]] | None = None,
        missing_keys: Sequence[str] = (),
        reply_cost: object = None,
    ) -> None:
        super().__init__("litellm")
        self.__spec__ = importlib.machinery.ModuleSpec("litellm", loader=None)
        self.script = list(script)
        self.prices = dict(prices or {})
        self.missing_keys = list(missing_keys)
        self.reply_cost = reply_cost
        self.calls: list[dict[str, Any]] = []
        self.validated: list[dict[str, Any]] = []

    def completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if callable(outcome) and not isinstance(outcome, BaseException):
            outcome = outcome(kwargs)
        if isinstance(outcome, BaseException):
            raise outcome
        return response(outcome, model=str(kwargs.get("model")))

    def completion_cost(self, completion_response: Any) -> object:
        if self.reply_cost is not None:
            if isinstance(self.reply_cost, BaseException):
                raise self.reply_cost
            return self.reply_cost
        prompt, completion = self.cost_per_token(
            model=completion_response.model,
            prompt_tokens=completion_response.usage.prompt_tokens,
            completion_tokens=completion_response.usage.completion_tokens,
        )
        return prompt + completion

    def cost_per_token(self, model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, float]:
        if model not in self.prices:
            raise ValueError(f"This model isn't mapped yet. model={model}")
        prompt, completion = self.prices[model]
        return prompt * prompt_tokens, completion * completion_tokens

    def validate_environment(self, model: str, api_base: str | None = None) -> dict[str, object]:
        self.validated.append({"model": model, "api_base": api_base})
        return {"keys_in_environment": not self.missing_keys, "missing_keys": list(self.missing_keys)}


def install(monkeypatch: pytest.MonkeyPatch, fake: FakeLiteLLM | None = None) -> FakeLiteLLM:
    """Put ``fake`` (a default one if None) in ``sys.modules`` as ``litellm`` until the test ends."""
    fake = fake if fake is not None else FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake
