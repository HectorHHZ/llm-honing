"""A :class:`honing.core.ChatModel` over litellm, for hosted APIs and local servers alike.

litellm's provider routing lets one client reach a hosted API or a local
SGLang, vLLM or Ollama server: ``model`` is a litellm model string and
``api_base`` points it at a server. litellm is an optional extra and is
imported only when a call is made, so ``honing`` imports without it.

Retries happen here, not through litellm's ``num_retries``. On the synchronous
path litellm retries every ``openai.APIError``, 4xx included, with no wait, and
needs ``tenacity``, which it does not declare. So litellm makes one attempt per
call, and this client retries only timeouts, dropped connections, rate limits
and server errors.

``cost_usd`` uses the price map bundled with the pinned litellm: the client
sets ``LITELLM_LOCAL_MODEL_COST_MAP=True`` before importing litellm, unless the
variable is already set, so nothing is downloaded at import. A process that
imports litellm before the first call decides for itself. The map prices a
hosted provider, so a reply from a server named by ``api_base`` has no known
cost, and neither does a model the map prices at zero, as it does ``ollama/``.

A litellm error carries the provider's response body in its message. Every
failure is raised as :class:`ChatModelError`, whose message names only the
model, the error class and the status code, and which does not chain the
litellm error: a method logging the traceback cannot leak a body or the key.
"""

from __future__ import annotations

import importlib.util
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import ModuleType

from honing.core import ChatMessage, ChatReply, ContractError, HoningError, require

# Statuses worth another attempt: a timeout, a rate limit, or a server error.
# Every other 4xx means the request itself is wrong and would fail again.
_RETRYABLE_STATUSES = frozenset({408, 429})


class ChatModelError(HoningError):
    """A chat completion failed. The message never holds a request body, a response body, or a key."""

    def __init__(
        self, model: str, error_type: str, status_code: int | None, attempts: int, hint: str | None = None
    ) -> None:
        self.model = model
        self.error_type = error_type
        self.status_code = status_code
        self.attempts = attempts
        status = "no status" if status_code is None else f"status {status_code}"
        plural = "" if attempts == 1 else "s"
        message = f"chat completion for {model!r} failed after {attempts} attempt{plural}: {error_type} ({status})"
        super().__init__(message if hint is None else f"{message}; {hint}")


@dataclass(frozen=True)
class LiteLLMChatModel:
    """Calls ``litellm.completion`` and reports the reply's tokens and cost.

    ``api_key_env`` names the environment variable that holds the key. The key
    is read on every call and passed to that call only, never to litellm's
    global settings. Without ``api_key_env`` no key is passed, and litellm
    applies its provider's default (for ``openai/`` models, ``OPENAI_API_KEY``).
    An ``openai/`` model needs a key even with ``api_base`` set; a local server
    that needs no key is reached as ``hosted_vllm/<name>`` without ``api_key_env``.
    A call that fails while that default key is unset is not retried.

    ``max_attempts`` counts the first attempt. Retry ``n`` waits
    ``retry_backoff_seconds`` times ``2 ** (n - 1)``: once, then twice, then
    four times that.
    ``cost_usd`` is ``None`` when ``api_base`` is set, when litellm has no price
    for ``model``, or when its price is zero.
    """

    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    top_p: float | None = None
    seed: int | None = None
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False, compare=False)

    def __post_init__(self) -> None:
        require(bool(self.model), "model must be a litellm model string")
        require(
            math.isfinite(self.timeout_seconds) and self.timeout_seconds > 0,
            f"timeout_seconds must be > 0, got {self.timeout_seconds}",
        )
        require(self.max_attempts >= 1, f"max_attempts must be >= 1, got {self.max_attempts}")
        require(
            math.isfinite(self.retry_backoff_seconds) and self.retry_backoff_seconds >= 0,
            f"retry_backoff_seconds must be >= 0, got {self.retry_backoff_seconds}",
        )
        require(self.top_p is None or 0.0 < self.top_p <= 1.0, f"top_p must be in (0, 1], got {self.top_p}")
        if importlib.util.find_spec("litellm") is None:
            raise ImportError("LiteLLMChatModel needs the 'litellm' extra: pip install 'llm-honing[litellm]'")
        # Fail at construction, not on the first reflection: a method may treat
        # a failed call as "no proposal" and carry on spending episodes.
        self._api_key()

    def settings(self) -> Mapping[str, object]:
        """The client's configuration as JSON-native values, for a method to record. Holds no key."""
        return {
            "client": "litellm",
            "model": self.model,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "max_attempts": self.max_attempts,
            "top_p": self.top_p,
            "seed": self.seed,
        }

    def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int, temperature: float) -> ChatReply:
        # Read once, when litellm is first imported: without it litellm downloads
        # the latest price map from GitHub. Set first to choose otherwise.
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        import litellm

        api_key = self._api_key()
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=[{"role": message.role, "content": message.content} for message in messages],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=self.top_p,
                    seed=self.seed,
                    api_base=self.api_base,
                    api_key=api_key,
                    timeout=self.timeout_seconds,
                    # One attempt per litellm call; this loop owns retries.
                    num_retries=0,
                    max_retries=0,
                )
            except Exception as error:
                status = _status_code(error)
                missing = [] if api_key is not None else _missing_keys(litellm, self, error, status)
                if missing:
                    # litellm reports an unset provider key as a 500, which a retry cannot fix.
                    hint = (
                        f"{', '.join(missing)} is unset; set api_key_env, or use a hosted_vllm/ model"
                        " for a local server that needs no key"
                    )
                    raise ChatModelError(self.model, "MissingAPIKey", status, attempt, hint) from None
                if attempt < self.max_attempts and _retryable(litellm, error, status):
                    self.sleep(self.retry_backoff_seconds * 2 ** (attempt - 1))
                    continue
                # ``from None``: the litellm error's message holds the response body.
                raise ChatModelError(self.model, type(error).__name__, status, attempt) from None
            return self._reply(litellm, response, attempt)
        raise AssertionError("unreachable: the last attempt returns or raises")

    def _api_key(self) -> str | None:
        if self.api_key_env is None:
            return None
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ContractError(f"environment variable {self.api_key_env} is unset or empty; it must hold the API key")
        return key

    def _reply(self, litellm: ModuleType, response: object, attempts: int) -> ChatReply:
        choices = getattr(response, "choices", None) or []
        text = getattr(getattr(choices[0], "message", None), "content", None) if choices else None
        if not isinstance(text, str):
            raise ChatModelError(self.model, "NoTextInReply", None, attempts)
        usage = getattr(response, "usage", None)
        return ChatReply(
            text=text,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cost_usd=None if self.api_base is not None else _cost_usd(litellm, response),
        )


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def _retryable(litellm: ModuleType, error: Exception, status: int | None) -> bool:
    if isinstance(error, (litellm.Timeout, litellm.APIConnectionError)):
        return True
    return status is not None and (status in _RETRYABLE_STATUSES or 500 <= status < 600)


def _missing_keys(litellm: ModuleType, chat: LiteLLMChatModel, error: Exception, status: int | None) -> list[str]:
    """The provider key variables litellm wants but cannot find, when they may explain ``error``.

    Checked only after a failure, so a provider that authenticates another way
    (an AWS profile, application default credentials) is never refused up front.
    """
    if isinstance(error, (litellm.Timeout, litellm.APIConnectionError)) or status not in (None, 401, 500):
        return []
    try:
        report = litellm.validate_environment(model=chat.model, api_base=chat.api_base)
    except Exception:
        return []
    missing = report.get("missing_keys") if isinstance(report, Mapping) else None
    return [str(name) for name in missing] if isinstance(missing, list) else []


def completion_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """litellm's price for these tokens of ``model``, or ``None`` when it has none.

    For calls that do not go through :class:`LiteLLMChatModel`, such as an
    agent's own calls to its policy, priced from the same bundled map.
    """
    require(input_tokens >= 0 and output_tokens >= 0, "token counts must be >= 0")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm

    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model, prompt_tokens=input_tokens, completion_tokens=output_tokens
        )
    except Exception:
        # As in _cost_usd: litellm raises for a model missing from its price map.
        return None
    cost = prompt_cost + completion_cost
    if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        return None
    return float(cost)


def _cost_usd(litellm: ModuleType, response: object) -> float | None:
    """litellm's price for the reply, or ``None`` when it has none, or none above zero, for the model."""
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        # litellm raises for a model missing from its price map, which is every local model.
        return None
    # A zero price is litellm's entry for a local provider such as ollama/, not a free call.
    if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
        return None
    return float(cost)
