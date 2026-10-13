"""SGLang as Honing's serving engine (RFC #103): one engine process per checkpoint in its own environment, the
readouts that turn its log-probabilities into decisions, and policy endpoints that start it for any checkpoint a
runner asks for. Its OpenAI-compatible routes are called as a ChatModel through
:class:`honing.harness.models.LiteLLMChatModel`."""

from honing.backends.sglang.checkpoint import checkpoint_checksum, hf_snapshot
from honing.backends.sglang.endpoints import ServingSettings, SGLangPolicyEndpoints
from honing.backends.sglang.engine import SGLangEngine, SGLangError, SGLangSettings

__all__ = [
    "SGLangEngine",
    "SGLangError",
    "SGLangPolicyEndpoints",
    "SGLangSettings",
    "ServingSettings",
    "checkpoint_checksum",
    "hf_snapshot",
]
