"""Chat calls sent through Reef, so that Reef records each one itself (#224).

Reef answers an OpenAI-compatible call on ``/reef/scenarios/<scenario>/v1/chat/completions``. It forwards the
call to the model its deployment names (``inference.upstream-url`` and ``inference.upstream-model``), with the
provider key Reef holds, stores the exchange as an inference record of the scenario, and returns the provider's
answer with the record's id in ``x-reef-agent-record-id``. The request's ``x-reef-tag-<name>`` headers become the
record's ``metadata.tags``. Reef replaces the ``model`` the client sends with its upstream model, and refuses a
scenario that does not exist (404).

So a client pointed at :attr:`ReefChatRoute.base_url` needs nothing else: the scenario is in the URL, and the key
it sends is Reef's token. A client that can add headers also sends :meth:`ReefChatRoute.headers`, Honing's tags.
Nothing here is on unless a config names a route, and a client without one calls its own server as before.

Follows Reef's ``scenario_inference`` route and ``InferenceReplica.answer`` (reef/api/inference_service.py), which
read the scenario from the path and the tags with ``parse_request_headers`` (reef/api/wire.py), and its
``inference_proxy`` runtime (``InferenceProxyRuntime``, reef/inference/http.py), which forwards the call, Reef
bc8c150. Differs: a route sends only Honing's tags (``TAG_NAMES``), checked as a decision record's are.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any
from urllib.parse import quote

from honing.backends.reef.records import tag_headers
from honing.core import ContractError, require
from honing.harness.models import LiteLLMChatModel

#: The variable that holds Reef's token, as the decision server's recorder reads it too.
TOKEN_ENV = "REEF_TOKEN"


@dataclass(frozen=True)
class ReefChatRoute:
    """A Reef service and one of its scenarios, which chat calls go through to be recorded.

    ``url`` is Reef's root, such as ``http://127.0.0.1:8977``. ``scenario`` names the Reef scenario the records go
    to, which must exist before the first call. ``tags`` are Honing's tags for every call. ``token_env`` names the
    variable that holds Reef's token, or is None for a Reef that takes none.
    """

    url: str
    scenario: str
    tags: Mapping[str, str] = field(default_factory=dict)
    token_env: str | None = TOKEN_ENV

    def __post_init__(self) -> None:
        require(self.url.startswith(("http://", "https://")), f"the Reef url must be an HTTP URL, got {self.url!r}")
        # Reef's rule for a scenario name (check_owner, reef/storage/data/store.py), and no surrounding spaces.
        require(
            bool(self.scenario)
            and self.scenario == self.scenario.strip()
            and "/" not in self.scenario
            and self.scenario not in (".", ".."),
            f"the Reef scenario must be a name without '/', got {self.scenario!r}",
        )
        require(self.token_env is None or bool(self.token_env), "token_env must be None or a variable's name")
        tag_headers(self.tags)  # refuses, before any call, a tag that a decision record would refuse
        object.__setattr__(self, "tags", MappingProxyType(dict(self.tags)))

    @property
    def base_url(self) -> str:
        """The OpenAI-compatible base URL that records each call in the scenario: ``<url>/reef/scenarios/<s>/v1``."""
        return f"{self.url.rstrip('/')}/reef/scenarios/{quote(self.scenario, safe='')}/v1"

    def headers(self) -> dict[str, str]:
        """The tags as ``x-reef-tag-*`` headers, which Reef stores in each record's ``metadata.tags``."""
        return tag_headers(self.tags)


def chat_model(
    context: object, reef: ReefChatRoute | Mapping[str, Any] | None = None, **fields: Any
) -> LiteLLMChatModel:
    """A :class:`~honing.harness.models.LiteLLMChatModel` for a pipeline config, sending its calls through Reef.

    Without ``reef`` the client is ``LiteLLMChatModel(**fields)``, as if the config had not named this factory.
    With it (a route, or its fields), the client calls the route's URL with Reef's token and the route's tags.
    Reef's deployment then names the model's server and its key, so ``fields`` may not name ``api_base``,
    ``api_key_env`` or ``extra_headers``. ``model`` still picks litellm's provider: ``hosted_vllm/<name>`` reaches
    Reef as an OpenAI-compatible server, whatever the name.

    Args:
        context: The pipeline context, unused.
        reef: The route, or None to call the model's server directly.
        **fields: :class:`~honing.harness.models.LiteLLMChatModel` fields.

    Raises:
        ContractError: The route is invalid, or ``fields`` name what the route decides.
    """
    if reef is None:
        return LiteLLMChatModel(**fields)
    route = reef if isinstance(reef, ReefChatRoute) else ReefChatRoute(**reef)
    decided = sorted({"api_base", "api_key_env", "extra_headers"} & set(fields))
    if decided:
        raise ContractError(f"with reef, Reef's deployment names the model's server and key, so drop {decided}")
    return LiteLLMChatModel(
        **fields, api_base=route.base_url, api_key_env=route.token_env, extra_headers=route.headers()
    )


__all__ = ["TOKEN_ENV", "ReefChatRoute", "chat_model"]
