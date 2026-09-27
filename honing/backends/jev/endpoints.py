"""Find the decision server serving a policy."""

from __future__ import annotations

from collections.abc import Mapping

from honing.core import ContractError, PolicyEndpoint, PolicyRef


class JevEndpoints:
    """A fixed map from checkpoint checksum to the decision server that loaded it.

    Satisfies :class:`~honing.core.PolicyEndpoints`. Starting and stopping
    servers is the caller's job; this only records where each one is.
    """

    def __init__(self, endpoints: Mapping[str, PolicyEndpoint]) -> None:
        for checksum, endpoint in endpoints.items():
            if endpoint.runtime != "jev_decision":
                raise ContractError(f"checkpoint {checksum!r} maps to a {endpoint.runtime} endpoint, not a Jev server")
        self._endpoints = dict(endpoints)

    def resolve(self, policy: PolicyRef) -> PolicyEndpoint:
        try:
            return self._endpoints[policy.checkpoint.checksum]
        except KeyError:
            raise ContractError(f"no decision server serves checkpoint {policy.checkpoint.checksum!r}") from None
