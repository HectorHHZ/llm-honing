"""Errors Honing raises on purpose, and the two checks every value object uses.

Every error names the contract or rule it enforces, so a refusal reads as a
decision rather than a crash.
"""

from __future__ import annotations

from collections.abc import Collection


class HoningError(Exception):
    """Base class for the errors Honing raises deliberately."""


class ContractError(HoningError, ValueError):
    """A value violates a declared contract.

    Raised when a value object is constructed or when components are wired
    together, before any training, editing, or episode work starts.
    """


class CampaignRuleError(HoningError):
    """A campaign ordering rule refused a step before it touched any weights.

    ``rule`` is the name of the function in :mod:`honing.campaign.rules` that
    refused, so logs and tests can match on it without parsing the message.
    """

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(f"{rule}: {message}")
        self.rule = rule


class ReleaseError(HoningError):
    """A release operation names a release that is absent or no longer current."""


def require(condition: bool, message: str) -> None:
    """Raise :class:`ContractError` with ``message`` unless ``condition`` holds."""
    if not condition:
        raise ContractError(message)


def require_choice(value: object, choices: Collection[object], name: str) -> None:
    """Raise :class:`ContractError` unless ``value`` is one of ``choices``.

    ``Literal`` annotations are not enforced at runtime, so every value object
    that declares one checks it here instead.
    """
    if value not in choices:
        raise ContractError(f"{name} must be one of {sorted(map(str, choices))}, got {value!r}")
