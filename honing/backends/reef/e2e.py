"""Component factories for an e2e config that trains on Reef's job data (#227)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from honing.backends.reef.provider import ReefDataProvider
from honing.core import ContractError


def data(context: Any, spool_dir: str, wait_s: float = 0.0) -> ReefDataProvider:
    """The job data Reef spooled to ``spool_dir``, which may name environment variables.

    A relative path is taken from the repository root. ``wait_s`` is how long
    a train move waits for Reef to spool its next job.
    """
    path = Path(os.path.expandvars(spool_dir))
    if "$" in str(path):
        raise ContractError(f"spool_dir {spool_dir} names an unset environment variable")
    return ReefDataProvider(path if path.is_absolute() else Path(context.root) / path, wait_s=wait_s)
