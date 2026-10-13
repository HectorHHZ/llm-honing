"""Smoke tests for the installed package."""

from __future__ import annotations

import subprocess
import sys

import pytest

import honing

# ``sys.modules[name] = None`` makes ``import name`` fail as if it were not installed.
WITHOUT_LITELLM = """
import sys
sys.modules["litellm"] = None
import honing, honing.core, honing.harness.models
try:
    honing.harness.models.LiteLLMChatModel("openai/gpt-4o-mini")
except ImportError as error:
    print(error)
"""


@pytest.mark.unit
def test_the_package_imports_and_reports_a_version() -> None:
    assert isinstance(honing.__version__, str)
    assert honing.__version__


@pytest.mark.unit
def test_the_package_docstring_names_the_planes() -> None:
    # The module docstring is the first thing a reader of the source sees; it
    # must stay in step with the planes the specification defines.
    docstring = honing.__doc__ or ""
    for plane in ("core", "compression", "train", "data", "harness", "campaign", "serve"):
        assert f"``{plane}``" in docstring, f"the {plane!r} plane is not described"


@pytest.mark.integration  # starts a subprocess
def test_the_package_imports_without_the_litellm_extra() -> None:
    result = subprocess.run([sys.executable, "-c", WITHOUT_LITELLM], capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LiteLLMChatModel needs the 'litellm' extra: pip install 'llm-honing[litellm]'"
