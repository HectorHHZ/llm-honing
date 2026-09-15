"""Smoke tests for the installed package."""

from __future__ import annotations

import pytest

import honing


@pytest.mark.unit
def test_the_package_imports_and_reports_a_version() -> None:
    assert isinstance(honing.__version__, str)
    assert honing.__version__


@pytest.mark.unit
def test_the_package_docstring_names_the_planes() -> None:
    # The module docstring is the first thing a reader of the source sees; it
    # must stay in step with the planes the specification defines.
    docstring = honing.__doc__ or ""
    for plane in ("prune", "train", "data", "harness", "serve"):
        assert f"``{plane}``" in docstring, f"the {plane!r} plane is not described"
