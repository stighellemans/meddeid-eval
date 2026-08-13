"""Test fixtures. Adds src/ to the path so tests run without `pip install`."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from meddeid_eval.stability.lookups import NameLookups  # noqa: E402


@pytest.fixture
def lookups() -> NameLookups:
    """A small deterministic lookup set for focused name tests."""
    return NameLookups(
        prefixes=["dr.", "prof.", "dhr.", "mevr.", "dr", "prof"],
        first_names=["Jan", "Sofie", "Marie", "Piet", "Noor"],
        surnames=["Janssens", "Peeters", "Mertens", "Claes"],
        interfixes=["van", "van den", "de"],
        interfix_surnames=["Berg", "Velde"],
        source="test-fixture",
    )
