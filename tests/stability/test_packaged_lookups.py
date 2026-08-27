from meddeid_eval.stability.lookups import load_lookups
from meddeid_eval.stability.providers import get_locale_provider


def test_loads_full_packaged_nl_be_lookups() -> None:
    lookups = load_lookups()
    assert len(lookups.first_names) > 10_000
    assert len(lookups.surnames) > 10_000
    assert "meddeid-language-nl" in lookups.source
    assert "fallback" not in lookups.source.lower()


def test_loads_separate_packaged_english_lookups() -> None:
    gb = load_lookups(provider=get_locale_provider("en-GB"))
    us = load_lookups(provider=get_locale_provider("en-US"))
    assert len(gb.surnames) >= 1000
    assert len(us.surnames) >= 160000
    assert "Mac" in gb.interfixes
    assert "del" in us.interfixes
    assert "en-GB" in gb.source
    assert "en-US" in us.source
