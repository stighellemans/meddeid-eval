from meddeid_eval.stability.lookups import load_lookups


def test_loads_full_packaged_nl_be_lookups() -> None:
    lookups = load_lookups()
    assert len(lookups.first_names) > 10_000
    assert len(lookups.surnames) > 10_000
    assert "meddeid-language-nl" in lookups.source
    assert "fallback" not in lookups.source.lower()
