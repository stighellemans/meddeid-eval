from __future__ import annotations

from meddeid_eval.stability.dates import (
    format_variants,
    looks_like_date,
    shift_years,
    value_shift_variants,
    years_in_span,
)


def test_shift_years_preserves_format():
    assert shift_years("12-05-1983", 1) == "12-05-1984"
    assert shift_years("opname 2001, controle 2003", 10) == "opname 2011, controle 2013"


def test_years_in_span():
    assert years_in_span("12-05-1983") == [1983]
    assert years_in_span("geen jaar hier") == []


def test_value_shift_variants_range_and_skip_identity():
    out = value_shift_variants("geb. 1983", 1980, 1985, 1)
    years = [y for _, y in out]
    # 1980..1985 minus the identity (1983)
    assert years == [1980, 1981, 1982, 1984, 1985]
    assert ("geb. 1980", 1980) in out


def test_value_shift_variants_empty_without_year():
    assert value_shift_variants("12/05", 1980, 2000, 5) == []


def test_format_variants_changes_format_same_value():
    out = format_variants("12 mei 1983", "Date")
    assert out, "expected at least one alternative format"
    for rendered, profile in out:
        assert rendered != "12 mei 1983"


def test_looks_like_date():
    assert looks_like_date("12-05-1983")
    assert looks_like_date("12 mei 1983")
    assert not looks_like_date("43 jr")
    assert not looks_like_date("volgende week")
