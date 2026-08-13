from __future__ import annotations

from meddeid_eval.stability import names as N


def test_parse_first_last(lookups):
    parts = N.parse_name_span("Jan Janssens", lookups)
    assert parts is not None
    assert parts.order == "first_last"
    assert parts.first_name == "Jan"
    assert parts.last_name == "Janssens"


def test_parse_title(lookups):
    parts = N.parse_name_span("dr. Jan Janssens", lookups)
    assert parts is not None
    assert parts.title.lower().startswith("dr")
    assert parts.first_name == "Jan"


def test_parse_first_only_and_last_only(lookups):
    p1 = N.parse_name_span("Sofie", lookups)
    assert p1 is not None and p1.order == "first_only"
    p2 = N.parse_name_span("Peeters", lookups)
    assert p2 is not None and p2.order == "last_only"


def test_recase_modes():
    assert N.recase("Jan Janssens", "lower") == "jan janssens"
    assert N.recase("jan janssens", "full") == "Jan Janssens"
    assert N.recase("jan janssens", "first_only") == "Jan janssens"
    assert N.recase("jan janssens", "upper") == "JAN JANSSENS"
    # separators preserved
    assert N.recase("Janssens, Jan", "lower") == "janssens, jan"
    assert N.recase("Janssens, Jan", "upper") == "JANSSENS, JAN"


def test_render_formats(lookups):
    parts = N.parse_name_span("Jan Janssens", lookups)
    assert N.render_format(parts, "full_name") == "Jan Janssens"
    assert N.render_format(parts, "first_only") == "Jan"
    assert N.render_format(parts, "initials_only") == "J.J."
    assert N.render_format(parts, "first_initials") == "Jan J."
    assert N.render_format(parts, "title_dr", title="dr.") == "dr. Jan Janssens"


def test_render_format_inapplicable_returns_none(lookups):
    parts = N.parse_name_span("Peeters", lookups)  # last-only
    assert N.render_format(parts, "full_name") is None
    assert N.render_format(parts, "first_only") is None
    # title still works on the available body
    assert N.render_format(parts, "title_dr", title="dr.") == "dr. Peeters"


def test_format_name_roundtrips_original(lookups):
    parts = N.parse_name_span("Jan Janssens", lookups)
    assert N.format_name(parts, parts.first_name, parts.last_name) == "Jan Janssens"


def test_choose_generated_name_differs(lookups):
    import random
    parts = N.parse_name_span("Jan Janssens", lookups)
    first, last = N.choose_generated_name(parts, lookups, random.Random(0))
    assert (first.lower(), last.lower()) != ("jan", "janssens")
