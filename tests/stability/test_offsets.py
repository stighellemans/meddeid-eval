from __future__ import annotations

import pytest

from meddeid_eval.stability.offsets import apply_replacements, assert_offsets, offsets_valid, replace_one


def test_replace_one_remaps_offsets():
    text = "Hi Jan Janssens."
    span = {"begin": 3, "end": 15, "label": "Name:Patient", "text": "Jan Janssens"}
    assert text[3:15] == "Jan Janssens"
    new_text, new_span = replace_one(text, span, "Piet Peeters")
    assert new_text == "Hi Piet Peeters."
    assert new_text[new_span["begin"]:new_span["end"]] == "Piet Peeters"
    assert_offsets(new_text, new_span)


def test_replace_one_shorter_and_longer():
    text = "a XX b"
    span = {"begin": 2, "end": 4, "label": "Date", "text": "XX"}
    for repl in ("Y", "ZZZZZ", ""):
        nt, ns = replace_one(text, span, repl)
        assert nt[ns["begin"]:ns["end"]] == repl
        assert offsets_valid(nt, ns) or repl == ""


def test_apply_replacements_multi_span_orders_and_remaps():
    text = "AAA name1 BBB name2 CCC"
    s1 = {"begin": 4, "end": 9, "label": "Name:Patient", "text": "name1"}
    s2 = {"begin": 14, "end": 19, "label": "Name:Caregiver", "text": "name2"}
    nt, spans = apply_replacements(text, [s1, s2], ["X", "YYYY"])
    assert nt == "AAA X BBB YYYY CCC"
    for s in spans:
        assert_offsets(nt, s)


def test_overlapping_replacements_raise():
    text = "abcdef"
    s1 = {"begin": 0, "end": 4, "label": "X", "text": "abcd"}
    s2 = {"begin": 2, "end": 6, "label": "X", "text": "cdef"}
    with pytest.raises(ValueError):
        apply_replacements(text, [s1, s2], ["1", "2"])
