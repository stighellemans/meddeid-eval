import pytest

from meddeid_eval.span_edits import evaluate_span_edits


def test_span_edits_counts_add_delete_and_label_edit():
    result = evaluate_span_edits(
        [
            {"begin": 0, "end": 3, "label": "Name:Other"},
            {"begin": 8, "end": 9, "label": "Date"},
        ],
        [
            {"begin": 0, "end": 3, "label": "Name:Patient"},
            {"begin": 4, "end": 7, "label": "ID:Patient"},
        ],
    )
    assert result["counts"] == {"Addition": 1, "Deletion": 1, "Edit": 1, "total_ops": 3}


def test_span_edits_rejects_duplicate_boundaries():
    with pytest.raises(ValueError, match="duplicate span boundary"):
        evaluate_span_edits(
            [{"begin": 0, "end": 1, "label": "Date"}, {"begin": 0, "end": 1, "label": "Date"}],
            [],
        )
