from __future__ import annotations

import pytest

from meddeid_eval.metadata import inject_name_spans, resolve, to_postprocess


def test_canonical_identity_metadata_reaches_postprocess_and_injection():
    metadata = {
        "patient": {
            "given_name": "Jan",
            "family_name": "Peeters",
            "birth_date": "1980-01-02",
        },
        "caregivers": [{"given_name": "Noor", "family_name": "Aerts"}],
    }
    resolved = resolve({"metadata": metadata}, {"source": "from_input"})

    assert to_postprocess(resolved)["patient"]["birth_date"] == "1980-01-02"
    spans = inject_name_spans([], "Jan Peeters sprak Noor Aerts.", resolved)
    assert {(span["text"], span["label"]) for span in spans} == {
        ("Jan Peeters", "Name:Patient"),
        ("Noor Aerts", "Name:Caregiver"),
    }


@pytest.mark.parametrize("retired_key", ["patient_name", "caregiver_names"])
def test_retired_identity_metadata_is_rejected(retired_key):
    with pytest.raises(ValueError, match="retired metadata key"):
        resolve({"metadata": {retired_key: {}}}, {"source": "from_input"})
