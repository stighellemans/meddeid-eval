from meddeid_eval.metrics import score_documents


def test_score_documents_exact_and_character_metrics() -> None:
    gold = [
        {
            "document_id": "d1",
            "spans": [{
                "begin": 0,
                "end": 3,
                "label": "Name:Patient",
                "subannotations": [{"begin": 0, "end": 3, "category": "given"}],
            }],
        }
    ]
    predicted = [
        {
            "document_id": "d1",
            "spans": [{"begin": 0, "end": 3, "label": "Name:Patient"}],
        }
    ]
    result = score_documents(gold, predicted)
    assert result["exact_f1"] == 1.0
    assert result["character_recall"] == 1.0
    assert result["core_pii_recall"] == 1.0


def test_core_pii_recall_uses_subannotations_and_is_label_agnostic() -> None:
    gold = [
        {
            "document_id": "d1",
            "spans": [{
                "begin": 0,
                "end": 5,
                "label": "Name:Patient",
                "subannotations": [
                    {"begin": 0, "end": 3, "category": "given"},
                    {"begin": 3, "end": 5, "category": "formatting"},
                ],
            }],
        }
    ]
    predicted = [
        {
            "document_id": "d1",
            "spans": [{"begin": 0, "end": 3, "label": "Name:Other"}],
        }
    ]

    result = score_documents(gold, predicted)

    assert result["character_recall"] == 0.0
    assert result["core_pii_recall"] == 1.0


def test_core_pii_recall_reads_legacy_top_level_subannotations() -> None:
    gold = [
        {
            "document_id": "d1",
            "spans": [{"begin": 0, "end": 4, "label": "Name:Patient"}],
            "subannotations": [
                {"begin": 0, "end": 3, "category": "name_identifier"},
                {"begin": 3, "end": 4, "category": "formatting"},
            ],
        }
    ]
    predicted = [
        {
            "document_id": "d1",
            "spans": [{"begin": 0, "end": 2, "label": "Name:Other"}],
        }
    ]

    result = score_documents(gold, predicted)

    assert result["core_pii_recall"] == 2 / 3


def test_nested_subannotations_take_precedence_over_legacy_top_level_data() -> None:
    gold = [
        {
            "document_id": "d1",
            "spans": [
                {
                    "begin": 0,
                    "end": 4,
                    "label": "Name:Patient",
                    "subannotations": [
                        {"begin": 0, "end": 2, "category": "name_identifier"}
                    ],
                }
            ],
            "subannotations": [
                {"begin": 0, "end": 4, "category": "name_identifier"}
            ],
        }
    ]
    predicted = [
        {
            "document_id": "d1",
            "spans": [{"begin": 0, "end": 2, "label": "Name:Other"}],
        }
    ]

    assert score_documents(gold, predicted)["core_pii_recall"] == 1.0
