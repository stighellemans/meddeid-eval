from meddeid_eval.metrics import score_documents


def test_score_documents_exact_and_character_metrics() -> None:
    gold = [
        {
            "document_id": "d1",
            "spans": [
                {
                    "begin": 0,
                    "end": 3,
                    "label": "Name:Patient",
                    "subannotations": [{"begin": 0, "end": 3, "category": "given"}],
                }
            ],
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
            "spans": [
                {
                    "begin": 0,
                    "end": 5,
                    "label": "Name:Patient",
                    "subannotations": [
                        {"begin": 0, "end": 3, "category": "given"},
                        {"begin": 3, "end": 5, "category": "formatting"},
                    ],
                }
            ],
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
            "subannotations": [{"begin": 0, "end": 4, "category": "name_identifier"}],
        }
    ]
    predicted = [
        {
            "document_id": "d1",
            "spans": [{"begin": 0, "end": 2, "label": "Name:Other"}],
        }
    ]

    assert score_documents(gold, predicted)["core_pii_recall"] == 1.0


def test_score_documents_emits_privacy_safe_detailed_tables() -> None:
    gold = [
        {
            "document_id": "d1",
            "text": "Jan bezocht UZA.",
            "spans": [
                {
                    "begin": 0,
                    "end": 3,
                    "label": "Name:Patient",
                    "subannotations": [{"begin": 0, "end": 3, "category": "given"}],
                },
                {
                    "begin": 12,
                    "end": 15,
                    "label": "Organization:Healthcare",
                    "subannotations": [
                        {"begin": 12, "end": 15, "category": "institution"}
                    ],
                },
            ],
        }
    ]
    predicted = [
        {
            "document_id": "d1",
            "spans": [
                {"begin": 0, "end": 3, "label": "Name:Other"},
                {"begin": 4, "end": 11, "label": "Profession"},
            ],
        }
    ]

    result = score_documents(gold, predicted)

    assert result["core_pii_recall"] == 0.5
    assert result["non_pii_redacted_chars"] == 7
    assert result["non_pii_redaction_rate"] == 7 / 10
    by_label = {
        row["gold_label"]: row for row in result["details"]["recall_by_gold_label"]
    }
    assert by_label["Name:Patient"]["core_pii_recall"] == 1.0
    assert by_label["Organization:Healthcare"]["core_pii_recall"] == 0.0
    by_category = {
        row["subannotation_category"]: row
        for row in result["details"]["recall_by_subannotation_category"]
    }
    assert by_category["given"]["matched_core_pii_chars"] == 3
    assert result["details"]["non_pii_redaction_by_predicted_label"] == [
        {"prediction_label": "Profession", "non_pii_redacted_chars": 7},
        {"prediction_label": "Name:Other", "non_pii_redacted_chars": 0},
    ]
    assert result["details"]["label_confusion_chars"] == [
        {"gold_label": "Name:Patient", "prediction_label": "Name:Other", "chars": 3}
    ]
    assert result["details"]["exact_label_confusion"] == [
        {
            "gold_label": "Name:Patient",
            "prediction_label": "Name:Other",
            "spans": 1,
        }
    ]
