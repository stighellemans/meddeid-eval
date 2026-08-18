from __future__ import annotations

import json

import pytest

from meddeid_eval.cli import main


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_score_metadata_and_plot_cli_round_trip(tmp_path, capsys) -> None:
    pytest.importorskip("matplotlib")
    gold_path = tmp_path / "gold.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    score_path = tmp_path / "score.json"
    plot_dir = tmp_path / "plots"
    gold = [
        {
            "document_id": "d1",
            "text": "Jan werkt.",
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
    _write_jsonl(gold_path, gold)
    _write_jsonl(
        predictions_path,
        [
            {
                "document_id": "d1",
                "spans": [{"begin": 0, "end": 3, "label": "Name:Patient"}],
            }
        ],
    )

    assert (
        main(
            [
                "score",
                "--gold",
                str(gold_path),
                "--predictions",
                str(predictions_path),
                "--output",
                str(score_path),
                "--name",
                "Test system",
                "--seconds",
                "1.25",
                "--device",
                "cpu",
                "--method-type",
                "neural",
            ]
        )
        == 0
    )
    score = json.loads(score_path.read_text(encoding="utf-8"))
    assert score["run"] == {
        "device": "cpu",
        "method_type": "neural",
        "name": "Test system",
        "seconds": 1.25,
    }
    assert score["details"]["exact_label_confusion"][0]["spans"] == 1

    assert (
        main(
            [
                "plot",
                "--scores",
                str(score_path),
                "--output-dir",
                str(plot_dir),
                "--formats",
                "png",
                "--dpi",
                "120",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert {path.rsplit("/", 1)[-1] for path in summary["plots"]} >= {
        "performance_overview.png",
        "recall_by_gold_label.png",
        "recall_by_subannotation.png",
        "exact_label_confusion.png",
        "accuracy_vs_runtime.png",
    }
