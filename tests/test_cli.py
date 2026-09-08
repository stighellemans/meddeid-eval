from __future__ import annotations

import json
import shutil

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
        "exact_metrics_by_label.png",
        "recall_by_gold_label.png",
        "subannotation_coverage_matrix.png",
        "exact_label_confusion.png",
        "accuracy_vs_runtime.png",
    }


def test_score_creates_output_parent_directory(tmp_path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    score_path = tmp_path / "results" / "score.json"
    rows = [{"document_id": "d1", "text": "No identifiers.", "spans": []}]
    _write_jsonl(gold_path, rows)
    _write_jsonl(predictions_path, rows)

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
            ]
        )
        == 0
    )

    assert json.loads(score_path.read_text(encoding="utf-8"))["documents"] == 1


def test_battery_infers_public_baseline_from_gold_profile(
    tmp_path, monkeypatch, capsys
) -> None:
    pytest.importorskip("matplotlib")
    gold_path = tmp_path / "gold.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "comparison"
    gold = [
        {
            "document_id": "d1",
            "text": "Jan",
            "metadata": {"lang": "nl-BE"},
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
                "text": "Jan",
                "spans": [{"begin": 0, "end": 3, "label": "Name:Other"}],
            }
        ],
    )

    def fake_run(command, *, check):
        assert check is True
        assert "stighellemans/meddeid-dutch-synth" in command
        assert command[command.index("--language-profile") + 1] == "nl-BE"
        target = command[command.index("--output") + 1]
        shutil.copyfile(gold_path, target)

    monkeypatch.setattr("meddeid_eval.cli.subprocess.run", fake_run)

    assert (
        main(
            [
                "battery",
                "--gold",
                str(gold_path),
                "--predictions",
                str(predictions_path),
                "--name",
                "Fine-tuned",
                "--output-dir",
                str(output_dir),
                "--formats",
                "png",
                "--bootstrap-replicates",
                "20",
            ]
        )
        == 0
    )

    comparison = json.loads(capsys.readouterr().out)
    assert comparison["language_profiles"] == ["nl-BE"]
    assert comparison["baseline_model"] == "stighellemans/meddeid-dutch-synth"
    assert json.loads((output_dir / "baseline.json").read_text())["exact_f1"] == 1.0
    assert json.loads((output_dir / "candidate.json").read_text())["exact_f1"] == 0.0
    assert (output_dir / "plots" / "exact_label_confusion.png").is_file()
    assert (output_dir / "tables" / "bootstrap_estimates.csv").is_file()
    assert (output_dir / "tables" / "recall_by_subannotation_category.csv").is_file()
    assert (output_dir / "REPORT.md").is_file()


def test_pseudonymization_cli_reuses_saved_predictions(tmp_path, capsys) -> None:
    text = "Datum 01/02/2020."
    begin = text.index("01/02/2020")
    gold_path = tmp_path / "gold.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    output_dir = tmp_path / "pseudonymization"
    _write_jsonl(
        gold_path,
        [
            {
                "document_id": "d1",
                "text": text,
                "spans": [
                    {
                        "begin": begin,
                        "end": begin + len("01/02/2020"),
                        "label": "Date",
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        predictions_path,
        [
            {
                "document_id": "d1",
                "spans": [
                    {
                        "begin": begin,
                        "end": begin + len("01/02/2020"),
                        "label": "Date",
                    }
                ],
            }
        ],
    )

    assert (
        main(
            [
                "pseudonymization",
                "--gold",
                str(gold_path),
                "--predictions",
                str(predictions_path),
                "--output-dir",
                str(output_dir),
                "--language-profile",
                "nl-BE",
                "--document-creation-date",
                "2025-01-15",
                "--date-shift-days",
                "371",
                "--name",
                "test-run",
                "--generate-missing-replacements",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["privacy_check"] == "passed"
    assert payload["summary"][0]["end_to_end_failure_rate"] == 0.0
