from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .metrics import score_documents

DEFAULT_BASELINE_MODELS = {
    "nl-BE": "stighellemans/meddeid-dutch-synth",
    "nl-NL": "stighellemans/meddeid-dutch-synth",
    "en-GB": "stighellemans/meddeid-english-synth",
    "en-US": "stighellemans/meddeid-english-synth",
}


def read_jsonl(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _profiles_from_gold(
    rows: list[dict], explicit_profile: str | None = None
) -> tuple[str, ...]:
    if explicit_profile:
        return (explicit_profile.replace("_", "-"),)
    profiles = {
        str(row.get("metadata", {}).get("lang") or "").replace("_", "-") for row in rows
    }
    profiles.discard("")
    if not profiles:
        raise ValueError(
            "could not infer a language profile from gold metadata.lang; "
            "pass --language-profile"
        )
    return tuple(sorted(profiles))


def _default_baseline_model(profiles: tuple[str, ...]) -> str:
    missing = [
        profile for profile in profiles if profile not in DEFAULT_BASELINE_MODELS
    ]
    if missing:
        raise ValueError(
            "no public baseline is registered for language profile(s): "
            + ", ".join(missing)
        )
    models = {DEFAULT_BASELINE_MODELS[profile] for profile in profiles}
    if len(models) != 1:
        raise ValueError(
            "the gold data requires multiple public baseline models; "
            "split the evaluation by language profile"
        )
    return models.pop()


def _write_score(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _prediction_manifest(path: str | Path) -> dict:
    prediction_path = Path(path)
    manifest_path = prediction_path.with_name(prediction_path.name + ".manifest.json")
    if not manifest_path.is_file():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _run_metadata(name: str, manifest: dict, **extra: object) -> dict:
    model = manifest.get("model", {})
    runtime = manifest.get("runtime", {})
    timing = manifest.get("timing", {})
    return {
        key: value
        for key, value in {
            "name": name,
            "method_type": "neural",
            "seconds": timing.get("elapsed_seconds"),
            "device": runtime.get("device"),
            "model": model.get("source") or model.get("name"),
            "revision": model.get("resolved_revision"),
            "bundle_sha256": model.get("bundle_sha256"),
            **extra,
        }.items()
        if value is not None
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meddeid-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--gold", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--output")
    score.add_argument("--name", help="system/run name stored in the score artifact")
    score.add_argument(
        "--seconds", type=float, help="optional measured evaluation time"
    )
    score.add_argument(
        "--device", help="optional hardware label, for example cpu or gpu"
    )
    score.add_argument(
        "--method-type", choices=("human", "rule", "neural", "generative", "unknown")
    )
    battery = sub.add_parser(
        "battery",
        help="run the paper-style evaluation battery with a profile baseline",
    )
    for evaluation in (battery,):
        evaluation.add_argument("--gold", required=True)
        evaluation.add_argument("--predictions", required=True)
        evaluation.add_argument("--name", default="candidate")
        evaluation.add_argument("--output-dir", required=True)
        evaluation.add_argument(
            "--language-profile", choices=tuple(DEFAULT_BASELINE_MODELS)
        )
        evaluation.add_argument(
            "--baseline-model",
            help="override the public baseline inferred from the language profile",
        )
        evaluation.add_argument("--baseline-revision")
        evaluation.add_argument("--device", choices=("cpu", "mps", "cuda"))
        evaluation.add_argument("--bootstrap-replicates", type=int, default=10_000)
        evaluation.add_argument("--bootstrap-seed", type=int, default=20_260_821)
        evaluation.add_argument(
            "--formats", default="png,pdf", help="comma-separated png,pdf,svg"
        )
        evaluation.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    stability = sub.add_parser("stability")
    stability.add_argument("args", nargs=argparse.REMAINDER)
    plot = sub.add_parser("plot", help="render comparison plots from score artifacts")
    plot.add_argument("--scores", nargs="+", required=True, help="score JSON files")
    plot.add_argument("--output-dir", required=True)
    plot.add_argument(
        "--formats", default="png,pdf", help="comma-separated png,pdf,svg"
    )
    plot.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    pseudonymization = sub.add_parser(
        "pseudonymization",
        help="evaluate date/age pseudonymization from predicted spans",
    )
    pseudonymization.add_argument("--gold", required=True)
    pseudonymization.add_argument("--predictions", required=True)
    pseudonymization.add_argument("--output-dir", required=True)
    pseudonymization.add_argument(
        "--language-profile",
        required=True,
        choices=("nl-BE", "nl-NL", "en-GB", "en-US"),
    )
    pseudonymization.add_argument("--document-creation-date", required=True)
    pseudonymization.add_argument("--date-shift-days", required=True, type=int)
    pseudonymization.add_argument(
        "--name",
        default="predictions",
        help="privacy-safe source id stored in the aggregate methodology",
    )
    pseudonymization.add_argument(
        "--generate-missing-replacements",
        action="store_true",
        help=(
            "apply the selected MedDeID language profile when prediction spans "
            "do not already contain replacement values"
        ),
    )
    args = parser.parse_args(argv)
    if args.command == "stability":
        from .stability.cli import main as stability_main

        return stability_main(args.args)
    if args.command == "battery":
        from .battery import (
            document_clustered_bootstrap,
            validate_inputs,
            write_report,
            write_tables,
        )
        from .benchmark_plots import render_comparison_plots

        gold_rows = read_jsonl(args.gold)
        profiles = _profiles_from_gold(gold_rows, args.language_profile)
        baseline_model = args.baseline_model or _default_baseline_model(profiles)
        destination = Path(args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        baseline_predictions = destination / "baseline-predictions.jsonl"
        meddeid_executable = Path(sys.executable).with_name("meddeid")
        command = [
            str(meddeid_executable),
            "batch",
            args.gold,
            "--output",
            str(baseline_predictions),
            "--model",
            baseline_model,
            "--quiet",
            "--overwrite",
        ]
        if args.baseline_revision:
            command.extend(["--revision", args.baseline_revision])
        if len(profiles) == 1:
            command.extend(["--language-profile", profiles[0]])
        if args.device:
            command.extend(["--device", args.device])
        try:
            subprocess.run(command, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "public-baseline inference failed; install "
                "'meddeid-eval[infer,plots]' and verify model access"
            ) from exc

        candidate_rows = read_jsonl(args.predictions)
        baseline_rows = read_jsonl(str(baseline_predictions))
        candidate_manifest = _prediction_manifest(args.predictions)
        baseline_manifest = _prediction_manifest(baseline_predictions)
        validate_inputs(
            gold_rows,
            {"public baseline": baseline_rows, args.name: candidate_rows},
        )
        candidate_payload = score_documents(gold_rows, candidate_rows)
        candidate_payload["run"] = _run_metadata(args.name, candidate_manifest)
        baseline_payload = score_documents(gold_rows, baseline_rows)
        baseline_payload["run"] = _run_metadata(
            f"Public baseline ({', '.join(profiles)})",
            baseline_manifest,
            model=baseline_model,
        )
        candidate_score = destination / "candidate.json"
        baseline_score = destination / "baseline.json"
        _write_score(candidate_score, candidate_payload)
        _write_score(baseline_score, baseline_payload)
        bootstrap = document_clustered_bootstrap(
            gold_rows,
            {
                baseline_payload["run"]["name"]: baseline_rows,
                candidate_payload["run"]["name"]: candidate_rows,
            },
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
        _write_score(destination / "bootstrap.json", bootstrap)
        table_paths = write_tables(
            destination / "tables", [baseline_payload, candidate_payload], bootstrap
        )
        formats = [value.strip() for value in args.formats.split(",") if value.strip()]
        plot_paths = render_comparison_plots(
            [baseline_payload, candidate_payload],
            destination / "plots",
            formats=formats,
            dpi=args.dpi,
        )
        report_path = destination / "REPORT.md"
        write_report(
            report_path,
            profiles=profiles,
            baseline_model=baseline_model,
            payloads=[baseline_payload, candidate_payload],
            bootstrap=bootstrap,
        )
        comparison = {
            "language_profiles": list(profiles),
            "baseline_model": baseline_model,
            "baseline_resolved_revision": baseline_payload["run"].get("revision"),
            "baseline_score": str(baseline_score),
            "candidate_score": str(candidate_score),
            "bootstrap": str(destination / "bootstrap.json"),
            "tables": [str(path) for path in table_paths],
            "plots": [str(path) for path in plot_paths],
            "report": str(report_path),
        }
        (destination / "battery.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(comparison, indent=2))
        return 0
    if args.command == "plot":
        from .benchmark_plots import render_comparison_plots

        payloads = [
            json.loads(Path(path).read_text(encoding="utf-8")) for path in args.scores
        ]
        formats = [value.strip() for value in args.formats.split(",") if value.strip()]
        paths = render_comparison_plots(
            payloads, args.output_dir, formats=formats, dpi=args.dpi
        )
        print(json.dumps({"plots": [str(path) for path in paths]}, indent=2))
        return 0
    if args.command == "pseudonymization":
        from .pseudonymization import (
            EvaluationSettings,
            aggregate_tables,
            evaluate_predicted_pseudonymization,
            write_safe_export,
        )

        settings = EvaluationSettings(
            language_profile=args.language_profile,
            document_creation_date=args.document_creation_date,
            date_shift_days=args.date_shift_days,
            generate_missing_replacements=args.generate_missing_replacements,
        )
        outcomes = evaluate_predicted_pseudonymization(
            read_jsonl(args.gold), read_jsonl(args.predictions), settings
        )
        manifest = write_safe_export(
            args.output_dir, outcomes, settings, prediction_source=args.name
        )
        summary, _ = aggregate_tables(outcomes)
        print(
            json.dumps(
                {
                    "summary": summary,
                    "output_dir": str(Path(args.output_dir)),
                    "privacy_check": manifest["privacy_check"],
                },
                indent=2,
            )
        )
        return 0
    payload = score_documents(read_jsonl(args.gold), read_jsonl(args.predictions))
    if args.name or args.seconds is not None or args.device or args.method_type:
        payload["run"] = {
            key: value
            for key, value in {
                "name": args.name,
                "seconds": args.seconds,
                "device": args.device,
                "method_type": args.method_type,
            }.items()
            if value is not None
        }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        _write_score(Path(args.output), payload)
    else:
        print(rendered)
    return 0
