from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import score_documents


def read_jsonl(path: str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    stability = sub.add_parser("stability")
    stability.add_argument("args", nargs=argparse.REMAINDER)
    plot = sub.add_parser("plot", help="render comparison plots from score artifacts")
    plot.add_argument("--scores", nargs="+", required=True, help="score JSON files")
    plot.add_argument("--output-dir", required=True)
    plot.add_argument(
        "--formats", default="png,pdf", help="comma-separated png,pdf,svg"
    )
    plot.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    args = parser.parse_args(argv)
    if args.command == "stability":
        from .stability.cli import main as stability_main

        return stability_main(args.args)
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
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0
