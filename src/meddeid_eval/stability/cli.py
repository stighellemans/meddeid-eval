"""Stability subcommand: expand, infer, analyze, adjust, or run."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .config import load_config


def _csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _print(summary: Any) -> None:
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


def _scope_files(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        scope, separator, path = value.partition("=")
        if not separator or not scope or not path:
            raise ValueError(f"invalid --scope {value!r}; expected NAME=PATH")
        if scope in result:
            raise ValueError(f"duplicate --scope {scope!r}")
        result[scope] = path
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meddeid-eval stability", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config", required=True, help="path to a stability YAML config"
        )
        p.add_argument(
            "--max-docs",
            type=int,
            default=None,
            help="override config max_docs (0 = all)",
        )

    p_expand = sub.add_parser("expand", help="perturb the dataset -> variants.jsonl")
    add_common(p_expand)
    p_expand.add_argument(
        "--dry-run", action="store_true", help="count variants without writing"
    )

    p_infer = sub.add_parser("infer", help="run models over variants -> predictions/")
    add_common(p_infer)
    p_infer.add_argument("--only", help="comma-separated model ids to run")

    p_analyze = sub.add_parser("analyze", help="predictions -> report + CSV + plots")
    add_common(p_analyze)
    p_analyze.add_argument("--only", help="comma-separated model ids to include")

    p_adjust = sub.add_parser("adjust", help="apply cross-scope BH correction")
    p_adjust.add_argument(
        "--scope",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="scope and stability_analysis.json; pass once per scope",
    )
    p_adjust.add_argument("--output-dir", required=True)

    p_plot = sub.add_parser("plot", help="render plots from a raw or adjusted analysis")
    p_plot.add_argument("--analysis", required=True, help="stability_analysis JSON")
    p_plot.add_argument("--output-dir", required=True)
    p_plot.add_argument("--formats", default="png,pdf")
    p_plot.add_argument("--dpi", type=int, default=300)

    p_run = sub.add_parser("run", help="expand + infer + analyze")
    add_common(p_run)
    p_run.add_argument("--only", help="comma-separated model ids")

    args = parser.parse_args(argv)
    if args.cmd == "adjust":
        from .multiple_testing import run_adjust

        try:
            _print(run_adjust(_scope_files(args.scope), args.output_dir))
        except ValueError as exc:
            parser.error(str(exc))
        return 0
    if args.cmd == "plot":
        from .io import read_json
        from .plots import render_stability_plots

        formats = [value.strip() for value in args.formats.split(",") if value.strip()]
        paths = render_stability_plots(
            args.output_dir, read_json(args.analysis), formats=formats, dpi=args.dpi
        )
        _print({"plots": [str(path) for path in paths]})
        return 0

    cfg = load_config(args.config)
    if args.max_docs is not None:
        cfg.max_docs = args.max_docs

    if args.cmd == "expand":
        from .expand import run_expand

        _print(run_expand(cfg, dry_run=args.dry_run))
    elif args.cmd == "infer":
        from .infer import run_infer

        _print(run_infer(cfg, only=_csv_list(args.only)))
    elif args.cmd == "analyze":
        from .analyze import run_analyze

        _print(run_analyze(cfg, only=_csv_list(args.only)))
    elif args.cmd == "run":
        from .analyze import run_analyze
        from .expand import run_expand
        from .infer import run_infer

        only = _csv_list(args.only)
        print("[1/3] expand", file=sys.stderr)
        _print(run_expand(cfg))
        print("[2/3] infer", file=sys.stderr)
        _print(run_infer(cfg, only=only))
        print("[3/3] analyze", file=sys.stderr)
        _print(run_analyze(cfg, only=only))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
