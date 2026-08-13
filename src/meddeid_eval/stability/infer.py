"""Run stability variants through the installed MedDeID inference package."""

from __future__ import annotations

from .config import ModelCfg, StabilityConfig
from .io import read_jsonl, write_json, write_jsonl


def run_model(model: ModelCfg, cfg: StabilityConfig, variants: list[dict]) -> dict:
    try:
        from meddeid import Deidentifier
    except ImportError as exc:
        raise RuntimeError("install meddeid-eval[infer] for stability inference") from exc

    engine = Deidentifier.from_pretrained(
        model.checkpoint,
        device=None if cfg.device in ("auto", "", None) else cfg.device,
    )
    output = []
    try:
        for variant in variants:
            result = engine(variant["text"])
            output.append({"variant_id": variant["variant_id"], "spans": result.spans})
    finally:
        engine.close()
    out_path = cfg.output_dir / "predictions" / f"{model.id}.jsonl"
    write_jsonl(out_path, output)
    return {
        "model": model.id,
        "checkpoint": model.checkpoint,
        "variants": len(variants),
        "predictions": str(out_path),
        "total_predicted_spans": sum(len(item["spans"]) for item in output),
    }


def run_infer(cfg: StabilityConfig, only: list[str] | None = None) -> dict:
    variants = read_jsonl(cfg.output_dir / "variants.jsonl")
    models = [model for model in cfg.models if not only or model.id in only]
    if not models:
        raise ValueError("no configured models selected")
    results = [run_model(model, cfg, variants) for model in models]
    summary = {"models": results, "variants": len(variants)}
    write_json(cfg.output_dir / "inference_summary.json", summary)
    return summary
