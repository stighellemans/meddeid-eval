"""Measure how stable de-identification models are to name and date rewrites.

Three decoupled stages:

    expand   dataset JSONL  -> variants.jsonl   (perturbation, one-factor-at-a-time)
    infer    variants.jsonl -> predictions/*.jsonl  (installed meddeid package)
    analyze  predictions    -> stability_report.{md,json} + CSVs + plots

The perturbation and analysis stages are near-stdlib (+ PyYAML, optional matplotlib)
so they run on any device; only `infer` needs the torch/transformers/`meddeid`
stack. Device-specific paths (checkpoints, dataset) live entirely in the config.
"""

__version__ = "0.1.0"
