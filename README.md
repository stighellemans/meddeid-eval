# meddeid-eval

Reproducible evaluation for clinical de-identification. `meddeid-eval` computes
exact-span metrics, character-level recall, core-PII recall, non-PII redaction
rate, and stability results from canonical MedDeID JSONL files.

See the [suite evaluation workflow](https://meddeid.github.io/workflows/train-and-evaluate/#evaluate-predictions)
for the end-to-end handoff. This repository remains authoritative for metric
definitions, commands, stability configuration, and plotting support.

## Installation

```bash
python -m pip install meddeid-eval
```

Optional extras add model inference and plotting support:

```bash
python -m pip install 'meddeid-eval[infer,plots]'
```

## Usage

```bash
meddeid-eval score --gold meddeid-dutch-synthetic-benchmark.jsonl --predictions predictions.jsonl
meddeid-eval stability expand --config stability.yaml
```

Gold and prediction files are matched by `document_id` and use half-open
`[begin, end)` Unicode-code-point offsets. The score command reports exact
precision, recall, and F1 together with character coverage and redaction
metrics.

Core-PII recall is the label-agnostic fraction of protocol-defined core PII
characters covered by any predicted redaction. Each primary gold span owns a
nested `subannotations` list. `formatting`, `additional_info`, `medical_info`,
`title`, and `time` segments are excluded from the denominator.

Stability perturbations use the complete `nl-BE` resources from
`meddeid-language-nl`; incomplete resources are reported as installation
errors.

## External comparators

Comparison systems run in their own environments. Export their predictions in
the canonical MedDeID JSONL schema and evaluate them with the same `score`
command. Belgian DEDUCE is not installed by `meddeid-eval`.

## Development

```bash
pip install -e '.[dev]'
pytest
```

## Licence

AGPL-3.0-only. External comparison systems retain their own licence terms.
