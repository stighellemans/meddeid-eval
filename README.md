# meddeid-eval

Reproducible evaluation for clinical de-identification. `meddeid-eval` computes
exact-span metrics, character-level recall, core-PII recall, non-PII redaction
rate, and stability results from canonical MedDeID JSONL files.

See the [suite evaluation workflow](https://stighellemans.github.io/meddeid/workflows/train-and-evaluate/#evaluate-predictions)
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
meddeid-eval score \
  --gold meddeid-dutch-synthetic-benchmark.jsonl \
  --predictions predictions.jsonl \
  --name meddeid-dutch-synth \
  --seconds 18.4 \
  --device gpu \
  --output results/meddeid-dutch-synth.json
meddeid-eval stability expand --config stability.yaml
```

Gold and prediction files are matched by `document_id` and use half-open
`[begin, end)` Unicode-code-point offsets. The score command reports exact
precision, recall, and F1 together with character coverage and redaction
metrics.

The score artifact also contains privacy-safe aggregate tables for recall by
gold label, recall by sub-annotation category, non-PII redactions by predicted
label, character-level label overlap, and exact-boundary label confusion. Source
text and document identifiers are never included. `non_pii_redaction_rate` is the fraction of characters
outside annotated PII spans covered by a prediction.

### Predicted-span pseudonymization

Measure the actual downstream date/age failure rate from saved predictions,
without rerunning inference:

```bash
meddeid-eval pseudonymization \
  --gold meddeid-dutch-synthetic-benchmark.jsonl \
  --predictions predictions.jsonl \
  --language-profile nl-BE \
  --document-creation-date 2025-01-15 \
  --date-shift-days 371 \
  --name meddeid-dutch-synth \
  --output-dir results/pseudonymization
```

The headline denominator is every gold `Date` and `Age_Birthdate` span. A
target is end-to-end valid only when one predicted span fully covers it, assigns
the correct label, and contains the protocol-valid replacement produced by the
selected language profile. The report separately gives the residual-exposure
rate: a wrong-label placeholder can remove the identifier while still failing
the clinically useful pseudonymization protocol.

MedDeID inference output already contains each span's exact `replacement`.
For canonical predictions from another detector that contain boundaries and
labels but no replacements, add `--generate-missing-replacements` to apply the
current MedDeID substitution layer during evaluation. The command writes only
aggregate, controlled-vocabulary CSV/JSON files and verifies a privacy manifest;
source text, document identifiers, and row-level outcomes are never exported.

### Comparison plots

Record a unique `--name` in each score artifact and render one or more systems:

```bash
meddeid-eval plot \
  --scores results/meddeid-dutch-synth.json results/comparator.json \
  --output-dir results/plots
```

The command writes PNG and vector PDF by default: a performance overview,
gold-label and sub-annotation recall heatmaps, non-PII-redaction and exact-label
confusion heatmaps, and an accuracy-versus-runtime plot when `--seconds` is available. Use
`--formats png,pdf,svg` and `--dpi 600` to override export settings.

Core-PII recall is the label-agnostic fraction of protocol-defined core PII
characters covered by any predicted redaction. Each primary gold span owns a
nested `subannotations` list. `formatting`, `additional_info`, `medical_info`,
`title`, and `time` segments are excluded from the denominator.

Stability perturbations use the configured locale provider and its complete
packaged resources. Dutch supports explicit `nl-BE` and `nl-NL` locale
selection; English must be selected as
either `en-GB` or `en-US` because bare `en` is ambiguous:

```yaml
dataset: annotations.jsonl
output_dir: results/stability
language_profile: en-GB
```

The same provider owns name lookup selection and date/age interpretation, so
GB DMY and US MDY behavior cannot fall back to Dutch globals.

`stability analyze` writes semantically ordered grouped bars, a year-shift line
plot with note-cluster bootstrap intervals, and a paired degradation forest in
both PNG and PDF. It pools roles by counts, marks missing observations as
missing rather than zero, and reports pair and contributing-note counts.

For confirmatory claims spanning the three prespecified benchmarks, apply one
Benjamini-Hochberg family per model after all analyses finish:

```bash
meddeid-eval stability adjust \
  --scope uza=results/uza/stability_analysis.json \
  --scope synthetic=results/synthetic/stability_analysis.json \
  --scope primary-care=results/primary-care/stability_analysis.json \
  --output-dir results/adjusted
```

Raw analyses remain unchanged; adjusted copies, a JSON audit manifest, and a
flat CSV are written to the output directory.

To place BH-adjusted significance markers in the degradation forest, rerender
an adjusted analysis:

```bash
meddeid-eval stability plot \
  --analysis results/adjusted/uza.stability_analysis.adjusted.json \
  --output-dir results/adjusted/uza-plots
```

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
