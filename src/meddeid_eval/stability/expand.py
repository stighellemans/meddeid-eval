"""Stage A — one-factor-at-a-time (OFAT) perturbation.

For every target span we emit a **baseline** variant (original text) plus, for
each enabled dimension, one variant per dimension value with exactly that span
rewritten. Perturbing a single target at a time isolates the measured effect and
keeps counts additive. Every emitted variant is offset-checked
(``text[begin:end] == target.text``) before it is written.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from . import names as N
from .config import StabilityConfig
from .dataset import coverage, load_rows
from .dates import age_variants, format_variants, looks_like_date, value_shift_variants
from .io import write_json, write_jsonl
from .lookups import NameLookups, load_lookups
from .offsets import assert_offsets, replace_one
from .spans import aliases_for, canonical_label, doc_id_of, label_matches, raw_spans, role_of


def _target(span: dict[str, Any], *, role: str, kind: str, span_index: int,
            dimension: str, value: str, base_text: str) -> dict[str, Any]:
    return {
        "begin": int(span["begin"]), "end": int(span["end"]),
        "label": span["label"], "text": span["text"],
        "role": role, "kind": kind, "span_index": span_index,
        "dimension": dimension, "dimension_value": value, "base_text": base_text,
    }


def _emit(document_id: str, dataset: str, text: str, target: dict[str, Any]) -> dict[str, Any]:
    vid = f"{document_id}::{target['span_index']}::{target['dimension']}::{target['dimension_value']}"
    return {"variant_id": vid, "document_id": document_id, "dataset": dataset, "text": text, "target": target}


def _name_variants(text, base_span, parts, role, span_index, cfg, lookups, rng, stats) -> Iterator[dict]:
    doc_id = base_span["_doc_id"]
    dataset = base_span["_dataset"]
    base_text = base_span["text"]
    seen = {base_text}

    def emit(new_text, new_span, dimension, value):
        assert_offsets(new_text, new_span)
        stats[f"name::{dimension}"] += 1
        yield _emit(doc_id, dataset, new_text,
                    _target(new_span, role=role, kind="name", span_index=span_index,
                            dimension=dimension, value=value, base_text=base_text))

    # baseline (original text/span unchanged)
    stats["name::baseline"] += 1
    yield _emit(doc_id, dataset, text,
                _target(base_span, role=role, kind="name", span_index=span_index,
                        dimension="baseline", value="original", base_text=base_text))

    # capitalization — same name & format, different casing
    canonical = N.format_name(parts, parts.first_name, parts.last_name)
    for mode in cfg.name.capitalization:
        repl = N.recase(canonical, mode)
        if not repl or repl in seen:
            continue
        seen.add(repl)
        new_text, new_span = replace_one(text, base_span, repl)
        yield from emit(new_text, new_span, "capitalization", mode)

    # format — same name, different composition (proper casing)
    for fmt in cfg.name.formats:
        repl = N.render_format(parts, fmt, title=cfg.name.title)
        if not repl or repl in seen:
            continue
        seen.add(repl)
        new_text, new_span = replace_one(text, base_span, repl)
        yield from emit(new_text, new_span, "format", fmt)

    # name source = other nl-BE profile names, keeping the original format
    other_seen: set[str] = set()
    for trial in range(cfg.name.other_trials):
        first, last = N.choose_generated_name(parts, lookups, rng)
        repl = N.format_name(parts, first, last)
        if not repl or repl == base_text or repl in other_seen:
            continue
        other_seen.add(repl)
        new_text, new_span = replace_one(text, base_span, repl)
        assert_offsets(new_text, new_span)
        stats["name::other"] += 1
        yield _emit(doc_id, dataset, new_text,
                    _target(new_span, role=role, kind="name", span_index=span_index,
                            dimension="name_source", value=f"other_{trial}", base_text=base_text))


def _date_variants(text, base_span, role, span_index, cfg, stats) -> Iterator[dict]:
    doc_id = base_span["_doc_id"]
    dataset = base_span["_dataset"]
    base_text = base_span["text"]
    label = base_span["label"]
    is_date = looks_like_date(base_text, label)
    kind = "date" if is_date else "age"
    role = role if is_date else "age"

    # baseline (tagged with the resolved kind so counts line up)
    stats[f"{kind}::baseline"] += 1
    yield _emit(doc_id, dataset, text,
                _target(base_span, role=role, kind=kind, span_index=span_index,
                        dimension="baseline", value="original", base_text=base_text))

    if is_date:
        vs = cfg.date.value_shift
        for repl, year in value_shift_variants(base_text, vs.year_min, vs.year_max, vs.step):
            new_text, new_span = replace_one(text, base_span, repl)
            assert_offsets(new_text, new_span)
            stats["date::value_shift"] += 1
            yield _emit(doc_id, dataset, new_text,
                        _target(new_span, role=role, kind=kind, span_index=span_index,
                                dimension="date_value_shift", value=str(year), base_text=base_text))
        for repl, profile in format_variants(base_text, label, cfg.date.formats):
            new_text, new_span = replace_one(text, base_span, repl)
            assert_offsets(new_text, new_span)
            stats["date::format"] += 1
            yield _emit(doc_id, dataset, new_text,
                        _target(new_span, role=role, kind=kind, span_index=span_index,
                                dimension="date_format", value=profile, base_text=base_text))
    else:
        # age phrasing (e.g. "43 jr") — no date value to shift; vary the wording
        for repl, tag in age_variants(base_text):
            new_text, new_span = replace_one(text, base_span, repl)
            assert_offsets(new_text, new_span)
            stats["age::format"] += 1
            yield _emit(doc_id, dataset, new_text,
                        _target(new_span, role="age", kind="age", span_index=span_index,
                                dimension="age_format", value=tag, base_text=base_text))


def iter_variants(rows: list[dict[str, Any]], cfg: StabilityConfig, lookups: NameLookups,
                  stats: Counter) -> Iterator[dict]:
    dataset = cfg.dataset.stem
    name_aliases = aliases_for(cfg.name.labels) if cfg.name.enabled else set()
    date_aliases = aliases_for(list(cfg.date.labels) + list(cfg.date.age_labels)) if cfg.date.enabled else set()
    name_patterns = set(cfg.name.patterns)
    rng = random.Random(cfg.seed)

    for idx, row in enumerate(rows):
        text = str(row.get("text", ""))
        doc_id = doc_id_of(row, idx)
        span_index = -1
        for span in raw_spans(row):
            try:
                begin, end = int(span.get("begin", span.get("start", -1))), int(span.get("end", -1))
            except (TypeError, ValueError):
                continue
            if begin < 0 or end <= begin or end > len(text):
                continue
            label = canonical_label(span)
            base = {"begin": begin, "end": end, "label": label, "text": text[begin:end],
                    "_doc_id": doc_id, "_dataset": dataset}

            if cfg.name.enabled and label_matches(span, name_aliases):
                parts = N.parse_name_span(base["text"], lookups, enabled_patterns=name_patterns)
                if parts is None:
                    stats["name::parse_failed"] += 1
                    continue
                span_index += 1
                role = role_of(label) or "name"
                yield from _name_variants(text, base, parts, role, span_index, cfg, lookups, rng, stats)
            elif cfg.date.enabled and label_matches(span, date_aliases):
                span_index += 1
                role = "date"
                yield from _date_variants(text, base, role, span_index, cfg, stats)


def run_expand(cfg: StabilityConfig, dry_run: bool = False) -> dict[str, Any]:
    rows = load_rows(cfg.dataset, cfg.text_source)
    if cfg.max_docs > 0:
        rows = rows[: cfg.max_docs]
    cov = coverage(rows)
    if cov["without_text"]:
        print(f"[expand] warning: {cov['without_text']}/{cov['rows']} docs have no text "
              f"(set `text_source` in the config to join a document_id->text map)", flush=True)
    lookups = load_lookups()
    stats: Counter = Counter()

    out_path = cfg.output_dir / "variants.jsonl"
    if dry_run:
        # count without writing
        for _ in iter_variants(rows, cfg, lookups, stats):
            stats["variants_total"] += 1
        return {"dry_run": True, "input": str(cfg.dataset), "coverage": cov,
                "lookup_source": lookups.source, "counts": dict(stats)}

    def gen() -> Iterator[dict]:
        for v in iter_variants(rows, cfg, lookups, stats):
            stats["variants_total"] += 1
            yield v

    write_jsonl(out_path, gen())
    summary = {
        "input": str(cfg.dataset), "coverage": cov, "variants": str(out_path),
        "lookup_source": lookups.source, "seed": cfg.seed, "counts": dict(stats),
    }
    write_json(cfg.output_dir / "expand_summary.json", summary)
    return summary
