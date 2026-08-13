"""Per-document metadata channel.

Known identifiers (patient / caregiver names, addresses, document date) can be
injected into comparator adapters and the language-profile post-processor. This
is the known-identifiers setting common in DEID.

Source is configured (``metadata.source`` in battery.yaml):
  - ``none``        : no metadata
  - ``from_input``  : each doc carries an explicit ``metadata`` key-value object
                      (shape below). This is the ONLY source -- metadata is never
                      derived from the gold annotation spans.

Canonical per-doc metadata shape::

    {
      "patient": {"given_name": "...", "family_name": "...",
                  "birth_date": "YYYY-MM-DD"},
      "caregivers": [{"given_name": "...", "family_name": "..."}, ...],
      "document_creation_date": "YYYY-MM-DD"
    }
"""
from __future__ import annotations

from typing import Any

from meddeid_core.normalize import normalize_metadata

from .schema import make_span


def _uniq(seq) -> list[str]:
    out, seen = [], set()
    for s in seq:
        s = (s or "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def resolve(doc: dict, cfg: dict) -> dict:
    """Return the per-doc metadata dict for the configured source. Metadata must
    be an explicit doc ``metadata`` object (``from_input``); it is never derived
    from the gold annotation spans."""
    source = (cfg or {}).get("source", "none")
    if source == "none":
        return {}
    if source == "from_input":
        return normalize_metadata(doc.get("metadata") or {})
    raise ValueError(
        f"unknown metadata.source: {source!r} (supported: 'none', 'from_input')")


# --- consumers -------------------------------------------------------------

def to_postprocess(meta: dict) -> dict:
    """Shape canonical identity metadata for the post-processor."""
    out: dict[str, Any] = {}
    p = meta.get("patient")
    if isinstance(p, dict) and p:
        out["patient"] = dict(p)
    cgs = meta.get("caregivers") or []
    if cgs:
        out["caregivers"] = [dict(c) for c in cgs if isinstance(c, dict)]
    for key in ("document_creation_date", "date_shift_days", "known_values"):
        if meta.get(key) is not None:
            out[key] = meta[key]
    return out


def _full_names(person: dict) -> list[str]:
    names = list(person.get("aliases") or [])
    full_name = " ".join(
        value
        for value in (person.get("given_name"), person.get("family_name"))
        if value
    ).strip()
    if full_name:
        names.append(full_name)
    return _uniq(names)


def inject_name_spans(spans: list[dict], text: str, meta: dict) -> list[dict]:
    """Metadata-driven recall boost: add patient/caregiver name spans (exact
    literal occurrences) that the model missed, without overlapping existing spans."""
    spans = list(spans)
    occupied = [(s["begin"], s["end"]) for s in spans]

    def add(names: list[str], label: str):
        for name in names:
            if len(name) < 2:
                continue
            start = 0
            while True:
                i = text.find(name, start)
                if i < 0:
                    break
                j = i + len(name)
                if not any(b < j and i < e for b, e in occupied):
                    spans.append(make_span(i, j, label, text[i:j]))
                    occupied.append((i, j))
                start = j

    if meta.get("patient"):
        add(_full_names(meta["patient"]), "Name:Patient")
    for c in meta.get("caregivers") or []:
        add(_full_names(c), "Name:Caregiver")
    return spans
