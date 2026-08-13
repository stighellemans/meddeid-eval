"""Date perturbation: value-shift (year) and format re-rendering.

- **value-shift** reuses the robustness job's ``shift_years`` (regex year bump),
  which preserves the exact written format and only changes the year number.
- **format** re-renders the same underlying date value across Dutch format
  profiles, via the vendored ``date_age_variants`` (``parse_date_text`` +
  ``format_named_date_profile``) from the canonical synthetic-data helpers.
"""
from __future__ import annotations

import re

from meddeid_language_nl.date_age_variants import (
    age_text_variant,
    format_named_date_profile,
    parse_date_text,
)

YEAR_RE = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-9]{2})(?!\d)")

# Day-precision Dutch format profiles (from date_age_variants.format_named_date_profile).
DATE_FORMAT_PROFILES = [
    "numeric_slash_long", "numeric_dash_long", "numeric_dot_long",
    "textual_full", "textual_abbr_dot", "textual_hyphen",
    "weekday_textual", "numeric_slash_short",
]
AGE_VARIANT_COUNT = 6


def years_in_span(text: str) -> list[int]:
    return [int(m.group(1)) for m in YEAR_RE.finditer(text)]


def shift_years(text: str, delta: int) -> str:
    return YEAR_RE.sub(lambda m: f"{int(m.group(1)) + delta:04d}", text)


def value_shift_variants(span_text: str, year_min: int, year_max: int, step: int) -> list[tuple[str, int]]:
    """(variant_text, transformed_year) for each target year in range, holding
    the written format fixed. Empty when the span carries no 4-digit year."""
    years = years_in_span(span_text)
    if not years:
        return []
    base = years[0]
    out: list[tuple[str, int]] = []
    for target in range(year_min, year_max + 1, max(1, step)):
        delta = target - base
        shifted = shift_years(span_text, delta)
        if shifted != span_text:
            out.append((shifted, target))
    return out


def format_variants(span_text: str, label: str = "Date", profiles: list[str] | None = None) -> list[tuple[str, str]]:
    """(variant_text, profile_name) re-rendering the *same* date value in other
    formats. Empty when the text can't be parsed to a concrete calendar date."""
    parsed = parse_date_text(span_text, label=label)
    if parsed is None or parsed.precision == "relative":
        return []
    value = parsed.value
    out: list[tuple[str, str]] = []
    seen = {span_text.strip()}
    for profile in (profiles or DATE_FORMAT_PROFILES):
        try:
            rendered = format_named_date_profile(value, profile)
        except (IndexError, ValueError):
            continue
        if rendered and rendered not in seen:
            seen.add(rendered)
            out.append((rendered, profile))
    return out


def age_variants(span_text: str, count: int = AGE_VARIANT_COUNT) -> list[tuple[str, str]]:
    """(variant_text, 'age:i') for age phrasings like '43 jr' that aren't dates."""
    out: list[tuple[str, str]] = []
    seen = {span_text.strip()}
    for i in range(count):
        rendered = age_text_variant(span_text, i)
        if rendered and rendered not in seen:
            seen.add(rendered)
            out.append((rendered, f"age_{i}"))
    return out


def looks_like_date(span_text: str, label: str = "Date") -> bool:
    parsed = parse_date_text(span_text, label=label)
    return parsed is not None and parsed.precision != "relative"
