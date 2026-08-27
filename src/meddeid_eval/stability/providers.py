"""Locale providers for stability perturbations.

The stability helpers depend on this small adapter instead of importing one
language pack directly.  Imports remain lazy so users only need the packs for
the profiles they select.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable


DateVariants = Callable[[str, str, list[str] | None], list[tuple[str, str]]]
AgeVariants = Callable[[str, int], list[tuple[str, str]]]


@dataclass(frozen=True)
class LocaleProvider:
    profile_id: str
    parse_date: Callable[[str, str], object | None]
    date_variants: DateVariants
    age_variants: AgeVariants
    lookup_values: Callable[[str], tuple[str, ...]]
    lookup_source: Callable[[str | None], str]


def normalize_profile_id(profile_id: str) -> str:
    normalized = str(profile_id).strip().replace("_", "-").lower()
    if normalized == "nl-be":
        return "nl-BE"
    if normalized == "nl-nl":
        return "nl-NL"
    if normalized == "en-gb":
        return "en-GB"
    if normalized == "en-us":
        return "en-US"
    raise ValueError(
        f"unsupported stability language profile {profile_id!r}; "
        "available: nl-BE, nl-NL, en-GB, en-US"
    )


def _nl_provider(profile_id: str) -> LocaleProvider:
    from meddeid_language_nl import lookup_source, lookup_values
    from meddeid_language_nl.date_age_variants import (
        age_text_variant,
        format_named_date_profile,
        parse_date_text,
    )

    def parse(text: str, label: str) -> object | None:
        return parse_date_text(text, label=label)

    def date_variants(
        text: str, label: str, profiles: list[str] | None
    ) -> list[tuple[str, str]]:
        parsed = parse_date_text(text, label=label)
        if parsed is None or parsed.precision == "relative":
            return []
        profile_names = profiles or [
            "numeric_slash_long", "numeric_dash_long", "numeric_dot_long",
            "textual_full", "textual_abbr_dot", "textual_hyphen",
            "weekday_textual", "numeric_slash_short",
        ]
        out: list[tuple[str, str]] = []
        seen = {text.strip()}
        for profile in profile_names:
            try:
                rendered = format_named_date_profile(parsed.value, profile)
            except (IndexError, ValueError):
                continue
            if rendered and rendered not in seen:
                seen.add(rendered)
                out.append((rendered, profile))
        return out

    def age_variants(text: str, count: int) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        seen = {text.strip()}
        for index in range(count):
            rendered = age_text_variant(text, index)
            if rendered and rendered not in seen:
                seen.add(rendered)
                out.append((rendered, f"age_{index}"))
        return out

    return LocaleProvider(
        profile_id=profile_id,
        parse_date=parse,
        date_variants=date_variants,
        age_variants=age_variants,
        lookup_values=lambda category: tuple(lookup_values(profile_id, category)),
        lookup_source=lambda category=None: lookup_source(profile_id),
    )


def _english_provider(profile_id: str) -> LocaleProvider:
    from meddeid_language_en import (
        date_age_variants,
        lookup_source,
        lookup_values,
        parse_date_text,
    )

    def parse(text: str, label: str) -> object | None:
        return parse_date_text(text, profile_id=profile_id, label=label)

    def date_variants(
        text: str, label: str, profiles: list[str] | None
    ) -> list[tuple[str, str]]:
        del profiles  # Locale packs define their own safe, canonical formats.
        return [
            (rendered, f"{profile_id.lower()}_{index}")
            for index, rendered in enumerate(
                date_age_variants(text, profile_id=profile_id, label=label)
            )
        ]

    def age_variants(text: str, count: int) -> list[tuple[str, str]]:
        return [
            (rendered, f"age_{index}")
            for index, rendered in enumerate(
                date_age_variants(
                    text, profile_id=profile_id, label="Age_Birthdate"
                )[:count]
            )
        ]

    return LocaleProvider(
        profile_id=profile_id,
        parse_date=parse,
        date_variants=date_variants,
        age_variants=age_variants,
        lookup_values=lambda category: tuple(
            lookup_values(profile_id, category)
        ),
        lookup_source=lambda category=None: lookup_source(
            profile_id, category
        ),
    )


@lru_cache(maxsize=4)
def get_locale_provider(profile_id: str = "nl-BE") -> LocaleProvider:
    profile_id = normalize_profile_id(profile_id)
    if profile_id in {"nl-BE", "nl-NL"}:
        return _nl_provider(profile_id)
    return _english_provider(profile_id)
