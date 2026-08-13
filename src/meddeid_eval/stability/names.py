"""Name parsing, composition rendering, and variant generation.

The parser (`parse_name_span`), composition renderer (`format_name`) and the
generated/donor helpers are lifted from the prior robustness job
(``robustness.py``) so name handling stays identical to what you validated there.
On top of that this module adds the explicit **capitalization** and **format**
renderers that realise the stability dimensions you asked for.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from .lookups import NameLookups

INITIAL_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ]\.?$")

DEFAULT_NAME_PATTERNS = [
    "first_last", "last_first", "first_only", "last_only", "last_comma_first",
    "initial_last", "last_initial", "last_comma_initial", "first_middle_initial_last",
]

# The stability dimensions (used by the expander and documented in configs).
CAPITALIZATION_MODES = ["full", "first_only", "lower", "upper"]
NAME_FORMATS = ["same", "full_name", "first_only", "initials_only", "first_initials", "title_dr"]
DEFAULT_TITLE = "dr."


@dataclass(frozen=True)
class NameParts:
    title: str
    first_name: str = ""
    last_name: str = ""
    order: str = "first_last"
    first_initial: str = ""
    first_style: str = ""
    last_style: str = ""
    initial_style: str = ""
    initial_has_period: bool = True


# --------------------------------------------------------------------------- #
# low-level helpers (verbatim from robustness.py)                             #
# --------------------------------------------------------------------------- #
def list_maps(values: list[str]) -> dict[str, str]:
    return {value.lower(): value for value in values}


def starts_with_lookup(text: str, values: list[str]) -> tuple[str, str] | None:
    lowered = text.lower()
    for value in sorted(values, key=len, reverse=True):
        raw = value.strip()
        if not raw:
            continue
        low = raw.lower()
        if lowered == low:
            return text[: len(raw)], ""
        if lowered.startswith(low + " "):
            return text[: len(raw)], text[len(raw):].strip()
        if raw.endswith(".") and lowered.startswith(low):
            return text[: len(raw)], text[len(raw):].strip()
    return None


def is_lookup_last_name(raw, surname_lookup, interfix_lookup, interfix_surname_lookup) -> bool:
    lowered = raw.lower().strip()
    if lowered in surname_lookup:
        return True
    parts = lowered.split()
    for split_at in range(1, len(parts)):
        interfix = " ".join(parts[:split_at])
        surname = " ".join(parts[split_at:])
        if interfix in interfix_lookup and (surname in surname_lookup or surname in interfix_surname_lookup):
            return True
    return False


def is_initial(raw: str) -> bool:
    return bool(INITIAL_RE.match(raw.strip()))


def clean_initial(raw: str) -> str:
    item = raw.strip()
    return item[0].upper() if item else ""


def has_initial_period(raw: str) -> bool:
    return raw.strip().endswith(".")


def apply_case_pattern(value: str, pattern: str) -> str:
    letters = [char for char in pattern if char.isalpha()]
    if not value or not letters:
        return value
    if all(char.isupper() for char in letters):
        return value.upper()
    if all(char.islower() for char in letters):
        return value.lower()
    return value


def format_initial(source: str, pattern: str, has_period: bool) -> str:
    letter = (source.strip() or "X")[0]
    letter = letter.lower() if (pattern and pattern[0].islower()) else letter.upper()
    return f"{letter}{'.' if has_period else ''}"


# --------------------------------------------------------------------------- #
# parser (verbatim logic from robustness.py)                                  #
# --------------------------------------------------------------------------- #
def parse_name_span(text, lookups: NameLookups, enabled_patterns: set[str] | None = None) -> NameParts | None:
    if enabled_patterns is None:
        enabled_patterns = set(DEFAULT_NAME_PATTERNS)
    stripped = " ".join(text.strip().split())
    if not stripped:
        return None

    titles: list[str] = []
    rest = stripped
    while True:
        title_match = starts_with_lookup(rest, lookups.prefixes)
        if title_match is None:
            break
        title, next_rest = title_match
        if not next_rest or next_rest == rest:
            break
        titles.append(title)
        rest = next_rest
    if not rest:
        return None

    first_lookup = list_maps(lookups.first_names)
    first_name_lookup = set(first_lookup)
    surname_lookup = set(list_maps(lookups.surnames))
    interfix_lookup = {item.lower() for item in lookups.interfixes}
    interfix_surname_lookup = {item.lower() for item in lookups.interfix_surnames}
    title = " ".join(titles)

    if "," in rest:
        left, right = [part.strip() for part in rest.split(",", 1)]
        right_first = starts_with_lookup(right, lookups.first_names)
        if ("last_comma_first" in enabled_patterns and right_first is not None and not right_first[1]
                and is_lookup_last_name(left, surname_lookup, interfix_lookup, interfix_surname_lookup)):
            return NameParts(title=title, first_name=first_lookup[right_first[0].lower()], last_name=left,
                             order="last_comma_first", first_style=right_first[0], last_style=left)
        if ("last_comma_initial" in enabled_patterns and is_initial(right)
                and is_lookup_last_name(left, surname_lookup, interfix_lookup, interfix_surname_lookup)):
            return NameParts(title=title, last_name=left, order="last_comma_initial",
                             first_initial=clean_initial(right), last_style=left, initial_style=right,
                             initial_has_period=has_initial_period(right))

    first_match = starts_with_lookup(rest, lookups.first_names)
    if first_match is not None:
        first_raw, remainder = first_match
        if not remainder and "first_only" in enabled_patterns:
            return NameParts(title=title, first_name=first_lookup[first_raw.lower()],
                             order="first_only", first_style=first_raw)
        if ("first_last" in enabled_patterns
                and is_lookup_last_name(remainder, surname_lookup, interfix_lookup, interfix_surname_lookup)):
            return NameParts(title=title, first_name=first_lookup[first_raw.lower()], last_name=remainder,
                             order="first_last", first_style=first_raw, last_style=remainder)
        remainder_parts = remainder.split()
        if ("first_middle_initial_last" in enabled_patterns and len(remainder_parts) >= 2
                and is_initial(remainder_parts[0])):
            last_name = " ".join(remainder_parts[1:])
            if is_lookup_last_name(last_name, surname_lookup, interfix_lookup, interfix_surname_lookup):
                return NameParts(title=title, first_name=first_lookup[first_raw.lower()], last_name=last_name,
                                 order="first_middle_initial_last", first_initial=clean_initial(remainder_parts[0]),
                                 first_style=first_raw, last_style=last_name, initial_style=remainder_parts[0],
                                 initial_has_period=has_initial_period(remainder_parts[0]))

    parts = rest.split()
    if len(parts) >= 2 and is_initial(parts[0]):
        last_name = " ".join(parts[1:])
        if ("initial_last" in enabled_patterns
                and is_lookup_last_name(last_name, surname_lookup, interfix_lookup, interfix_surname_lookup)):
            return NameParts(title=title, last_name=last_name, order="initial_last",
                             first_initial=clean_initial(parts[0]), last_style=last_name,
                             initial_style=parts[0], initial_has_period=has_initial_period(parts[0]))
    if len(parts) >= 2 and is_initial(parts[-1]):
        last_name = " ".join(parts[:-1])
        if ("last_initial" in enabled_patterns
                and is_lookup_last_name(last_name, surname_lookup, interfix_lookup, interfix_surname_lookup)):
            return NameParts(title=title, last_name=last_name, order="last_initial",
                             first_initial=clean_initial(parts[-1]), last_style=last_name,
                             initial_style=parts[-1], initial_has_period=has_initial_period(parts[-1]))
    if ("last_only" in enabled_patterns
            and is_lookup_last_name(rest, surname_lookup, interfix_lookup, interfix_surname_lookup)):
        return NameParts(title=title, last_name=rest, order="last_only", last_style=rest)

    for split_at in range(1, len(parts)):
        last_name = " ".join(parts[:split_at])
        first_name = " ".join(parts[split_at:])
        first_key = first_name.lower()
        if ("last_first" in enabled_patterns and first_key in first_name_lookup
                and is_lookup_last_name(last_name, surname_lookup, interfix_lookup, interfix_surname_lookup)):
            return NameParts(title=title, first_name=first_lookup[first_key], last_name=last_name,
                             order="last_first", first_style=first_name, last_style=last_name)
    return None


# --------------------------------------------------------------------------- #
# composition renderer (verbatim from robustness.py)                          #
# --------------------------------------------------------------------------- #
def format_name(parts: NameParts, first_name: str, last_name: str) -> str:
    first_name = apply_case_pattern(first_name, parts.first_style)
    last_name = apply_case_pattern(last_name, parts.last_style)
    initial_source = first_name or parts.first_name or parts.first_initial or "X"
    initial = format_initial(initial_source, parts.initial_style, parts.initial_has_period)
    if parts.order == "first_only":
        body = first_name
    elif parts.order == "last_only":
        body = last_name
    elif parts.order == "initial_last":
        body = f"{initial} {last_name}"
    elif parts.order == "last_initial":
        body = f"{last_name} {initial}"
    elif parts.order == "last_comma_initial":
        body = f"{last_name}, {initial}"
    elif parts.order == "last_comma_first":
        body = f"{last_name}, {first_name}"
    elif parts.order == "first_middle_initial_last":
        middle_initial = format_initial(parts.first_initial or initial_source, parts.initial_style, parts.initial_has_period)
        body = f"{first_name} {middle_initial} {last_name}"
    elif parts.order == "last_first":
        body = f"{last_name} {first_name}"
    else:
        body = f"{first_name} {last_name}"
    return f"{parts.title} {body}" if parts.title else body


# --------------------------------------------------------------------------- #
# generated / donor name helpers (verbatim from robustness.py)                #
# --------------------------------------------------------------------------- #
def choose_generated_name(parts: NameParts, lookups: NameLookups, rng: random.Random) -> tuple[str, str]:
    for _ in range(100):
        first_name = rng.choice(lookups.first_names)
        last_name = rng.choice(lookups.surnames)
        if parts.order == "first_only" and first_name.lower() != parts.first_name.lower():
            return first_name, ""
        if parts.order == "last_only" and last_name.lower() != parts.last_name.lower():
            return "", last_name
        if parts.order in {"initial_last", "last_initial", "last_comma_initial"}:
            if last_name.lower() != parts.last_name.lower():
                return first_name, last_name
        if (parts.order not in {"first_only", "last_only", "initial_last", "last_initial", "last_comma_initial"}
                and (first_name.lower() != parts.first_name.lower() or last_name.lower() != parts.last_name.lower())):
            return first_name, last_name
    if parts.order == "first_only":
        return rng.choice(lookups.first_names), ""
    if parts.order == "last_only":
        return "", rng.choice(lookups.surnames)
    return rng.choice(lookups.first_names), rng.choice(lookups.surnames)


# --------------------------------------------------------------------------- #
# NEW: explicit stability renderers (capitalization + format dimensions)      #
# --------------------------------------------------------------------------- #
def recase(value: str, mode: str) -> str:
    """Recolour casing while preserving separators/structure.

    full       -> Title Case each word ("jan janssens" -> "Jan Janssens")
    first_only -> only the first alphabetic char upper, rest lower
    lower      -> all lowercase
    upper      -> ALL UPPERCASE ("jan janssens" -> "JAN JANSSENS")
    """
    if mode == "lower":
        return value.lower()
    if mode == "upper":
        return value.upper()
    if mode == "full":
        return re.sub(r"[^\W\d_]+", lambda m: m.group(0).capitalize(), value)
    if mode == "first_only":
        lowered = value.lower()
        for i, ch in enumerate(lowered):
            if ch.isalpha():
                return lowered[:i] + ch.upper() + lowered[i + 1:]
        return lowered
    return value


def _initial(token: str, period: bool = True) -> str:
    token = token.strip()
    if not token:
        return ""
    return token[0].upper() + ("." if period else "")


def render_format(parts: NameParts, fmt: str, *, title: str = DEFAULT_TITLE) -> str | None:
    """Render the parsed name in an explicit target format (proper casing).

    Returns None when the format is inapplicable to the available name parts
    (e.g. ``full_name`` for a last-name-only span) so the caller can skip it.
    """
    first = parts.first_name.strip()
    last = parts.last_name.strip()
    if fmt == "same":
        return format_name(parts, parts.first_name, parts.last_name)
    if fmt == "full_name":
        if not (first and last):
            return None
        return f"{first} {last}"
    if fmt == "first_only":
        if not first:
            return None
        return first
    if fmt == "initials_only":
        fi, li = _initial(first), _initial(last)
        if not (fi and li):
            return None
        return f"{fi}{li}"
    if fmt == "first_initials":
        li = _initial(last)
        if not (first and li):
            return None
        return f"{first} {li}"
    if fmt == "title_dr":
        body = " ".join(t for t in (first, last) if t) or last or first
        if not body:
            return None
        return f"{title} {body}"
    return None
