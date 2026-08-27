"""Locale-provider name lookups used by stability perturbations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .providers import LocaleProvider, get_locale_provider


@dataclass
class NameLookups:
    prefixes: list[str]
    first_names: list[str]
    surnames: list[str]
    interfixes: list[str]
    interfix_surnames: list[str]
    source: str


def _read_items(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            out.append(item)
    return out


def _from_names_dir(names_dir: Path, source: str) -> NameLookups | None:
    prefixes = _read_items(names_dir / "lst_prefix/items.txt")
    first_names = _read_items(names_dir / "lst_first_name/items.txt")
    surnames = _read_items(names_dir / "lst_surname/items.txt")
    if not (prefixes and first_names and surnames):
        return None
    return NameLookups(
        prefixes=prefixes,
        first_names=first_names,
        surnames=surnames,
        interfixes=_read_items(names_dir / "lst_interfix/items.txt"),
        interfix_surnames=_read_items(names_dir / "lst_interfix_surname/items.txt"),
        source=source,
    )


def load_lookups(
    lookup_dir: str | None = None, *, provider: LocaleProvider | None = None
) -> NameLookups:
    selected = provider or get_locale_provider()
    if lookup_dir:
        root = Path(lookup_dir).expanduser()
        for names_dir in (root, root / "names"):
            got = _from_names_dir(names_dir, str(names_dir))
            if got is not None:
                return got
        raise RuntimeError(
            f"no complete {selected.profile_id} name lookups under {root}"
        )

    if selected.profile_id in {"nl-BE", "nl-NL"}:
        interfixes = list(selected.lookup_values("interfixes"))
        interfix_surnames = list(selected.lookup_values("interfix_surnames"))
    else:
        interfixes = list(selected.lookup_values("surname_particles"))
        interfix_surnames = []

    return NameLookups(
        prefixes=list(selected.lookup_values("prefixes")),
        first_names=list(selected.lookup_values("first_names")),
        surnames=list(selected.lookup_values("family_names")),
        interfixes=interfixes,
        interfix_surnames=interfix_surnames,
        source=selected.lookup_source(None),
    )


def load_lookups_from_cfg(cfg: dict[str, Any]) -> NameLookups:
    provider = get_locale_provider(str(cfg.get("language_profile", "nl-BE")))
    return load_lookups(
        str(cfg.get("language_lookup_dir", "")).strip() or None,
        provider=provider,
    )
