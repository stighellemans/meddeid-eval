"""Config schema + loader (YAML -> dataclasses).

Relative ``dataset`` and ``output_dir`` are resolved
against the config file's directory so a config is portable: copy the repo to a
new device, edit the paths block, and every stage follows.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dates import DATE_FORMAT_PROFILES
from .names import CAPITALIZATION_MODES, DEFAULT_NAME_PATTERNS, DEFAULT_TITLE, NAME_FORMATS

# ${VAR}, ${VAR:-default}, or $VAR  — lets a config point at a shared data endpoint
# via an environment variable (e.g. dataset: ${DEID_ANNOTATIONS_PATH:-/abs/default}).
_ENV_RE = re.compile(r"\$\{([A-Za-z_]\w*)(?::-([^}]*))?\}|\$([A-Za-z_]\w*)")


def _stability_name_labels(labels: list[str]) -> list[str]:
    """Remove the unsupported Name:Other perturbation target."""
    return [
        str(label) for label in labels
        if re.sub(r"[^a-z0-9]+", "", str(label).lower()) != "nameother"
    ]


def _expand_env(value: str) -> str:
    def repl(m: re.Match) -> str:
        name = m.group(1) or m.group(3)
        default = m.group(2) if m.group(2) is not None else ""
        return os.environ.get(name, default)
    return _ENV_RE.sub(repl, value)


def _expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, list):
        return [_expand(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    return obj


@dataclass
class ValueShiftCfg:
    year_min: int = 1980
    year_max: int = 2050
    step: int = 1


@dataclass
class NameCfg:
    enabled: bool = True
    labels: list[str] = field(default_factory=lambda: ["Name:Patient", "Name:Caregiver"])
    patterns: list[str] = field(default_factory=lambda: list(DEFAULT_NAME_PATTERNS))
    capitalization: list[str] = field(default_factory=lambda: list(CAPITALIZATION_MODES))
    formats: list[str] = field(default_factory=lambda: list(NAME_FORMATS))
    other_trials: int = 25
    title: str = DEFAULT_TITLE


@dataclass
class DateCfg:
    enabled: bool = True
    labels: list[str] = field(default_factory=lambda: ["Date"])
    age_labels: list[str] = field(default_factory=lambda: ["Age_Birthdate"])
    value_shift: ValueShiftCfg = field(default_factory=ValueShiftCfg)
    formats: list[str] = field(default_factory=lambda: list(DATE_FORMAT_PROFILES))


@dataclass
class ModelCfg:
    id: str
    checkpoint: str
    base_model: str = "DTAI-KULeuven/robbert-2023-dutch-base"
    runner: str = "robbert"
    entity_labels: list[str] = field(default_factory=list)
    bio_labels: list[str] | None = None
    max_length: int = 512
    overlap: int = 64


@dataclass
class StabilityConfig:
    dataset: Path
    output_dir: Path
    seed: int = 42
    max_docs: int = 0
    device: str = "auto"
    text_source: Path | None = None  # optional map (document_id -> text) for split datasets
    name: NameCfg = field(default_factory=NameCfg)
    date: DateCfg = field(default_factory=DateCfg)
    models: list[ModelCfg] = field(default_factory=list)
    source: Path | None = None  # the config file this was loaded from


def _resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path: str | Path) -> StabilityConfig:
    import yaml  # PyYAML

    cfg_path = Path(path).expanduser().resolve()
    base = cfg_path.parent
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw = _expand(raw)  # resolve ${ENV_VAR} references (shared data endpoint, etc.)

    name_raw = raw.get("name", {}) or {}
    date_raw = raw.get("date", {}) or {}
    vs_raw = date_raw.get("value_shift", {}) or {}

    models: list[ModelCfg] = []
    for m in raw.get("models", []) or []:
        params = dict(m.get("params", {}))  # tolerate deid-battery-style nesting
        models.append(ModelCfg(
            id=str(m["id"]),
            checkpoint=str(m.get("checkpoint") or params.get("checkpoint")),
            base_model=str(m.get("base_model") or params.get("base_model") or "DTAI-KULeuven/robbert-2023-dutch-base"),
            runner=str(m.get("runner", "robbert")),
            entity_labels=list(m.get("entity_labels") or params.get("entity_labels") or []),
            bio_labels=m.get("bio_labels") or params.get("bio_labels"),
            max_length=int(m.get("max_length") or params.get("max_length") or 512),
            overlap=int(m.get("overlap") or params.get("overlap") or 64),
        ))

    dataset = _resolve(base, raw.get("dataset"))
    if dataset is None:
        raise ValueError("config must set `dataset` (path to the JSONL to perturb)")
    output_dir = _resolve(base, raw.get("output_dir") or "outputs") or (base / "outputs")

    return StabilityConfig(
        dataset=dataset,
        output_dir=output_dir,
        seed=int(raw.get("seed", 42)),
        max_docs=int(raw.get("max_docs", 0)),
        device=str(raw.get("device", "auto")),
        text_source=_resolve(base, raw.get("text_source")),
        name=NameCfg(
            enabled=bool(name_raw.get("enabled", True)),
            labels=_stability_name_labels(list(name_raw.get("labels", NameCfg().labels))),
            patterns=list(name_raw.get("patterns", DEFAULT_NAME_PATTERNS)),
            capitalization=list(name_raw.get("capitalization", CAPITALIZATION_MODES)),
            formats=list(name_raw.get("formats", NAME_FORMATS)),
            other_trials=int(name_raw.get("other_trials", 25)),
            title=str(name_raw.get("title", DEFAULT_TITLE)),
        ),
        date=DateCfg(
            enabled=bool(date_raw.get("enabled", True)),
            labels=list(date_raw.get("labels", ["Date"])),
            age_labels=list(date_raw.get("age_labels", ["Age_Birthdate"])),
            value_shift=ValueShiftCfg(
                year_min=int(vs_raw.get("year_min", 1980)),
                year_max=int(vs_raw.get("year_max", 2050)),
                step=int(vs_raw.get("step", 1)),
            ),
            formats=list(date_raw.get("formats", DATE_FORMAT_PROFILES)),
        ),
        models=models,
        source=cfg_path,
    )
