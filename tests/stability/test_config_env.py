from __future__ import annotations

from meddeid_eval.stability.config import NameCfg, load_config

_CFG = """
dataset: ${DEID_TEST_DATA:-/fallback/data.jsonl}
output_dir: ../outputs/x
models: []
"""


def _write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_default_name_targets_exclude_name_other():
    assert NameCfg().labels == ["Name:Patient", "Name:Caregiver"]


def test_config_cannot_reenable_name_other(tmp_path):
    cfg = load_config(_write(
        tmp_path,
        "dataset: /d.jsonl\nname:\n  labels: [Name:Patient, Name:Other, Name:Caregiver]\nmodels: []\n",
    ))
    assert cfg.name.labels == ["Name:Patient", "Name:Caregiver"]


def test_env_default_used_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("DEID_TEST_DATA", raising=False)
    cfg = load_config(_write(tmp_path, _CFG))
    assert cfg.dataset.name == "data.jsonl"
    assert str(cfg.dataset) == "/fallback/data.jsonl"


def test_env_value_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("DEID_TEST_DATA", "/env/chosen.jsonl")
    cfg = load_config(_write(tmp_path, _CFG))
    assert str(cfg.dataset) == "/env/chosen.jsonl"


def test_env_expands_in_text_source(tmp_path, monkeypatch):
    monkeypatch.setenv("DEID_TEXTS", "/env/texts.json")
    p = _write(tmp_path, "dataset: /d.jsonl\ntext_source: ${DEID_TEXTS}\nmodels: []\n")
    cfg = load_config(p)
    assert str(cfg.text_source) == "/env/texts.json"
