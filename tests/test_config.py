"""The config system is load-bearing for reproducibility, so it is tested hard.

The single most valuable test here is that an unknown key raises. A silently
ignored ``--train.leraning_rate`` produces a run you believe used your setting
and did not, and nothing downstream can detect that.
"""

from __future__ import annotations

import pytest
import yaml

from chartqa_dt.config import (
    Config,
    apply_overrides,
    build_config,
    dataclass_field_names,
    dump_resolved,
    from_dict,
    git_provenance,
    load_yaml_tree,
    to_dict,
)


def test_defaults_match_the_plan():
    c = Config()
    assert c.model.lora_r == 16                    # IDEA.md 8
    assert c.model.lora_on_vision and c.model.lora_on_language   # rule 3
    assert c.model.max_seq_len == 1024
    assert c.model.load_in_4bit is True
    assert c.train.per_device_batch * c.train.grad_accum == 8     # effective batch 8
    assert c.train.save_every_steps == 100
    assert c.eval.split == "val"                   # rule 1: test is sealed by default
    assert c.hub.private is True                   # rule 8


def test_override_types_are_coerced():
    c = build_config(None, ["--train.lr", "5e-5", "--data.max_examples=2000",
                            "--model.lora_on_vision", "false", "--eval.seeds", "0,1"])
    assert isinstance(c.train.lr, float) and c.train.lr == 5e-5
    assert isinstance(c.data.max_examples, int) and c.data.max_examples == 2000
    assert c.model.lora_on_vision is False
    assert c.eval.seeds == [0, 1]


def test_unknown_key_is_fatal():
    with pytest.raises(KeyError, match="unknown config key"):
        from_dict(Config, {"train": {"leraning_rate": 1}})
    with pytest.raises(KeyError, match="unknown config key"):
        from_dict(Config, {"trian": {}})


def test_non_integer_for_int_field_is_fatal():
    with pytest.raises(TypeError, match="not an integer"):
        from_dict(Config, {"train": {"per_device_batch": 0.5}})


def test_optional_field_accepts_none():
    c = from_dict(Config, {"data": {"max_examples": None}})
    assert c.data.max_examples is None


def test_override_without_value_raises():
    with pytest.raises(ValueError, match="no value"):
        apply_overrides({}, ["--train.lr"])


def test_yaml_base_inheritance(tmp_path):
    (tmp_path / "a.yaml").write_text("train:\n  lr: 1.0e-4\n  epochs: 1\n")
    (tmp_path / "b.yaml").write_text("_base_: a.yaml\nrun_name: child\ntrain:\n  lr: 5.0e-5\n")
    tree = load_yaml_tree(tmp_path / "b.yaml")
    assert tree["train"]["lr"] == 5e-5      # child wins
    assert tree["train"]["epochs"] == 1     # base survives
    assert tree["run_name"] == "child"


def test_circular_base_raises(tmp_path):
    (tmp_path / "a.yaml").write_text("_base_: b.yaml\n")
    (tmp_path / "b.yaml").write_text("_base_: a.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        load_yaml_tree(tmp_path / "a.yaml")


def test_every_shipped_config_loads(repo_root):
    paths = sorted((repo_root / "configs").glob("*.yaml"))
    assert len(paths) >= 9, "Appendix G names nine config files"
    for p in paths:
        cfg = build_config(p)
        assert isinstance(cfg, Config)


def test_shipped_eval_configs_never_default_to_test_split(repo_root):
    """Rule 1: a stray default of split=test is how a sealed split gets opened."""
    for name in ("eval_chartqa.yaml", "eval_refchartqa.yaml"):
        cfg = build_config(repo_root / "configs" / name)
        assert cfg.eval.split == "val", f"{name} must default to val, not test"


def test_dump_resolved_writes_config_and_provenance(tmp_path):
    cfg = build_config(None, ["--run_name", "unit"])
    path = dump_resolved(cfg, tmp_path)
    assert path.name == "resolved_config.yaml"
    payload = yaml.safe_load(path.read_text())
    assert payload["config"]["run_name"] == "unit"
    assert set(payload["provenance"]) >= {"git_sha", "git_branch", "git_dirty"}


def test_dump_resolved_never_records_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("CDT_SECRET_TOKEN", "super-secret")
    monkeypatch.setenv("CDT_WANDB_API_KEY", "also-secret")
    monkeypatch.setenv("CDT_PLATFORM", "local")
    text = dump_resolved(build_config(None), tmp_path).read_text()
    assert "super-secret" not in text
    assert "also-secret" not in text


def test_roundtrip_to_dict_from_dict():
    c = build_config(None, ["--train.lr", "3e-5"])
    assert from_dict(Config, to_dict(c)) == c


def test_field_listing_is_complete():
    names = dataclass_field_names()
    for expected in ("train.lr", "model.lora_r", "data.stage1_cap", "eval.split", "hub.private", "seed"):
        assert expected in names


def test_git_provenance_shape():
    p = git_provenance()
    assert set(p) >= {"git_sha", "git_branch", "git_dirty", "git_dirty_files"}
