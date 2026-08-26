"""Non-negotiable rule 7 is enforced in code, not by discipline.

ChartQA is GPL-3.0 and RefChartQA is AGPL-3.0. One stray qualitative-example
PNG in a results directory would put a licensed chart image on the Hub, and it
would be committed history by the time anyone noticed.
"""

from __future__ import annotations

import pytest

from chartqa_dt.hub import HubError, HubStore, assert_no_dataset_content, get_token


def test_clean_directory_passes(tmp_path):
    (tmp_path / "results.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")  # adapters are fine
    assert_no_dataset_content(tmp_path)


@pytest.mark.parametrize("name", ["chart.png", "a.jpg", "x.jpeg", "d.parquet", "ChartQA Dataset.zip", "t.arrow"])
def test_dataset_content_is_refused(tmp_path, name):
    (tmp_path / name).write_bytes(b"x")
    with pytest.raises(HubError, match="rule 7"):
        assert_no_dataset_content(tmp_path)


def test_nested_dataset_content_is_refused(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "leaked.png").write_bytes(b"x")
    with pytest.raises(HubError, match="rule 7"):
        assert_no_dataset_content(tmp_path)


@pytest.mark.parametrize("rel", ["report/figures/fig.png", "demo/examples/own_chart.png", "qualitative/a.png"])
def test_self_generated_figures_are_allowed(tmp_path, rel):
    p = tmp_path / rel
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    assert_no_dataset_content(tmp_path)


def test_store_is_disabled_without_a_token(clean_env):
    s = HubStore(repo_id="user/repo")
    assert s.enabled is False
    assert "no HF token" in s.status()
    with pytest.raises(HubError, match="hub disabled"):
        s.ensure_repo()


def test_store_defaults_to_private(clean_env, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    s = HubStore(repo_id="user/repo")
    assert s.private is True          # rule 8
    assert s.enabled is True
    assert "private=True" in s.status()


def test_push_is_non_strict_when_asked(clean_env, tmp_path):
    s = HubStore(repo_id="user/repo")          # disabled: no token
    assert s.push_dir(tmp_path, "x", strict=False) is False
    assert s.pull_dir("x", tmp_path, strict=False) is None
    assert s.exists("x") is False


def test_get_token_prefers_environment(clean_env, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "  hf_from_env  ")
    assert get_token() == "hf_from_env"
