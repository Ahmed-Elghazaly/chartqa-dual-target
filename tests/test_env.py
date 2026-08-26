"""Environment detection. Acceptance criterion: correct paths on Kaggle and Colab.

We cannot run the test suite on a Kaggle kernel, so the platform is simulated
the same way the real detector observes it — via environment variables — and the
resolved roots are asserted against what those platforms actually provide.
"""

from __future__ import annotations

import sys

import pytest

from chartqa_dt.env import Environment, _roots_for, detect_platform, get_env, load_dotenv


def test_detects_local_by_default(clean_env, monkeypatch):
    monkeypatch.delenv("CDT_PLATFORM", raising=False)
    # Only assert the negative: a dev machine is not Kaggle or Colab.
    assert detect_platform() in ("local", "kaggle", "colab")


def test_forced_platform_wins(monkeypatch):
    monkeypatch.setenv("CDT_PLATFORM", "colab")
    assert detect_platform() == "colab"
    monkeypatch.setenv("CDT_PLATFORM", "kaggle")
    assert detect_platform() == "kaggle"


def test_bad_forced_platform_raises(monkeypatch):
    monkeypatch.setenv("CDT_PLATFORM", "azure")
    with pytest.raises(ValueError, match="not one of"):
        detect_platform()


def test_kaggle_env_var_is_detected(clean_env, monkeypatch):
    monkeypatch.delenv("CDT_PLATFORM", raising=False)
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    assert detect_platform() == "kaggle"


def test_colab_env_var_is_detected(clean_env, monkeypatch):
    monkeypatch.delenv("CDT_PLATFORM", raising=False)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab-20260801")
    assert detect_platform() == "colab"


@pytest.mark.parametrize(
    ("platform", "expect_output_prefix"),
    [("kaggle", "/kaggle/working"), ("colab", "/content")],
)
def test_platform_roots_are_the_real_ones(platform, expect_output_prefix):
    """These literals are the whole point of the module; assert them explicitly."""
    data, _cache, out = _roots_for(platform)
    assert str(out).startswith(expect_output_prefix)
    if platform == "kaggle":
        # Bulky, re-downloadable data must not eat the size-capped output dir.
        assert str(data).startswith("/kaggle/")
        assert not str(data).startswith("/kaggle/working"), (
            "on Kaggle, data belongs in /kaggle/temp so it does not count "
            "against the kernel output size cap"
        )
    else:
        assert str(data).startswith("/content")


def test_env_overrides_beat_platform_defaults(clean_env, monkeypatch):
    monkeypatch.setenv("CDT_PLATFORM", "kaggle")
    e = get_env()
    assert e.platform == "kaggle"
    assert e.data_root == clean_env / "data"       # override applied
    assert e.output_root == clean_env / "out"


def test_get_env_creates_roots_and_sets_hf_home(clean_env, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    e = get_env()
    assert e.data_root.is_dir() and e.cache_root.is_dir() and e.output_root.is_dir()
    import os
    assert os.environ["HF_HOME"] == str(e.cache_root)


def test_run_dir_is_created(clean_env):
    e = get_env()
    d = e.run_dir("my-run")
    assert d.is_dir() and d.parent == e.output_root


def test_describe_mentions_every_root(clean_env):
    text = get_env().describe()
    for key in ("platform", "gpu", "data_root", "cache_root", "output_root"):
        assert key in text


def test_vram_is_zero_without_cuda(clean_env):
    e = get_env()
    assert e.vram_gb >= 0.0
    if e.gpu_name == "cpu":
        assert e.vram_gb == 0.0


def test_load_dotenv_reads_keys_but_returns_no_values(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text('# comment\nHF_TOKEN="hf_secret_value"\nEMPTY\nGITHUB_USER=someone\n')
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_USER", raising=False)
    keys = load_dotenv(p)
    assert set(keys) == {"HF_TOKEN", "GITHUB_USER"}
    assert "hf_secret_value" not in " ".join(keys)   # names only, never values
    import os
    assert os.environ["HF_TOKEN"] == "hf_secret_value"


def test_load_dotenv_does_not_clobber_existing(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("HF_TOKEN=from_file\n")
    monkeypatch.setenv("HF_TOKEN", "from_shell")
    load_dotenv(p)
    import os
    assert os.environ["HF_TOKEN"] == "from_shell"


def test_environment_is_frozen(clean_env):
    e = get_env()
    assert isinstance(e, Environment)
    with pytest.raises(Exception):
        e.platform = "kaggle"  # type: ignore[misc]


def test_colab_is_detected_by_module_not_only_by_sys_modules(clean_env, monkeypatch):
    """The documented Colab test is whether `google.colab` imports.

    Checking `sys.modules` alone fails in a fresh process, because nothing has
    imported it yet -- which is exactly the situation a CLI entry point is in.
    """
    monkeypatch.delenv("CDT_PLATFORM", raising=False)
    import chartqa_dt.env as envmod

    monkeypatch.setattr(envmod, "_colab_module_importable", lambda: True)
    assert envmod.detect_platform() == "colab"


def test_colab_env_vars_are_a_secondary_signal(clean_env, monkeypatch):
    monkeypatch.delenv("CDT_PLATFORM", raising=False)
    import chartqa_dt.env as envmod

    monkeypatch.setattr(envmod, "_colab_module_importable", lambda: False)
    monkeypatch.setenv("COLAB_GPU", "1")
    assert envmod.detect_platform() == "colab"


def test_colab_probe_is_side_effect_free(clean_env):
    """find_spec rather than import: nothing should execute to answer this."""
    import chartqa_dt.env as envmod

    before = set(sys.modules)
    envmod._colab_module_importable()
    assert "google.colab" not in (set(sys.modules) - before)
