from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Isolate a test from the host's platform and any real credentials."""
    for var in ("CDT_PLATFORM", "CDT_DATA_ROOT", "CDT_CACHE_ROOT", "CDT_OUTPUT_ROOT",
                "KAGGLE_KERNEL_RUN_TYPE", "COLAB_RELEASE_TAG",
                "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "WANDB_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CDT_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("CDT_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("CDT_OUTPUT_ROOT", str(tmp_path / "out"))
    return tmp_path


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT
