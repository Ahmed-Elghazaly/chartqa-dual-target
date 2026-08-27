"""Every source file this project needs must actually be in the repository.

This exists because it happened. `.gitignore` carried an unanchored `data/` rule — meant
for the top-level dataset directory under rule 7 — and it silently matched
`src/chartqa_dt/data/`, the entire loaders package. Nine commits went by with every local
test passing, because the files were on disk; CI failed each time with
`ModuleNotFoundError: No module named 'chartqa_dt.data'`.

The failure mode is nasty in a specific way: *the code works everywhere it is run by the
person who wrote it, and exists nowhere else.* Kaggle pulls from GitHub, so training would
have failed on a machine with no local copy — and if this one were lost, so was Phase 3.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "tests", "scripts")
SOURCE_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".cfg"}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=False).stdout


def _is_git_repo() -> bool:
    return (ROOT / ".git").exists()


def source_files() -> list[Path]:
    out = []
    for directory in SOURCE_DIRS:
        for path in (ROOT / directory).rglob("*"):
            if (path.is_file() and path.suffix in SOURCE_SUFFIXES
                    and "__pycache__" not in path.parts):
                out.append(path)
    return sorted(out)


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_no_source_file_is_ignored_by_git():
    """The exact check that would have caught it on the first commit."""
    files = source_files()
    assert files, "found no source files at all — the glob is wrong"
    rel = [str(p.relative_to(ROOT)) for p in files]
    ignored = _git("check-ignore", *rel).split()
    assert not ignored, (
        f"{len(ignored)} source file(s) are excluded by .gitignore and would never reach "
        f"the repository: {ignored[:8]}. If a rule is meant for a top-level directory, "
        f"anchor it with a leading slash."
    )


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_every_python_package_under_src_is_tracked():
    """A package missing from git imports fine locally and nowhere else."""
    tracked = set(_git("ls-files", "src").split())
    missing = [str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("__init__.py")
               if "__pycache__" not in p.parts
               and str(p.relative_to(ROOT)) not in tracked]
    assert not missing, f"untracked packages: {missing}"


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_the_rule_seven_ignore_is_anchored():
    """Rule 7 must exclude the dataset directory without eating source directories."""
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    assert "/data/" in lines, "the dataset ignore must be anchored to the repo root"
    assert "data/" not in lines, (
        "an unanchored `data/` matches every directory named data, including "
        "src/chartqa_dt/data/"
    )


def test_the_modules_phase_three_depends_on_all_import():
    """A blunt smoke test: if any of these vanish, Phase 4 cannot start."""
    import importlib

    for name in ("chartqa_dt.data.records", "chartqa_dt.data.chartqa",
                 "chartqa_dt.data.refchartqa", "chartqa_dt.data.dedup",
                 "chartqa_dt.data.mixture", "chartqa_dt.data.download",
                 "chartqa_dt.data.sources", "chartqa_dt.data.remote_zip",
                 "chartqa_dt.synth.generator", "chartqa_dt.plans.mining"):
        assert importlib.import_module(name) is not None
