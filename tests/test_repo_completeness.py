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


#: Files that live in the repo and that tests or a clean checkout depend on. Every one of
#: these sits under `data/`, which rule 7 excludes wholesale — so each is an explicit
#: exception, and each exception has to be checked rather than assumed to work.
REQUIRED_ARTEFACTS = (
    "data/MANIFEST.json",
    "data/sealed_images.json",
    "data/refchartqa_audit.jsonl",
    "data/mixture_stage1.json",
    "data/mixture_stage2.json",
)


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_required_artefacts_are_not_excluded_by_gitignore():
    """The second gitignore failure, and the subtler one.

    Git **cannot re-include a file whose parent directory is excluded**. So `/data/`
    followed by `!/data/MANIFEST.json` silently does nothing, and the exception lines read
    as if they work. `/data/*` excludes the contents while leaving the directory visible,
    which is what makes re-inclusion possible at all.

    Every one of these files existed locally, so every local test passed; CI failed on
    `FileNotFoundError: data/MANIFEST.json`.
    """
    present = [a for a in REQUIRED_ARTEFACTS if (ROOT / a).exists()]
    if not present:
        pytest.skip("no data artefacts built in this checkout")
    ignored = _git("check-ignore", *present).split()
    assert not ignored, (
        f"{len(ignored)} required artefact(s) are excluded by .gitignore and will not "
        f"reach a clean checkout: {ignored}. A `!` exception cannot rescue a file inside "
        f"an excluded directory — exclude the contents (`/data/*`) instead."
    )


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_rule_seven_still_excludes_dataset_content():
    """The exceptions must not have opened a hole. Rule 7 is non-negotiable."""
    must_be_ignored = ("data/chart.png", "data/images/x.png",
                       "data/refchartqa/train/y.png", "data/hf/blob.parquet",
                       "data/tables/gold.csv")
    not_ignored = [p for p in must_be_ignored
                   if not _git("check-ignore", p).strip()]
    assert not not_ignored, f"rule 7 no longer covers: {not_ignored}"


@pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")
def test_the_rule_seven_ignore_is_anchored():
    """Rule 7 must exclude the dataset directory without eating source directories."""
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    assert "/data/*" in lines, (
        "the dataset ignore must be anchored (`/data/`, not `data/`) AND must exclude the "
        "directory's contents (`/data/*`), or the `!` exceptions below it are no-ops"
    )
    assert "data/" not in lines, (
        "an unanchored `data/` matches every directory named data, including "
        "src/chartqa_dt/data/"
    )
    assert "/data/" not in lines, (
        "`/data/` excludes the directory itself, and git cannot re-include a file from "
        "inside an excluded directory — use `/data/*`"
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


#: Fields that carry dataset content rather than derived statistics. Rule 7 permits
#: "scripts, IDs, hashes, adapters and derived statistics only", and ChartQA is GPL-3.0
#: while RefChartQA is AGPL-3.0.
FORBIDDEN_FIELDS = ("question", "answer", "query", "label", "response", "table",
                    "image_path")


def test_no_committed_artefact_carries_dataset_text():
    """Rule 7, checked on content and not only on file type.

    `assert_no_dataset_content` screens *file types* — png, zip, parquet — so a JSONL full
    of questions and gold answers passes it while plainly being dataset content. The audit
    file was exactly that until this test existed. Judgements stay auditable because `id`
    identifies each row: anyone with the dataset can recover the question and re-judge.
    """
    import json as _json

    for name in ("data/refchartqa_audit.jsonl", "data/mixture_stage1.json",
                 "data/mixture_stage2.json", "data/sealed_images.json",
                 "data/MANIFEST.json"):
        path = ROOT / name
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            rows = [_json.loads(line) for line in
                    path.read_text(encoding="utf-8").splitlines() if line]
        else:
            rows = [_json.loads(path.read_text(encoding="utf-8"))]
        for row in rows:
            present = [f for f in FORBIDDEN_FIELDS if f in row]
            assert not present, f"{name} carries dataset content: {present}"


def test_the_mixture_files_carry_ids_not_content():
    """A mixture must be reproducible without redistributing anything."""
    import json as _json

    for name in ("data/mixture_stage1.json", "data/mixture_stage2.json"):
        path = ROOT / name
        if not path.exists():
            continue
        data = _json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == {"composition", "record_ids", "keys"}
        assert len(data["record_ids"]) == data["composition"]["total"]
        assert all(":" in k for k in data["keys"]), "keys must be dedup keys"
