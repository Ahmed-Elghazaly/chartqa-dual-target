"""Vendored third-party code must be byte-identical to upstream.

This test exists because git already broke it once. `.gitattributes` originally
carried `* text=auto eol=lf`, which silently rewrote the official RefChartQA
evaluator's CRLF line endings on commit: the working copy hashed to
d0c9f87d... (matching upstream) while the blob git stored hashed to 5ab767f5...

That matters because DECISIONS.md 0003 makes the official evaluator the scorer
of record. "We ran the official evaluator" is only true if the bytes are the
official ones, and a whitespace-only diff is still a diff you would have to
defend. A hash check is cheap; discovering this after reporting a number is not.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

VENDOR_DIRS = [
    Path("verification/refchartqa_eval"),
    Path("src/chartqa_dt/eval/official/vendor"),
]


def _provenance_dirs(repo_root: Path) -> list[Path]:
    return [d for d in (repo_root / v for v in VENDOR_DIRS) if (d / "PROVENANCE.json").is_file()]


def test_at_least_one_vendored_evaluator_is_recorded(repo_root):
    assert _provenance_dirs(repo_root), "no vendored evaluator with a PROVENANCE.json was found"


def test_vendored_files_match_their_recorded_hashes(repo_root):
    checked = 0
    for d in _provenance_dirs(repo_root):
        prov = json.loads((d / "PROVENANCE.json").read_text(encoding="utf-8"))
        for name, meta in prov["files"].items():
            path = d / name
            assert path.is_file(), f"{path} is recorded in PROVENANCE.json but missing"
            raw = path.read_bytes()
            actual = hashlib.sha256(raw).hexdigest()
            assert actual == meta["sha256"], (
                f"{path} has been modified.\n"
                f"  expected sha256 {meta['sha256']}\n"
                f"  actual   sha256 {actual}\n"
                "Vendored evaluators must stay byte-identical to upstream (DECISIONS.md 0003). "
                "If the change is deliberate, re-download from the recorded URL and update "
                "PROVENANCE.json in the same commit."
            )
            assert len(raw) == meta["bytes"], f"{path}: size drifted from the recorded value"
            checked += 1
    assert checked >= 3, "expected at least the three RefChartQA evaluator files"


def test_gitattributes_protects_every_vendor_dir(repo_root):
    """The normalisation that broke this once must stay disabled."""
    text = (repo_root / ".gitattributes").read_text(encoding="utf-8")
    for d in VENDOR_DIRS:
        pattern = f"{d.as_posix()}/**"
        assert pattern in text, f"{pattern} must be marked -text in .gitattributes"
        line = next(ln for ln in text.splitlines() if ln.strip().startswith(pattern))
        assert "-text" in line, f"{pattern} must be marked -text, got: {line!r}"


@pytest.mark.parametrize("expected", ["<grounding-sep>", "<box>", "bins - 1"])
def test_official_evaluator_contract_is_present(repo_root, expected):
    """Guards the three facts our scoring adapter is built against (phase0 F2)."""
    src = (repo_root / "verification/refchartqa_eval/evaluate.py").read_text(
        encoding="utf-8", errors="replace"
    )
    assert expected in src
