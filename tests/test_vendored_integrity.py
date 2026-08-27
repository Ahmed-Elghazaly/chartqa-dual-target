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
    Path("verification/chartqa_eval"),
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


def test_the_official_chartqa_metric_is_vendored_and_matches_ours(repo_root):
    """`PLAN.md` 4.2 names *both* official evaluators; this is the ChartQA one.

    The RefChartQA evaluator carries a copy of `relaxed_correctness`, so checking only
    that one leaves the ChartQA claim resting on the assumption that the copy is faithful.
    Vendoring the original makes it checkable: the source is compared line for line
    against our implementation's documented behaviour.

    The file is read as text rather than imported — `pix2struct/metrics.py` pulls in heavy
    dependencies that this project does not have and does not need.
    """
    path = repo_root / "verification/chartqa_eval/metrics.py"
    if not path.exists():
        pytest.skip("the ChartQA metric has not been vendored in this checkout")
    src = path.read_text(encoding="utf-8")

    body = src[src.index("def relaxed_correctness"):]
    body = body[:body.index("\ndef ", 1)] if "\ndef " in body[1:] else body

    # The three behaviours our implementation reproduces, asserted against the source.
    assert 'return float(text.rstrip("%")) / 100.0' in body, "percent handling"
    assert 'return float(text)' in body and '.replace(","' not in body, \
        "the official parser does NOT strip thousands separators"
    assert "if prediction_float is not None and target_float:" in body, \
        "the zero guard is a truthiness test, not `is not None` — see DECISIONS.md 0053"
    assert "return prediction.lower() == target.lower()" in body, \
        "the string branch does not strip whitespace"


def test_our_relaxed_correctness_agrees_with_the_vendored_chartqa_metric(repo_root):
    """Execute the official function itself, in isolation, and compare outputs."""
    path = repo_root / "verification/chartqa_eval/metrics.py"
    if not path.exists():
        pytest.skip("the ChartQA metric has not been vendored in this checkout")

    from chartqa_dt.eval.metrics import relaxed_correctness

    src = path.read_text(encoding="utf-8")
    start = src.index("def relaxed_correctness")
    end = src.index("\ndef ", start + 1)
    namespace: dict = {"Optional": object}
    exec(compile(src[start:end], str(path), "exec"), namespace)
    official = namespace["relaxed_correctness"]

    cases = [("10", "10.4"), ("10", "10.6"), ("0", "0"), ("0", "0.0"), ("0", "0.1"),
             ("50%", "0.5"), ("0.5", "50%"), ("Yes", "yes"), ("Yes", "Yes."),
             ("1,234", "1234"), ("1,234", "1,234"), (" Yes ", "Yes"), ("abc", "ABC"),
             ("2020", "2020.0"), ("-5", "-5.1"), ("100", "104"), ("100", "106")]
    bad = [(t, p) for t, p in cases
           if bool(official(t, p)) != relaxed_correctness(t, p)]
    assert not bad, f"disagreement with the official ChartQA metric: {bad}"
