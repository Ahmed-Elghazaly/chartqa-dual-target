"""The GPU-budget parser.

Kaggle allows ~30 GPU-hours a week and Phase 6 alone needs 6-10, so this tracker
guards a real constraint. Its first version summed every duration-shaped cell in
RUNS.md -- including the gate table's thresholds and the budget table's own
figures -- and reported 47.9 hours used against a 30-hour quota.

A tracker that cannot be trusted is worse than none: it trains you to ignore it,
and the one time it is right you will ignore it then too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
gpu_budget = pytest.importorskip("gpu_budget")


@pytest.mark.parametrize(
    ("cell", "hours"),
    [
        ("25 min", 25 / 60), ("~0.5 min", 0.5 / 60), ("3 min", 0.05),
        ("1.2 h", 1.2), ("~2 h", 2.0), ("30 s", 30 / 3600),
    ],
)
def test_durations_parse(cell, hours):
    assert gpu_budget.parse_duration(cell) == pytest.approx(hours)


@pytest.mark.parametrize("cell", ["", "—", "_running_", "1.48", "≤ 10 h", "pass", "USD 0.00"])
def test_non_durations_are_ignored(cell):
    assert gpu_budget.parse_duration(cell) is None


def test_only_run_rows_are_counted(tmp_path, monkeypatch):
    """The exact shape that produced 47.9 h: other tables also contain 'N h' cells."""
    doc = tmp_path / "RUNS.md"
    doc.write_text(
        "| # | Date | Host | What | Wall | Peak | Outcome | Notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
        "| 1 | d | Kaggle | x | 30 min | 1.48 | ERROR | n |\n"
        "| 2 | d | Kaggle | x | 1.5 h | 1.48 | COMPLETE | n |\n"
        "| 3 | d | Kaggle | x | _running_ | | | |\n"
        "\n## Budget\n"
        "| Item | Spent | Cap |\n|---|---|---|\n"
        "| Kaggle GPU hours this week | ~0.9 h | ~30 h |\n"
        "| Committed ahead | ~17 h | — |\n"
        "\n## Gates\n"
        "| Gate | Threshold | Measured |\n|---|---|---|\n"
        "| Projected full run | 10 h | 7.22 h |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gpu_budget, "RUNS_MD", doc)
    hours, sessions = gpu_budget.from_runs_md()
    assert sessions == 3, "three run rows, including the one still running"
    assert hours == pytest.approx(0.5 + 1.5), (
        f"got {hours:.2f} h; the budget and gate tables must not be counted"
    )


def test_missing_file_is_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_budget, "RUNS_MD", tmp_path / "nope.md")
    assert gpu_budget.from_runs_md() == (0.0, 0)


def test_real_runs_md_is_within_the_quota(repo_root, monkeypatch):
    """Guards against the tracker silently drifting into nonsense again."""
    monkeypatch.setattr(gpu_budget, "RUNS_MD", repo_root / "RUNS.md")
    hours, sessions = gpu_budget.from_runs_md()
    assert sessions >= 1
    assert 0.0 <= hours <= gpu_budget.WEEKLY_QUOTA_HOURS, (
        f"parsed {hours:.2f} h from RUNS.md, which is outside the {gpu_budget.WEEKLY_QUOTA_HOURS} h quota"
    )
