"""Track Kaggle GPU-hour consumption against the weekly quota.

Kaggle allows roughly **30 GPU-hours per week**, reset weekly. Phase 6's training
run alone is budgeted at 6-10 hours, so the quota is a real constraint rather
than a formality, and a wasted long run costs a meaningful fraction of a week.

Kaggle exposes quota through an accelerator-quota endpoint; where that is
unavailable this falls back to the durations recorded in RUNS.md, which is why
every session must be logged there.

Run:  python scripts/gpu_budget.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

WEEKLY_QUOTA_HOURS = 30.0
RUNS_MD = Path(__file__).resolve().parents[1] / "RUNS.md"


def from_kaggle_api() -> tuple[float, float] | None:
    """(used_hours, quota_hours) from Kaggle, or None if unavailable."""
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except Exception:  # noqa: BLE001
        return None
    for name in ("get_accelerator_quota_statistics", "accelerator_quota_statistics"):
        fn = getattr(api, name, None)
        if fn is None:
            continue
        try:
            stats = fn()
        except Exception:  # noqa: BLE001
            continue
        used = getattr(stats, "used_seconds", None) or getattr(stats, "usedSeconds", None)
        quota = getattr(stats, "quota_seconds", None) or getattr(stats, "quotaSeconds", None)
        if used is not None and quota:
            return float(used) / 3600.0, float(quota) / 3600.0
    return None


def parse_duration(cell: str) -> float | None:
    """Hours from a cell like '25 min', '~0.5 min', '3 min', '1.2 h'. None otherwise."""
    cell = cell.strip().lstrip("~").strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|min|mins|m|s|sec|secs)", cell, re.I)
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2).lower()
    if unit.startswith("h"):
        return value
    if unit in ("m", "min", "mins"):
        return value / 60.0
    return value / 3600.0


def from_runs_md() -> tuple[float, int]:
    """(hours, n_sessions) summed from the RUN TABLE only.

    Deliberately strict about which rows and which column. A looser parser
    summed the gate table and the budget table as well and reported 47.9 hours
    against a 30-hour quota — a tracker that cannot be trusted is worse than none,
    because it invites ignoring a real warning later.

    A run row starts with an integer session number; the wall time is column 5.
    """
    if not RUNS_MD.is_file():
        return 0.0, 0
    total, n = 0.0, 0
    for line in RUNS_MD.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or not cells[0].isdigit():
            continue
        n += 1
        if (hours := parse_duration(cells[4])) is not None:
            total += hours
    return total, n


def main() -> int:
    api = from_kaggle_api()
    logged, sessions = from_runs_md()

    print("Kaggle GPU budget")
    print("-" * 56)
    if api:
        used, quota = api
        print(f"  reported by Kaggle : {used:6.2f} h used of {quota:.0f} h")
        remaining = quota - used
    else:
        print("  reported by Kaggle : (quota endpoint unavailable)")
        used, quota = logged, WEEKLY_QUOTA_HOURS
        remaining = quota - used
    print(f"  logged in RUNS.md  : {logged:6.2f} h across {sessions} session(s)")
    print(f"  remaining (est.)   : {remaining:6.2f} h of {quota:.0f} h weekly")
    print()
    print("  committed ahead:")
    for name, hours in (("Phase 2 measurement (running)", 1.0),
                        ("Phase 5 zero-shot, both protocols", 3.0),
                        ("Phase 6 stage 1 + stage 2", 8.0),
                        ("Phase 6 direct-answer control", 3.0),
                        ("Phase 7 test evaluation", 3.0)):
        print(f"    {name:<36} ~{hours:4.1f} h")
    print(f"    {'total':<36} ~{18.0:4.1f} h")
    if remaining < 18.0:
        print("\n  WARNING: less remaining than the committed plan needs.")
        print("  Long runs must be resumable and must not be repeated for avoidable reasons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
