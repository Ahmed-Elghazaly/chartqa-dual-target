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


def from_kaggle_api() -> dict | None:
    """Authoritative quota for THIS account, straight from Kaggle.

    The method is `quota_view()`. An earlier version of this function guessed at
    `get_accelerator_quota_statistics` and silently reported "endpoint
    unavailable" for days — the same read-the-API-first lesson as everywhere else,
    and a reminder that a fallback path hides a bug rather than reporting it.
    """
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        q = api.quota_view()
    except Exception:  # noqa: BLE001
        return None

    gpu = getattr(q, "gpu_quota", None)
    if gpu is None:
        return None

    def hours(value) -> float:
        return value.total_seconds() / 3600.0 if hasattr(value, "total_seconds") else float(value or 0)

    return {
        "used": hours(getattr(gpu, "time_used", 0)),
        "reserved": hours(getattr(gpu, "time_reserved", 0)),
        "total": hours(getattr(gpu, "total_time_allowed", 0)),
        "refresh": getattr(q, "quota_refresh_time", None),
    }


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
    print("-" * 60)
    if api:
        remaining = api["total"] - api["used"] - api["reserved"]
        reserved_note = f", {api['reserved']:.2f} h reserved by running sessions" if api["reserved"] else ""
        print(f"  reported by Kaggle : {api['used']:6.2f} h used{reserved_note} of {api['total']:.0f} h")
        if api["refresh"]:
            print(f"  weekly window resets: {api['refresh']}")
        quota = api["total"]
    else:
        print("  reported by Kaggle : (unavailable — falling back to RUNS.md)")
        remaining = WEEKLY_QUOTA_HOURS - logged
        quota = WEEKLY_QUOTA_HOURS
    print(f"  logged in RUNS.md  : {logged:6.2f} h across {sessions} session(s)")
    print(f"  remaining          : {remaining:6.2f} h of {quota:.0f} h")

    committed = [
        ("Phase 5 zero-shot, both protocols", 3.0),
        ("Phase 6 stage 1 + stage 2", 10.0),
        ("Phase 6 direct-answer control", 3.0),
        ("Phase 7 test evaluation", 3.0),
    ]
    print("\n  committed ahead:")
    for name, h in committed:
        print(f"    {name:<36} ~{h:4.1f} h")
    total_ahead = sum(h for _, h in committed)
    print(f"    {'total':<36} ~{total_ahead:4.1f} h")

    if remaining < total_ahead:
        print(f"\n  NOTE: {remaining:.1f} h remain this window against ~{total_ahead:.1f} h committed.")
        print("  The window resets weekly, and every long job is resumable and pushes")
        print("  checkpoints on save, so work spans windows rather than being lost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
