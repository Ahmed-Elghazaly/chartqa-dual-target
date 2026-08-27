"""Report the live Kaggle GPU quota against what this project still has to spend.

Kaggle is the authority on how much quota is left, so this asks it —
`KaggleApi.quota_view()` — rather than keeping a parallel tally.

An earlier version of this script parsed durations out of `RUNS.md` and totalled
them. That was written because a first attempt at the quota API guessed the
method name, failed, and silently fell back. Two bugs came out of that: the
fallback parser summed the gate and budget tables as well as the run rows and
reported 47.9 h used against a 30 h quota, and the "endpoint unavailable" message
hid the real cause for hours. Both are gone. `RUNS.md` remains the session record
`PLAN.md` Appendix F requires — what ran, where, and what it produced — but it is
no longer a second source of truth for hours.

Run:  python scripts/gpu_budget.py
"""

from __future__ import annotations

import os
from pathlib import Path

# What the remaining phases are expected to cost, from the Phase 2 measurements.
COMMITTED_AHEAD: list[tuple[str, float]] = [
    ("Phase 5 zero-shot, both protocols", 3.0),
    ("Phase 6 stage 1 + stage 2", 10.0),
    ("Phase 6 direct-answer control", 3.0),
    ("Phase 7 test evaluation", 3.0),
]


def gpu_quota() -> dict | None:
    """Live GPU quota for this account, or None if Kaggle cannot be reached."""
    os.environ.setdefault("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        stats = api.quota_view()
    except Exception as exc:  # noqa: BLE001
        print(f"  (cannot reach Kaggle: {type(exc).__name__}: {exc})")
        return None

    gpu = getattr(stats, "gpu_quota", None)
    if gpu is None:
        return None

    def hours(value) -> float:
        return value.total_seconds() / 3600.0 if hasattr(value, "total_seconds") else float(value or 0)

    return {
        "used": hours(getattr(gpu, "time_used", 0)),
        "reserved": hours(getattr(gpu, "time_reserved", 0)),
        "total": hours(getattr(gpu, "total_time_allowed", 0)),
        "refresh": getattr(stats, "quota_refresh_time", None),
    }


def main() -> int:
    print("Kaggle GPU quota")
    print("-" * 58)
    q = gpu_quota()
    if q is None:
        return 1

    remaining = q["total"] - q["used"] - q["reserved"]
    reserved = f"   reserved by running sessions: {q['reserved']:.2f} h" if q["reserved"] else ""
    print(f"  used      : {q['used']:6.2f} h of {q['total']:.0f} h{reserved}")
    print(f"  remaining : {remaining:6.2f} h")
    if q["refresh"]:
        print(f"  window resets: {q['refresh']}")

    print("\n  committed ahead:")
    for name, h in COMMITTED_AHEAD:
        print(f"    {name:<36} ~{h:4.1f} h")
    total = sum(h for _, h in COMMITTED_AHEAD)
    print(f"    {'total':<36} ~{total:4.1f} h")

    if remaining < total:
        print(f"\n  {remaining:.1f} h remain this window against ~{total:.1f} h committed.")
        print("  The window resets weekly, and every long job is resumable with checkpoints")
        print("  pushed on save (verified, DECISIONS.md 0033), so work spans windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
