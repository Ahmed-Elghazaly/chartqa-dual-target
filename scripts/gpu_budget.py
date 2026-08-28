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
#: Measured on a T4 at 512 px, 4-bit (`verification/measured_facts.json`). Everything below
#: is derived from these two numbers rather than estimated, because the previous estimates
#: were out by 4x on Phase 5 and nobody noticed until the quota was nearly spent.
#: `PLAN.md`: the free tier allows this much GPU per account per week.
WEEKLY_QUOTA_H = 30.0
SECONDS_PER_STEP = 11.903          # phase2.seconds_per_step
SECONDS_PER_STRUCTURED_ITEM = 11.44   # phase5.variant_selection_5_2.median_latency_s
SECONDS_PER_PLAIN_ITEM = 1.2
#: A fine-tuned model reads the 117-token training prompt rather than the 980-token
#: zero-shot one, and emits the compact 141-token record rather than 198 tokens with
#: run-ons. Scaled from the measured zero-shot latency by the token ratio; it is an
#: estimate and is labelled as one, because nothing has been fine-tuned yet.
SECONDS_PER_FINETUNED_ITEM = 7.0

#: Steps come from `cli/train.steps_for`: stage 1 is one pass over 10,304 records at
#: effective batch 8, stage 2 takes the rest of the 24,000-presentation budget.
STAGE1_STEPS, STAGE2_STEPS = 1288, 1712


def _train_h(steps: int) -> float:
    return steps * SECONDS_PER_STEP / 3600


def _gen_h(n: int, seconds: float = SECONDS_PER_STRUCTURED_ITEM) -> float:
    return n * seconds / 3600


COMMITTED_AHEAD: list[tuple[str, float]] = [
    ("Phase 5.3 ChartQA zero-shot, 1,920 x 2 arms",
     _gen_h(1920) + _gen_h(1920, SECONDS_PER_PLAIN_ITEM)),
    ("Phase 5.4 RefChartQA zero-shot, 1,800 rows", _gen_h(1800)),
    ("Phase 6 stage 1, one pass", _train_h(STAGE1_STEPS)),
    ("Phase 6 stage 2, pre-registered arm", _train_h(STAGE2_STEPS)),
    ("Phase 6 stage 2, plan-rich arm", _train_h(STAGE2_STEPS)),
    ("Phase 6 direct-answer control, both stages",
     _train_h(STAGE1_STEPS + STAGE2_STEPS)),
    # Phase 7 is the largest line, so it is costed per arm rather than as one number.
    # The plain arm is nearly free (32-token cap) and is the condition the published 79.1
    # was measured under, so the reproduction attempt costs almost nothing; the structured
    # arms are what the budget actually buys.
    ("Phase 7 ChartQA test, plain arm (the 79.1 condition)",
     _gen_h(2500, SECONDS_PER_PLAIN_ITEM)),
    ("Phase 7 ChartQA test, structured x 3 systems",
     _gen_h(2500) + 2 * _gen_h(2500, SECONDS_PER_FINETUNED_ITEM)),
    ("Phase 7 RefChartQA test, 1,800 sampled x 3 systems",
     _gen_h(1800) + 2 * _gen_h(1800, SECONDS_PER_FINETUNED_ITEM)),
]

#: Deferred by Ahmed until the core result exists: they document or harden a result rather
#: than produce one (`STATUS.md`, open items).
DEFERRED: list[tuple[str, float]] = [
    ("three training seeds", 3 * _train_h(STAGE1_STEPS + STAGE2_STEPS)),
    ("RefChartQA scaling ladder, 4k / 10k / 25k rows", 3 * _train_h(STAGE2_STEPS)),
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

    print("\n  committed ahead (derived from measured s/step and s/item):")
    for name, h in COMMITTED_AHEAD:
        print(f"    {name:<36} ~{h:4.1f} h")
    total = sum(h for _, h in COMMITTED_AHEAD)
    print(f"    {'total':<36} ~{total:4.1f} h")

    if remaining < total:
        print(f"\n  {remaining:.1f} h remain this window against ~{total:.1f} h committed.")
        weeks = total / WEEKLY_QUOTA_H
        print(f"  That is {weeks:.1f} account-weeks at {WEEKLY_QUOTA_H:.0f} h/week: "
              f"one account over {weeks:.1f} weeks, or one week across "
              f"{int(weeks) + 1} team accounts.")
        print("\n  deferred (not counted above):")
        for name, h in DEFERRED:
            print(f"    {name:<52}~{h:5.1f} h")
        print(f"    {'total deferred':<52}~{sum(h for _, h in DEFERRED):5.1f} h")
        print("  The window resets weekly, and every long job is resumable with checkpoints")
        print("  pushed on save (verified, DECISIONS.md 0033), so work spans windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
