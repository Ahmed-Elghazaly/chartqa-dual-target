#!/usr/bin/env python3
"""AUDIT · how much GROUNDING supervision is discarded for lacking a plan?

`PLAN.md` 6.1 makes stage 1 **grounding only** — teach the model where to point, before
teaching it to reason. But `targets.build_record` requires a plan and refuses without one,
so a record with gold boxes and no derivable plan is dropped from every mixture.

That is a design choice, not a necessity, and this measures what it costs.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.data.records import ELEMENTS_KEY  # noqa: E402
from chartqa_dt.env import get_env  # noqa: E402
from chartqa_dt.train.targets import TargetError, build_target  # noqa: E402


def main() -> int:
    from scripts.build_mixtures import refchartqa_records

    root = Path(get_env().data_root)
    records = refchartqa_records(cap=100000, cache=root / "refchartqa_train.jsonl")

    stats = collections.Counter()
    for r in records:
        stats["records"] += 1
        has_boxes = bool(r.boxes)
        aligned = bool(r.meta.get("aligned_to_chartqa"))
        els = [e for e in (r.meta.get(ELEMENTS_KEY) or []) if isinstance(e, dict)]
        named = bool(els)
        try:
            build_target(r)
            stats["usable_with_a_plan"] += 1
            continue
        except TargetError as exc:
            reason = str(exc)
        if not has_boxes:
            stats["no_boxes_at_all"] += 1
            continue
        if "no mined plan" in reason:
            stats["HAS BOXES, no plan"] += 1
            stats["  ...and semantically named" if named else "  ...still item1"] += 1
            stats["  ...aligned" if aligned else "  ...unaligned"] += 1
        else:
            stats["refused_for_another_reason"] += 1

    n = stats["records"]
    print(f"\nRefChartQA cached records: {n:,}\n")
    for k, v in stats.items():
        if k == "records":
            continue
        print(f"  {k:<34}{v:>7,}  ({100 * v / n:5.1f}%)")

    lost = stats["HAS BOXES, no plan"]
    from chartqa_dt.config import build_config  # noqa: F401
    full = 55789
    print("\n  These have gold grounding and are dropped only for lacking a plan.")
    print(f"  Projected over the full RefChartQA train split: "
          f"~{int(full * lost / n):,} records of real grounding supervision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
