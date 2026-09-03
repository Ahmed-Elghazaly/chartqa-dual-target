#!/usr/bin/env python3
"""AUDIT · are the newly-refused RefChartQA records refused CORRECTLY?

Enrichment gave RefChartQA evidence a real value from ChartQA instead of the answer copied
in. Yield fell 2,063 -> 1,272 and 791 records now fail the round-trip. Either those records
were previously passing a circular check, or enrichment broke them. Look, do not assume.
"""
from __future__ import annotations

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
    shown = 0
    kinds = {"single_box": 0, "multi_box": 0}
    for r in records:
        if not r.meta.get("aligned_to_chartqa"):
            continue
        try:
            build_target(r)
            continue
        except TargetError as exc:
            if "does not reproduce" not in str(exc):
                continue
        els = r.meta.get(ELEMENTS_KEY) or []
        kinds["single_box" if len(els) == 1 else "multi_box"] += 1
        if shown < 6:
            shown += 1
            print(f"\n--- {r.record_id} ---")
            print(f"  question   : {r.question[:100]}")
            print(f"  gold answer: {r.answer!r}")
            print(f"  the {len(els)} region(s) RefChartQA marks, now identified:")
            for e in els[:4]:
                print(f"      {e.get('label')!r:<32} value {e.get('value')!r:<10} "
                      f"(IoU {e.get('match_iou')})")
            print("  -> refused: lookup over that region does not give the answer")
    print(f"\n\nnewly refused, by evidence count: {kinds}")
    print("\nInterpretation: a record is refused when the region marked does not itself "
          "contain the answer — i.e. the answer is DERIVED, so `lookup` was never the "
          "right plan for it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
