#!/usr/bin/env python3
"""AUDIT · Idea 7 — how often does a "unique" mined plan use the WRONG operands?

`mine_plan` accepts an operation when exactly one reproduces the gold answer. That is
numerical agreement, not semantic correctness. RefChartQA independently states which regions
a correct answer uses, so on aligned records the two can be compared — giving a measurement
of the miner's semantic error rate at scale, without hand-labelling.

This prints the disagreements so they can be judged rather than trusted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord  # noqa: E402
from chartqa_dt.env import get_env  # noqa: E402
from chartqa_dt.plans.mining import mine_plan  # noqa: E402
from chartqa_dt.train.targets import plan_labels  # noqa: E402


def main() -> int:
    root = Path(get_env().data_root)
    aligned = {}
    for line in (root / "refchartqa_aligned.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            a = json.loads(line)
            aligned[a["record_id"]] = a
    records = {r.record_id: r for r in (
        ChartRecord.from_dict(json.loads(line))
        for line in (root / "refchartqa_train.jsonl").read_text(
            encoding="utf-8").splitlines() if line)}

    shown = 0
    n_unique = n_bad = 0
    for rid, a in aligned.items():
        r = records.get(rid)
        table = a.get("table")
        if r is None or not table or r.answer is None:
            continue
        rows = [table.get("columns") or [], *(table.get("rows") or [])]
        if len(rows) < 2:
            continue
        mined = mine_plan(rows, r.answer)
        if mined.plan is None:
            continue
        n_unique += 1
        used = {str(x) for x in plan_labels(mined.plan)}
        marked = {str(e.get("label")) for e in a[ELEMENTS_KEY]}
        if used and used <= marked:
            continue
        n_bad += 1
        if shown < 6:
            shown += 1
            print(f"\n--- {rid} ---")
            print(f"  question : {r.question[:95]}")
            print(f"  answer   : {r.answer!r}")
            print(f"  miner says: {json.dumps(mined.plan)}")
            print(f"  it reads  : {sorted(used)}")
            print(f"  RefChartQA marks: {sorted(marked)}")
            print("  -> the plan reaches the right number from marks the annotation "
                  "does not consider relevant")
    print(f"\n\nrecords where the miner found a UNIQUE plan : {n_unique:,}")
    print(f"of those, operands disagree with the gold grounding: {n_bad:,} "
          f"({100 * n_bad / max(n_unique, 1):.1f}%)")
    print("\nThis is the deterministic miner's SEMANTIC error rate, measured against an "
          "independent gold source rather than estimated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
