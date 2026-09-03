#!/usr/bin/env python3
"""MEASUREMENT · where the supervision is actually lost: expressiveness, or mining?

Claude judged 60 random ChartQA questions and found 56 (93.3%) expressible in the current
DSL (`audit/judge_dsl_sample.py`). This runs the DETERMINISTIC miner on the same 60 and
compares, which separates the two failure modes on identical records:

  * a question judged inexpressible  -> the DSL is the constraint; a better miner cannot help
  * expressible but the miner refuses -> mining is the constraint; the question text would help

Same records, same gold tables, so the two rates are directly comparable.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from audit.judge_dsl_sample import JUDGEMENTS  # noqa: E402
from chartqa_dt.plans.mining import mine_plan  # noqa: E402


def main() -> int:
    rows = [json.loads(x) for x in
            Path("audit/dsl_expressiveness_sample.jsonl").read_text(encoding="utf-8")
            .splitlines() if x]

    outcomes, reasons = collections.Counter(), collections.Counter()
    lost = []
    for i, r in enumerate(rows):
        mined = mine_plan([r["columns"], *r["rows"]], r["answer"])
        judged = JUDGEMENTS.get(i, ("UNJUDGED", ""))[0]
        got = mined.plan is not None
        outcomes[(judged, got)] += 1
        if not got:
            reasons[mined.status] += 1
            if judged == "ok":
                lost.append((i, JUDGEMENTS[i][1], mined.status,
                             r["question"]))

    n = len(rows)
    mined_ok = sum(v for (_, g), v in outcomes.items() if g)
    expressible = sum(v for (j, _), v in outcomes.items() if j == "ok")
    print(f"the same {n} random questions, judged and then mined\n")
    print(f"  expressible in the DSL (Claude's judgement) : {expressible}/{n}  "
          f"({100 * expressible / n:.1f}%)")
    print(f"  the deterministic miner settles             : {mined_ok}/{n}  "
          f"({100 * mined_ok / n:.1f}%)")
    print(f"  expressible but NOT mined                   : {len(lost)}/{n}  "
          f"({100 * len(lost) / n:.1f}%)   <-- the recoverable gap\n")
    print("  why the miner refused:")
    for why, k in reasons.most_common():
        print(f"    {why:<26}{k:>4}")
    by_op = collections.Counter(op for _, op, _, _ in lost)
    print("\n  what the lost supervision would have been:")
    for op, k in by_op.most_common():
        print(f"    {op:<18}{k:>4}")
    print("\n  a few of the lost ones:")
    for i, op, why, q in lost[:8]:
        print(f"    [{i}] wanted {op}, miner said {why}")
        print(f"         {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
