#!/usr/bin/env python3
"""Draw the calibration sample an LLM miner is scored on.

`DECISIONS.md` 0078 established that RefChartQA's grounding gives **gold operand identity**:
for an aligned record we know which regions a correct answer uses. That makes it possible to
score a proposed plan on more than arithmetic — it must also read the right marks.

This draws a reproducible sample from the records the *deterministic* miner could not settle,
because those are exactly the cases a question-reading teacher is supposed to fix. Sampling
is seeded and the ids are recorded, so the measurement is repeatable and cannot be
cherry-picked after the fact.

Emits one JSON object per record with everything a teacher needs and nothing it should not
see: the question, the table, the marked regions with their values, and the DSL. **The gold
answer is included** because the task is *"which operation explains this answer"*, not
*"what is the answer"* — the miner is recovering a rationale for a known label, which is the
standard weak-supervision setting.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.data.records import ELEMENTS_KEY  # noqa: E402
from chartqa_dt.env import get_env  # noqa: E402
from chartqa_dt.plans.executor import NEEDS_TABLE, OPS  # noqa: E402
from chartqa_dt.train.targets import TargetError, build_target  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="audit/llm_mining_sample.jsonl")
    args = ap.parse_args()

    from scripts.build_mixtures import refchartqa_records

    root = Path(get_env().data_root)
    records = refchartqa_records(cap=100000, cache=root / "refchartqa_train.jsonl")

    pool = []
    for r in records:
        if not r.meta.get("aligned_to_chartqa") or r.answer is None:
            continue
        els = [e for e in (r.meta.get(ELEMENTS_KEY) or []) if isinstance(e, dict)]
        if not els or len(els) > 8:
            continue
        try:
            build_target(r)
            continue                     # already supervised; not what we are testing
        except TargetError as exc:
            if "no mined plan" not in str(exc):
                continue
        pool.append((r, els))

    print(f"eligible: {len(pool):,} aligned records the deterministic miner could not settle")
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))

    usable_ops = sorted(OPS - set(NEEDS_TABLE))
    out = Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        for r, els in sample:
            fh.write(json.dumps({
                "record_id": r.record_id,
                "question": r.question,
                "gold_answer": r.answer,
                "marked_regions": [{"label": str(e.get("label")), "value": e.get("value"),
                                    "unit": e.get("unit")} for e in els],
                "table": r.table,
                "allowed_operations": usable_ops,
            }, ensure_ascii=False) + "\n")
    print(f"sample of {len(sample)} (seed {args.seed}) -> {out}")
    print("\nallowed operations:", ", ".join(usable_ops))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
