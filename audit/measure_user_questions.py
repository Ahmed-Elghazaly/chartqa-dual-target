#!/usr/bin/env python3
"""AUDIT · three claims to check before arguing about them.

1. Are ChartQA and RefChartQA "the same questions"?
2. Does the miner give up on label answers that `argmax`/`argmin` could actually produce?
3. What do the mixture caps cost, given how much data we now have?
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.config import build_config  # noqa: E402
from chartqa_dt.data.records import ELEMENTS_KEY, normalise_question  # noqa: E402
from chartqa_dt.env import get_env  # noqa: E402
from chartqa_dt.eval.metrics import to_float  # noqa: E402
from chartqa_dt.plans.executor import EvidenceItem, execute  # noqa: E402


def main() -> int:
    from scripts.build_mixtures import archive_path, chartqa_records, refchartqa_records

    from chartqa_dt.data.chartqa import ArchiveReader
    from chartqa_dt.data.mixture import CHARTQA_DRAW

    cfg = build_config(None)
    root = Path(get_env().data_root)

    # ---- 1. question overlap -------------------------------------------------
    cq = chartqa_records(ArchiveReader(archive_path()), limit=CHARTQA_DRAW, seed=cfg.seed)
    ref = refchartqa_records(cap=100000, cache=root / "refchartqa_train.jsonl")
    cq_keys = {(r.image_sha256, normalise_question(r.question)) for r in cq}
    cq_images = {r.image_sha256 for r in cq}
    same_q = sum(1 for r in ref
                 if (r.image_sha256, normalise_question(r.question)) in cq_keys)
    same_img = sum(1 for r in ref if r.image_sha256 in cq_images)
    print("1. ARE THEY THE SAME QUESTIONS?")
    print(f"   ChartQA records drawn        : {len(cq):,}")
    print(f"   RefChartQA records cached    : {len(ref):,}")
    print(f"   same IMAGE as a ChartQA row  : {same_img:,}  ({100 * same_img / len(ref):.1f}%)")
    print(f"   same image AND same question : {same_q:,}  ({100 * same_q / len(ref):.1f}%)")

    # ---- 2. label answers the miner refuses ---------------------------------
    print("\n2. LABEL ANSWERS THE MINER GIVES UP ON")
    stats = collections.Counter()
    examples = []
    for r in ref:
        if r.answer is None or to_float(r.answer) is not None:
            continue                                   # numeric answer, not this case
        els = [e for e in (r.meta.get(ELEMENTS_KEY) or []) if isinstance(e, dict)]
        if len(els) < 2:
            continue
        stats["label_answer_with_multiple_marks"] += 1
        evidence = [EvidenceItem(str(e.get("label")), e.get("value"), e.get("unit"))
                    for e in els]
        for op in ("argmax", "argmin"):
            try:
                got = execute({"op": op, "args": []}, evidence)
            except Exception:                          # noqa: BLE001
                continue
            if isinstance(got, str) and got.strip().lower() == str(r.answer).strip().lower():
                stats[f"explained by {op}"] += 1
                if len(examples) < 4:
                    examples.append({"q": r.question[:76], "a": r.answer, "op": op})
                break
    n = stats["label_answer_with_multiple_marks"]
    print(f"   records whose answer is a LABEL and which mark >1 region: {n:,}")
    for k in ("explained by argmax", "explained by argmin"):
        print(f"     {k:<26}{stats[k]:>5,}")
    total = stats["explained by argmax"] + stats["explained by argmin"]
    print(f"   -> {total:,} of {n:,} ({100 * total / max(n, 1):.1f}%) ARE explained by an "
          f"operation the miner never tries, because it")
    print("      returns `non_numeric` the moment the gold answer is not a number.")
    for e in examples:
        print(f"        {e['op']:<7} {e['a']!r:<12} {e['q']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
