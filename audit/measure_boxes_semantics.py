#!/usr/bin/env python3
"""AUDIT · Idea 1 — does `ChartRecord.boxes` have one meaning? Measured, not argued.

Writers of the field disagree about what it holds:

    chartqa.py     boxes = every element in the chart      (~12.7 per chart)
    refchartqa.py  boxes = this question's gold grounding  (~1-2)
    synthetic      boxes = this question's exact evidence  (~1-2)
    dedup.py       boxes = union of whichever two merged

This script measures the consequences on the mixtures that would actually be
trained and monitored on.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.config import build_config  # noqa: E402
from chartqa_dt.data.records import ELEMENTS_KEY  # noqa: E402
from chartqa_dt.env import get_env  # noqa: E402
from chartqa_dt.train.targets import TargetError, build_target, plan_labels  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mixture", default="data/mixture_stage1.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import argparse as _a

    from chartqa_dt.cli.train import _all_source_records
    ctx = _a.Namespace(args=_a.Namespace(mixture=args.mixture),
                       cfg=build_config(None), env=get_env())
    by_id = {r.record_id: r for r in _all_source_records(ctx)}
    ids = json.loads(Path(args.mixture).read_text(encoding="utf-8"))["record_ids"]

    rows = []
    for rid in ids:
        r = by_id.get(rid)
        if r is None:
            continue
        elements = [e for e in (r.meta.get(ELEMENTS_KEY) or []) if isinstance(e, dict)]
        merged = sorted(r.meta.get("merged_from") or [r.source])
        try:
            ev = json.loads(build_target(r))["evidence"]
            n_ev, usable = len(ev), True
        except TargetError:
            n_ev, usable = 0, False
        rows.append({
            "source": r.source, "merged_from": "+".join(merged),
            "n_boxes": len(r.boxes or []), "n_elements": len(elements),
            "has_plan": bool(r.plan), "n_plan_labels": len(set(plan_labels(r.plan))),
            "n_evidence": n_ev, "usable": usable,
        })

    print(f"\nmixture: {args.mixture}   records rehydrated: {len(rows):,}\n")

    # 1 — what does `boxes` hold, by source?
    print("What `record.boxes` actually contains, by source:")
    print(f"  {'source':<24}{'n':>7}{'median boxes':>14}{'median elements':>17}")
    by_src = collections.defaultdict(list)
    for r in rows:
        by_src[r["merged_from"]].append(r)
    for src, rs in sorted(by_src.items()):
        mb = statistics.median(x["n_boxes"] for x in rs)
        me = statistics.median(x["n_elements"] for x in rs)
        print(f"  {src:<24}{len(rs):>7,}{mb:>14.0f}{me:>17.0f}")

    # 2 — the monitoring metric uses record.boxes as ground truth for AP.
    print("\nIf `record.boxes` is used as grounding ground truth (cli/train.py does):")
    for src, rs in sorted(by_src.items()):
        usable = [x for x in rs if x["usable"] and x["n_boxes"]]
        if not usable:
            continue
        inflated = [x for x in usable if x["n_boxes"] > x["n_evidence"]]
        ratio = statistics.median(x["n_boxes"] / max(x["n_evidence"], 1) for x in usable)
        print(f"  {src:<24}{len(inflated):>6,}/{len(usable):<6,} records have MORE gt boxes "
              f"than the target emits   (median {ratio:.1f}x)")

    # 3 — records holding BOTH gold grounding and semantic elements
    both = [r for r in rows if "chartqa" in r["merged_from"]
            and "refchartqa" in r["merged_from"]]
    print(f"\nRecords carrying BOTH ChartQA elements and RefChartQA gold grounding: "
          f"{len(both):,}")
    if both:
        print(f"  of those, usable targets: {sum(r['usable'] for r in both):,}")
        print("  their gold question-grounding is currently DISCARDED: the target is built "
              "from elements filtered by the mined plan.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        print(f"\n  per-record rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
