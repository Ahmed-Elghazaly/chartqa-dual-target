#!/usr/bin/env python3
"""AUDIT · C1 mechanism 2 — what do the percent-scale records actually look like?

`_table_values` parses a table cell with `to_float`, which divides a "%" value by 100.
The annotation stores the same number unscaled. Before deciding whether that is a bug or a
consistent convention, look at what the plan and the gold answer do with it.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.config import build_config  # noqa: E402
from chartqa_dt.data.records import ELEMENTS_KEY  # noqa: E402
from chartqa_dt.env import get_env  # noqa: E402
from chartqa_dt.eval.metrics import to_float  # noqa: E402
from chartqa_dt.plans.roundtrip import check_record  # noqa: E402
from chartqa_dt.train.targets import (  # noqa: E402
    TargetError,
    _table_values,
    build_target,
    plan_labels,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mixture", default="data/mixture_stage2.json")
    args = ap.parse_args()
    import argparse as _a

    from chartqa_dt.cli.train import _all_source_records
    ctx = _a.Namespace(args=_a.Namespace(mixture=args.mixture),
                       cfg=build_config(None), env=get_env())
    by_id = {r.record_id: r for r in _all_source_records(ctx)}
    ids = json.loads(Path(args.mixture).read_text(encoding="utf-8"))["record_ids"]

    kinds = collections.Counter()
    shown = 0
    for rid in ids:
        r = by_id.get(rid)
        if r is None or r.source != "chartqa" or not r.plan:
            continue
        els = [e for e in (r.meta.get(ELEMENTS_KEY) or []) if isinstance(e, dict)]
        if not els:
            continue
        try:
            target = json.loads(build_target(r))
        except TargetError:
            continue
        tv = _table_values(r)
        by_label = {}
        for e in els:
            by_label.setdefault(str(e.get("label")), e)
        hit = None
        for label in dict.fromkeys(plan_labels(r.plan)):
            el = by_label.get(label)
            if el is None:
                continue
            a, b = to_float(tv.get(label)), to_float(el.get("value"))
            if a is not None and b is not None and abs(a * 100 - b) < 1e-6:
                hit = (label, a, b, el.get("value"))
                break
        if hit is None:
            continue
        kinds[str(r.plan.get("op"))] += 1
        rt = check_record(target)
        kinds[f"roundtrip:{rt.outcome}"] += 1
        if shown < 5:
            shown += 1
            print(f"\n--- {r.record_id}  op={r.plan.get('op')} ---")
            print(f"  question       : {r.question[:80]}")
            print(f"  gold answer    : {r.answer!r}")
            print(f"  label {hit[0]!r}: table -> {hit[1]}   annotation -> {hit[3]!r}")
            print(f"  target evidence: {[(e['label'], e['value']) for e in target['evidence']][:3]}")
            print(f"  round-trip     : {rt.outcome}  executed={rt.executed!r} stated={rt.stated!r}")

    print("\n\nacross all percent-scale records that ship a target:")
    for k, v in sorted(kinds.items()):
        print(f"  {k:<26}{v:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
