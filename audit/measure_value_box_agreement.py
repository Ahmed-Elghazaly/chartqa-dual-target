#!/usr/bin/env python3
"""AUDIT · the value/box join — do an evidence entry's value and box describe the SAME mark?

`_evidence_from` builds an entry as:

    label  = the plan's label
    value  = _table_values(record)[label]        -> FIRST numeric column of that table row
    bbox   = by_label[label]["bbox"]             -> FIRST element in ANNOTATION order

On a single-series chart those coincide. On a multi-series chart they need not: 74.2% of
ChartQA charts have a label naming more than one element (audit/label_ambiguity.json).

If the emitted value belongs to a different mark than the emitted box, the target teaches
the model to point at one bar and report another bar's number.

This measures how often that happens on records that actually produce targets.
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
from chartqa_dt.train.targets import (  # noqa: E402
    TargetError,
    _table_values,
    build_target,
    plan_labels,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mixture", default="data/mixture_stage2.json")
    ap.add_argument("--out", default="audit/value_box_agreement.json")
    args = ap.parse_args()

    import argparse as _a

    from chartqa_dt.cli.train import _all_source_records
    ctx = _a.Namespace(args=_a.Namespace(mixture=args.mixture),
                       cfg=build_config(None), env=get_env())
    by_id = {r.record_id: r for r in _all_source_records(ctx)}
    ids = json.loads(Path(args.mixture).read_text(encoding="utf-8"))["record_ids"]

    stats = collections.Counter()
    examples = []
    for rid in ids:
        r = by_id.get(rid)
        if r is None or r.source != "chartqa":
            continue
        elements = [e for e in (r.meta.get(ELEMENTS_KEY) or []) if isinstance(e, dict)]
        if not elements or not r.plan:
            continue
        try:
            build_target(r)                      # only count records that ship
        except TargetError:
            continue
        stats["records"] += 1

        table_values = _table_values(r)
        by_label: dict[str, dict] = {}
        for e in elements:
            by_label.setdefault(str(e.get("label")), e)
        counts = collections.Counter(str(e.get("label")) for e in elements)

        record_bad = False
        for label in dict.fromkeys(plan_labels(r.plan)):
            el = by_label.get(label)
            if el is None:
                continue
            emitted = to_float(table_values.get(label))
            boxed = to_float(el.get("value"))
            if emitted is None or boxed is None:
                stats["unparseable_value"] += 1
                continue
            stats["evidence_entries"] += 1
            ambiguous = counts[label] > 1
            stats["entries_on_ambiguous_label"] += ambiguous
            if abs(emitted - boxed) > 1e-6:
                stats["MISMATCH"] += 1
                # Two distinct mechanisms, and they need different fixes.
                if abs(emitted * 100.0 - boxed) < 1e-6:
                    stats["cause_percent_scale"] += 1
                elif abs(emitted - boxed) / max(abs(boxed), 1e-9) < 0.02:
                    stats["cause_rounding"] += 1
                else:
                    stats["cause_genuine_disagreement"] += 1
                stats["mismatch_on_ambiguous_label"] += ambiguous
                record_bad = True
                if len(examples) < 8:
                    same = [e for e in elements if str(e.get("label")) == label]
                    examples.append({
                        "record": r.record_id, "label": label,
                        "emitted_value_from_table": emitted,
                        "value_of_the_boxed_element": boxed,
                        "label_names_n_elements": counts[label],
                        "all_values_for_that_label": [e.get("value") for e in same],
                        "series": [e.get("series") for e in same],
                        "question": r.question[:90],
                    })
        stats["records_with_a_mismatch"] += record_bad

    n = stats["evidence_entries"]
    print(f"\nChartQA records in {args.mixture} that build a target: {stats['records']:,}")
    print(f"evidence entries examined: {n:,}\n")
    print(f"  entries whose label names >1 element : {stats['entries_on_ambiguous_label']:,}"
          f"  ({100 * stats['entries_on_ambiguous_label'] / max(n, 1):.1f}%)")
    print("  entries where the emitted VALUE does not match the BOXED element:")
    print(f"      {stats['MISMATCH']:,} of {n:,}  ({100 * stats['MISMATCH'] / max(n, 1):.1f}%)")
    print(f"      of those, on an ambiguous label: {stats['mismatch_on_ambiguous_label']:,}")
    print(f"  records shipping at least one mismatched entry: "
          f"{stats['records_with_a_mismatch']:,} of {stats['records']:,}"
          f"  ({100 * stats['records_with_a_mismatch'] / max(stats['records'], 1):.1f}%)")
    if examples:
        print("\nexamples — the model is taught to box one mark and state another's number:")
        for e in examples[:5]:
            print(f"\n  {e['label']!r} names {e['label_names_n_elements']} marks "
                  f"{e['series']} with values {e['all_values_for_that_label']}")
            print(f"     target emits value {e['emitted_value_from_table']} "
                  f"but boxes the mark whose value is {e['value_of_the_boxed_element']}")
            print(f"     Q: {e['question']}")
    Path(args.out).write_text(json.dumps({"stats": dict(stats), "examples": examples},
                                         indent=1) + "\n", encoding="utf-8")
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
