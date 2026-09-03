#!/usr/bin/env python3
"""AUDIT · ChartQA element labels are TRUNCATED relative to its table labels.

Discovered while checking mined plans against RefChartQA's gold grounding: the miner reads
`'MSCI Global, excluding U.S.'` from the table while the annotation calls the same mark
`'MSCI Global, excluding'`. The annotation appears to store the label *as drawn*, clipped to
the space available.

That breaks any join on label equality, and this project joins on label equality in three
places:

  * `targets._evidence_from` — `by_label[plan_label]`, so a truncated element makes the
    record fail with "the plan references X, which has no element box";
  * `align_refchartqa.mine_grounded_plan` — comparing mined operands with marked regions;
  * the emitted evidence label itself, which is what the model is trained to produce.

Measured here so the fix is sized rather than guessed.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--out", default="audit/label_truncation.json")
    args = ap.parse_args()

    from scripts.build_mixtures import archive_path, chartqa_records

    from chartqa_dt.data.chartqa import ArchiveReader

    cfg = build_config(None)
    stats = collections.Counter()
    examples = []
    seen_images = set()
    for r in chartqa_records(ArchiveReader(archive_path()), limit=args.limit, seed=cfg.seed):
        if r.image_sha256 in seen_images or not r.table:
            continue
        seen_images.add(r.image_sha256)
        els = [e for e in (r.meta.get(ELEMENTS_KEY) or []) if isinstance(e, dict)]
        if not els:
            continue
        table_labels = {str(row[0]).strip() for row in (r.table.get("rows") or []) if row}
        if not table_labels:
            continue
        stats["charts"] += 1
        for e in els:
            label = str(e.get("label")).strip()
            stats["elements"] += 1
            if label in table_labels:
                stats["exact"] += 1
                continue
            prefixes = [t for t in table_labels if t.startswith(label) and len(t) > len(label)]
            if prefixes:
                stats["element_is_a_PREFIX_of_a_table_label"] += 1
                if len(examples) < 8:
                    examples.append({"element": label, "table": sorted(prefixes)[0],
                                     "chart": r.image_sha256[:12]})
            elif any(label.startswith(t) for t in table_labels):
                stats["table_label_is_a_prefix_of_the_element"] += 1
            else:
                stats["no_relation"] += 1

    n = stats["elements"]
    print(f"\ncharts inspected: {stats['charts']:,}   elements: {n:,}\n")
    for k in ("exact", "element_is_a_PREFIX_of_a_table_label",
              "table_label_is_a_prefix_of_the_element", "no_relation"):
        print(f"  {k:<42}{stats[k]:>7,}  ({100 * stats[k] / max(n, 1):5.1f}%)")
    print("\nexamples of truncation:")
    for e in examples:
        print(f"  element {e['element']!r}\n     table {e['table']!r}")
    Path(args.out).write_text(json.dumps({"stats": dict(stats), "examples": examples},
                                         indent=1) + "\n", encoding="utf-8")
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
