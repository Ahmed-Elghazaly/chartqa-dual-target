#!/usr/bin/env python3
"""AUDIT · Idea 2 — are chart element labels actually unique?

The target builder resolves a plan's label against the chart's elements with
`by_label.setdefault(label, element)` — FIRST match wins. The executor resolves the
same label with `{e.label: e for e in evidence}` — LAST match wins. Neither carries
the `series` the element belongs to, and the output schema has no field for it.

On a grouped or multi-series chart, "2019" names one bar per series. If that happens
often, a mined plan referencing "2019" is ambiguous, and the two sides above resolve it
differently.

Measured over real ChartQA train annotations.
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

from chartqa_dt.data.chartqa import (  # noqa: E402
    ArchiveReader,
    annotation_boxes,
    image_path,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--out", default="audit/label_ambiguity.json")
    args = ap.parse_args()

    from scripts.build_mixtures import archive_path

    reader = ArchiveReader(archive_path())
    # ArchiveReader keeps the member set privately; it exposes `exists`/`read` only.
    names = sorted(n for n in reader._names
                   if n.startswith("ChartQA Dataset/train/annotations/")
                   and n.endswith(".json"))
    names = names[:args.limit]

    stats = collections.Counter()
    by_type = collections.defaultdict(collections.Counter)
    dup_examples = []
    series_counts = collections.Counter()

    for name in names:
        stem = Path(name).stem
        try:
            ann = reader.read_json(name)
            img = reader.read(image_path("train", stem + ".png"))
        except Exception:                       # noqa: BLE001 — missing member
            stats["unreadable"] += 1
            continue
        from io import BytesIO

        from PIL import Image
        w, h = Image.open(BytesIO(img)).size
        els = annotation_boxes(ann, w, h)
        if not els:
            stats["no_elements"] += 1
            continue
        ctype = str(ann.get("type"))
        labels = [str(e.get("label")) for e in els]
        counts = collections.Counter(labels)
        dups = {k: v for k, v in counts.items() if v > 1}
        n_series = len({e.get("series") for e in els})
        series_counts[n_series] += 1
        stats["charts"] += 1
        by_type[ctype]["charts"] += 1
        if dups:
            stats["charts_with_duplicate_labels"] += 1
            by_type[ctype]["dup"] += 1
            if len(dup_examples) < 6:
                lab = max(dups, key=dups.get)
                dup_examples.append({
                    "chart": stem, "type": ctype, "label": lab, "times": dups[lab],
                    "series": sorted({str(e.get("series")) for e in els
                                      if str(e.get("label")) == lab}),
                    "values": [e.get("value") for e in els
                               if str(e.get("label")) == lab],
                })

    n = stats["charts"]
    print(f"\nChartQA train annotations inspected: {n:,}\n")
    print(f"charts where at least one label names MORE THAN ONE element: "
          f"{stats['charts_with_duplicate_labels']:,}  "
          f"({100 * stats['charts_with_duplicate_labels'] / max(n, 1):.1f}%)")
    print("\nby chart type:")
    for t, c in sorted(by_type.items()):
        print(f"  {t:<12}{c['dup']:>6,} / {c['charts']:<7,}"
              f"  ({100 * c['dup'] / max(c['charts'], 1):5.1f}%)")
    print("\nnumber of distinct series per chart:")
    for k, v in sorted(series_counts.items()):
        print(f"  {k} series: {v:>6,} charts")
    print("\nexamples of an ambiguous label:")
    for e in dup_examples:
        print(f"  {e['type']:<8} label {e['label']!r} names {e['times']} elements "
              f"in series {e['series']} with values {e['values']}")

    Path(args.out).write_text(json.dumps(
        {"n_charts": n, "charts_with_duplicate_labels": stats["charts_with_duplicate_labels"],
         "by_type": {t: dict(c) for t, c in by_type.items()},
         "series_per_chart": dict(series_counts),
         "examples": dup_examples}, indent=1) + "\n", encoding="utf-8")
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
