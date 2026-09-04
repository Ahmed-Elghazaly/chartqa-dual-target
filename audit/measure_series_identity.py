#!/usr/bin/env python3
"""MEASUREMENT · would the series name make chart element labels unique?

`AUDIT.md` H3: labels are not unique, and the two sides that resolve them disagree —
`train.targets` keeps the FIRST element with a given label (`by_label.setdefault`) and
`plans.executor` keeps the LAST (`{e.label: e for e in evidence}`). On a grouped chart,
"2019" names one bar per series, so a plan saying `lookup('2019')` means two different bars
depending on which module is reading it.

Running the LLM teacher over 40 unbiased records made this the largest single blocker:
6 of its 15 refusals were "this label appears N times and nothing says which".

The annotation already carries the answer. `chartqa.py::_series_elements` writes
`"series": model.get("name")` on every element and **nothing downstream reads it** — the
schema has no field for it, the target builder drops it, the executor never sees it. Before
proposing that it be carried through, this measures whether it would actually help: how
often labels collide, how often the series name is present, and whether (series, label) is
unique where the label alone is not.
"""
from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scripts.build_mixtures import archive_path

    from chartqa_dt.data.chartqa import (
        ArchiveReader,
        annotation_boxes,
        annotation_path,
        image_path,
    )

    charts = dup_charts = series_named = pair_unique = pair_not_unique = 0
    dup_sizes: collections.Counter[int] = collections.Counter()
    seen_images: set[str] = set()
    with ArchiveReader(archive_path()) as reader:
        pool = [r for k in ("human", "machine") for r in reader.qa_rows("train", k)]
        random.Random(args.seed).shuffle(pool)
        for row in pool:
            if charts >= args.limit:
                break
            name = row["imgname"]
            if name in seen_images:
                continue
            ann, img = annotation_path("train", name), image_path("train", name)
            if not (reader.exists(ann) and reader.exists(img)):
                continue
            w, h = reader.image_size(img)
            elements = annotation_boxes(reader.read_json(ann), w, h)
            if not elements:
                continue
            seen_images.add(name)
            charts += 1
            labels = [str(e.get("label")) for e in elements]
            counts = collections.Counter(labels)
            worst = max(counts.values())
            if worst == 1:
                continue
            dup_charts += 1
            dup_sizes[worst] += 1
            if all(e.get("series") is not None for e in elements):
                series_named += 1
            pairs = [(str(e.get("series")), str(e.get("label"))) for e in elements]
            if len(set(pairs)) == len(pairs):
                pair_unique += 1
            else:
                pair_not_unique += 1

    print(f"{charts:,} distinct ChartQA train charts (seed {args.seed})\n")
    print(f"  a label appears more than once : {dup_charts:,}  "
          f"({100 * dup_charts / charts:.1f}%)")
    if not dup_charts:
        return 0
    print(f"  of those, every element has a series name : {series_named:,}  "
          f"({100 * series_named / dup_charts:.1f}%)")
    print(f"\n  (series, label) is unique      : {pair_unique:,}  "
          f"({100 * pair_unique / dup_charts:.1f}% of colliding charts)   <-- fixable")
    print(f"  still collides even with series: {pair_not_unique:,}  "
          f"({100 * pair_not_unique / dup_charts:.1f}%)")
    print("\n  how many times the worst label repeats:")
    for k, v in sorted(dup_sizes.items())[:6]:
        print(f"    {k}x{'':<4}{v:>6,}  ({100 * v / dup_charts:>5.1f}% of colliding charts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
