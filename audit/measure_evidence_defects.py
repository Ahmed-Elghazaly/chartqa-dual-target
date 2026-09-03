#!/usr/bin/env python3
"""MEASUREMENT · three defects that reject correct plans, sized over real ChartQA.

Running the LLM-mining pipeline end-to-end on 40 unbiased records accepted 0 of 25 correct
proposals. None of the failures were the teacher's. Each is measured here at scale before
anything is changed, because each fix trades away some strictness and the trade should be
priced.

  1. MAX_EVIDENCE = 8   A chart with more elements than the schema allows is rejected
                        whatever the plan says -- and rejected as `malformed_plan`, which
                        blames the teacher for a schema limit.
  2. thousands written with a space   `'3 071'` is what the annotation carries; `to_number`
                        raises on it. `scripts/align_refchartqa.py` already normalises these,
                        but that fix was local to the aligner and never reached the executor.
  3. percent scale      An element value of `'5.3%'` parses to 0.053 while the gold answer
                        `'5.3'` parses to 5.3. `to_float` divides by 100 for a reason that
                        was verified (0045: the official metric scores `'81.9%'` as 0.819),
                        so the convention is right and the COMPARISON is what is wrong.
"""
from __future__ import annotations

import argparse
import collections
import random
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.eval.metrics import to_float  # noqa: E402
from chartqa_dt.plans.schema import MAX_EVIDENCE  # noqa: E402

SPACED = re.compile(r"^-?\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?$")


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

    sizes: list[int] = []
    spaced_charts = pct_answer_mismatch = charts = rows_seen = 0
    unparsable = collections.Counter()
    with ArchiveReader(archive_path()) as reader:
        pool = [r for k in ("human", "machine") for r in reader.qa_rows("train", k)]
        random.Random(args.seed).shuffle(pool)
        for row in pool:
            if rows_seen >= args.limit:
                break
            ann, img = (annotation_path("train", row["imgname"]),
                        image_path("train", row["imgname"]))
            if not (reader.exists(ann) and reader.exists(img)):
                continue
            w, h = reader.image_size(img)
            elements = annotation_boxes(reader.read_json(ann), w, h)
            if not elements:
                continue
            rows_seen += 1
            charts += 1
            sizes.append(len(elements))
            values = [str(e.get("value")) for e in elements if e.get("value") is not None]
            if any(SPACED.match(v) for v in values):
                spaced_charts += 1
            for v in values:
                if to_float(v) is None:
                    unparsable[re.sub(r"\d", "#", v)[:18]] += 1
            # a percent-valued chart whose gold answer carries no percent sign
            answer = str(row.get("label", ""))
            if values and all(v.endswith("%") for v in values) and not answer.endswith("%"):
                pct_answer_mismatch += 1

    over = sum(1 for s in sizes if s > MAX_EVIDENCE)
    sizes.sort()
    print(f"{charts:,} real ChartQA train charts (seed {args.seed})\n")
    print(f"  1. elements per chart: median {sizes[len(sizes) // 2]}, "
          f"mean {sum(sizes) / len(sizes):.1f}, max {sizes[-1]}")
    print(f"     over MAX_EVIDENCE={MAX_EVIDENCE}: {over:,}/{charts:,} "
          f"({100 * over / charts:.1f}%)  <-- rejected whatever the plan says\n")
    print(f"  2. charts with space-separated thousands: {spaced_charts:,}/{charts:,} "
          f"({100 * spaced_charts / charts:.1f}%)  <-- executor raises")
    if unparsable:
        print("     shapes `to_float` cannot read (digits shown as #):")
        for shape, k in unparsable.most_common(6):
            print(f"       {shape!r:<22}{k:>6,}")
    print(f"\n  3. all-percent chart, gold answer without a %: "
          f"{pct_answer_mismatch:,}/{charts:,} ({100 * pct_answer_mismatch / charts:.1f}%)"
          "  <-- executed value is 100x too small")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
