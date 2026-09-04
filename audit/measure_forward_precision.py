#!/usr/bin/env python3
"""MEASUREMENT · is forward construction as PRECISE as the miner it replaces?

Forward construction triples the yield (14.9% -> 44.9%). That is worthless if the plans are
worse, so this measures agreement on the records where both methods commit.

Where `plans.mining` returns `unique`, exactly one operation reproduces the answer — the
operation is not in doubt. If `plans.forward` builds a different one on the same record,
one of them is wrong, and since the miner's verdict is forced by arithmetic it is forward
that must justify itself. Every disagreement is printed in full so it can be judged rather
than counted.

`sum2` / `mean2` are the miner's own names for a fold over a two-column flattening; they are
compared against `sum` / `mean` rather than treated as different operations.
"""
from __future__ import annotations

import argparse
import collections
import math
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.data.records import qualified_labels  # noqa: E402
from chartqa_dt.plans import forward  # noqa: E402
from chartqa_dt.plans.mining import mine_plan  # noqa: E402

#: The miner distinguishes flattenings; forward does not, and the distinction is internal.
ALIASES = {"sum2": "sum", "mean2": "mean", "median2": "median", "count2": "count"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c, h = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scripts.build_mixtures import archive_path

    from chartqa_dt.data.chartqa import (
        ArchiveReader,
        annotation_boxes,
        annotation_path,
        image_path,
        parse_table,
        table_path,
    )

    agree = disagree = seen = 0
    pairs: collections.Counter[tuple[str, str]] = collections.Counter()
    shown: list[tuple[str, str, str, str]] = []

    with ArchiveReader(archive_path()) as reader:
        pool = [r for k in ("human", "machine") for r in reader.qa_rows("train", k)]
        random.Random(args.seed).shuffle(pool)
        for row in pool:
            if seen >= args.limit:
                break
            name = row["imgname"]
            ann, img, tbl = (annotation_path("train", name), image_path("train", name),
                             table_path("train", name))
            if not (reader.exists(ann) and reader.exists(img) and reader.exists(tbl)):
                continue
            try:
                table = parse_table(reader.read_text(tbl))
            except ValueError:
                continue
            w, h = reader.image_size(img)
            elements = annotation_boxes(reader.read_json(ann), w, h)
            if not elements:
                continue
            seen += 1
            question, answer = str(row["query"]), str(row.get("label", ""))
            mined = mine_plan([table["columns"], *table["rows"]], answer)
            if mined.status != "unique" or not mined.op:
                continue
            evidence = [{"label": n, "value": e.get("value"), "unit": e.get("unit")}
                        for n, e in zip(qualified_labels(elements), elements)]
            built = forward.build(question, answer=answer, evidence=evidence)
            if not built.ok:
                continue
            truth = ALIASES.get(mined.op, mined.op)
            got = ALIASES.get(built.op or "", built.op or "")
            pairs[(truth, got)] += 1
            if truth == got:
                agree += 1
            else:
                disagree += 1
                if len(shown) < 10:
                    shown.append((question, answer, truth, got))

    judged = agree + disagree
    lo, hi = wilson(agree, judged)
    print(f"{seen:,} records; both methods produced a plan on {judged:,} of them\n")
    print(f"  same operation      : {agree:,}")
    print(f"  different operation : {disagree:,}")
    if judged:
        print(f"  AGREEMENT           : {100 * agree / judged:.1f}%  "
              f"(95% CI {100 * lo:.1f}–{100 * hi:.1f}%)")
    print("\n  most common pairings (miner -> forward):")
    for (t, g), k in pairs.most_common(8):
        mark = "" if t == g else "   <-- differs"
        print(f"    {t:<12} -> {g:<12}{k:>5,}{mark}")
    if shown:
        print("\n  every disagreement, to be judged rather than counted:")
        for q, a, t, g in shown:
            print(f"    miner {t!r} vs forward {g!r}, answer {a!r}\n      {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
