#!/usr/bin/env python3
"""MEASUREMENT · forwards vs backwards, on identical records.

`plans.mining` asks *which operations reproduce this answer?* and refuses when several do —
53.9% of ChartQA rows. `plans.forward` asks *what does the question ask for?*, builds that
plan, and checks it against the answer. The ambiguity that dominates the first cannot arise
in the second.

Both are run over the same random sample so the comparison is a difference, not two
separately quoted numbers. Evidence is the chart's own annotated elements — what the target
builder and the executor actually see — not the table, so a plan counted here is a plan that
can really become a target.
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

from chartqa_dt.data.records import qualified_labels  # noqa: E402
from chartqa_dt.plans import forward  # noqa: E402
from chartqa_dt.plans.mining import mine_plan  # noqa: E402


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

    back = fwd = both = only_fwd = only_back = seen = 0
    fwd_ops: collections.Counter[str] = collections.Counter()
    fwd_fail: collections.Counter[str] = collections.Counter()
    examples: list[tuple[str, str, dict]] = []

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
            evidence = [{"label": n, "value": e.get("value"), "unit": e.get("unit")}
                        for n, e in zip(qualified_labels(elements), elements)]

            got_back = mine_plan([table["columns"], *table["rows"]], answer).plan is not None
            built = forward.build(question, answer=answer, evidence=evidence)

            back += got_back
            fwd += built.ok
            both += got_back and built.ok
            only_fwd += built.ok and not got_back
            only_back += got_back and not built.ok
            if built.ok:
                fwd_ops[built.op or "?"] += 1
                if not got_back and len(examples) < 6:
                    examples.append((question, answer, built.plan or {}))
            else:
                fwd_fail[built.reason.split(";")[0][:52]] += 1

    print(f"{seen:,} real ChartQA train records, same sample for both (seed {args.seed})\n")
    print(f"  BACKWARDS (plans.mining)  : {back:>6,}  ({100 * back / seen:>5.1f}%)")
    print(f"  FORWARDS  (plans.forward) : {fwd:>6,}  ({100 * fwd / seen:>5.1f}%)"
          f"   {'+' if fwd > back else ''}{100 * (fwd - back) / seen:.1f} points")
    print(f"\n  both agree a plan exists : {both:,}")
    print(f"  only forwards            : {only_fwd:,}   <-- recovered")
    print(f"  only backwards           : {only_back:,}   <-- lost, worth inspecting")
    print("\n  what forwards built:")
    for op, k in fwd_ops.most_common():
        print(f"    {op:<16}{k:>6,}  ({100 * k / max(fwd, 1):>5.1f}%)")
    print("\n  why forwards declined:")
    for why, k in fwd_fail.most_common(6):
        print(f"    {k:>6,}  {why}")
    if examples:
        print("\n  recovered examples (backwards refused these):")
        for q, a, plan in examples:
            print(f"    {plan}  == {a!r}\n      {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
