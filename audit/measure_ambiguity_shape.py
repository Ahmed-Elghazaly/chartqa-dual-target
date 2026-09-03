#!/usr/bin/env python3
"""MEASUREMENT · what the miner's `ambiguous` refusal actually is.

`ambiguous` is the miner's dominant refusal, and it is easy to misread. It does NOT mean
two cells hold the answer. It means two OPERATIONS reproduce it (`mining.py:282`, returned
when `len(hits) > 1`), so the uniqueness rule cannot say which one the question asked for.

On a sorted bar chart -- which is most of ChartQA -- the top row's value is simultaneously
`lookup(<that label>)` and `max` of the column. A question naming the label wants the
lookup; a question saying "highest" wants the extremum. The table alone cannot tell them
apart; one word of the question can.

This counts the shapes of that collision over real ChartQA training rows, so the value of
letting a reader see the question is a number rather than an argument.
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

from chartqa_dt.plans.mining import mine_plan  # noqa: E402

LOOKUPISH = {"lookup"}
EXTREMUM = {"max", "min", "argmax", "argmin"}


def shape(ops: list[str]) -> str:
    s = set(ops)
    if s & LOOKUPISH and s & EXTREMUM and not (s - LOOKUPISH - EXTREMUM):
        return "lookup_vs_extremum"
    if s <= EXTREMUM:
        return "extremum_vs_extremum"
    if s & LOOKUPISH:
        return "lookup_vs_other"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scripts.build_mixtures import archive_path

    from chartqa_dt.data.chartqa import ArchiveReader, parse_table, table_path

    status = collections.Counter()
    shapes = collections.Counter()
    pairs = collections.Counter()
    by_kind: collections.Counter[str] = collections.Counter()
    with ArchiveReader(archive_path()) as reader:
        seen = 0
        # Shuffle across BOTH question kinds. Iterating in order would take human rows
        # only until the limit, and human questions are the harder half.
        pool = [(k, r) for k in ("human", "machine") for r in reader.qa_rows("train", k)]
        random.Random(args.seed).shuffle(pool)
        for kind, row in pool:
                if seen >= args.limit:
                    break
                tbl = table_path("train", row["imgname"])
                if not reader.exists(tbl):
                    continue
                try:
                    table = parse_table(reader.read_text(tbl))
                except ValueError:
                    continue
                seen += 1
                by_kind[kind] += 1
                m = mine_plan([table["columns"], *table["rows"]], str(row.get("label", "")))
                status[m.status] += 1
                if m.status == "ambiguous":
                    shapes[shape(m.ops_matched)] += 1
                    pairs["+".join(sorted(m.ops_matched))] += 1

    print(f"deterministic miner over {seen:,} real ChartQA train rows "
          f"(random across both kinds: "
          f"{by_kind['human']:,} human, {by_kind['machine']:,} machine)\n")
    for st, k in status.most_common():
        print(f"  {st:<18}{k:>7,}  ({100 * k / seen:>5.1f}%)")
    amb = status["ambiguous"]
    print(f"\n  the {amb:,} `ambiguous` refusals, by what collided:")
    for sh, k in shapes.most_common():
        print(f"    {sh:<22}{k:>7,}  ({100 * k / amb:>5.1f}% of ambiguous, "
              f"{100 * k / seen:>4.1f}% of all)")
    print("\n  most common colliding operation sets:")
    for p, k in pairs.most_common(8):
        print(f"    {p:<34}{k:>7,}")
    recoverable = shapes["lookup_vs_extremum"] + shapes["extremum_vs_extremum"]
    print(f"\n  collisions a single word of the question resolves: {recoverable:,} "
          f"({100 * recoverable / seen:.1f}% of all rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
