#!/usr/bin/env python3
"""MEASUREMENT · when the operation is unique, are its OPERANDS?

`Prompt.md` Idea 8 asks for special attention to a case the audit had not measured:

    one operation TYPE is unique
    but
    multiple concrete programs of that type produce the same answer

The miner's uniqueness rule counts **operations** (`mining.py` returns `ambiguous` when more
than one op reproduces the answer). It does not ask whether, within the single surviving
operation, more than one choice of operands also reproduces it. A plan can be the right
operation on the wrong pair — `difference("Alpha","Beta")` and `difference("Gamma","Delta")`
are different claims about which marks the question is about, and a target built from the
wrong one points at two marks the question never mentioned.

This counts how often that happens, over real ChartQA tables, for the two-operand operations
where it is possible: `difference`, `ratio` and `percent_change`.

Only records the miner calls `unique` are examined — by construction the ambiguity measured
here is invisible to it.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.plans.mining import matches_gold, mine_plan, to_number  # noqa: E402

#: Operations whose meaning depends on WHICH two operands were chosen.
PAIRWISE = {"difference", "ratio", "percent_change"}


def concrete_matches(op: str, values: list[tuple[str, float]], target) -> list[tuple[str, str]]:
    """Every ordered operand pair of `op` that reproduces the gold answer."""
    out = []
    for (la, a), (lb, b) in itertools.permutations(values, 2):
        if op == "difference":
            got = a - b
        elif op == "ratio":
            if b == 0:
                continue
            got = a / b
        else:
            if b == 0:
                continue
            got = 100.0 * (a - b) / b
        if matches_gold(got, target):
            out.append((la, lb))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scripts.build_mixtures import archive_path

    from chartqa_dt.data.chartqa import ArchiveReader, parse_table, table_path

    unique_pairwise = 0
    counts: collections.Counter[int] = collections.Counter()
    examples: list[tuple[str, str, str, list]] = []
    seen = 0

    with ArchiveReader(archive_path()) as reader:
        pool = [r for k in ("human", "machine") for r in reader.qa_rows("train", k)]
        random.Random(args.seed).shuffle(pool)
        for row in pool:
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
            answer = str(row.get("label", ""))
            mined = mine_plan([table["columns"], *table["rows"]], answer)
            if mined.status != "unique" or mined.op not in PAIRWISE:
                continue
            unique_pairwise += 1
            values = [(str(r[0]), v) for r in table["rows"]
                      for c in r[1:] if (v := to_number(c)) is not None]
            pairs = concrete_matches(mined.op, values, answer)
            counts[len(pairs)] += 1
            if len(pairs) > 1 and len(examples) < 5:
                examples.append((str(row["query"]), answer, mined.op, pairs[:4]))

    print(f"{seen:,} real ChartQA train rows (seed {args.seed})\n")
    print(f"  the miner called the operation UNIQUE and it is pairwise : {unique_pairwise:,}")
    if not unique_pairwise:
        return 0
    ambiguous = sum(v for k, v in counts.items() if k > 1)
    print(f"  of those, MORE THAN ONE operand pair reproduces the answer: "
          f"{ambiguous:,}  ({100 * ambiguous / unique_pairwise:.1f}%)   <-- invisible to the "
          f"uniqueness rule")
    print("\n  how many operand pairs reproduce it:")
    for k, v in sorted(counts.items()):
        print(f"    {k:>3} pair(s){'':<4}{v:>6,}  ({100 * v / unique_pairwise:>5.1f}%)")
    if examples:
        print("\n  examples — the operation was certain, the operands were not:")
        for q, a, op, pairs in examples:
            print(f"    {op} == {a!r} via {pairs}")
            print(f"      {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
