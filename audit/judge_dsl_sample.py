#!/usr/bin/env python3
"""AUDIT · judging the unbiased sample: can the DSL express these questions at all?

Sixty ChartQA training questions drawn at random (`audit/make_unbiased_dsl_sample.py`,
seed 0), each read against its gold table and judged by Claude for whether SOME plan in
the current DSL computes the gold answer. This is the expressiveness question only --
whether a miner could FIND that plan is a separate matter, measured elsewhere.

The judgements are listed one per record with the operator that would serve, so a reader
can disagree with any single call rather than with the total.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

#: index -> (verdict, operator-or-reason)
JUDGEMENTS: dict[int, tuple[str, str]] = {
    **dict.fromkeys((0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 25, 26, 27, 28, 31, 32, 34, 35, 36, 37, 38, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 56, 57, 58), ("ok", "lookup")),
    **dict.fromkeys((11, 14, 16, 18, 20, 22, 23, 30, 40, 52, 53, 59), ("ok", "argmax/argmin")),
    15: ("ok", "ratio"), 17: ("ok", "sum"), 19: ("ok", "difference"),
    21: ("ok", "mean"), 29: ("ok", "sum"), 45: ("ok", "max"),
    54: ("ok", "count"), 55: ("ok", "count"),
    # --- not expressible
    4:  ("blocked", "reverse lookup: 'which country has the value 3.91t' -> a LABEL"),
    33: ("blocked", "boolean: 'is the largest bar 41?' -> Yes"),
    39: ("blocked", "conditional count: 'how many workers answering Yes < 70%' -> `filter`"),
    24: ("unanswerable", "gold answer 'Chevrolet' (14.97%) contradicts the table's "
                         "own maximum 'Suzuki' (18.45%)"),
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main() -> int:
    rows = [json.loads(x) for x in
            Path("audit/dsl_expressiveness_sample.jsonl").read_text(encoding="utf-8")
            .splitlines() if x]
    verdicts = collections.Counter()
    ops = collections.Counter()
    problems = []
    for i in range(len(rows)):
        v, why = JUDGEMENTS.get(i, ("UNJUDGED", ""))
        verdicts[v] += 1
        if v == "ok":
            ops[why] += 1
        else:
            problems.append((i, v, why, rows[i]["question"]))

    n = len(rows)
    ok = verdicts["ok"]
    lo, hi = wilson(ok, n)
    print(f"unbiased sample: {n} random ChartQA train questions (seed 0)\n")
    print(f"  expressible in the current DSL : {ok}/{n}  "
          f"({100 * ok / n:.1f}%, 95% CI {100 * lo:.1f}-{100 * hi:.1f}%)")
    for v, k in verdicts.most_common():
        if v != "ok":
            print(f"  {v:<31}: {k}/{n}  ({100 * k / n:.1f}%)")
    print("\n  operator that would serve:")
    for op, k in ops.most_common():
        print(f"    {op:<16}{k:>4}  ({100 * k / n:>4.1f}%)")
    print("\n  the ones that fail:")
    for i, v, why, q in problems:
        print(f"    [{i}] {v}: {why}")
        print(f"         {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
