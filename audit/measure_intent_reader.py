#!/usr/bin/env python3
"""MEASUREMENT · can reading the QUESTION break the miner's ties, without an LLM per record?

`AUDIT.md` H4 measured the loss: 53.9% of rows are refused as `ambiguous`, meaning several
operations reproduce the gold answer and the table cannot say which was asked for. The
obvious fix is to have a language model read each question — but that is a paid API call per
record, and it is not obviously *better* than a rule that can be tested.

So this tests the rule first, on ground truth that costs nothing:

  **Where the miner says `unique`, the correct operation is already known.** Exactly one
  operation reproduces the answer, so that operation is what the question must have asked
  for. `plans.intent` never looks at the answer or at any value — only at the wording — so
  checking it against those verdicts measures it rather than restating it.

Three outcomes per record: the reader AGREES with the known operation, DISAGREES with it
(a real error), or ABSTAINS. Precision is agree / (agree + disagree); abstention is not an
error, it is the designed behaviour when the wording does not decide.

Then the same reader is applied to the `ambiguous` records to see how many ties it breaks.
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

from chartqa_dt.plans.intent import disambiguate, intended_operations  # noqa: E402
from chartqa_dt.plans.mining import mine_plan  # noqa: E402


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

    from chartqa_dt.data.chartqa import ArchiveReader, parse_table, table_path

    agree = disagree = abstain = 0
    wrong_examples: list[tuple[str, str, set[str]]] = []
    amb_total = amb_resolved = 0
    resolved_as: collections.Counter[str] = collections.Counter()
    resolved_examples: list[tuple[str, str, list[str]]] = []
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
            question = str(row["query"])
            labels = [str(r[0]) for r in table["rows"] if r]
            mined = mine_plan([table["columns"], *table["rows"]], row.get("label"))

            if mined.status == "unique" and mined.op:
                guess = intended_operations(question, labels=labels)
                if not guess:
                    abstain += 1
                elif mined.op in guess:
                    agree += 1
                else:
                    disagree += 1
                    if len(wrong_examples) < 6:
                        wrong_examples.append((question, mined.op, guess))

            elif mined.status == "ambiguous":
                amb_total += 1
                pick = disambiguate(mined.ops_matched, question, labels=labels)
                if pick:
                    amb_resolved += 1
                    resolved_as[pick] += 1
                    if len(resolved_examples) < 6:
                        resolved_examples.append((question, pick, mined.ops_matched))

    print(f"{seen:,} real ChartQA train questions (seed {args.seed})\n")
    judged = agree + disagree
    lo, hi = wilson(agree, judged)
    print("A. checked against the miner's `unique` verdicts — free ground truth")
    print(f"   agrees with the known operation : {agree:,}")
    print(f"   disagrees (a real error)        : {disagree:,}")
    print(f"   abstains (wording does not say) : {abstain:,}")
    if judged:
        print(f"   PRECISION where it commits      : {100 * agree / judged:.1f}%  "
              f"(95% CI {100 * lo:.1f}–{100 * hi:.1f}%, n={judged:,})")
    if wrong_examples:
        print("\n   where it was wrong:")
        for q, truth, guess in wrong_examples:
            print(f"     said {sorted(guess)}, truth {truth!r}\n       {q}")

    print(f"\nB. applied to the {amb_total:,} `ambiguous` records the miner refuses")
    if amb_total:
        print(f"   ties broken : {amb_resolved:,}  ({100 * amb_resolved / amb_total:.1f}%)"
              f"   <-- supervision recovered")
        print(f"   still tied  : {amb_total - amb_resolved:,}")
        print("\n   what the broken ties resolved to:")
        for op, k in resolved_as.most_common():
            print(f"     {op:<16}{k:>6,}")
        print("\n   examples:")
        for q, pick, cands in resolved_examples:
            print(f"     {sorted(cands)} -> {pick!r}\n       {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
