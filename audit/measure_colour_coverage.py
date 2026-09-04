#!/usr/bin/env python3
"""MEASUREMENT · can the annotation's colours answer the questions that mention colour?

**21.8% of human-written ChartQA questions mention a colour** against 0.5% of machine ones
(`DECISIONS.md` 0086), and human questions are half the test split and half the headline
metric. Every ChartQA annotation carries a `colors` list per series and nothing in this
project has ever read it.

Before building on that, the hypothesis has to survive contact with the data: when a person
writes *"the dark blue bar"*, does the chart actually have a series whose hex maps to those
words? A mapper that names colours plausibly but disagrees with how people describe them
would quietly select the wrong marks.

Three outcomes per colour-mentioning question: the chart has a series matching the words
(**usable**), the chart has colours but none match (**mismatch** — the mapper or the wording
is off), or the annotation carries no usable colour at all (**absent**).
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

from chartqa_dt.data.colours import distinct_palette, mentioned_in, names_for  # noqa: E402

COLOUR_WORD = re.compile(
    r"\b(colou?rs?|blue|red|green|orange|purple|brown|grey|gray|yellow|pink|black|navy|"
    r"teal|violet|maroon|cyan|magenta|silver|white)\b", re.I)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scripts.build_mixtures import archive_path

    from chartqa_dt.data.chartqa import ArchiveReader, annotation_path

    outcome: collections.Counter[str] = collections.Counter()
    misses: list[tuple[str, list[str]]] = []
    hits: list[tuple[str, str, list[str]]] = []
    asks_which_colour = 0

    with ArchiveReader(archive_path()) as reader:
        rows = [r for r in reader.qa_rows("train", "human") if COLOUR_WORD.search(str(r["query"]))]
        random.Random(args.seed).shuffle(rows)
        for row in rows[:args.limit]:
            ann = annotation_path("train", row["imgname"])
            if not reader.exists(ann):
                outcome["annotation missing"] += 1
                continue
            models = (reader.read_json(ann).get("models") or [])
            palette = distinct_palette(models)
            question = str(row["query"])
            if re.search(r"\b(what|which)\s+colou?r\b", question, re.I):
                asks_which_colour += 1
            if not palette:
                outcome["no usable colour in the annotation"] += 1
                continue
            labels = [str(x) for m in models for x in (m.get("x") or [])] + \
                     [str(m.get("name")) for m in models]
            matched = mentioned_in(question, palette, labels)
            if matched:
                outcome["a series matches the words"] += 1
                if len(hits) < 5:
                    hits.append((question, sorted(matched)[0],
                                 sorted(names_for(sorted(matched)[0]))))
            else:
                outcome["colours present, none match"] += 1
                if len(misses) < 8:
                    misses.append((question, palette[:5]))

    total = sum(outcome.values())
    print(f"{total:,} human questions that mention a colour (seed {args.seed})\n")
    for k, v in outcome.most_common():
        print(f"  {k:<38}{v:>6,}  ({100 * v / max(total, 1):>5.1f}%)")
    print(f"\n  of these, {asks_which_colour:,} ask *which colour* something is "
          f"({100 * asks_which_colour / max(total, 1):.1f}%) — those need the NAME as the "
          f"answer,\n  the rest use the colour to pick which marks the question is about.")
    if hits:
        print("\n  matched:")
        for q, hexc, words in hits:
            print(f"    {hexc} {words}\n      {q}")
    if misses:
        print("\n  colours present but nothing matched — the honest failures:")
        for q, pal in misses:
            print(f"    palette {pal}\n      {q}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
