#!/usr/bin/env python3
"""MEASUREMENT · which question types the DSL cannot express, across the whole corpus.

Decision 0080 found three missing operators on a 40-record sample. Forty records is a
sample, not a measurement: it says the gap exists, not what it is worth. This counts the
same categories across every question in ChartQA and RefChartQA, so each proposed operator
carries a number.

Classification leans on the GOLD ANSWER wherever it can, because the answer is unambiguous
where question wording is not: an answer of exactly "Yes"/"No" is a boolean question no
matter how the question is phrased. Only where the answer cannot decide does it fall back
to question wording, and those cases are reported separately so a reader can discount them.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.eval.metrics import to_float  # noqa: E402

#: "second highest", "3rd largest", "second most popular" -- a rank, not an extremum.
RANK = re.compile(
    r"\b(second|third|fourth|fifth|2nd|3rd|4th|5th)\b[^?]{0,40}?"
    r"\b(highest|largest|greatest|biggest|most|lowest|smallest|least|top|ranked|rank)\b",
    re.I)
#: the same, written the other way round: "ranked second", "the 3rd from the top"
RANK_REV = re.compile(r"\b(rank(ed|s)?|position)\b[^?]{0,20}\b"
                      r"(second|third|fourth|fifth|2nd|3rd|4th|5th|two|three)\b", re.I)
EXTREMUM = re.compile(r"\b(highest|largest|greatest|maximum|max|biggest|most|top|"
                      r"lowest|smallest|minimum|min|least|fewest|bottom)\b", re.I)
#: "which/what ... " asking for a name rather than a number
ASKS_LABEL = re.compile(r"^\s*(which|what|who|in which|for which|name the)\b", re.I)
DIFF_OF = re.compile(r"\b(difference|gap|ratio|sum|total|average|mean)\b[^?]{0,30}"
                     r"\bbetween\b", re.I)


def classify(question: str, answer: str) -> str:
    q, a = question or "", (answer or "").strip()
    low = a.lower()

    if low in {"yes", "no"}:
        return "boolean_yes_no"
    if RANK.search(q) or RANK_REV.search(q):
        return "rank_n"

    answer_is_number = to_float(a) is not None
    if not answer_is_number:
        # the answer names something. If the question also states a number, the model is
        # being asked to go from a value back to its label.
        numbers_in_q = re.findall(r"\d[\d,\.]*", q)
        if ASKS_LABEL.match(q) and numbers_in_q and not EXTREMUM.search(q):
            return "reverse_lookup"
        if EXTREMUM.search(q) and DIFF_OF.search(q):
            return "argmax_over_computed"
        if EXTREMUM.search(q):
            return "argmax_argmin"
        return "label_other"

    if DIFF_OF.search(q):
        return "arithmetic_between"
    return "numeric_other"


def chartqa_qa() -> list[tuple[str, str]]:
    """Every ChartQA question/answer pair, all splits, both human and machine."""
    from scripts.build_mixtures import archive_path

    from chartqa_dt.data.chartqa import ArchiveReader
    out: list[tuple[str, str]] = []
    with ArchiveReader(archive_path()) as reader:
        for split in ("train", "val", "test"):
            for kind in ("human", "machine"):
                for row in reader.qa_rows(split, kind):
                    out.append((str(row["query"]), str(row.get("label", ""))))
    return out


def refchartqa_qa() -> list[tuple[str, str]]:
    """RefChartQA training rows, from the streamed cache the mixture builder reads."""
    cache = Path.home() / ".cache/chartqa_dt/data/refchartqa_train.jsonl"
    if not cache.exists():
        return []
    out = []
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            out.append((str(r.get("question", "")), str(r.get("answer", ""))))
    return out


def main() -> int:
    counts: dict[str, collections.Counter[str]] = {}
    for source, pairs in (("chartqa", chartqa_qa()), ("refchartqa", refchartqa_qa())):
        c: collections.Counter[str] = collections.Counter()
        for question, answer in pairs:
            c[classify(question, answer)] += 1
        counts[source] = c

    EXPRESSIBLE = {"argmax_argmin", "arithmetic_between", "numeric_other", "label_other"}
    MISSING = {"boolean_yes_no": "no Yes/No comparison operator",
               "rank_n": "`rank` declared in OPS but unimplemented",
               "reverse_lookup": "no value -> label operator",
               "argmax_over_computed": "argmax takes labels, not a computed series"}

    print("question types the DSL cannot express, over every question in both datasets\n")
    total_all = sum(sum(c.values()) for c in counts.values())
    hdr = f"  {'category':<24}{'ChartQA':>10}{'RefChartQA':>12}{'total':>9}{'share':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    blocked = 0
    for cat in list(MISSING) + sorted(EXPRESSIBLE):
        a, b = counts["chartqa"][cat], counts["refchartqa"][cat]
        if not (a + b):
            continue
        mark = "  <-- BLOCKED" if cat in MISSING else ""
        if cat in MISSING:
            blocked += a + b
        print(f"  {cat:<24}{a:>10,}{b:>12,}{a + b:>9,}{100 * (a + b) / total_all:>7.1f}%{mark}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'TOTAL':<24}{sum(counts['chartqa'].values()):>10,}"
          f"{sum(counts['refchartqa'].values()):>12,}{total_all:>9,}")
    print(f"\n  not expressible in the current DSL: {blocked:,} of {total_all:,} "
          f"({100 * blocked / total_all:.1f}%)\n")
    print("  what each missing operator would unlock:")
    for cat, why in MISSING.items():
        n = counts["chartqa"][cat] + counts["refchartqa"][cat]
        if n:
            print(f"    {n:>6,}  ({100 * n / total_all:>4.1f}%)  {cat:<22}{why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
