"""What operation the QUESTION asks for, read from its wording alone.

`AUDIT.md` H4: the deterministic miner refuses 53.9% of ChartQA training rows as
`ambiguous`, and `ambiguous` does not mean two cells hold the answer — it means two
*operations* reproduce it. Half those collisions are `lookup` against an extremum
(26.6% of all rows; `lookup+max` alone 775 times in a 4,000-row sample). ChartQA charts are
usually sorted and questions often ask about the top row, so the answer cell is
simultaneously `lookup(<its label>)` and `max` of its column.

*"How many internet users did Nigeria have"* wants the lookup. *"Which country had the
most?"* wants the extremum. **The two are identical in the table and one word apart in the
question**, and the miner never reads the question.

**This module reads only the question — never the answer, never the values.** That is what
makes it safe to check against the miner's own `unique` verdicts: where exactly one
operation reproduces the answer, that operation is known ground truth, and agreement is a
real measurement rather than a restatement. Thousands of free labels, no circularity.

It is deliberately **not** a plan generator. The miner enumerates and validates candidates;
this only says which of them the question was asking for, and abstains when it cannot tell.
Abstention is the common case by design — a wrong tie-break produces supervision that
teaches the wrong reasoning, which is worse than none (`PLAN.md` 3.6).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

#: A question asking which ITEM is extreme wants the label back: `argmax` / `argmin`.
#: One asking for the VALUE of the extreme item wants `max` / `min`.
_MOST = r"(?:highest|largest|greatest|biggest|most|maximum|max|top|leading|peak|peaked|best)"
_LEAST = r"(?:lowest|smallest|least|fewest|minimum|min|bottom|worst)"

#: "which country", "what year", "who", "in which month" — the answer is a NAME.
_ASKS_NAME = re.compile(
    r"^\s*(?:in\s+|for\s+|at\s+|on\s+|during\s+)?(?:which|what|who|whom|where|when|name)\b",
    re.I)
#: "what is the value of", "how much", "how many" — the answer is a NUMBER.
_ASKS_VALUE = re.compile(r"\b(?:how\s+much|how\s+many|what\s+(?:is|was)\s+the\s+"
                         r"(?:value|number|amount|total|share|percentage|percent))\b", re.I)

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("median", re.compile(r"\bmedian\b", re.I)),
    ("mean", re.compile(r"\b(?:average|mean|avg)\b", re.I)),
    ("ratio", re.compile(r"\bratio\b|\btimes\s+(?:as|more|larger|greater)\b", re.I)),
    ("percent_change", re.compile(r"\b(?:percent(?:age)?\s+(?:change|increase|decrease)|"
                                  r"grow(?:th)?\s+rate)\b", re.I)),
    # "how much more", but also "how much percentage is X more than Y" and "X less than Y":
    # a comparison of two named things is a difference however the sentence is arranged.
    ("difference", re.compile(r"\b(?:difference|gap)\b|"
                              r"\b(?:more|less|higher|lower|greater|bigger|smaller|fewer)\s+"
                              r"than\b", re.I)),
    ("sum", re.compile(r"\b(?:total|sum|combined|together|altogether|add(?:ed|ing)?\s+up)\b",
                       re.I)),
    ("trend", re.compile(r"\btrend\b|\b(?:increasing|decreasing)\s+(?:trend|overall)\b", re.I)),
)

#: "how many bars are there", "how many segments", "how many categories are shown"
_COUNTS_MARKS = re.compile(
    r"\bhow\s+many\s+(?:\w+\s+){0,2}?"
    r"(?:bars?|bar|segments?|categories|category|colors?|colours?|slices?|wedges?|"
    r"groups?|items?|entries|lines?|points?)\b[^?]{0,30}"
    r"(?:are\s+there|shown|displayed|in\s+the\s+(?:graph|chart|figure|plot))?", re.I)


def _normalise(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def labels_named_in(question: str, labels: Iterable[str]) -> list[str]:
    """Which evidence labels the question mentions outright.

    The strongest signal there is for `lookup`: a question that says *"Nigeria"* is asking
    about Nigeria, whatever else happens to be true of Nigeria's bar. Matching is on
    normalised word boundaries so `'2019'` does not match `'2019*'`'s neighbour `'12019'`,
    and one-character labels are ignored as too easy to hit by accident.
    """
    q = f" {_normalise(question)} "
    found = []
    for label in labels:
        norm = _normalise(label)
        if len(norm) < 2:
            continue
        if f" {norm} " in q:
            found.append(label)
    return found


_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def restricts_to_a_subset(question: str, labels: Iterable[str]) -> bool:
    """Does the question single out part of the chart that we could not resolve to a label?

    A fold — `max()`, `mean()`, `argmax()` — reads the WHOLE chart. That is only a faithful
    reading when the question asks about the whole chart. *"How many people were in Norway's
    largest age group between 45 and 69 years old in 2021?"* names a series and a year; a
    global `max()` happened to return the right number and was still the wrong plan, because
    it answers a different question.

    The check is deliberately narrow: the question names a year that some label contains, yet
    no label matched outright. That means the chart is indexed by something the question
    restricted and we failed to pin down — so a fold cannot be trusted, and abstaining costs
    only the records where we were about to guess.
    """
    years = set(_YEAR.findall(str(question)))
    if not years:
        return False
    if labels_named_in(question, labels):
        return False
    return any(y in str(label) for label in labels for y in years)


def intended_operations(question: str, *, labels: Iterable[str] = ()) -> set[str]:
    """The operations this question could be asking for. Empty means "cannot tell".

    Never consults the answer or any value — only the wording and, for `lookup`, whether
    the question names one of the chart's own labels.
    """
    q = str(question or "")
    out: set[str] = set()

    for op, pattern in _PATTERNS:
        if pattern.search(q):
            out.add(op)

    if _COUNTS_MARKS.search(q):
        out.add("count")

    wants_most = re.search(rf"\b{_MOST}\b", q, re.I) is not None
    wants_least = re.search(rf"\b{_LEAST}\b", q, re.I) is not None
    if wants_most or wants_least:
        asks_name = bool(_ASKS_NAME.match(q))
        asks_value = bool(_ASKS_VALUE.search(q))
        if wants_most:
            out.add("argmax" if asks_name and not asks_value else "max")
            if not asks_name and not asks_value:
                out.add("argmax")      # genuinely ambiguous phrasing; keep both
        if wants_least:
            out.add("argmin" if asks_name and not asks_value else "min")
            if not asks_name and not asks_value:
                out.add("argmin")

    # A named label always admits `lookup`, EVEN when an extremum word is present.
    # Suppressing it cost 86.1% precision: *"What was the peak number of overseas visits to
    # the UK in 2019?"* names 2019 and says "peak", and the answer is the 2019 cell. Adding
    # both lets the intersection with the miner's validated candidates decide, which is the
    # whole point of narrowing rather than generating.
    named = labels_named_in(q, labels)
    if len(named) == 1:
        out.add("lookup")

    # A composite question — "the sum of the highest and lowest value" — asks for an
    # arithmetic operation OVER extrema. Both families fire, neither is the whole answer,
    # and guessing either is wrong. Abstain instead.
    aggregates = out & {"sum", "mean", "median", "ratio", "difference", "percent_change"}
    extrema = out & {"max", "min", "argmax", "argmin"}
    if aggregates and extrema:
        return set()

    # A fold reads the whole chart, so drop the folds when the question restricted it to a
    # part we could not identify. What remains may be empty, which is the correct answer.
    labels = list(labels)
    if restricts_to_a_subset(q, labels):
        out -= {"max", "min", "argmax", "argmin", "mean", "median", "sum", "count", "trend"}
    return out


def disambiguate(candidates: Iterable[str], question: str, *,
                 labels: Iterable[str] = ()) -> str | None:
    """Pick the one operation the question asked for, or `None` to abstain.

    `candidates` is what the miner proved reproduces the gold answer. This narrows that set
    by intent; it can never widen it, so an operation the arithmetic rejected can never be
    chosen. Anything other than exactly one survivor abstains.
    """
    pool = set(candidates)
    if len(pool) == 1:
        return next(iter(pool))
    wanted = intended_operations(question, labels=labels) & pool
    return next(iter(wanted)) if len(wanted) == 1 else None


def summarise(counts: Mapping[str, int]) -> str:      # pragma: no cover - reporting only
    total = sum(counts.values()) or 1
    return "\n".join(f"    {k:<16}{v:>6,}  ({100 * v / total:>5.1f}%)"
                     for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))


__all__ = ["disambiguate", "intended_operations", "labels_named_in", "summarise"]
