"""Building the plan the question asks for, then checking it against the answer.

**The direction is the point.** `plans.mining` works backwards: it asks *which operations
reproduce this gold answer?*, and refuses when more than one does. That refusal is 53.9% of
ChartQA training rows (`AUDIT.md` H4), and it is not a bug to be patched — it is what
working backwards must do. On a sorted bar chart the top row's value is simultaneously
`lookup(<its label>)` and `max` of its column, so an answer-first search will always find
both and can never choose.

This module goes the other way:

    read the question  ->  build the plan it asks for  ->  does it reproduce the answer?

and the ambiguity simply does not arise. If the question names Nigeria and
`lookup("Nigeria")` yields the gold answer, that plan is a faithful reading of the question
— it is irrelevant that `max()` also yields it. Uniqueness was never the property we wanted;
**fidelity to the question** was, and the arithmetic check is what keeps it honest.

Two independent things must hold before a plan is kept, and neither substitutes for the
other:

  * **fidelity** — `plans.intent` read the operation from the wording, never from the answer
  * **arithmetic** — the plan executes against the record's own evidence and reproduces the
    gold answer at the answer's own precision (`mining.matches_gold`, not the 5% scoring
    tolerance: 5% of the year 2014 is a century, `DECISIONS.md` 0045)

A plan failing either is discarded, never repaired. What this cannot catch is a question
misread in a way that still lands on the right number; that residue is measured rather than
assumed, and it is bounded by how often `intent` commits at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chartqa_dt.plans.executor import EvidenceItem, execute
from chartqa_dt.plans.intent import intended_operations, labels_named_in
from chartqa_dt.plans.mining import matches_gold

#: Tried in this order when the wording admits several readings. A question that names one
#: of the chart's own labels is asking about that label first and foremost, so `lookup` is
#: tried before any aggregate; the extrema come next because they are the most specific
#: remaining reading; open folds last, since they fit almost anything and so prove least.
PRIORITY: tuple[str, ...] = (
    "lookup", "argmax", "argmin", "max", "min",
    "difference", "ratio", "percent_change",
    "median", "mean", "sum", "count", "trend",
)

#: Operations built with no arguments, folding over the whole evidence list.
_FOLDS = frozenset({"max", "min", "argmax", "argmin", "median", "mean", "sum", "count",
                    "trend"})


@dataclass
class Built:
    """One construction attempt, with why it was or was not kept."""

    plan: dict | None = None
    op: str | None = None
    reason: str = ""
    executed: Any = None

    @property
    def ok(self) -> bool:
        return self.plan is not None


def _candidate_plans(op: str, question: str, labels: Sequence[str]) -> list[dict]:
    """Every plan of this operation the question plausibly means. Usually one, often none."""
    named = labels_named_in(question, labels)
    if op == "lookup":
        return [{"op": "lookup", "args": [named[0]]}] if len(named) == 1 else []
    if op in ("difference", "ratio", "percent_change"):
        if len(named) != 2:
            return []
        a, b = named
        # Which way round is not stated by the wording, so both orders are offered and the
        # arithmetic check decides. For `ratio` and `percent_change` the two differ; for
        # `difference` they differ in sign, which the gold answer settles.
        return [{"op": op, "args": [a, b]}, {"op": op, "args": [b, a]}]
    if op in _FOLDS:
        return [{"op": op, "args": []}]
    return []


def build(question: str, *, answer: Any, evidence: Sequence[Mapping[str, Any]]) -> Built:
    """Construct the plan this question asks for, or explain why none survived.

    The answer is used **only** to check a constructed plan, never to choose which plan to
    construct. That separation is what makes the result supervision rather than a restatement
    of the label.
    """
    labels = [str(e.get("label")) for e in evidence]
    items = [EvidenceItem(str(e.get("label")), e.get("value"), e.get("unit"))
             for e in evidence]

    wanted = intended_operations(question, labels=labels)
    if not wanted:
        return Built(reason="the wording does not say which operation is meant")

    tried = 0
    for op in PRIORITY:
        if op not in wanted:
            continue
        for plan in _candidate_plans(op, question, labels):
            tried += 1
            try:
                got = execute(plan, items)
            except Exception:                    # noqa: BLE001 — any refusal is a rejection
                continue
            if got is None:
                continue
            agrees = (str(got).strip().lower() == str(answer).strip().lower()
                      if isinstance(got, str) else matches_gold(got, answer))
            if agrees:
                return Built(plan=plan, op=op, executed=got, reason="ok")
    if not tried:
        return Built(reason=f"wording suggests {sorted(wanted)}, but no plan of those "
                            f"shapes could be formed from the labels")
    return Built(reason=f"built {tried} plan(s) from {sorted(wanted)}; none reproduced "
                        f"the answer {answer!r}")


__all__ = ["PRIORITY", "Built", "build"]
