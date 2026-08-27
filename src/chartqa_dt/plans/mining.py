"""Recover a typed plan for a real ChartQA question, or refuse.

`PLAN.md` Appendix E: enumerate whitelisted operations over the gold table and
accept a plan **only when exactly one operation type reproduces the recorded
answer**. Ambiguity is the normal case, not an edge case — `IDEA.md` §5.1 measures
roughly 16.5% unique for human questions and 1.9% for machine-generated ones.

Two things Appendix E leaves open, both settled here.

**Which cells are candidates (`DECISIONS.md` 0030).** Appendix E takes a flat
`list[(label, value)]`. That is unambiguous for a two-column table and undefined
for a wide one, and 28% of ChartQA tables are wide (3–9 columns). The choice
matters because uniqueness is the only thing deciding whether a plan is kept:
measured on **training** data, yield ranges from 4.2% (per row) to 14.2% (all
cells) — a 3.4× spread. So the flattening is an explicit parameter, every mode is
measurable, and the yield is reported as a range rather than a point.

**What the mined plan looks like.** Appendix E returns an operation *type*.
A trainable example needs a concrete tree, so the operands that produced the match
are recovered and the tree is **verified by executing it** — a mined plan that does
not reproduce its own answer is discarded.
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field
from typing import Any, Literal

from chartqa_dt.plans.executor import EvidenceItem, ExecutorError, execute

Flattening = Literal["all_cells", "per_row", "per_column", "union"]
FLATTENINGS: tuple[Flattening, ...] = ("all_cells", "per_row", "per_column", "union")

# Appendix E's search space. `sum2` and `mean2` are pair variants it enumerates
# separately, because a two-operand sum is a different plan from a whole-column sum.
UNARY_OPS = ("count", "sum", "mean", "median", "min", "max")
BINARY_OPS = ("difference", "ratio", "percent_change")
PAIR_OPS = ("sum2", "mean2")


def close(a: Any, b: Any, rel: float = 0.05) -> bool:
    """ChartQA's 5% tolerance, so a mined plan agrees with the metric that scores it."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if b == 0:
        return a == 0
    return abs(a - b) / abs(b) <= rel


def to_number(cell: Any) -> float | None:
    """Parse a table cell, or None. Never raises — most cells are labels."""
    try:
        return float(str(cell).replace(",", "").replace("$", "").rstrip("%").strip())
    except (TypeError, ValueError):
        return None


# A candidate set of one value cannot discriminate between operations: with a
# single value, lookup, sum, mean, median, min and max all return it. Such a set
# contributes only ambiguity, never information, so it is dropped.
#
# This matters concretely. 72% of ChartQA tables are two-column, and `per_row` on
# a two-column table produces exactly these degenerate singletons — which is why
# per_row measured only 4.2% yield, and why `union` (9.2%) scored WORSE than
# all_cells (14.2%) despite being the stricter rule. The strictness was an
# artefact, not a property of the data.
MIN_CANDIDATE_VALUES = 2


def candidate_sets(rows: list[list[str]], mode: Flattening) -> list[list[tuple[str, float]]]:
    """Candidate (label, value) sets for a table, under one flattening.

    ``rows`` includes the header. Returns a LIST of sets because ``per_row`` and
    ``per_column`` produce several, and a plan is unique only if it is unique
    across all of them. Sets smaller than ``MIN_CANDIDATE_VALUES`` are dropped.
    """
    if len(rows) < 2:
        return []
    header, body = rows[0], rows[1:]

    def keep(sets: list[list[tuple[str, float]]]) -> list[list[tuple[str, float]]]:
        return [s for s in sets if len(s) >= MIN_CANDIDATE_VALUES]

    if mode == "union":
        out: list[list[tuple[str, float]]] = []
        seen: set[tuple] = set()
        for m in ("all_cells", "per_row", "per_column"):
            for vals in candidate_sets(rows, m):  # type: ignore[arg-type]
                sig = tuple(vals)
                if sig not in seen:               # per_column == all_cells for 2-column tables
                    seen.add(sig)
                    out.append(vals)
        return out

    if mode == "all_cells":
        vals = [(str(r[0]), v) for r in body for c in r[1:] if (v := to_number(c)) is not None]
        return keep([vals])

    if mode == "per_row":
        out = []
        for r in body:
            vals = [(header[i] if i < len(header) else str(i), v)
                    for i, c in enumerate(r[1:], start=1) if (v := to_number(c)) is not None]
            out.append(vals)
        return keep(out)

    if mode == "per_column":
        out = []
        ncols = max(len(r) for r in rows)
        for j in range(1, ncols):
            out.append([(str(r[0]), v) for r in body
                        if j < len(r) and (v := to_number(r[j])) is not None])
        return keep(out)

    raise ValueError(f"unknown flattening: {mode!r}")


def enumerate_plan_ops(values: list[tuple[str, float]], target: Any) -> set[str]:
    """Distinct operation TYPES whose result matches the target.

    Types rather than concrete trees, deliberately: two lookups of different cells
    are the same *kind* of plan, and the question is whether the question
    determines the operation, not the operand.
    """
    v = [x for _, x in values]
    hits: set[str] = set()
    if not v:
        return hits
    if any(close(x, target) for x in v):
        hits.add("lookup")
    if close(len(v), target):
        hits.add("count")
    if close(sum(v), target):
        hits.add("sum")
    if close(statistics.fmean(v), target):
        hits.add("mean")
    if close(statistics.median(v), target):
        hits.add("median")
    if close(max(v), target):
        hits.add("max")
    if close(min(v), target):
        hits.add("min")
    for a, b in itertools.permutations(range(len(v)), 2):
        x, y = v[a], v[b]
        if close(x - y, target):
            hits.add("difference")
        if y != 0 and close(x / y, target):
            hits.add("ratio")
        if y != 0 and close(100 * (x - y) / y, target):
            hits.add("percent_change")
        if close(x + y, target):
            hits.add("sum2")
        if close((x + y) / 2, target):
            hits.add("mean2")
    return hits


@dataclass
class MinedPlan:
    status: Literal["unique", "ambiguous", "none", "non_numeric"]
    op: str | None = None
    plan: dict | None = None
    evidence: list[dict] = field(default_factory=list)
    ops_matched: list[str] = field(default_factory=list)
    flattening: str = ""


def _build_tree(op: str, values: list[tuple[str, float]], target: Any) -> tuple[dict, list[dict]] | None:
    """Recover the concrete tree and the operands that produced the match."""
    def ev(labels):
        return [{"label": lab, "value": val} for lab, val in labels]

    if op == "lookup":
        for lab, val in values:
            if close(val, target):
                return {"op": "lookup", "args": [lab]}, ev([(lab, val)])
        return None

    if op in ("difference", "ratio", "percent_change", "sum2", "mean2"):
        real = {"sum2": "sum", "mean2": "mean"}.get(op, op)
        for (la, va), (lb, vb) in itertools.permutations(values, 2):
            got = {"difference": va - vb,
                   "ratio": va / vb if vb else None,
                   "percent_change": 100 * (va - vb) / vb if vb else None,
                   "sum2": va + vb,
                   "mean2": (va + vb) / 2}[op]
            if got is not None and close(got, target):
                return {"op": real, "args": [la, lb]}, ev([(la, va), (lb, vb)])
        return None

    if op in ("sum", "mean", "median", "min", "max", "count"):
        labels = [lab for lab, _ in values]
        if len(set(labels)) != len(labels):        # a lookup would be ambiguous
            return None
        return {"op": op, "args": labels}, ev(values)

    return None


def mine_plan(rows: list[list[str]], target: Any, *, flattening: Flattening = "union") -> MinedPlan:
    """Accept a plan only when exactly one operation type reproduces the answer."""
    if to_number(target) is None:
        return MinedPlan(status="non_numeric", flattening=flattening)

    sets = candidate_sets(rows, flattening)
    if not sets:
        return MinedPlan(status="none", flattening=flattening)

    hits: set[str] = set()
    per_set: list[tuple[list[tuple[str, float]], set[str]]] = []
    for values in sets:
        h = enumerate_plan_ops(values, target)
        per_set.append((values, h))
        hits |= h

    if not hits:
        return MinedPlan(status="none", flattening=flattening)
    if len(hits) > 1:
        return MinedPlan(status="ambiguous", ops_matched=sorted(hits), flattening=flattening)

    op = next(iter(hits))
    for values, h in per_set:
        if op not in h:
            continue
        built = _build_tree(op, values, target)
        if built is None:
            continue
        plan, evidence = built
        # A mined plan that cannot reproduce its own answer is not a plan.
        try:
            got = execute(plan, [EvidenceItem(e["label"], e["value"]) for e in evidence])
        except ExecutorError:
            continue
        if close(got, target):
            return MinedPlan(status="unique", op=op, plan=plan, evidence=evidence,
                             ops_matched=[op], flattening=flattening)

    # The type matched but no concrete tree verified: not a usable example.
    return MinedPlan(status="none", ops_matched=[op], flattening=flattening)
