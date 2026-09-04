"""The deterministic interpreter that recomputes the answer from a typed plan.

`PLAN.md` Appendix B supplies this, and it is reproduced faithfully **except for
one deliberate, recorded correction** (`DECISIONS.md` 0016).

The correction. In Appendix B a bare string argument means "evidence label" in
``argmin``/``argmax``/``check_units`` and "numeric literal" in
``sum``/``mean``/``difference``/``ratio``. With evidence ``2019=245, 2018=210``:

    argmax(["2019", "2018"])   ->  "2019"    (labels)
    mean(["2019", "2018"])     ->   2018.5   (numbers)   <- averages the LABELS
    mean(lookup 2019, lookup 2018) -> 227.5              <- what was meant

The failure profile is the dangerous one: a non-numeric label raises loudly, while
a numeric-looking label — years, counts, quantities, which is what chart
categories overwhelmingly are — silently returns a plausible wrong number. And
``{"op": "mean", "args": ["2019", "2018"]}`` is the most natural way a model
writes "the average of 2019 and 2018".

So here a bare string **always** resolves through the evidence list, and a numeric
literal must be a JSON number — which is what JSON gives you anyway.

Nothing in this module evaluates generated code. Arbitrary execution is impossible
by construction rather than by sandboxing, and every failure raises rather than
returning a fallback, so invalid plans are *counted* (non-negotiable rule 4).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

#: How deep a plan may nest. Measured over real targets, the median depth is 1-2 and the
#: deepest thing anyone has needed is `argmax(difference(a, b))` at 3 -- so 4 is one level of
#: headroom rather than a guess. It also bounds the executor's recursion, which is why it is
#: checked once at the top rather than trusted per node.
MAX_DEPTH = 4

#: Between a series name and a label in a qualified element name — `"Democratic · 2019"`.
#: Defined here rather than in `data.records`, which formats it, because `within` has to
#: parse it back out and this module imports nothing from the project.
SERIES_SEPARATOR = " · "

OPS: frozenset[str] = frozenset({
    "lookup", "filter", "count", "sum", "mean", "median", "difference",
    "ratio", "percent_change", "min", "max", "argmin", "argmax", "rank",
    "compare", "trend", "boolean", "multiple_choice", "unanswerable",
    "within",
})

# Operations Appendix B leaves unimplemented because they need table context.
NEEDS_TABLE: frozenset[str] = frozenset({"filter", "rank", "multiple_choice"})

#: What the schema admits and the prompt offers: operations that can actually run.
#:
#: `OPS` is the DSL's **vocabulary** — every operation the language names, including the three
#: it declares and does not implement. Offering those to the model is a trap: the prompt said
#: `rank` was allowed, `OUTPUT_SCHEMA` accepted it, and the executor refused it, so a model
#: emitting one produced a **schema-valid record that could not execute** — spending
#: probability mass on an operation that can never succeed and inflating the executor-failure
#: bucket with our own doing. `Prompt.md` Idea 10 asks for exactly this case by name
#: ("currently schema-valid but non-executable operators"); `DECISIONS.md` 0109.
EXECUTABLE_OPS: frozenset[str] = OPS - NEEDS_TABLE


class ExecutorError(Exception):
    """Any invalid plan. Never swallowed — every occurrence is counted."""


@dataclass
class EvidenceItem:
    label: str
    value: float | str | None
    unit: str | None = None


def plan_depth(node: Any) -> int:
    """Depth of a typed tree. Computed, never trusted from the model."""
    if not isinstance(node, dict):
        return 0
    args = node.get("args") or []
    return 1 + max([plan_depth(a) for a in args], default=0)


#: Every space character that appears as a thousands separator in Statista charts:
#: ordinary, non-breaking, narrow non-breaking, thin.
_SEPARATORS = str.maketrans({",": "", "$": "", " ": "", "\xa0": "", "\u202f": "",
                             "\u2009": ""})


def parse_numeric(x: Any) -> float | None:
    """**The** parser for a chart value. One function, because two disagreed by 100x.

    `mining.to_number` stripped a trailing `%` and `executor.to_number` divided by it, so a
    plan mined against a table value of `5.3` was executed against an evidence value of
    `0.053` and the round-trip failed on every percentage chart. Measured: 21.4% of ChartQA
    charts are all-percent, and **0 of 32,719 ChartQA gold answers and 0 of 3,996 RefChartQA
    answers carry a `%` sign** — so the divided form could never match an answer, and the
    scale that agrees with the data is the undivided one. `%` is dropped here and kept in
    `EvidenceItem.unit`, where `check_units` can still see it.

    This is a different question from `eval.metrics.to_float`, which parses gold ANSWERS and
    stays byte-faithful to the official evaluator, division and all (`DECISIONS.md` 0045).
    That function is unchanged.

    Thousands separators are removed, including the four space characters Statista uses:
    20.7% of ChartQA charts carry at least one value like `'3 071'`, which the executor
    previously refused outright. Separators are stripped before parsing and the result must
    still be a number, so `'5 apples'` is still rejected.
    """
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return None if math.isnan(x) or math.isinf(x) else float(x)
    if not isinstance(x, str):
        return None
    s = x.strip().translate(_SEPARATORS).rstrip("%")
    try:
        v = float(s)
    except ValueError:
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def plan_labels(plan: Any) -> list[str]:
    """Every evidence label a plan refers to, in order, depth-first.

    `within`'s first argument names a **series**, not an element, so it is skipped: treating
    it as a label would have every `within` plan rejected for an operand that is not in the
    evidence, which is true and beside the point.
    """
    out: list[str] = []
    if not isinstance(plan, dict):
        return out
    args = list(plan.get("args") or [])
    if plan.get("op") == "within" and args:
        args = args[1:]
    for arg in args:
        if isinstance(arg, str):
            out.append(arg)
        elif isinstance(arg, dict):
            out.extend(plan_labels(arg))
    return out


#: Operations that fold over **every** evidence item when their `args` are empty. This is
#: the compact form `DECISIONS.md` 0041 introduced so an L3 aggregate could stay inside the
#: schema's `maxItems: 4`. Its consequence is that such a plan's meaning depends on what is
#: in the evidence list, which is why evidence selection has to know about it.
FOLD_OPS = frozenset({"sum", "mean", "median", "min", "max", "count",
                      "argmin", "argmax", "trend", "within"})


def folds_over_evidence(plan: Any) -> bool:
    """Whether any node in the tree folds over the whole evidence list.

    `{"op": "mean", "args": []}` means *the mean of everything on the chart*. Selecting
    evidence by the labels a plan names — right for every other plan — hands such a node a
    one-item list, and the fold quietly returns that item instead of the aggregate.
    """
    if not isinstance(plan, dict):
        return False
    op, args = plan.get("op"), plan.get("args") or []
    if op in FOLD_OPS and not [a for a in args if isinstance(a, str)]:
        return True
    return any(folds_over_evidence(a) for a in args if isinstance(a, dict))


def to_number(x: Any) -> float:
    """Coerce to a finite float, or raise.

    Booleans are rejected explicitly: in Python ``True`` is ``1``, so a boolean
    slipping into an arithmetic slot would compute silently.
    """
    if isinstance(x, bool):
        raise ExecutorError("boolean where number expected")
    v = parse_numeric(x)
    if v is None:
        if isinstance(x, (int, float)):
            raise ExecutorError("non-finite number")
        raise ExecutorError(f"not numeric: {x!r}")
    return v


def execute(node: Any, evidence: list[EvidenceItem], *, _depth_checked: bool = False) -> Any:
    """Evaluate a typed expression tree against an evidence list."""
    if not _depth_checked:
        d = plan_depth(node)
        if d > MAX_DEPTH:
            raise ExecutorError(f"plan depth {d} exceeds {MAX_DEPTH}")

    if not isinstance(node, dict):
        return node

    op = node.get("op")
    if op not in OPS:
        raise ExecutorError(f"unknown op: {op!r}")
    args = node.get("args") or []

    by_label = {e.label: e for e in evidence}

    def resolve(a: Any) -> Any:
        """A bare string is ALWAYS an evidence label (decision 0016)."""
        if isinstance(a, dict):
            return execute(a, evidence, _depth_checked=True)
        if isinstance(a, str):
            if a not in by_label:
                raise ExecutorError(f"lookup of unknown evidence label: {a!r}")
            return by_label[a].value
        return a

    def numbers(seq: list[Any]) -> list[float]:
        return [to_number(resolve(a)) for a in seq]

    def check_units(seq: list[Any]) -> None:
        units = {by_label[a].unit for a in seq
                 if isinstance(a, str) and a in by_label and by_label[a].unit is not None}
        if len(units) > 1:
            raise ExecutorError(f"unit mismatch: {sorted(units)}")

    def all_values() -> list[float]:
        return [to_number(e.value) for e in evidence]

    def labels_or_all(seq: list[Any]) -> list[str]:
        chosen = [a for a in seq if isinstance(a, str)]
        return chosen or [e.label for e in evidence]

    if op == "unanswerable":
        return None

    if op == "lookup":
        if len(args) != 1 or not isinstance(args[0], str):
            raise ExecutorError("lookup takes exactly one string label")
        return to_number(resolve(args[0]))

    if op == "count":
        return float(len(args)) if args else float(len(evidence))

    if op in ("sum", "mean", "median", "min", "max"):
        check_units(args)
        values = numbers(args) if args else all_values()
        if not values:
            raise ExecutorError(f"{op} over empty set")
        return {"sum": sum, "mean": statistics.fmean, "median": statistics.median,
                "min": min, "max": max}[op](values)

    if op in ("difference", "ratio", "percent_change"):
        if len(args) != 2:
            raise ExecutorError(f"{op} takes exactly 2 arguments")
        check_units(args)
        a, b = numbers(args)
        if op == "difference":
            return a - b
        if b == 0:
            raise ExecutorError("division by zero")
        return a / b if op == "ratio" else 100.0 * (a - b) / b

    if op in ("argmin", "argmax"):
        chosen = labels_or_all(args)
        if not chosen:
            raise ExecutorError(f"{op} over empty set")
        missing = [c for c in chosen if c not in by_label]
        if missing:
            raise ExecutorError(f"lookup of unknown evidence label: {missing[0]!r}")
        pairs = [(c, to_number(by_label[c].value)) for c in chosen]
        return (min if op == "argmin" else max)(pairs, key=lambda p: p[1])[0]

    if op == "compare":
        if len(args) != 2:
            raise ExecutorError("compare takes exactly 2 arguments")
        a, b = numbers(args)
        return "greater" if a > b else ("less" if a < b else "equal")

    if op == "trend":
        values = numbers(args) if args else all_values()
        if len(values) < 2:
            raise ExecutorError("trend needs at least 2 values")
        d = values[-1] - values[0]
        return "increasing" if d > 0 else ("decreasing" if d < 0 else "flat")

    if op == "boolean":
        if len(args) != 1:
            raise ExecutorError("boolean takes exactly 1 argument")
        return bool(resolve(args[0]))

    if op == "within":
        # *"Which year has the highest number in hyperscale?"* -- an argmax over ONE series,
        # not over the chart. Measured on 40 human-written questions read by hand, this was
        # the single most-requested missing operation (6 of 40), and 8.6% of human questions
        # ask for a fold restricted to a series against 0.1% of machine ones
        # (`DECISIONS.md` 0090).
        if len(args) != 2 or not isinstance(args[0], str) or not isinstance(args[1], dict):
            raise ExecutorError(
                "within takes a series name and one nested operation, "
                'as {"op":"within","args":["Hyperscale",{"op":"argmax","args":[]}]}')
        prefix = f"{args[0]}{SERIES_SEPARATOR}"
        # The series prefix is STRIPPED from the subset's labels. Inside one series the
        # identifying part of a name is the bare label, so `argmax` returns "2021" and not
        # "Hyperscale · 2021" -- which is what the gold answer says.
        subset = [EvidenceItem(e.label[len(prefix):], e.value, e.unit)
                  for e in evidence if e.label.startswith(prefix)]
        if not subset:
            raise ExecutorError(f"no evidence belongs to the series {args[0]!r}")
        return execute(args[1], subset, _depth_checked=True)

    if op in NEEDS_TABLE:
        raise ExecutorError(
            f"{op} requires table context and is not enabled; "
            "PLAN.md Appendix B defers it until it has regression tests"
        )

    raise ExecutorError(f"unhandled op: {op!r}")   # pragma: no cover - OPS guards this
