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

MAX_DEPTH = 4

OPS: frozenset[str] = frozenset({
    "lookup", "filter", "count", "sum", "mean", "median", "difference",
    "ratio", "percent_change", "min", "max", "argmin", "argmax", "rank",
    "compare", "trend", "boolean", "multiple_choice", "unanswerable",
})

# Operations Appendix B leaves unimplemented because they need table context.
NEEDS_TABLE: frozenset[str] = frozenset({"filter", "rank", "multiple_choice"})


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


def to_number(x: Any) -> float:
    """Coerce to a finite float, or raise.

    Booleans are rejected explicitly: in Python ``True`` is ``1``, so a boolean
    slipping into an arithmetic slot would compute silently.
    """
    if isinstance(x, bool):
        raise ExecutorError("boolean where number expected")
    if isinstance(x, (int, float)):
        if math.isnan(x) or math.isinf(x):
            raise ExecutorError("non-finite number")
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        pct = s.endswith("%")
        if pct:
            s = s[:-1]
        try:
            v = float(s)
        except ValueError as e:
            raise ExecutorError(f"not numeric: {x!r}") from e
        return v / 100.0 if pct else v
    raise ExecutorError(f"not numeric: {x!r}")


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

    if op in NEEDS_TABLE:
        raise ExecutorError(
            f"{op} requires table context and is not enabled; "
            "PLAN.md Appendix B defers it until it has regression tests"
        )

    raise ExecutorError(f"unhandled op: {op!r}")   # pragma: no cover - OPS guards this
