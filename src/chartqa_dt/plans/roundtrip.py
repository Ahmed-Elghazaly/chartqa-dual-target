"""Does the model's own plan reproduce the model's own answer?

This is the project's central claim in one function. `IDEA.md`'s whole premise is that a
model can emit a *typed expression tree* alongside its answer, and that a deterministic CPU
executor can then recompute that answer — making the arithmetic checkable rather than
asserted. If the emitted plan does not reproduce the emitted answer, the plan is decoration.

It is therefore a headline number, not a diagnostic, and it is measured from the very first
zero-shot baseline so the trained model has something to be compared against.

Four outcomes, and they are deliberately distinguished because they call for different
fixes:

* **agrees**       — the executor's result matches `model_answer`. The claim holds.
* **disagrees**    — the plan runs and produces something else. A *reasoning* error: the
                     model chose the wrong operation, or read the wrong values.
* **raises**       — the plan is not executable at all (wrong arity, unknown label). A
                     *format* error the prompt can usually fix.
* **no plan**      — nothing to check; the record was already a failure.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from chartqa_dt.eval.metrics import relaxed_correctness
from chartqa_dt.plans.executor import EvidenceItem, execute

Outcome = str  # "agrees" | "disagrees" | "raises" | "no_plan"


@dataclass
class RoundTrip:
    outcome: Outcome
    executed: Any = None
    stated: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome == "agrees"


def check_record(record: dict[str, Any]) -> RoundTrip:
    """Run one record's plan against its own evidence and compare."""
    plan = record.get("plan")
    stated = str(record.get("model_answer", ""))
    if not isinstance(plan, dict) or not plan.get("op"):
        return RoundTrip("no_plan", stated=stated)

    evidence = [
        EvidenceItem(str(e.get("label")), e.get("value"), e.get("unit"))
        for e in (record.get("evidence") or []) if isinstance(e, dict)
    ]
    try:
        got = execute(plan, evidence)
    except Exception as exc:  # noqa: BLE001 - any executor refusal is the same outcome
        return RoundTrip("raises", stated=stated, error=f"{type(exc).__name__}: {exc}")

    # Compared with the official tolerance, not exactly: the model states a rounded
    # answer ("35") for a computed 35.0001, and calling that a disagreement would
    # measure formatting rather than reasoning.
    agrees = relaxed_correctness(stated, "" if got is None else str(got))
    return RoundTrip("agrees" if agrees else "disagrees", executed=got, stated=stated)


@dataclass
class RoundTripStats:
    total: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)

    def add(self, result: RoundTrip) -> None:
        self.total += 1
        self.counts[result.outcome] = self.counts.get(result.outcome, 0) + 1
        if result.outcome == "raises":
            key = result.error.split(":")[-1].strip()[:60]
            self.errors[key] = self.errors.get(key, 0) + 1

    @property
    def agreement(self) -> float:
        """Of all records, the share whose plan reproduces their answer."""
        return self.counts.get("agrees", 0) / self.total if self.total else 0.0

    @property
    def executable(self) -> float:
        """Of records with a plan, the share that run at all."""
        with_plan = self.total - self.counts.get("no_plan", 0)
        if not with_plan:
            return 0.0
        return (self.counts.get("agrees", 0) + self.counts.get("disagrees", 0)) / with_plan

    def describe(self) -> str:
        lines = [f"  round-trip      : {self.counts.get('agrees', 0)}/{self.total} "
                 f"({100 * self.agreement:.1f}%) — the plan reproduces the answer",
                 f"  executable      : {100 * self.executable:.1f}% of records with a plan"]
        for outcome in ("disagrees", "raises", "no_plan"):
            n = self.counts.get(outcome, 0)
            if n:
                lines.append(f"    {outcome:<10} {n:>5}")
        for err, n in sorted(self.errors.items(), key=lambda kv: -kv[1])[:5]:
            lines.append(f"      {n:>3}  {err}")
        return "\n".join(lines)


def check_many(records: Iterable[dict[str, Any]]) -> tuple[list[RoundTrip], RoundTripStats]:
    out: list[RoundTrip] = []
    stats = RoundTripStats()
    for record in records:
        result = check_record(record)
        stats.add(result)
        out.append(result)
    return out, stats


def disagreement_examples(records: Sequence[dict[str, Any]], limit: int = 8
                          ) -> list[dict[str, Any]]:
    """The cases worth reading, which is how the op-choice confusion was found."""
    out = []
    for record in records:
        result = check_record(record)
        if result.outcome in ("disagrees", "raises") and len(out) < limit:
            out.append({"plan": record.get("plan"), "outcome": result.outcome,
                        "executed": result.executed, "stated": result.stated,
                        "error": result.error})
    return out


__all__ = ["RoundTrip", "RoundTripStats", "check_many", "check_record",
           "disagreement_examples"]
