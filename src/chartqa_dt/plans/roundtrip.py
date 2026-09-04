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

from chartqa_dt.eval.metrics import relaxed_correctness, to_float
from chartqa_dt.plans.executor import EvidenceItem, execute

#: The official relaxed tolerance, `max_relative_change` in the published evaluator.
RELAXED_TOLERANCE = 0.05

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

    return RoundTrip("agrees" if answers_agree(stated, got) else "disagrees",
                     executed=got, stated=stated)


def answers_agree(stated: str, got: Any) -> bool:
    """Does a plan's result reproduce the answer stated beside it?

    Compared with the official 5% tolerance rather than exactly: a record states a rounded
    answer ("35") for a computed 35.0001, and calling that a disagreement would measure
    formatting rather than reasoning.

    **Zero is handled here rather than deferred to `relaxed_correctness`.** The official
    implementation computes a *relative* error and guards the division with a truthiness
    test, so a target of zero falls back to string equality and
    ``relaxed_correctness("0", "0.0")`` is `False`. That behaviour is faithful to the
    published evaluator and `eval/metrics.py` keeps it exactly, because a reported score
    must match what the benchmark's own code would produce.

    This is not scoring. It asks whether a plan reproduces its own answer, and there a
    correct result of zero is a correct result. Inheriting the quirk discarded **512
    synthetic L4 records** — every one of them a valid `difference` whose two operands were
    equal, which is precisely the case a compositional example should cover
    (`DECISIONS.md` 0071).

    The relative test is also made symmetric, since neither side is a gold reference.
    """
    if got is None:
        return relaxed_correctness(stated, "")
    a, b = to_float(stated), to_float(str(got))
    if a is None or b is None:
        return relaxed_correctness(stated, str(got))
    scale = max(abs(a), abs(b))
    return scale == 0.0 or abs(a - b) <= RELAXED_TOLERANCE * scale


#: How a scored answer is chosen from a generated record. **This is the project's central
#: claim, made testable.** `README.md` says *"a small deterministic CPU interpreter re-runs
#: that program, so the arithmetic never depends on the model doing mental maths"* — but every
#: evaluation path scores `model_answer`, the model's own string, and the executor is only ever
#: consulted as a diagnostic (`DECISIONS.md` 0096). 0059 states the weaker and accurate version:
#: the executor makes the arithmetic *"checkable rather than asserted"*.
#:
#: Which policy is better is an empirical question nobody has asked, and it can be answered
#: from **one** set of generations at no extra cost, because the executed value is already
#: computed for the round-trip.
ANSWER_POLICIES: tuple[str, ...] = ("stated", "executed", "executed_or_stated")


def _as_answer(value: Any) -> str:
    """Format an executed value the way an answer is written.

    A whole float loses its `.0`: the official metric reads `"0"` and `"0.0"` as **different**
    answers because of a truthiness guard it contains (`eval.metrics`, faithful to upstream),
    so emitting `"245.0"` where the gold says `"245"` would lose marks to formatting rather
    than to arithmetic.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def answer_under(policy: str, record: dict[str, Any]) -> str:
    """The answer string a scoring policy would submit for one generated record.

    * `stated` — the model's own `model_answer`. **What every path scores today.**
    * `executed` — the executor's output, and the empty string when the plan does not run.
      The strict reading of the project's claim: the model transcribes, the CPU computes.
    * `executed_or_stated` — the executor's output where the plan runs, the stated answer
      otherwise. The practical reading: never worse than today unless the executor is wrong.

    Nothing here changes what is scored. It makes the three comparable on identical
    generations, which is the only way to find out whether the interpreter earns its place.
    """
    if policy not in ANSWER_POLICIES:
        raise ValueError(f"unknown answer policy {policy!r}; expected one of {ANSWER_POLICIES}")
    stated = str(record.get("model_answer", ""))
    if policy == "stated":
        return stated
    trip = check_record(record)
    executed = _as_answer(trip.executed) if trip.outcome in ("agrees", "disagrees") else ""
    if policy == "executed":
        return executed
    return executed or stated


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


__all__ = ["RELAXED_TOLERANCE", "RoundTrip", "RoundTripStats", "answers_agree",
           "check_many", "check_record",
           "disagreement_examples"]
