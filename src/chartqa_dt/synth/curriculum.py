"""Questions, exact answers and exact typed plans, from data we control.

`PLAN.md` 3.5 makes synthetic charts the **primary** source of typed-plan
supervision, because the uniqueness rule admits only about 5.7% of real ChartQA
questions (`IDEA.md` §5.1). Here the answer and the plan are known *by
construction* — no inference, no ambiguity — which is the whole reason this path
exists.

Difficulty levels, per `PLAN.md` 3.5:

* **L1** — a single lookup.
* **L2** — a two-value comparison or difference.
* **L3** — an aggregate over the series.
* **L4** — a nested two-operation plan.

Two invariants hold for every generated item, and both are asserted rather than
assumed: the plan **executes to the answer**, and every label the plan references
appears in the evidence. A synthetic example whose own plan does not reproduce its
own answer would teach the model something false with perfect confidence.

Plans use the corrected executor semantics (`DECISIONS.md` 0016): a bare string
argument is always an evidence label.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Literal

from chartqa_dt.plans.executor import EvidenceItem, execute

Level = Literal["L1", "L2", "L3", "L4"]
LEVELS: tuple[Level, ...] = ("L1", "L2", "L3", "L4")


def format_answer(value: Any) -> str:
    """Render an answer the way the official metric will compare it.

    The canonical `relaxed_correctness` compares a gold answer of ``"0"`` as a
    **string** (`DECISIONS.md` 0015), so ``"0.0"`` would score incorrect. Integral
    values are therefore emitted without a decimal part.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return ""
    f = float(value)
    if abs(f - round(f)) < 1e-9:
        return str(round(f))
    return f"{f:.2f}".rstrip("0").rstrip(".")


@dataclass
class SynthQuestion:
    level: Level
    question: str
    answer: str
    plan: dict
    evidence_labels: list[str]
    unit: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _labels_for(plan: dict) -> list[str]:
    out: list[str] = []
    if not isinstance(plan, dict):
        return out
    for a in plan.get("args") or []:
        if isinstance(a, str):
            out.append(a)
        elif isinstance(a, dict):
            out += _labels_for(a)
    return out


def build_question(
    level: Level,
    series: list[tuple[str, float]],
    rng: random.Random,
    *,
    unit: str | None = None,
    quantity: str = "value",
) -> SynthQuestion | None:
    """One question at the requested level, or None if the data cannot support it."""
    if len(series) < 2:
        return None
    by_label = dict(series)
    labels = [lab for lab, _ in series]
    aggregate_over_all = False

    if level == "L1":
        lab = rng.choice(labels)
        plan = {"op": "lookup", "args": [lab]}
        q = rng.choice([
            f"What is the {quantity} for {lab}?",
            f"What {quantity} is shown for {lab}?",
            f"How much is {lab}?",
        ])
        answer = by_label[lab]

    elif level == "L2":
        a, b = rng.sample(labels, 2)
        if by_label[a] < by_label[b]:
            a, b = b, a
        style = rng.choice(["difference", "ratio", "compare"])
        if style == "difference":
            plan = {"op": "difference", "args": [a, b]}
            q = rng.choice([
                f"How much more is {a} than {b}?",
                f"What is the difference between {a} and {b}?",
            ])
            answer = by_label[a] - by_label[b]
        elif style == "ratio" and by_label[b] != 0:
            plan = {"op": "ratio", "args": [a, b]}
            q = f"What is the ratio of {a} to {b}?"
            answer = by_label[a] / by_label[b]
        else:
            plan = {"op": "compare", "args": [a, b]}
            q = f"Is {a} greater or less than {b}?"
            answer = "greater" if by_label[a] > by_label[b] else (
                "less" if by_label[a] < by_label[b] else "equal")

    elif level == "L3":
        op = rng.choice(["sum", "mean", "max", "min", "count", "argmax", "argmin"])
        # Empty args is the executor's own idiom for "fold over all the evidence"
        # (`PLAN.md` Appendix B). Listing every label instead would blow the schema's
        # `maxItems: 4` on args as soon as a chart has five categories.
        plan = {"op": op, "args": []}
        aggregate_over_all = True
        values = [v for _, v in series]
        if op == "sum":
            q, answer = f"What is the total {quantity} across all categories?", sum(values)
        elif op == "mean":
            q, answer = f"What is the average {quantity}?", statistics.fmean(values)
        elif op == "max":
            q, answer = f"What is the highest {quantity} shown?", max(values)
        elif op == "min":
            q, answer = f"What is the lowest {quantity} shown?", min(values)
        elif op == "count":
            q, answer = "How many categories are shown?", float(len(values))
        elif op == "argmax":
            q, answer = f"Which category has the highest {quantity}?", max(series, key=lambda p: p[1])[0]
        else:
            q, answer = f"Which category has the lowest {quantity}?", min(series, key=lambda p: p[1])[0]

    elif level == "L4":
        lab = rng.choice(labels)
        values = [v for _, v in series]
        style = rng.choice(["vs_mean", "vs_max", "share"])
        aggregate_over_all = True
        if style == "vs_mean":
            plan = {"op": "difference", "args": [lab, {"op": "mean", "args": []}]}
            q = f"How far is {lab} from the average?"
            answer = by_label[lab] - statistics.fmean(values)
        elif style == "vs_max":
            plan = {"op": "difference", "args": [{"op": "max", "args": []}, lab]}
            q = f"How much lower is {lab} than the highest category?"
            answer = max(values) - by_label[lab]
        else:
            total = sum(values)
            if total == 0:
                return None
            plan = {"op": "ratio", "args": [lab, {"op": "sum", "args": []}]}
            q = f"What fraction of the total does {lab} represent?"
            answer = by_label[lab] / total
    else:
        raise ValueError(f"unknown level: {level!r}")

    # An aggregate folds over whatever evidence is present, so the evidence list is the
    # plan's real argument there and must be the whole series, in chart order.
    used = labels if aggregate_over_all else sorted(set(_labels_for(plan)))
    evidence = [EvidenceItem(lab, by_label[lab], unit) for lab in used]

    # Invariant: the plan must reproduce the answer. A synthetic example whose own
    # plan disagrees with its own answer teaches something false with confidence.
    try:
        got = execute(plan, evidence)
    except Exception:  # noqa: BLE001 - a generator bug, surfaced by returning None
        return None
    if isinstance(answer, str):
        if str(got) != answer:
            return None
    elif got is None or abs(float(got) - float(answer)) > 1e-6 * max(1.0, abs(float(answer))):
        return None

    return SynthQuestion(level=level, question=q, answer=format_answer(answer), plan=plan,
                         evidence_labels=used, unit=unit,
                         meta={"style": locals().get("style", ""), "n_series": len(series)})
