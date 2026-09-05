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


#: What the chart's categories *are*, inferred from the labels themselves.
#:
#: **Measured** over ChartQA's 28,299 training questions: the four most common openings
#: are *"what was the"* (6,485), *"what is the"* (3,291), *"what percentage of"* (1,459)
#: and *"how much did"* (750), and the list goes on with *"in what year"*, *"how many
#: people"*, *"which country was"*, *"who is the"*. Real questions name the kind of thing
#: they ask about. Synthetic questions said *"category"* every time, which is a word that
#: appears in almost no real question (`DECISIONS.md` 0122).
ENTITY_NOUNS: tuple[tuple[str, str], ...] = (
    ("year", "year"),
    ("quarter", "quarter"),
    ("month", "month"),
    ("country", "country"),
    ("state", "state"),
    ("age group", "age group"),
    ("category", "category"),
)

_MONTHS = {"jan", "feb", "mar", "apr", "may", "jun",
           "jul", "aug", "sep", "oct", "nov", "dec"}
_COUNTRIES = {"argentina", "australia", "brazil", "canada", "china", "france",
              "germany", "india", "italy", "japan", "mexico", "nigeria", "norway",
              "spain", "sweden", "united kingdom", "united states", "vietnam"}
_STATES = {"alabama", "alaska", "arizona", "california", "colorado", "florida",
           "georgia", "illinois", "michigan", "ohio", "oregon", "texas", "utah",
           "vermont", "virginia", "washington", "wyoming"}


def entity_noun(labels: list[str]) -> str:
    """The noun a person would use for these labels — "year", "country", "category".

    Cheap and deliberately conservative: anything it cannot recognise stays "category",
    which is what every question used to say.
    """
    lows = [x.strip().lower() for x in labels]
    if all(x.isdigit() and 1800 <= int(x) <= 2100 for x in lows):
        return "year"
    if all(x[:2] in {"q1", "q2", "q3", "q4"} for x in lows):
        return "quarter"
    if all(x[:3] in _MONTHS for x in lows):
        return "month"
    if sum(x in _COUNTRIES for x in lows) >= max(2, len(lows) // 2):
        return "country"
    if sum(x in _STATES for x in lows) >= max(2, len(lows) // 2):
        return "state"
    if all("-" in x or x.endswith("+") for x in lows):
        return "age group"
    return "category"


#: Past tense is the *majority* voice in ChartQA — *"what was the"* outnumbers *"what is
#: the"* roughly two to one — and synthetic data had none of it at all.
PAST_TENSE_SHARE = 0.55


#: Trailing clauses that lengthen a question the way real ones are lengthened. ChartQA's
#: questions run to a median of **11** words and a 90th percentile of **16**; synthetic
#: ran to 7 and never exceeded 10.
TAIL_CLAUSES: tuple[str, ...] = (
    "", "", "", " according to the chart", " in the chart",
    " shown in the graph", " based on the chart",
)


def _tense(rng: random.Random) -> tuple[str, str]:
    """`(is/was, does/did)` — one draw, so a question does not mix voices."""
    return ("was", "did") if rng.random() < PAST_TENSE_SHARE else ("is", "does")


#: Which aggregate an L3 question asks for.
#:
#: **Not uniform, and 0101 is why.** `PLAN.md` 6.1 grades stage 1 easy->hard, and 0101
#: settled what the grade means: L1-L2 give *uniform* coverage so the model meets every
#: operation, and **L3-L4 should look like ChartQA**. Against Claude's judgement of 60
#: random real questions (0091), `argmax`/`argmin` are **21.4%** of real questions and were
#: **7.3%** of synthetic — 2.9x under — while the rarer aggregates were over-weighted by
#: being drawn uniformly. Sampling seven operations with equal probability is a decision
#: about the prior over questions, and it was never made deliberately (`DECISIONS.md` 0123).
L3_OPERATION_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("argmax", 0.24),
    ("argmin", 0.20),
    ("max", 0.16),
    ("min", 0.13),
    ("mean", 0.11),
    ("sum", 0.09),
    ("count", 0.07),
)


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
    noun = entity_noun(labels)
    is_was, does_did = _tense(rng)
    tail = rng.choice(TAIL_CLAUSES)

    if level == "L1":
        lab = rng.choice(labels)
        plan = {"op": "lookup", "args": [lab]}
        q = rng.choice([
            f"What {is_was} the {quantity} for {lab}{tail}?",
            f"What {quantity} {is_was} shown for {lab}{tail}?",
            f"How much {is_was} {lab}{tail}?",
            f"What {is_was} the {quantity} of {lab}{tail}?",
            f"For the {noun} {lab}, what {is_was} the {quantity}{tail}?",
            f"How much {quantity} {is_was} recorded for {lab}{tail}?",
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
                f"How much more {is_was} {a} than {b}{tail}?",
                f"What {is_was} the difference between {a} and {b}{tail}?",
                f"By how much {is_was} {a} greater than {b}{tail}?",
                f"How much larger {is_was} the {quantity} for {a} than for {b}{tail}?",
                f"What {is_was} the gap between {a} and {b}{tail}?",
            ])
            answer = by_label[a] - by_label[b]
        elif style == "ratio" and by_label[b] != 0:
            plan = {"op": "ratio", "args": [a, b]}
            q = rng.choice([
                f"What {is_was} the ratio of {a} to {b}{tail}?",
                f"How many times larger {is_was} {a} than {b}{tail}?",
                f"What {is_was} the ratio between the {quantity} for {a} and for {b}{tail}?",
            ])
            answer = by_label[a] / by_label[b]
        else:
            plan = {"op": "compare", "args": [a, b]}
            q = rng.choice([
                f"{is_was.title()} {a} greater or less than {b}{tail}?",
                f"Which {is_was} larger, {a} or {b}{tail}?",
                f"{is_was.title()} the {quantity} for {a} greater or less than for {b}{tail}?",
            ])
            answer = "greater" if by_label[a] > by_label[b] else (
                "less" if by_label[a] < by_label[b] else "equal")

    elif level == "L3":
        op = rng.choices([name for name, _ in L3_OPERATION_WEIGHTS],
                         weights=[w for _, w in L3_OPERATION_WEIGHTS], k=1)[0]
        # Empty args is the executor's own idiom for "fold over all the evidence"
        # (`PLAN.md` Appendix B). Listing every label instead would blow the schema's
        # `maxItems: 4` on args as soon as a chart has five categories.
        plan = {"op": op, "args": []}
        aggregate_over_all = True
        values = [v for _, v in series]
        if op == "sum":
            q, answer = rng.choice([
                f"What {is_was} the total {quantity} across all {noun}s{tail}?",
                f"What {is_was} the sum of all the {quantity}s shown{tail}?",
                f"Added together, what {is_was} the total {quantity}{tail}?",
            ]), sum(values)
        elif op == "mean":
            q, answer = rng.choice([
                f"What {is_was} the average {quantity}{tail}?",
                f"What {is_was} the mean {quantity} across all {noun}s{tail}?",
                f"On average, what {is_was} the {quantity} per {noun}{tail}?",
            ]), statistics.fmean(values)
        elif op == "max":
            q, answer = rng.choice([
                f"What {is_was} the highest {quantity} shown{tail}?",
                f"What {is_was} the largest {quantity} in the chart{tail}?",
                f"What {is_was} the peak {quantity} recorded{tail}?",
            ]), max(values)
        elif op == "min":
            q, answer = rng.choice([
                f"What {is_was} the lowest {quantity} shown{tail}?",
                f"What {is_was} the smallest {quantity} in the chart{tail}?",
                f"What {is_was} the minimum {quantity} recorded{tail}?",
            ]), min(values)
        elif op == "count":
            q, answer = rng.choice([
                f"How many {noun}s are shown{tail}?",
                f"How many {noun}s appear in the chart{tail}?",
                f"What {is_was} the number of {noun}s shown{tail}?",
            ]), float(len(values))
        elif op == "argmax":
            q, answer = rng.choice([
                f"Which {noun} {'had' if is_was == 'was' else 'has'} the highest "
                f"{quantity}{tail}?",
                f"In which {noun} {is_was} the {quantity} highest{tail}?",
                f"Which {noun} recorded the largest {quantity}{tail}?",
            ]), max(series, key=lambda p: p[1])[0]
        else:
            q, answer = rng.choice([
                f"Which {noun} {'had' if is_was == 'was' else 'has'} the lowest "
                f"{quantity}{tail}?",
                f"In which {noun} {is_was} the {quantity} lowest{tail}?",
                f"Which {noun} recorded the smallest {quantity}{tail}?",
            ]), min(series, key=lambda p: p[1])[0]

    elif level == "L4":
        lab = rng.choice(labels)
        values = [v for _, v in series]
        style = rng.choice(["vs_mean", "vs_max", "share"])
        aggregate_over_all = True
        if style == "vs_mean":
            plan = {"op": "difference", "args": [lab, {"op": "mean", "args": []}]}
            q = rng.choice([
                f"How far {is_was} {lab} from the average{tail}?",
                f"By how much {does_did} {lab} differ from the mean {quantity}{tail}?",
                f"What {is_was} the difference between {lab} and the average{tail}?",
            ])
            answer = by_label[lab] - statistics.fmean(values)
        elif style == "vs_max":
            plan = {"op": "difference", "args": [{"op": "max", "args": []}, lab]}
            q = rng.choice([
                f"How much lower {is_was} {lab} than the highest {noun}{tail}?",
                f"How far below the maximum {is_was} {lab}{tail}?",
                f"What {is_was} the gap between {lab} and the highest {quantity}{tail}?",
            ])
            answer = max(values) - by_label[lab]
        else:
            total = sum(values)
            if total == 0:
                return None
            plan = {"op": "ratio", "args": [lab, {"op": "sum", "args": []}]}
            q = rng.choice([
                f"What fraction of the total {does_did} {lab} represent{tail}?",
                f"What share of the total {quantity} {is_was} {lab}{tail}?",
                # No trailing clause here: the template already ends in a preposition,
                # and "account for shown in the graph" is not English.
                f"What proportion of the total {does_did} {lab} account for?",
            ])
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
