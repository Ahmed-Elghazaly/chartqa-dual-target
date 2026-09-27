from __future__ import annotations

import itertools
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Literal

from chartqa_dt.plans.executor import MAX_FACTS, EvidenceItem, execute_label_plan

Level = Literal["L1", "L2", "L3", "L4"]
LEVELS: tuple[Level, ...] = ("L1", "L2", "L3", "L4")


def format_answer(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return ""
    f = float(value)
    if not math.isfinite(f):
        raise ValueError("synthetic answers must be finite")
    if abs(f - round(f)) < 1e-9:
        return str(round(f))
    return f"{f:.2f}".rstrip("0").rstrip(".")


@dataclass
class SynthQuestion:
    level: Level
    question: str
    answer: str
    plan: dict[str, Any]
    evidence_labels: list[str]
    unit: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _labels_for(plan: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if not isinstance(plan, dict):
        return out
    for a in plan.get("args") or []:
        if isinstance(a, str):
            out.append(a)
        elif isinstance(a, dict):
            out += _labels_for(a)
    return out


_MONTHS = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}
_COUNTRIES = {
    "argentina",
    "australia",
    "brazil",
    "canada",
    "china",
    "france",
    "germany",
    "india",
    "italy",
    "japan",
    "mexico",
    "nigeria",
    "norway",
    "spain",
    "sweden",
    "united kingdom",
    "united states",
    "vietnam",
}
_STATES = {
    "alabama",
    "alaska",
    "arizona",
    "california",
    "colorado",
    "florida",
    "georgia",
    "illinois",
    "michigan",
    "ohio",
    "oregon",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "wyoming",
}


def entity_noun(labels: list[str]) -> str:
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


PAST_TENSE_SHARE = 0.55


TAIL_CLAUSES: tuple[str, ...] = (
    "",
    "",
    "",
    " according to the chart",
    " in the chart",
    " shown in the graph",
    " based on the chart",
)


def _tense(rng: random.Random) -> tuple[str, str]:
    return ("was", "did") if rng.random() < PAST_TENSE_SHARE else ("is", "does")


L3_OPERATION_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("argmax", 0.24),
    ("argmin", 0.20),
    ("max", 0.16),
    ("min", 0.13),
    ("mean", 0.11),
    ("sum", 0.09),
    ("count", 0.07),
)

MAX_AGGREGATE_SCOPE_WORDS = 14


def _format_aggregate_scope(labels: list[str]) -> str:
    if len(labels) < 2:
        raise ValueError("an aggregate scope needs at least two labels")
    return " among " + ", ".join(labels[:-1]) + f", and {labels[-1]}"


COLOUR_REFERENCE_SHARE = 0.20


def colour_reference(labels: list[str], colours: list[str] | None, index: int) -> str | None:
    if not colours or index >= len(colours):
        return None
    from chartqa_dt.data.colours import describe_colour

    names = [describe_colour(c) for c in colours[: len(labels)]]
    mine = names[index]
    if not mine or names.count(mine) != 1:
        return None
    return mine


POSITION_REFERENCE_SHARE = 0.34

POSITION_WORDS: dict[str, tuple[str, str]] = {
    "vbar": ("leftmost", "rightmost"),
    "grouped_bar": ("leftmost", "rightmost"),
    "line": ("leftmost", "rightmost"),
    "multi_line": ("leftmost", "rightmost"),
    "area": ("leftmost", "rightmost"),
    "scatter": ("leftmost", "rightmost"),
    "hbar": ("topmost", "bottommost"),
}


def position_reference(labels: list[str], index: int, chart_type: str | None) -> str | None:
    words = POSITION_WORDS.get(chart_type or "")
    if not words or len(labels) < 2:
        return None
    if index == 0:
        return words[0]
    if index == len(labels) - 1:
        return words[1]
    return None


def build_question(
    level: Level,
    series: list[tuple[str, float]],
    rng: random.Random,
    *,
    unit: str | None = None,
    quantity: str = "value",
    colours: list[str] | None = None,
    mark: str = "bar",
    chart_type: str | None = None,
    l2_style: str | None = None,
    comparison_outcome: str | None = None,
) -> SynthQuestion | None:
    if level not in LEVELS:
        raise ValueError(f"unknown level: {level!r}")
    if any(not isinstance(label, str) or not label.strip() for label, _ in series):
        raise ValueError("synthetic series labels must be non-empty strings")
    labels = [label for label, _ in series]
    if len(labels) != len(set(labels)):
        raise ValueError("synthetic series labels must be unique")
    for _label, value in series:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("synthetic series values must be finite numbers")
        try:
            finite = math.isfinite(float(value))
        except (OverflowError, ValueError):
            finite = False
        if not finite:
            raise ValueError("synthetic series values must be finite numbers")
    if len(series) < 2:
        return None
    by_label = dict(series)
    aggregate_labels = labels
    if len(labels) > MAX_FACTS:
        selected = rng.sample(range(len(labels)), MAX_FACTS)
        while len(selected) > 2:
            candidate = [label for index, label in enumerate(labels) if index in set(selected)]
            if len(_format_aggregate_scope(candidate).split()) <= MAX_AGGREGATE_SCOPE_WORDS:
                break
            selected.pop()
        chosen = set(selected)
        aggregate_labels = [label for index, label in enumerate(labels) if index in chosen]
    aggregate_scope = ""
    if aggregate_labels != labels:
        aggregate_scope = _format_aggregate_scope(aggregate_labels)
        if len(aggregate_scope.split()) > MAX_AGGREGATE_SCOPE_WORDS:
            return None
    noun = entity_noun(labels)
    is_was, does_did = _tense(rng)
    tail = rng.choice(TAIL_CLAUSES)
    if aggregate_scope:
        tail = ""

    plan: dict[str, Any]
    question_text: str
    answer: float | str

    if level == "L1":
        lab = rng.choice(labels)
        plan = {"op": "lookup", "args": [lab]}
        if rng.random() < POSITION_REFERENCE_SHARE:
            where = position_reference(labels, labels.index(lab), chart_type)
            if where:
                question_text = rng.choice(
                    [
                        f"What {is_was} the {quantity} of the {where} {mark}{tail}?",
                        f"What {is_was} the {where} {quantity}{tail}?",
                        f"How much {is_was} the {where} {mark}{tail}?",
                    ]
                )
                return SynthQuestion(
                    level=level,
                    question=question_text,
                    answer=format_answer(by_label[lab]),
                    plan=plan,
                    evidence_labels=sorted(set(_labels_for(plan))),
                    unit=unit,
                    meta={"referring_expression": "position"},
                )
        if rng.random() < COLOUR_REFERENCE_SHARE:
            phrase = colour_reference(labels, colours, labels.index(lab))
            if phrase:
                question_text = rng.choice(
                    [
                        f"What {is_was} the {quantity} of the {phrase} {mark}{tail}?",
                        f"How much {is_was} the {phrase} {mark}{tail}?",
                        f"What {quantity} {is_was} shown for the {phrase} {mark}{tail}?",
                    ]
                )
                answer = by_label[lab]
                used = sorted(set(_labels_for(plan)))
                return SynthQuestion(
                    level=level,
                    question=question_text,
                    answer=format_answer(answer),
                    plan=plan,
                    evidence_labels=used,
                    unit=unit,
                    meta={"referring_expression": "colour"},
                )
        question_text = rng.choice(
            [
                f"What {is_was} the {quantity} for {lab}{tail}?",
                f"What {quantity} {is_was} shown for {lab}{tail}?",
                f"How much {is_was} {lab}{tail}?",
                f"What {is_was} the {quantity} of {lab}{tail}?",
                f"For the {noun} {lab}, what {is_was} the {quantity}{tail}?",
                f"How much {quantity} {is_was} recorded for {lab}{tail}?",
            ]
        )
        answer = by_label[lab]

    elif level == "L2":
        style = l2_style or rng.choice(
            ["signed_difference", "absolute_distance", "ratio", "compare", "winner"]
        )
        if style not in {
            "signed_difference",
            "absolute_distance",
            "ratio",
            "compare",
            "winner",
        }:
            raise ValueError(f"unknown L2 question style: {style!r}")
        if comparison_outcome not in {None, "greater", "less", "equal"}:
            raise ValueError(f"unknown comparison outcome: {comparison_outcome!r}")
        if comparison_outcome is not None and style != "compare":
            raise ValueError("comparison_outcome is valid only for the compare style")
        pairs = list(itertools.permutations(labels, 2))
        if style == "ratio":
            pairs = [(left, right) for left, right in pairs if by_label[right] != 0]
            if not pairs:
                return None
        if style == "compare" and comparison_outcome:
            pairs = [
                (left, right)
                for left, right in pairs
                if (
                    "greater"
                    if by_label[left] > by_label[right]
                    else "less"
                    if by_label[left] < by_label[right]
                    else "equal"
                )
                == comparison_outcome
            ]
            if not pairs:
                return None
        a, b = rng.choice(pairs)
        if style == "signed_difference":
            plan = {"op": "difference", "args": [a, b]}
            question_text = rng.choice(
                [
                    f"What {is_was} {a} minus {b}{tail}?",
                    f"Subtract the {quantity} for {b} from the {quantity} for {a}{tail}?",
                ]
            )
            answer = by_label[a] - by_label[b]
        elif style == "absolute_distance":
            plan = {"op": "absolute_difference", "args": [a, b]}
            question_text = rng.choice(
                [
                    f"How far apart were {a} and {b}{tail}?",
                    f"What {is_was} the absolute distance between {a} and {b}{tail}?",
                ]
            )
            answer = abs(by_label[a] - by_label[b])
        elif style == "ratio":
            plan = {"op": "ratio", "args": [a, b]}
            question_text = rng.choice(
                [
                    f"What {is_was} the ratio of {a} to {b}{tail}?",
                    f"How many times as large {is_was} {a} as {b}{tail}?",
                    f"What {is_was} the ratio between the {quantity} for {a} and for {b}{tail}?",
                ]
            )
            answer = by_label[a] / by_label[b]
        elif style == "compare":
            plan = {"op": "compare", "args": [a, b]}
            question_text = rng.choice(
                [
                    f"{is_was.title()} {a} greater than, less than, or equal to {b}{tail}?",
                    f"{is_was.title()} the {quantity} for {a} greater than, less than, "
                    f"or equal to that for {b}{tail}?",
                ]
            )
            answer = (
                "greater" if by_label[a] > by_label[b] else ("less" if by_label[a] < by_label[b] else "equal")
            )
        elif style == "winner":
            if by_label[a] == by_label[b]:
                return None
            plan = {"op": "argmax", "args": [a, b]}
            question_text = f"Which {is_was} larger, {a} or {b}{tail}?"
            answer = a if by_label[a] > by_label[b] else b
        else:
            raise AssertionError(style)

    elif level == "L3":
        op = rng.choices(
            [name for name, _ in L3_OPERATION_WEIGHTS], weights=[w for _, w in L3_OPERATION_WEIGHTS], k=1
        )[0]
        plan = {"op": op, "args": list(aggregate_labels)}
        aggregate_series = [(label, by_label[label]) for label in aggregate_labels]
        values = [value for _, value in aggregate_series]
        if op == "sum":
            question_text, answer = (
                rng.choice(
                    [
                        f"What {is_was} the total {quantity}{aggregate_scope}{tail}?",
                        f"What {is_was} the sum of the selected {quantity}s{aggregate_scope}{tail}?",
                        f"Added together, what {is_was} the total {quantity}{aggregate_scope}{tail}?",
                    ]
                ),
                sum(values),
            )
        elif op == "mean":
            question_text, answer = (
                rng.choice(
                    [
                        f"What {is_was} the average {quantity}{aggregate_scope}{tail}?",
                        f"What {is_was} the mean {quantity}{aggregate_scope}{tail}?",
                        f"On average, what {is_was} the {quantity}{aggregate_scope}{tail}?",
                    ]
                ),
                statistics.fmean(values),
            )
        elif op == "max":
            question_text, answer = (
                rng.choice(
                    [
                        f"What {is_was} the highest {quantity}{aggregate_scope}{tail}?",
                        f"What {is_was} the largest {quantity}{aggregate_scope}{tail}?",
                        f"What {is_was} the peak {quantity}{aggregate_scope}{tail}?",
                    ]
                ),
                max(values),
            )
        elif op == "min":
            question_text, answer = (
                rng.choice(
                    [
                        f"What {is_was} the lowest {quantity}{aggregate_scope}{tail}?",
                        f"What {is_was} the smallest {quantity}{aggregate_scope}{tail}?",
                        f"What {is_was} the minimum {quantity}{aggregate_scope}{tail}?",
                    ]
                ),
                min(values),
            )
        elif op == "count":
            question_text, answer = (
                rng.choice(
                    [
                        f"How many selected facts are listed{aggregate_scope}{tail}?",
                        f"How many values are included{aggregate_scope}{tail}?",
                        f"What {is_was} the number of selected values{aggregate_scope}{tail}?",
                    ]
                ),
                float(len(values)),
            )
        elif op in ("argmax", "argmin"):
            target = max(values) if op == "argmax" else min(values)
            if values.count(target) > 1:
                return None
            if op == "argmax":
                question_text, answer = (
                    rng.choice(
                        [
                            f"Which {noun} {'had' if is_was == 'was' else 'has'} the highest "
                            f"{quantity}{aggregate_scope}{tail}?",
                            f"In which {noun} {is_was} the {quantity} highest{aggregate_scope}{tail}?",
                            f"Which {noun} recorded the largest {quantity}{aggregate_scope}{tail}?",
                        ]
                    ),
                    max(aggregate_series, key=lambda p: p[1])[0],
                )
            else:
                question_text, answer = (
                    rng.choice(
                        [
                            f"Which {noun} {'had' if is_was == 'was' else 'has'} the lowest "
                            f"{quantity}{aggregate_scope}{tail}?",
                            f"In which {noun} {is_was} the {quantity} lowest{aggregate_scope}{tail}?",
                            f"Which {noun} recorded the smallest {quantity}{aggregate_scope}{tail}?",
                        ]
                    ),
                    min(aggregate_series, key=lambda p: p[1])[0],
                )
        else:
            raise ValueError(f"unhandled L3 operation: {op!r}")

    elif level == "L4":
        lab = rng.choice(aggregate_labels)
        values = [by_label[label] for label in aggregate_labels]
        aggregate_args = list(aggregate_labels)
        style = rng.choice(["vs_mean", "vs_max", "fraction", "percentage"])
        if style == "vs_mean":
            plan = {"op": "absolute_difference", "args": [lab, {"op": "mean", "args": aggregate_args}]}
            question_text = rng.choice(
                [
                    f"How far {is_was} {lab} from the average{aggregate_scope}{tail}?",
                    f"What {is_was} the absolute distance from {lab} to the mean{aggregate_scope}{tail}?",
                ]
            )
            answer = abs(by_label[lab] - statistics.fmean(values))
        elif style == "vs_max":
            below_max = [label for label in aggregate_labels if by_label[label] < max(values)]
            if not below_max:
                return None
            lab = rng.choice(below_max)
            plan = {"op": "difference", "args": [{"op": "max", "args": aggregate_args}, lab]}
            question_text = rng.choice(
                [
                    f"How much lower {is_was} {lab} than the highest {noun}{aggregate_scope}{tail}?",
                    f"How far below the maximum {is_was} {lab}{aggregate_scope}{tail}?",
                    f"What {is_was} the gap between {lab} and the highest {quantity}{aggregate_scope}{tail}?",
                ]
            )
            answer = max(values) - by_label[lab]
        else:
            total = sum(values)
            if total <= 0 or any(value < 0 for value in values):
                return None
            total_plan: dict[str, Any] = {"op": "sum", "args": aggregate_args}
            if style == "fraction":
                plan = {"op": "ratio", "args": [lab, total_plan]}
                question_text = (
                    f"What fraction of the selected total {does_did} {lab} represent{aggregate_scope}{tail}?"
                )
                answer = by_label[lab] / total
            else:
                plan = {"op": "percentage", "args": [lab, total_plan]}
                question_text = (
                    f"What percentage of the selected total {does_did} {lab} represent"
                    f"{aggregate_scope}{tail}?"
                )
                answer = 100 * by_label[lab] / total
    else:
        raise AssertionError(level)

    used = list(dict.fromkeys(_labels_for(plan)))
    evidence = [EvidenceItem(lab, by_label[lab], unit) for lab in used]

    try:
        got = execute_label_plan(plan, evidence)
    except Exception:
        return None
    if isinstance(answer, str):
        if str(got) != answer:
            return None
    elif got is None or abs(float(got) - float(answer)) > 1e-6 * max(1.0, abs(float(answer))):
        return None

    return SynthQuestion(
        level=level,
        question=question_text,
        answer=format_answer(answer),
        plan=plan,
        evidence_labels=used,
        unit=unit,
        meta={"style": locals().get("style", ""), "n_series": len(series)},
    )
