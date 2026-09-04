"""Building the plan a question asks for, instead of searching for one that fits the answer.

`plans.mining` works backwards — *which operations reproduce this answer?* — and refuses when
several do, which is 53.9% of ChartQA rows. `plans.forward` reads the question, builds that
plan, and checks it against the answer. Measured on the same 4,000 records: 14.9% -> 44.9%,
with 100% agreement on the 330 where both commit.

These tests pin the properties that make that safe, not the numbers.
"""
from __future__ import annotations

import pytest

from chartqa_dt.plans import forward
from chartqa_dt.plans.intent import (
    intended_operations,
    labels_named_in,
    restricts_to_a_subset,
)

SORTED_CHART = [
    {"label": "Nigeria", "value": 154.3, "unit": None},
    {"label": "Egypt", "value": 54.74, "unit": None},
    {"label": "Kenya", "value": 46.87, "unit": None},
]


# ------------------------------------------------- the collision that motivates all of this


def test_the_same_chart_gives_two_plans_for_two_questions():
    """The whole point. Nigeria is both `lookup('Nigeria')` and `max()` of its column, and
    an answer-first search can never tell which was asked for. One word of the question can.
    """
    named = forward.build("How many internet users did Nigeria have?",
                          answer="154.3", evidence=SORTED_CHART)
    extreme = forward.build("Which country had the most internet users?",
                            answer="Nigeria", evidence=SORTED_CHART)
    assert named.plan == {"op": "lookup", "args": ["Nigeria"]}
    assert extreme.plan == {"op": "argmax", "args": []}


def test_a_named_label_wins_over_an_extremum_word():
    """*"What was the peak number of visits in 2019?"* names 2019 AND says "peak". The answer
    is the 2019 cell. Suppressing `lookup` whenever an extremum word appeared cost 3 points
    of precision on real data."""
    ev = [{"label": "2018", "value": 3.0, "unit": None},
          {"label": "2019", "value": 9.0, "unit": None}]
    built = forward.build("What was the peak number of visits in 2019?",
                          answer="9", evidence=ev)
    assert built.plan == {"op": "lookup", "args": ["2019"]}


# ------------------------------------------------- the answer checks, it never chooses


def test_a_plan_that_does_not_reproduce_the_answer_is_refused():
    built = forward.build("How many internet users did Nigeria have?",
                          answer="999", evidence=SORTED_CHART)
    assert not built.ok
    assert "none reproduced" in built.reason


def test_the_answer_is_never_used_to_pick_the_operation():
    """`intended_operations` takes no answer at all -- that signature is the guarantee. If it
    could see the answer, agreement with the miner would restate the label, not measure it."""
    import inspect
    params = inspect.signature(intended_operations).parameters
    assert "answer" not in params
    assert set(params) == {"question", "labels"}


# ------------------------------------------------- abstaining is a correct outcome


def test_wording_that_does_not_decide_produces_nothing():
    built = forward.build("Tell me about this chart.", answer="154.3", evidence=SORTED_CHART)
    assert not built.ok
    assert "does not say" in built.reason


def test_a_composite_question_abstains_rather_than_guessing():
    """*"the sum of the highest value and the lowest value"* asks for arithmetic OVER extrema.
    Both families fire, neither is the whole answer, and either guess is wrong."""
    assert intended_operations("What is the sum of highest value and lowest value?",
                               labels=["a", "b"]) == set()


def test_a_fold_is_dropped_when_the_question_restricts_the_chart():
    """A global `max()` once returned the right number for *"Norway's largest age group
    between 45 and 69 years old in 2021"* -- right by accident, wrong as a reading."""
    labels = [f"{g} · {y}" for g in ("0-24 years", "45-69 years") for y in (2020, 2021)]
    q = "How many people were in Norway's largest age group between 45 and 69 in 2021?"
    assert restricts_to_a_subset(q, labels)
    ev = [{"label": x, "value": float(i), "unit": None} for i, x in enumerate(labels)]
    assert not forward.build(q, answer=str(float(len(labels) - 1)), evidence=ev).ok


def test_a_whole_chart_question_keeps_its_fold():
    """The guard must not fire on a question that really does ask about everything."""
    assert not restricts_to_a_subset("Which place shows the highest number of voters?",
                                     ["Eastern Europe", "World"])


def test_the_guard_only_fires_when_the_year_is_actually_in_the_labels():
    """A year mentioned in passing, on a chart not indexed by year, restricts nothing."""
    assert not restricts_to_a_subset("Which country led the market in 2019?",
                                     ["Nigeria", "Egypt"])


# ------------------------------------------------- label matching


@pytest.mark.parametrize("question,expected", [
    ("How many users did Nigeria have?", ["Nigeria"]),
    ("Compare Nigeria and Egypt", ["Nigeria", "Egypt"]),
    ("What about Chad?", []),
])
def test_labels_named_in(question, expected):
    assert labels_named_in(question, ["Nigeria", "Egypt", "Kenya"]) == expected


def test_two_named_labels_do_not_make_a_lookup():
    """`lookup` takes one label. A question naming two is asking to relate them."""
    assert "lookup" not in intended_operations("Compare Nigeria and Egypt",
                                               labels=["Nigeria", "Egypt"])


def test_a_difference_tries_both_orders_and_lets_the_answer_decide():
    """The wording rarely says which way round; the sign of the gold answer does."""
    ev = [{"label": "Alpha", "value": 10.0, "unit": None},
          {"label": "Beta", "value": 4.0, "unit": None}]
    q = "What is the difference between Alpha and Beta?"
    assert forward.build(q, answer="6", evidence=ev).plan == {
        "op": "difference", "args": ["Alpha", "Beta"]}
    assert forward.build(q, answer="-6", evidence=ev).plan == {
        "op": "difference", "args": ["Beta", "Alpha"]}


def test_lookup_is_tried_before_any_fold():
    """A question naming a label is about that label, whatever else is true of its value."""
    assert forward.PRIORITY.index("lookup") == 0
