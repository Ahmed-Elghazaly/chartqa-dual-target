"""The project's central claim, as a measurement — `chartqa_dt.plans.roundtrip`.

`IDEA.md`'s premise is that the model emits a typed expression tree next to its answer and
a deterministic executor recomputes that answer, making the arithmetic checkable rather
than asserted. If the emitted plan does not reproduce the emitted answer, the plan is
decoration. So this is a headline number and it is measured from the first zero-shot
baseline onward.

The four outcomes are kept distinct because they call for different fixes: `raises` is a
format error the prompt can usually correct, `disagrees` is a reasoning error that needs
training.
"""

from __future__ import annotations

import pytest

from chartqa_dt.plans.roundtrip import (
    RoundTripStats,
    check_many,
    check_record,
    disagreement_examples,
)


def record(plan, answer, evidence=None):
    return {
        "answerable": True,
        "evidence": evidence if evidence is not None else [
            {"label": "2018", "value": 210, "unit": None, "bbox": [0, 0, 10, 10]},
            {"label": "2019", "value": 245, "unit": None, "bbox": [0, 0, 10, 10]},
        ],
        "plan": plan,
        "model_answer": answer,
    }


def test_a_correct_plan_agrees():
    r = check_record(record({"op": "difference", "args": ["2019", "2018"]}, "35"))
    assert r.outcome == "agrees" and r.ok


def test_rounding_is_not_a_disagreement():
    """The model states a rounded answer; calling that wrong would measure formatting."""
    ev = [{"label": "a", "value": 1.0, "bbox": [0, 0, 1, 1]},
          {"label": "b", "value": 3.0, "bbox": [0, 0, 1, 1]}]
    r = check_record(record({"op": "ratio", "args": ["a", "b"]}, "0.33", ev))
    assert r.outcome == "agrees", f"executor gave {r.executed}"


def test_the_wrong_operation_disagrees_rather_than_raising():
    """The pattern the zero-shot probe showed: `lookup` where `argmax` was meant.

    "Which year was highest?" is answered by a *label*. `lookup` returns a *value*, so it
    runs happily and returns the wrong kind of thing — which is why this outcome has to
    be separated from a crash.
    """
    r = check_record(record({"op": "lookup", "args": ["2019"]}, "2019"))
    assert r.outcome == "disagrees"
    assert r.executed == 245.0 and r.stated == "2019"

    fixed = check_record(record({"op": "argmax", "args": []}, "2019"))
    assert fixed.outcome == "agrees", "argmax returns the label, which is the answer"


@pytest.mark.parametrize(("plan", "fragment"), [
    ({"op": "lookup", "args": ["2018", "2019"]}, "exactly one"),
    ({"op": "compare", "args": ["2018", "2019", "2020"]}, "exactly 2"),
    ({"op": "difference", "args": ["2018"]}, "exactly 2"),
    ({"op": "lookup", "args": ["missing-label"]}, ""),
])
def test_an_unexecutable_plan_raises_and_the_reason_is_kept(plan, fragment):
    r = check_record(record(plan, "1"))
    assert r.outcome == "raises"
    assert r.error, "a refusal must say why, or it cannot be acted on"
    if fragment:
        assert fragment in r.error


def test_a_missing_plan_is_its_own_outcome():
    """Already a failure elsewhere; counting it as a disagreement would double-count."""
    assert check_record({"model_answer": "5"}).outcome == "no_plan"
    assert check_record({"plan": {}, "model_answer": "5"}).outcome == "no_plan"
    assert check_record({"plan": "sum", "model_answer": "5"}).outcome == "no_plan"


def test_stats_separate_agreement_from_executability():
    """Two different things: a plan can run and still be wrong."""
    records = [
        record({"op": "difference", "args": ["2019", "2018"]}, "35"),      # agrees
        record({"op": "lookup", "args": ["2019"]}, "2019"),                # disagrees
        record({"op": "lookup", "args": ["a", "b"]}, "1"),                 # raises
        {"model_answer": "5"},                                             # no plan
    ]
    _, stats = check_many(records)
    assert stats.total == 4
    assert stats.agreement == pytest.approx(0.25), "1 of 4 records round-trips"
    assert stats.executable == pytest.approx(2 / 3), "2 of 3 plans run"
    assert stats.counts == {"agrees": 1, "disagrees": 1, "raises": 1, "no_plan": 1}
    assert stats.errors, "the executor's refusal is recorded by reason"


def test_empty_input_does_not_divide_by_zero():
    stats = RoundTripStats()
    assert stats.agreement == 0.0 and stats.executable == 0.0
    _, s = check_many([])
    assert s.total == 0 and s.agreement == 0.0


def test_disagreements_are_recoverable_for_reading():
    """Reading them is how the op-choice confusion was found in the first place."""
    records = [record({"op": "lookup", "args": ["2019"]}, "2019"),
               record({"op": "difference", "args": ["2019", "2018"]}, "35")]
    examples = disagreement_examples(records)
    assert len(examples) == 1
    assert examples[0]["plan"]["op"] == "lookup"
    assert examples[0]["stated"] == "2019" and examples[0]["executed"] == 245.0


def test_the_describe_output_names_the_claim():
    _, stats = check_many([record({"op": "difference", "args": ["2019", "2018"]}, "35")])
    text = stats.describe()
    assert "round-trip" in text and "reproduces the answer" in text
