"""The deterministic executor, including the correction to PLAN.md Appendix B.

The single most important test here is
`test_bare_string_is_always_an_evidence_label`. Appendix B, as written, makes
`mean(["2019", "2018"])` return 2018.5 — the mean of the label *text* — while
`argmax(["2019", "2018"])` treats the same arguments as labels. With numeric-looking
categories, which chart axes overwhelmingly are, that returns a plausible wrong
number and raises nothing. See DECISIONS.md 0016.
"""

from __future__ import annotations

import pytest

from chartqa_dt.plans.executor import (
    MAX_DEPTH,
    NEEDS_TABLE,
    OPS,
    EvidenceItem,
    ExecutorError,
    execute,
    plan_depth,
    to_number,
)

EV = [EvidenceItem("2019", 245, "millions"),
      EvidenceItem("2018", 210, "millions"),
      EvidenceItem("2020", 232, "millions")]


def lookup(label):
    return {"op": "lookup", "args": [label]}


# ------------------------------------------------- the Appendix B correction


def test_bare_string_is_always_an_evidence_label():
    """Appendix B returns 2018.5 here — the mean of the LABELS."""
    assert execute({"op": "mean", "args": ["2019", "2018"]}, EV) == pytest.approx(227.5)
    assert execute({"op": "sum", "args": ["2019", "2018"]}, EV) == pytest.approx(455.0)
    assert execute({"op": "difference", "args": ["2019", "2018"]}, EV) == pytest.approx(35.0)


def test_the_interpretation_is_consistent_across_every_operation():
    """The bug was that it differed between operations, not that either was wrong."""
    assert execute({"op": "argmax", "args": ["2019", "2018"]}, EV) == "2019"
    assert execute({"op": "max", "args": ["2019", "2018"]}, EV) == pytest.approx(245.0)
    assert execute({"op": "compare", "args": ["2019", "2018"]}, EV) == "greater"


def test_numeric_literals_must_be_json_numbers():
    assert execute({"op": "difference", "args": [100, 40]}, EV) == pytest.approx(60.0)
    assert execute({"op": "mean", "args": [10, 20, 30]}, EV) == pytest.approx(20.0)


def test_an_unknown_label_raises_rather_than_computing():
    """The dangerous case is a label that LOOKS numeric; it must still raise."""
    for missing in ("2021", "nope"):
        with pytest.raises(ExecutorError, match="unknown evidence label"):
            execute({"op": "mean", "args": ["2019", missing]}, EV)


# ------------------------------------------------------------ the worked example


def test_idea_worked_example():
    plan = {"op": "difference", "args": [lookup("2019"), lookup("2018")]}
    assert execute(plan, EV) == pytest.approx(35.0)


# ------------------------------------------------------------------- semantics


@pytest.mark.parametrize(
    ("plan", "expected"),
    [
        ({"op": "sum", "args": []}, 687.0),
        ({"op": "mean", "args": []}, 229.0),
        ({"op": "median", "args": []}, 232.0),
        ({"op": "min", "args": []}, 210.0),
        ({"op": "max", "args": []}, 245.0),
        ({"op": "count", "args": []}, 3.0),
        ({"op": "ratio", "args": [lookup("2019"), lookup("2018")]}, 245 / 210),
        ({"op": "percent_change", "args": [lookup("2019"), lookup("2018")]}, 100 * 35 / 210),
        ({"op": "argmin", "args": []}, "2018"),
        ({"op": "trend", "args": [lookup("2018"), lookup("2019")]}, "increasing"),
        ({"op": "unanswerable", "args": []}, None),
    ],
)
def test_operation_semantics(plan, expected):
    got = execute(plan, EV)
    if isinstance(expected, float):
        assert got == pytest.approx(expected)
    else:
        assert got == expected


def test_division_by_zero_raises():
    ev = [EvidenceItem("a", 1.0), EvidenceItem("b", 0.0)]
    for op in ("ratio", "percent_change"):
        with pytest.raises(ExecutorError, match="division by zero"):
            execute({"op": op, "args": ["a", "b"]}, ev)


def test_unit_mismatch_raises():
    ev = [EvidenceItem("a", 1.0, "kg"), EvidenceItem("b", 2.0, "lb")]
    with pytest.raises(ExecutorError, match="unit mismatch"):
        execute({"op": "sum", "args": ["a", "b"]}, ev)


def test_booleans_are_not_numbers():
    """True == 1 in Python, so a boolean in an arithmetic slot would compute silently."""
    with pytest.raises(ExecutorError, match="boolean where number expected"):
        to_number(True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_numbers_raise(bad):
    with pytest.raises(ExecutorError, match="non-finite"):
        to_number(bad)


def test_a_percent_keeps_the_scale_the_chart_is_drawn_in():
    """`'50%'` is 50, not 0.5.

    This assertion was the other way round, and it cost every percentage chart its
    supervision. `mining.to_number` stripped the `%` and this one divided by it, so a plan
    mined against a table value of 5.3 was executed against an evidence value of 0.053 and
    the round-trip failed. The scale that matches the data is the undivided one: **0 of
    32,719 ChartQA gold answers and 0 of 3,996 RefChartQA answers carry a `%` sign**, so
    the divided form could never agree with an answer.

    `eval.metrics.to_float` is a different function on different input -- it parses gold
    ANSWERS and stays byte-faithful to the official evaluator, division included.
    """
    assert to_number("50%") == pytest.approx(50.0)
    assert to_number("1,234") == pytest.approx(1234.0)
    assert to_number(" 42 ") == pytest.approx(42.0)


@pytest.mark.parametrize("text", ["3 071", "1\xa0234", "12\u202f345", "9\u2009876"])
def test_spaces_are_thousands_separators(text):
    """20.7% of ChartQA charts carry a value like `'3 071'`, in one of four space
    characters. The executor used to refuse all of them outright."""
    assert to_number(text) == pytest.approx(float(text.translate(
        {ord(c): None for c in " \xa0\u202f\u2009"})))


def test_stripping_separators_does_not_invent_a_number():
    with pytest.raises(ExecutorError, match="not numeric"):
        to_number("5 apples")


def test_the_miner_and_the_executor_agree_on_every_value():
    """The regression guard. These two parsers disagreed by a factor of 100 on every
    percentage, and nothing in the suite noticed, because each was tested alone."""
    from chartqa_dt.plans.mining import to_number as mining_to_number
    for cell in ["5.3%", "81.9%", "3 071", "1,234", "-4.5", "0", "1\xa0000", "$12"]:
        assert mining_to_number(cell) == pytest.approx(to_number(cell)), cell
    for cell in ["Nigeria", "", "n/a", "5 apples"]:
        assert mining_to_number(cell) is None, cell
        with pytest.raises(ExecutorError):
            to_number(cell)


# ----------------------------------------------------------------------- depth


def test_depth_is_computed_not_trusted():
    node = lookup("2019")
    for expected in (1, 2, 3, 4):
        assert plan_depth(node) == expected
        node = {"op": "sum", "args": [node, lookup("2018")]}
    assert plan_depth(node) == 5
    with pytest.raises(ExecutorError, match="exceeds"):
        execute(node, EV)


def test_depth_four_is_allowed():
    node = lookup("2019")
    for _ in range(3):
        node = {"op": "sum", "args": [node, lookup("2018")]}
    assert plan_depth(node) == MAX_DEPTH
    assert execute(node, EV) == pytest.approx(245 + 210 * 3)


# ------------------------------------------------------------ refusals


def test_unknown_operations_raise():
    for op in ("integrate", "eval", "exec", ""):
        with pytest.raises(ExecutorError, match="unknown op"):
            execute({"op": op, "args": []}, EV)


@pytest.mark.parametrize("op", sorted(NEEDS_TABLE))
def test_table_dependent_operations_refuse_rather_than_guess(op):
    with pytest.raises(ExecutorError, match="requires table context"):
        execute({"op": op, "args": []}, EV)


def test_no_operation_can_reach_code_execution():
    """Safety is by construction: only whitelisted ops exist, and none evaluate code."""
    assert "eval" not in OPS and "exec" not in OPS
    for payload in ("__import__('os').system('echo pwned')", "1+1"):
        with pytest.raises(ExecutorError):
            execute({"op": payload, "args": []}, EV)


def test_empty_evidence_raises_rather_than_returning_zero():
    for op in ("sum", "mean", "median", "min", "max"):
        with pytest.raises(ExecutorError, match="empty set"):
            execute({"op": op, "args": []}, [])
