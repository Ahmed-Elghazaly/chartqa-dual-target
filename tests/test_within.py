"""`within` — applying an operation to one series instead of the whole chart.

*"Which year has the highest number in hyperscale?"* is an argmax over ONE series. Reading 40
human-written ChartQA questions by hand, this was the single most-requested missing operation
(6 of 40), and a corpus count puts a series-restricted fold at **8.6% of human questions
against 0.1% of machine ones** — human questions being half the test split and half the
headline metric (`DECISIONS.md` 0090).
"""
from __future__ import annotations

import pytest

from chartqa_dt.plans.executor import (
    OPS,
    EvidenceItem,
    ExecutorError,
    execute,
    folds_over_evidence,
    plan_labels,
)

#: A grouped chart: two series, each drawn over the same three years.
GROUPED = [
    EvidenceItem("Hyperscale · 2019", 69.7),
    EvidenceItem("Hyperscale · 2020", 76.2),
    EvidenceItem("Hyperscale · 2021", 86.6),
    EvidenceItem("Traditional · 2019", 50.4),
    EvidenceItem("Traditional · 2020", 41.0),
    EvidenceItem("Traditional · 2021", 32.6),
]


def within(series, node):
    return {"op": "within", "args": [series, node]}


def test_it_folds_over_one_series_not_the_chart():
    """`argmax()` alone returns 'Hyperscale · 2021' (86.6, the chart's maximum). Asked about
    Traditional, the answer is a different year entirely."""
    assert execute(within("Traditional", {"op": "argmax", "args": []}), GROUPED) == "2019"
    assert execute(within("Hyperscale", {"op": "argmax", "args": []}), GROUPED) == "2021"


def test_the_series_prefix_is_stripped_from_what_comes_back():
    """The gold answer says '2021', not 'Hyperscale · 2021'. Inside one series the
    identifying part of a name is the bare label, so a plan that returned the qualified form
    would fail its own round-trip on every one of these questions."""
    got = execute(within("Hyperscale", {"op": "argmin", "args": []}), GROUPED)
    assert got == "2019"
    assert "·" not in got


def test_a_nested_operation_can_name_bare_labels():
    """Inside `within`, labels are the bare ones, so the nested plan uses those."""
    plan = within("Traditional", {"op": "difference", "args": ["2019", "2021"]})
    assert execute(plan, GROUPED) == pytest.approx(50.4 - 32.6)


def test_any_fold_works_inside_it():
    assert execute(within("Hyperscale", {"op": "max", "args": []}), GROUPED) == pytest.approx(86.6)
    assert execute(within("Traditional", {"op": "count", "args": []}), GROUPED) == 3
    assert execute(within("Hyperscale", {"op": "trend", "args": []}), GROUPED) == "increasing"
    assert execute(within("Traditional", {"op": "trend", "args": []}), GROUPED) == "decreasing"


# ------------------------------------------------- refusing rather than guessing


def test_an_unknown_series_raises():
    with pytest.raises(ExecutorError, match="no evidence belongs to the series"):
        execute(within("Cloud", {"op": "max", "args": []}), GROUPED)


@pytest.mark.parametrize("args", [
    ["Hyperscale"],                                    # no operation
    ["Hyperscale", {"op": "max", "args": []}, "extra"],
    [{"op": "max", "args": []}, "Hyperscale"],         # the wrong way round
    ["Hyperscale", "2021"],                            # second argument not an operation
])
def test_a_malformed_within_raises(args):
    with pytest.raises(ExecutorError, match="within takes a series name"):
        execute({"op": "within", "args": args}, GROUPED)


def test_it_refuses_a_chart_whose_labels_are_not_qualified():
    """An unqualified chart has one series, so `within` is meaningless there rather than a
    silent no-op that folds over everything."""
    flat = [EvidenceItem("2019", 1.0), EvidenceItem("2020", 2.0)]
    with pytest.raises(ExecutorError, match="no evidence belongs to the series"):
        execute(within("Hyperscale", {"op": "max", "args": []}), flat)


# ------------------------------------------------- how the rest of the system sees it


def test_the_series_name_is_not_treated_as_an_evidence_label():
    """`within`'s first argument names a series. Counting it as a label would have every
    `within` plan rejected for an operand that is not in the evidence -- true, and beside
    the point."""
    plan = within("Traditional", {"op": "difference", "args": ["2019", "2021"]})
    assert plan_labels(plan) == ["2019", "2021"]
    assert "Traditional" not in plan_labels(plan)


def test_it_counts_as_folding_over_the_evidence():
    """It has to filter the whole list, so the target must carry the whole list -- the same
    rule that stops a bare `argmax()` being handed a truncated chart."""
    assert folds_over_evidence(within("Hyperscale", {"op": "max", "args": []}))


def test_it_is_a_real_operation_the_schema_knows():
    assert "within" in OPS
    from chartqa_dt.plans.schema import OUTPUT_SCHEMA
    assert "within" in OUTPUT_SCHEMA["$defs"]["node"]["properties"]["op"]["enum"]


def test_the_prompt_offers_it_and_describes_it():
    from chartqa_dt.plans.teacher import OFFERED, SIGNATURES, build_system
    assert "within" in OFFERED
    assert "within" in SIGNATURES
    assert "within(series, op)" in build_system()
