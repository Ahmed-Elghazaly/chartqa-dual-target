"""Plan mining: recover a typed plan, or refuse.

The uniqueness rule is the only thing deciding whether a mined plan is kept, so
these tests cover both directions — a genuinely determined question must be
mined, and an underdetermined one must be refused. Refusing too little is worse
than refusing too much: an invented plan teaches the model a wrong association it
has no way to unlearn.
"""

from __future__ import annotations

import pytest

from chartqa_dt.plans.executor import EvidenceItem, execute
from chartqa_dt.plans.mining import (
    FLATTENINGS,
    MIN_CANDIDATE_VALUES,
    candidate_sets,
    close,
    enumerate_plan_ops,
    gold_tolerance,
    matches_gold,
    mine_plan,
    to_number,
)

TWO_COL = [["Country", "Value"], ["Haiti", "6.12"], ["Libya", "5.32"], ["Morocco", "5.11"]]
WIDE = [["Entity", "2005", "2006"], ["Myanmar", "59.61", "59.78"], ["Zambia", "27.44", "27.31"]]


# ------------------------------------------------------------- cell parsing


@pytest.mark.parametrize(("cell", "expected"),
                         [("6.12", 6.12), ("1,234", 1234.0), ("45%", 45.0),
                          ("$12", 12.0), (" 7 ", 7.0)])
def test_numeric_cells_parse(cell, expected):
    assert to_number(cell) == pytest.approx(expected)


@pytest.mark.parametrize("cell", ["Haiti", "", "n/a", None, "--"])
def test_label_cells_return_none_rather_than_raising(cell):
    assert to_number(cell) is None


def test_tolerance_matches_the_chartqa_metric():
    """Mined plans must agree with the metric that will score them: 5%."""
    assert close(10.0, 10.4)
    assert not close(10.0, 10.6)
    assert close(0, 0)
    assert not close(0.1, 0)


# --------------------------------------------------- flattening (decision 0030)


def test_singleton_candidate_sets_are_dropped():
    """One value makes lookup, sum, mean, median, min and max identical.

    72% of ChartQA tables are two-column, so `per_row` on them produces exactly
    these degenerate sets — which is why per_row measured 4.2% yield and why
    `union` scored WORSE than all_cells despite being the stricter rule.
    """
    assert MIN_CANDIDATE_VALUES == 2
    assert candidate_sets(TWO_COL, "per_row") == []
    for values in candidate_sets(WIDE, "per_row"):
        assert len(values) >= MIN_CANDIDATE_VALUES


def test_union_deduplicates_identical_sets():
    """On a two-column table per_column and all_cells produce the same set."""
    assert len(candidate_sets(TWO_COL, "union")) == 1
    assert len(candidate_sets(WIDE, "union")) == 5


@pytest.mark.parametrize("mode", FLATTENINGS)
def test_every_flattening_is_implemented(mode):
    assert isinstance(candidate_sets(WIDE, mode), list)


def test_unknown_flattening_raises():
    with pytest.raises(ValueError, match="unknown flattening"):
        candidate_sets(WIDE, "sideways")  # type: ignore[arg-type]


# ----------------------------------------------------------- the uniqueness rule


def test_a_determined_question_is_mined_with_a_concrete_tree():
    m = mine_plan(TWO_COL, 16.55)          # 6.12 + 5.32 + 5.11
    assert m.status == "unique" and m.op == "sum"
    assert m.plan == {"op": "sum", "args": ["Haiti", "Libya", "Morocco"]}
    assert len(m.evidence) == 3


def test_a_mined_plan_reproduces_its_own_answer():
    """A tree that does not execute to the recorded answer is not a plan."""
    m = mine_plan(TWO_COL, 16.55)
    got = execute(m.plan, [EvidenceItem(e["label"], e["value"]) for e in m.evidence])
    assert close(got, 16.55)


def test_the_worked_ambiguous_example_from_idea_is_refused():
    """IDEA.md 4: 2018=10, 2019=20, answer 10 — several plans give 10."""
    m = mine_plan([["x", "v"], ["2018", "10"], ["2019", "20"]], 10)
    assert m.status == "ambiguous"
    assert len(m.ops_matched) > 1
    assert m.plan is None, "an ambiguous question must never receive an invented plan"


def test_the_scoring_tolerance_does_not_manufacture_ambiguity():
    """Mining matches at the gold answer's own precision, not ChartQA's 5%.

    5.11 is within 5% of 5.32, so the scoring tolerance would call `min` a match and
    reject the question as ambiguous. The gold answer "5.32" was written to two
    decimals, so `min` is not a match and only genuinely indistinguishable operations
    remain (DECISIONS.md 0045).
    """
    m = mine_plan(TWO_COL, 5.32)
    assert "min" not in set(m.ops_matched), \
        "a value 4% away is not the answer, it is a different value"


def test_gold_tolerance_follows_the_answers_written_precision():
    assert gold_tolerance("48.6") == pytest.approx(0.05)
    assert gold_tolerance("2014") == pytest.approx(0.5)
    assert gold_tolerance("0.405") == pytest.approx(0.0005)
    assert gold_tolerance("1,234") == pytest.approx(0.5)
    assert gold_tolerance("15%") == pytest.approx(0.5)


def test_matches_gold_accepts_rounding_but_not_coincidence():
    assert matches_gold(48.62, "48.6"), "the table value rounds to the printed answer"
    assert not matches_gold(47.70, "48.6"), "0.9 away is a different quantity"
    assert matches_gold(2014.4, "2014")
    assert not matches_gold(2066.0, "2014"), \
        "within ChartQA's 5% of a year, but 52 years is not a rounding error"


def test_a_category_answer_is_never_explained_by_arithmetic():
    """"Which year has the most crime?" is answered by a label, not a computation.

    Year answers parse as numbers, so without this guard a `difference` landing near
    the year looks like a unique match — measured at 8.5% of mined plans.
    """
    rows = [["Year", "Value"], ["2012", "1000"], ["2013", "1200"], ["2014", "2014"]]
    m = mine_plan(rows, "2014")
    assert m.status == "category_answer"
    assert m.plan is None


def test_a_quantity_answer_that_is_not_a_label_still_mines():
    rows = [["Entity", "Value"], ["a", "10"], ["b", "20"], ["c", "30"]]
    m = mine_plan(rows, 60)
    assert m.status == "unique" and m.plan["op"] == "sum"


def test_a_non_numeric_answer_is_classified_not_mined():
    m = mine_plan(TWO_COL, "Blue")
    assert m.status == "non_numeric" and m.plan is None


def test_an_unreachable_answer_yields_no_plan():
    m = mine_plan(TWO_COL, 999999)
    assert m.status == "none" and m.plan is None


def test_an_empty_table_yields_no_plan():
    assert mine_plan([["header"]], 5).status == "none"


def test_a_corrupt_all_zero_table_is_refused():
    """IDEA.md 5.3 claims the uniqueness filter absorbs table corruption.

    An all-zero table matches many operations at once, so it is rejected — the
    protection is real, and this pins it.
    """
    zeros = [["k", "v"], ["a", "0"], ["b", "0"], ["c", "0"]]
    m = mine_plan(zeros, 0)
    assert m.status == "ambiguous"
    assert len(m.ops_matched) >= 5


# ------------------------------------------------------ operation enumeration


def test_enumerate_finds_every_matching_operation_type():
    values = [("a", 10.0), ("b", 20.0)]
    assert enumerate_plan_ops(values, 30.0) >= {"sum", "sum2"}
    assert "difference" in enumerate_plan_ops(values, 10.0)
    assert "ratio" in enumerate_plan_ops(values, 2.0)
    assert "percent_change" in enumerate_plan_ops(values, 100.0)
    assert "count" in enumerate_plan_ops(values, 2.0)


def test_enumerate_on_an_empty_set_returns_nothing():
    assert enumerate_plan_ops([], 5.0) == set()


def test_duplicate_labels_block_an_aggregate_plan():
    """An aggregate over duplicate labels would produce ambiguous lookups."""
    dup = [["k", "v"], ["a", "1"], ["a", "2"], ["b", "97"]]
    m = mine_plan(dup, 100.0)
    assert m.plan is None or len({e["label"] for e in m.evidence}) == len(m.evidence)
