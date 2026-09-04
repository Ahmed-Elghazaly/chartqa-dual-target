"""Telling apart plans that agree on this chart but mean different things.

`DECISIONS.md` 0080 found the blind spot: with one marked element, `argmax`, `argmin` and the
folds all return the same thing, so no arithmetic gate can separate them. 0097 closes it with
the technique weakly supervised semantic parsing uses for spurious programs — execute under
perturbed inputs and compare behaviour.
"""
from __future__ import annotations

from chartqa_dt.plans.distinguish import (
    fingerprint,
    indistinguishable_from,
    rivals_for,
)
from chartqa_dt.plans.executor import EvidenceItem

ONE = [EvidenceItem("Afghanistan", 41.17)]
THREE = [EvidenceItem("Nigeria", 154.3), EvidenceItem("Egypt", 54.7),
         EvidenceItem("Kenya", 46.9)]
EQUAL = [EvidenceItem("A", 5.0), EvidenceItem("B", 5.0), EvidenceItem("C", 5.0)]


def ops(rivals):
    return {r["op"] for r in rivals}


# ------------------------------------------------- the case 0080 could not close


def test_one_element_cannot_separate_argmax_from_argmin():
    """Both return the only label, on every input, so the chart says nothing about which
    the question meant."""
    assert ops(indistinguishable_from({"op": "argmax", "args": []}, ONE)) == {"argmin"}


def test_one_element_cannot_separate_the_value_folds():
    assert ops(indistinguishable_from({"op": "max", "args": []}, ONE)) == {
        "min", "mean", "sum", "median"}


# ------------------------------------------------- and where the chart DOES decide


def test_three_distinct_values_separate_everything():
    """Shuffling breaks the label-to-value association, which is exactly what tells
    `lookup` from an extremum."""
    assert indistinguishable_from({"op": "max", "args": []}, THREE) == []
    assert indistinguishable_from({"op": "lookup", "args": ["Nigeria"]}, THREE) == []


def test_it_measures_semantics_and_not_a_pattern():
    """With three EQUAL values, max/min/mean/median all return 5 and are genuinely
    inseparable -- but `sum` returns 15 and must not be swept in with them."""
    rivals = ops(indistinguishable_from({"op": "max", "args": []}, EQUAL))
    assert rivals == {"min", "mean", "median"}
    assert "sum" not in rivals


# ------------------------------------------------- properties of the fingerprint


def test_a_fingerprint_is_deterministic():
    """Same plan, same evidence, same seed -- or the comparison means nothing."""
    plan = {"op": "argmax", "args": []}
    assert fingerprint(plan, THREE) == fingerprint(plan, THREE)


def test_refusing_to_run_is_recorded_as_behaviour():
    """Two plans that raise in different places are not the same plan, so a refusal is part
    of the fingerprint rather than a reason to drop the trial."""
    fp = fingerprint({"op": "lookup", "args": ["absent"]}, THREE)
    assert all(kind == "raises" for kind, _ in fp)


def test_a_plan_that_never_runs_flags_nothing():
    """It has no discriminating behaviour, so calling everything indistinguishable from it
    would be noise rather than a finding."""
    assert indistinguishable_from({"op": "lookup", "args": ["absent"]}, THREE) == []


def test_empty_evidence_flags_nothing():
    assert indistinguishable_from({"op": "max", "args": []}, []) == []


def test_rivals_are_built_from_the_operands_the_plan_names():
    """An alternative must be executable wherever the proposal is, or the comparison is
    between a plan and a crash."""
    rivals = rivals_for({"op": "lookup", "args": ["Nigeria"]}, THREE)
    assert {"op": "lookup", "args": ["Nigeria"]} not in rivals, "not a rival to itself"
    assert all(r["op"] != "lookup" or r["args"][0] == "Nigeria" for r in rivals)


# ------------------------------------------------- how the verifier reports it


def test_the_verifier_records_it_without_rejecting():
    """Whether to refuse an underdetermined plan is a trade to price on real proposals, not
    to assume -- so it is reported on the verdict and the plan is still accepted."""
    from chartqa_dt.plans import llm_mining
    v = llm_mining.verify({"op": "argmax", "args": []}, answer="Afghanistan",
                          evidence=[{"label": "Afghanistan", "value": 41.17}])
    assert v.status == llm_mining.OK
    assert ops(v.underdetermined) == {"argmin"}


def test_a_well_determined_plan_carries_no_flag():
    from chartqa_dt.plans import llm_mining
    v = llm_mining.verify({"op": "argmax", "args": []}, answer="Nigeria",
                          evidence=[{"label": e.label, "value": e.value} for e in THREE])
    assert v.status == llm_mining.OK
    assert v.underdetermined == []
