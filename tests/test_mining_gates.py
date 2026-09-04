"""The five gates, one at a time and in combination.

`plans.llm_mining.verify` is the only thing standing between a language model's guess and the
training set. Its first end-to-end run accepted **0 of 25 correct proposals**, and not one of
the four causes was the model's fault — two of them were bugs in the gates themselves
(`DECISIONS.md` 0082): the evidence cap applied to the wrong set, and an empty marked-label
set read as *"nothing may be used"*.

So each gate is tested in isolation, in order, and for the two ways a gate can be wrong:
letting something through, and refusing something correct.
"""
from __future__ import annotations

import random

import pytest

from chartqa_dt.plans import llm_mining
from chartqa_dt.plans.schema import MAX_EVIDENCE

EV = [{"label": "2019", "value": 245.0}, {"label": "2018", "value": 198.0},
      {"label": "2017", "value": 150.0}]


def verify(plan, answer="245", evidence=None, marked=None):
    return llm_mining.verify(plan, answer=answer,
                             evidence=EV if evidence is None else evidence,
                             marked_labels=marked)


# ================================================================== gate 1 — shape


@pytest.mark.parametrize("plan,status", [
    ({"op": "lookup", "args": ["2019"]}, llm_mining.OK),
    ("not a dict", llm_mining.BAD_SHAPE),
    (None, llm_mining.BAD_SHAPE),
    ({"args": ["2019"]}, llm_mining.BAD_SHAPE),
    ({"op": 7, "args": []}, llm_mining.BAD_SHAPE),
    ({"op": "lookup"}, llm_mining.BAD_SHAPE),
    ({"op": "lookup", "args": "2019"}, llm_mining.BAD_SHAPE),
    ({"op": "invented", "args": []}, llm_mining.BAD_OP),
    ({"op": "lookup", "args": ["a", "b", "c", "d", "e"]}, llm_mining.TOO_MANY_ARGS),
])
def test_shape_gate(plan, status):
    assert verify(plan).status == status


def test_a_deferred_operation_is_rejected_at_shape_not_at_execution():
    """`rank` is in the DSL's vocabulary and not in `EXECUTABLE_OPS`. It should never reach
    the executor from a proposal (`DECISIONS.md` 0109)."""
    got = verify({"op": "rank", "args": []})
    assert got.status in (llm_mining.BAD_OP, llm_mining.RAISES)


def test_depth_is_checked_on_the_whole_tree_not_the_root():
    from chartqa_dt.plans.executor import MAX_DEPTH
    plan = {"op": "lookup", "args": ["2019"]}
    for _ in range(MAX_DEPTH + 1):
        plan = {"op": "difference", "args": [plan, {"op": "lookup", "args": ["2018"]}]}
    assert verify(plan).status == llm_mining.TOO_DEEP


# ========================================================= gate 2 — operands must exist


def test_an_operand_that_is_not_in_the_evidence_is_refused():
    got = verify({"op": "lookup", "args": ["2099"]})
    assert got.status == llm_mining.UNKNOWN_LABEL
    assert "2099" in got.detail, "the detail must name the missing label"


def test_a_nested_plans_operands_are_checked_too():
    plan = {"op": "difference", "args": [{"op": "lookup", "args": ["2099"]}, "2018"]}
    assert verify(plan).status == llm_mining.UNKNOWN_LABEL


def test_a_series_name_is_not_mistaken_for_a_missing_operand():
    """`within`'s first argument names a series, not an element. Counting it as a label
    would reject every `within` plan for an operand that is not in the evidence."""
    ev = [{"label": "Hyperscale · 2019", "value": 1.0},
          {"label": "Hyperscale · 2020", "value": 9.0}]
    got = llm_mining.verify(
        {"op": "within", "args": ["Hyperscale", {"op": "argmax", "args": []}]},
        answer="2020", evidence=ev)
    assert got.status == llm_mining.OK


# ============================================================ gate 3 — it has to execute


def test_a_plan_the_executor_refuses_is_rejected_with_the_reason():
    got = verify({"op": "ratio", "args": ["2019", "zero"]},
                 evidence=[*EV, {"label": "zero", "value": 0.0}])
    assert got.status == llm_mining.RAISES
    assert "ExecutorError" in got.detail


def test_a_value_the_parser_cannot_read_is_a_refusal_not_a_crash():
    got = verify({"op": "lookup", "args": ["odd"]}, answer="1",
                 evidence=[{"label": "odd", "value": "not a number"}])
    assert got.status == llm_mining.RAISES


# ==================================================== gate 4 — it has to reproduce the answer


def test_the_right_operation_on_the_wrong_operand_is_refused():
    assert verify({"op": "lookup", "args": ["2018"]}, answer="245").status == \
        llm_mining.WRONG_ANSWER


def test_the_answers_own_precision_decides_not_a_five_percent_tolerance():
    """5% of the year 2014 is a century (`DECISIONS.md` 0045)."""
    ev = [{"label": "a", "value": 2014.0}, {"label": "b", "value": 2010.0}]
    assert verify({"op": "lookup", "args": ["a"]}, answer="2014", evidence=ev).status == \
        llm_mining.OK
    assert verify({"op": "lookup", "args": ["b"]}, answer="2014", evidence=ev).status == \
        llm_mining.WRONG_ANSWER


def test_a_string_answer_is_compared_case_insensitively_and_trimmed():
    ev = [{"label": "Nigeria", "value": 9.0}, {"label": "Egypt", "value": 1.0}]
    assert verify({"op": "argmax", "args": []}, answer=" nigeria ", evidence=ev).status == \
        llm_mining.OK


# ============================================== gate 5 — the marked regions, where they exist


def test_operands_outside_the_marked_regions_are_refused():
    got = verify({"op": "lookup", "args": ["2018"]}, answer="198", marked={"2019"})
    assert got.status == llm_mining.WRONG_OPERANDS


def test_an_empty_marked_set_means_ungrounded_not_forbidden():
    """This bug rejected all 25 correct proposals in the first end-to-end run: an ungrounded
    ChartQA record passes `set()`, which is not `None`, and every operand was then 'outside
    the marked regions'."""
    assert verify({"op": "lookup", "args": ["2019"]}, marked=set()).status == llm_mining.OK
    assert verify({"op": "lookup", "args": ["2019"]}, marked=None).status == llm_mining.OK


# ================================================= the cap applies to what the plan needs


def test_a_lookup_on_a_large_chart_is_not_rejected_for_the_cap():
    """The cap was applied to the pool of candidates rather than to the evidence a plan
    needs, which rejected `lookup` on any chart with more elements than the cap — 64.4% of
    ChartQA — and reported it as a malformed plan."""
    big = [{"label": f"y{i}", "value": float(i)} for i in range(MAX_EVIDENCE + 20)]
    got = verify({"op": "lookup", "args": ["y3"]}, answer="3", evidence=big)
    assert got.status == llm_mining.OK


def test_a_fold_over_a_chart_larger_than_the_cap_is_refused_with_its_own_status():
    big = [{"label": f"y{i}", "value": float(i)} for i in range(MAX_EVIDENCE + 20)]
    got = verify({"op": "max", "args": []}, answer=str(float(MAX_EVIDENCE + 19)),
                 evidence=big)
    assert got.status == llm_mining.TOO_MUCH_EVIDENCE
    assert "needs" in got.detail


# ============================================================== order, and batch accounting


def test_the_gates_run_in_order_so_the_first_failure_is_the_reported_one():
    """A malformed plan naming an absent label on evidence that cannot execute should report
    the SHAPE failure — the earliest one — or the profile becomes meaningless."""
    assert verify({"op": "invented", "args": ["2099"]}).status == llm_mining.BAD_OP


def test_batch_statistics_account_for_every_proposal():
    proposals = [
        {"plan": {"op": "lookup", "args": ["2019"]}, "answer": "245", "evidence": EV},
        {"plan": {"op": "lookup", "args": ["2018"]}, "answer": "245", "evidence": EV},
        {"plan": "junk", "answer": "245", "evidence": EV},
    ]
    verdicts, stats = llm_mining.verify_many(proposals)
    assert len(verdicts) == len(proposals)
    assert stats.total == len(proposals)
    assert sum(stats.counts.values()) == len(proposals)
    assert stats.accepted == 1


def test_nothing_is_ever_repaired_into_acceptance():
    """A proposal that fails any gate is discarded, never fixed. Repairing would make the
    pipeline the author of its own supervision."""
    rng = random.Random(0)
    for _ in range(200):
        plan = {"op": rng.choice(["lookup", "difference", "max", "invented"]),
                "args": rng.sample(["2019", "2018", "nope"], rng.randint(0, 2))}
        got = verify(plan, answer=str(rng.choice([245, 47, 999])))
        if got.accepted:
            # an accepted verdict must carry back the EXACT plan it was given
            assert got.plan == plan
