"""Mining against records that already hold a plan — `DECISIONS.md` 0143.

Ahmed: *"we ll keep it but still use llm on them and compare them and if there r conflicts
we ll see them."* The 22,780 RefChartQA records carrying a derived `lookup` are the only
free validation set the project has: an independent method already committed to an answer,
so a disagreement is a signal about the **teacher** rather than about the chart.

The rule these tests hold: **the comparison reports, it never decides.** A prior plan must
not make a proposal more or less likely to be accepted, or the comparison stops being
evidence and becomes a filter that agrees with itself.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, ".")
from scripts.mine_plans import report_conflicts  # noqa: E402


def proposal(rid, plan, prior):
    return {"record_id": rid, "plan": plan, "prior_plan": prior,
            "answer": "1", "evidence": [], "marked_labels": None}


def accepted(ok=True):
    return SimpleNamespace(accepted=ok, status="ok", detail="")


def test_agreement_is_reported(capsys):
    p = [proposal("r1", {"op": "lookup", "args": ["A"]}, {"op": "lookup", "args": ["A"]})]
    report_conflicts(p, [accepted()])
    out = capsys.readouterr().out
    assert "agree" in out and "1 (100.0%)" in out.replace(",", "")


def test_a_different_operation_is_reported(capsys):
    p = [proposal("r1", {"op": "max", "args": []}, {"op": "lookup", "args": ["A"]})]
    report_conflicts(p, [accepted()])
    assert "different operation" in capsys.readouterr().out


def test_same_operation_different_operands_is_called_out(capsys):
    """The dangerous case: both reach the gold answer from different marks."""
    p = [proposal("r1", {"op": "lookup", "args": ["B"]}, {"op": "lookup", "args": ["A"]})]
    report_conflicts(p, [accepted()])
    out = capsys.readouterr().out
    assert "DIFFERENT operands" in out
    assert "numeric agreement is not semantic agreement" in out


def test_records_with_no_prior_plan_are_not_counted(capsys):
    p = [proposal("r1", {"op": "lookup", "args": ["A"]}, None)]
    report_conflicts(p, [accepted()])
    assert capsys.readouterr().out == ""


def test_rejected_proposals_are_not_compared(capsys):
    """A plan the verifier threw out says nothing about the prior one."""
    p = [proposal("r1", {"op": "max", "args": []}, {"op": "lookup", "args": ["A"]})]
    report_conflicts(p, [accepted(False)])
    assert capsys.readouterr().out == ""


def test_the_comparison_never_changes_what_is_kept():
    """The rule that makes this evidence rather than a self-agreeing filter."""
    import inspect

    src = inspect.getsource(report_conflicts)
    for verb in ("kept.append", "accepted =", "return kept", "verdict.accepted ="):
        assert verb not in src, f"report_conflicts must not decide anything: found {verb!r}"


def test_it_survives_a_malformed_reply():
    """The model can return a string or a refusal; comparison must not crash the run."""
    report_conflicts([proposal("r1", "not a plan", {"op": "lookup", "args": ["A"]})],
                     [accepted()])


def test_mining_can_be_pointed_at_refchartqa():
    """It could not before 0143 — `finished_records` read ChartQA only, so the records
    that carry a prior plan were the ones the teacher never saw."""
    import inspect

    from scripts.mine_plans import finished_records

    sig = inspect.signature(finished_records)
    assert "source" in sig.parameters
    src = inspect.getsource(finished_records)
    assert "refchartqa_records" in src


@pytest.mark.parametrize("source", ["chartqa", "refchartqa", "all"])
def test_every_source_choice_is_accepted(source):
    import inspect

    from scripts.mine_plans import main

    assert source in inspect.getsource(main)
