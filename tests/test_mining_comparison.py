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
from scripts.mine_plans import report_conflicts


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


# --- mining runs on the merged set, not on each pool (`DECISIONS.md` 0145) --------------

def _rec(question, *, cells=1, elements=1, source="chartqa", image="a"):
    from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord

    return ChartRecord(
        record_id=f"{source}:{image}:{question}", source=source, split="train",
        image_path="i.png", image_sha256=image * 64, question=question, answer="1",
        question_kind="human", table={"rows": [list(range(cells))]},
        meta={ELEMENTS_KEY: [{}] * elements})


def test_the_same_question_on_the_same_image_is_mined_once():
    """17,920 records share an (image, question) key — 78.8% of all ChartQA. Mining both
    pools separately pays for those twice and can return two different plans for one
    record."""
    from scripts.mine_plans import deduplicate_for_mining

    out = deduplicate_for_mining([_rec("q?", source="chartqa"),
                                  _rec("q?", source="refchartqa")])
    assert len(out) == 1


def test_the_richer_record_is_the_one_kept():
    """The prompt shows the question, the chart's data and the gold answer, so the copy
    with more table and more elements is the better prompt — regardless of source."""
    from scripts.mine_plans import deduplicate_for_mining

    thin = _rec("q?", cells=1, elements=1, source="chartqa")
    rich = _rec("q?", cells=9, elements=5, source="refchartqa")
    assert deduplicate_for_mining([thin, rich])[0].source == "refchartqa"
    assert deduplicate_for_mining([rich, thin])[0].source == "refchartqa"


def test_different_questions_on_one_image_are_both_kept():
    from scripts.mine_plans import deduplicate_for_mining

    assert len(deduplicate_for_mining([_rec("a?"), _rec("b?")])) == 2


def test_the_same_question_on_different_images_is_kept_twice():
    from scripts.mine_plans import deduplicate_for_mining

    assert len(deduplicate_for_mining([_rec("q?", image="a"),
                                       _rec("q?", image="b")])) == 2


def test_mining_uses_the_same_duplicate_key_as_training():
    """If mining and `data/dedup.py` disagreed about what a duplicate is, a record could be
    mined once and trained twice, or the reverse."""
    import inspect

    from scripts.mine_plans import deduplicate_for_mining

    assert "record.key" in inspect.getsource(deduplicate_for_mining)


def test_deduplication_is_stable_and_order_independent():
    from scripts.mine_plans import deduplicate_for_mining

    a, b, c = _rec("a?"), _rec("b?", cells=4), _rec("a?", cells=7)
    first = {r.record_id for r in deduplicate_for_mining([a, b, c])}
    second = {r.record_id for r in deduplicate_for_mining([c, b, a])}
    assert first == second
