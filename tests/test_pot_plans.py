"""RefChartQA's gold derivations, converted to our DSL — `DECISIONS.md` 0133.

35,304 records carry a program of thought and nothing read it, while a paid mining run was
queued to recover the same information. These tests hold two lines:

* the conversion is **deterministic and closed** — an unrecognised shape refuses rather
  than guesses, because a guessed shape can still execute to the right number, which is
  precisely the spurious program the gates cannot catch;
* the converted plan is a **candidate**, never a fact. It reaches a record only after
  executing against that record's own evidence and reproducing the gold answer.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from chartqa_dt.plans.pot import SHAPES, classify, quoted, steps_of, to_plan

RATIO = ("<comment># Get the value of 'N' in 'China', set to Y_1</comment>"
         "<step>Y_1=7562</step>"
         "<comment># Get the value of 'N' in 'Romania', set to Y_2</comment>"
         "<step>Y_2=9891</step>"
         "<comment># Divide Y_1 by Y_2, set to Answer</comment>"
         "<step>Answer=np.divide(Y_1, Y_2)</step>")

ARGMIN = ("<comment># Get the names of all 'C', set to X</comment><step>X=['a','b']</step>"
          "<comment># Get all the values of 'V', set to Y</comment><step>Y=[1,2]</step>"
          "<comment># Get the index that minimize Y, set to MinIndex</comment>"
          "<step>MinIndex=np.argmin(Y)</step>"
          "<comment># Get the value of 'C' at MinIndex, set to Answer</comment>"
          "<step>Answer=X[MinIndex]</step>")


# --- parsing --------------------------------------------------------------------------

def test_a_derivation_parses_into_comment_step_pairs():
    assert len(steps_of(RATIO)) == 3


def test_an_empty_or_absent_response_yields_nothing():
    assert steps_of("") == [] and steps_of(None) == []
    assert to_plan("") is None and to_plan(None) is None


def test_quoted_pulls_labels_in_order():
    assert quoted("the value of 'N' in 'China'") == ["N", "China"]


def test_quoted_handles_double_quotes():
    assert quoted('the value of "N" in "Chad"') == ["N", "Chad"]


# --- classification -------------------------------------------------------------------

@pytest.mark.parametrize("comment,expected", [
    ("Get the index that minimize Y, set to MinIndex", "argmin"),
    ("Get the index that maximize Y, set to MaxIndex", "argmax"),
    ("Get the value of 'N' in 'China', set to Y_1", "lookup"),
    ("Get all the values of 'V', set to Y", "fold_values"),
    ("Get the names of all 'C', set to X", "fold_labels"),
    ("Divide Y_1 by Y_2, set to Answer", "ratio"),
    ("Subtract Y_2 from Y_1, set to Answer", "difference"),
    ("Get the sum of Y, set to Answer", "sum"),
    ("Get the average of Y, set to Answer", "mean"),
    ("Get the maximum value of Y, set to Answer", "max"),
    ("Get the minimum value of Y, set to Answer", "min"),
    ("Do something nobody anticipated", "?"),
])
def test_a_step_is_classified_from_its_comment(comment, expected):
    assert classify(comment) == expected


def test_intent_is_read_from_the_comment_not_the_code():
    """The comment states intent (*"the index that minimize"*); the step states an
    implementation (`np.argmin`). A plan records intent."""
    assert classify("Get the index that minimize Y") == "argmin"


# --- conversion -----------------------------------------------------------------------

def test_a_ratio_derivation_becomes_a_ratio_plan():
    assert to_plan(RATIO) == {"op": "ratio", "args": ["China", "Romania"]}


def test_an_argmin_derivation_becomes_an_argmin_plan():
    assert to_plan(ARGMIN) == {"op": "argmin", "args": []}


@pytest.mark.parametrize("shape,op", list(SHAPES.items()))
def test_every_declared_shape_maps_to_a_plan(shape, op):
    assert op["op"] and isinstance(op["args"], list)


def test_an_unrecognised_shape_refuses_rather_than_guesses():
    """A guessed shape can still execute to the right number. That is the spurious program
    the gates cannot catch, so it must never be produced."""
    weird = ("<comment># Rotate the chart widdershins</comment><step>X=1</step>"
             "<comment># Consult the oracle</comment><step>Answer=42</step>")
    assert to_plan(weird) is None


def test_a_binary_shape_without_labels_refuses():
    no_labels = ("<comment># Get the value of something, set to Y_1</comment>"
                 "<step>Y_1=1</step>"
                 "<comment># Get the value of another, set to Y_2</comment>"
                 "<step>Y_2=2</step>"
                 "<comment># Divide Y_1 by Y_2, set to Answer</comment>"
                 "<step>Answer=1</step>")
    assert to_plan(no_labels) is None


def test_conversion_is_deterministic():
    assert to_plan(RATIO) == to_plan(RATIO)


def test_no_shape_maps_to_an_operation_the_executor_lacks():
    from chartqa_dt.plans.executor import OPS
    from chartqa_dt.plans.pot import BINARY_SHAPES

    for plan in SHAPES.values():
        assert plan["op"] in OPS
    for op in BINARY_SHAPES.values():
        assert op in OPS


# --- against the real cache -------------------------------------------------------------

CACHE = pathlib.Path.home() / ".cache/chartqa_dt/data/refchartqa_train.jsonl"


@pytest.mark.skipif(not CACHE.exists(), reason="RefChartQA cache not present")
def test_the_derivations_really_are_there_and_parse():
    """35,304 of 55,486, all parsing. If this drops, the source format changed."""
    have = parsed = 0
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        response = (json.loads(line).get("meta") or {}).get("response")
        if response:
            have += 1
            parsed += bool(steps_of(response))
    assert have > 30_000, f"only {have:,} derivations; expected ~35,000"
    assert parsed == have, f"{have - parsed:,} derivations no longer parse"


@pytest.mark.skipif(not CACHE.exists(), reason="RefChartQA cache not present")
def test_a_converted_plan_only_reaches_a_record_after_the_executor_agrees():
    """The gate that makes this safe: no plan is attached on the derivation's word."""
    import sys

    sys.path.insert(0, ".")
    from scripts.build_mixtures import refchartqa_records

    records = refchartqa_records(cap=3000, cache=CACHE)
    from chartqa_dt.eval.metrics import relaxed_correctness
    from chartqa_dt.plans.executor import EvidenceItem, ExecutorError, execute, parse_numeric

    checked = 0
    for record in records:
        if (record.meta or {}).get("plan_provenance") != "refchartqa_pot":
            continue
        evidence = [EvidenceItem(label=str(e["label"]), value=parse_numeric(e.get("value")),
                                 unit=e.get("unit"))
                    for e in (record.elements or []) if e.get("label") is not None]
        try:
            got = execute(record.plan, evidence)
        except ExecutorError:  # pragma: no cover - would be a bug in the gate
            pytest.fail(f"{record.record_id}: attached a plan that does not execute")
        if isinstance(got, float):
            rendered = (str(int(got)) if got == int(got)
                        else f"{got:.6f}".rstrip("0").rstrip("."))
        elif isinstance(got, bool):
            rendered = "Yes" if got else "No"
        else:
            rendered = str(got)
        assert relaxed_correctness(rendered, str(record.answer)), (
            f"{record.record_id}: an attached plan executes to {rendered!r} but the gold "
            f"answer is {record.answer!r} — the gate let through a plan it should not have")
        checked += 1
    assert checked > 10, f"only {checked} PoT-derived plans in 3,000 records"


@pytest.mark.skipif(not CACHE.exists(), reason="RefChartQA cache not present")
def test_no_attached_fold_is_over_a_single_element():
    """`max([x])` is `x`, so such a plan verifies by construction and teaches the model to
    emit one box and trivially fold it — the circular supervision `align_refchartqa.py`
    warns about. 97.6% of the first conversion's folds were exactly this (0133)."""
    import sys

    sys.path.insert(0, ".")
    from scripts.build_mixtures import refchartqa_records

    from chartqa_dt.plans.executor import FOLD_OPS

    records = refchartqa_records(cap=6000, cache=CACHE)
    offenders = []
    for record in records:
        if (record.meta or {}).get("plan_provenance") != "refchartqa_pot":
            continue
        plan = record.plan or {}
        if plan.get("op") in FOLD_OPS and not plan.get("args"):
            n = len([e for e in (record.elements or []) if e.get("label") is not None])
            if n < 2:
                offenders.append((record.record_id, plan["op"], n))
    assert not offenders, (
        f"{len(offenders)} folds over fewer than two elements, e.g. {offenders[:3]}")
