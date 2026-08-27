"""The output schema and the validation it delegates.

Phase 0 established that Appendix A's schema is sound and deliberately delegates
five checks. This tests both halves, and both measured hazards behind them:
a coordinate of exactly 1000 (silently discarded by the official evaluator) and
a long evidence list (dataset AP collapses from 1.0000 to 0.3243).
"""

from __future__ import annotations

import copy

import pytest

from chartqa_dt.plans.schema import (
    MAX_EVIDENCE,
    OUTPUT_SCHEMA,
    validate_beyond_schema,
    validate_record,
    validate_schema,
)

GOOD = {
    "answerable": True,
    "evidence": [
        {"label": "2019", "value": 245, "unit": "millions", "bbox": [412, 180, 486, 742]},
        {"label": "2018", "value": 210, "unit": "millions", "bbox": [318, 265, 392, 742]},
    ],
    "focus_bbox": [300, 160, 500, 760],
    "plan": {"op": "difference",
             "args": [{"op": "lookup", "args": ["2019"]}, {"op": "lookup", "args": ["2018"]}]},
    "model_answer": "35",
}


def mutate(**changes):
    r = copy.deepcopy(GOOD)
    r.update(changes)
    return r


def test_the_schema_is_valid_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(OUTPUT_SCHEMA)


def test_the_worked_example_from_idea_validates():
    result = validate_record(GOOD)
    assert result.ok, result.errors


# --------------------------------------------------- what the schema rejects


@pytest.mark.parametrize(
    "record",
    [
        mutate(model_answer=None),
        mutate(confidence=0.9),                                   # additionalProperties
        mutate(plan={"op": "integrate", "args": []}),             # unknown op
        mutate(evidence=[{"label": "a", "bbox": [0, 0, 1001, 10]}]),   # coord > 1000
        mutate(evidence=[{"label": str(i), "bbox": [0, 0, 10, 10]} for i in range(MAX_EVIDENCE + 1)]),
    ],
)
def test_schema_rejects(record):
    assert validate_schema(record), "the schema should have rejected this"


def test_a_missing_required_field_is_rejected():
    r = copy.deepcopy(GOOD)
    del r["model_answer"]
    assert validate_schema(r)


# ------------------------------------- what the schema delegates (the five rules)


def test_inverted_box_is_caught_beyond_the_schema():
    r = mutate(evidence=[{"label": "a", "bbox": [900, 100, 100, 900]}])
    assert not validate_schema(r), "precondition: the schema itself accepts this"
    assert any("x1" in e for e in validate_record(r).errors)


def test_zero_area_box_is_caught():
    r = mutate(evidence=[{"label": "a", "bbox": [100, 100, 100, 100]}])
    errors = validate_record(r).errors
    assert any("must be less than" in e for e in errors)


def test_plan_deeper_than_four_is_caught():
    node = {"op": "lookup", "args": ["2019"]}
    for _ in range(5):
        node = {"op": "sum", "args": [node]}
    assert any("depth" in e for e in validate_record(mutate(plan=node)).errors)


def test_lookup_of_a_label_not_in_evidence_is_caught():
    r = mutate(plan={"op": "lookup", "args": ["2020"]})
    assert any("not in evidence" in e for e in validate_record(r).errors)


def test_duplicate_evidence_labels_are_caught():
    """Two items with one label make every lookup of it ambiguous."""
    r = mutate(evidence=[{"label": "x", "bbox": [0, 0, 10, 10]},
                         {"label": "x", "bbox": [20, 20, 30, 30]}],
               plan={"op": "lookup", "args": ["x"]})
    assert any("duplicate label" in e for e in validate_record(r).errors)


# ------------------------------------------------- the two measured hazards


def test_coordinate_of_exactly_1000_warns():
    """The schema allows it; the official evaluator silently discards the box."""
    r = mutate(evidence=[{"label": "a", "bbox": [0, 0, 1000, 1000]}],
               plan={"op": "lookup", "args": ["a"]})
    assert validate_schema(r) == [], "the schema permits 1000; that is the hazard"
    warnings = validate_record(r).warnings
    assert any("discards the whole box" in w for w in warnings)


def test_a_long_evidence_list_warns_without_failing():
    """maxItems 8 is a hazard, not an allowance: dataset AP 1.0000 -> 0.3243."""
    ev = [{"label": f"l{i}", "bbox": [i, i, i + 10, i + 10]} for i in range(6)]
    r = mutate(evidence=ev, plan={"op": "lookup", "args": ["l0"]})
    result = validate_record(r)
    assert result.ok, "six items is legal, so it must not be an error"
    assert any("false positive" in w for w in result.warnings)


def test_three_or_fewer_evidence_items_do_not_warn():
    _, warnings = validate_beyond_schema(GOOD)
    assert not any("false positive" in w for w in warnings)


def test_validation_result_is_falsy_when_invalid():
    assert validate_record(GOOD)
    assert not validate_record(mutate(plan={"op": "lookup", "args": ["absent"]}))
