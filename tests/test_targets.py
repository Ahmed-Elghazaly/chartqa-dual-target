"""Training targets — the join between the data pipeline and the model.

This is the easiest place in the project to introduce a defect nothing catches. A target
that differs from what the parser accepts, by one key order or one float where an integer
belongs, teaches the model to produce output our own evaluator rejects — and the only
symptom is a disappointing score.

So `build_target` refuses to emit anything that does not survive our own pipeline, and the
tests below pin each failure that was found by measuring rather than by reading.
"""

from __future__ import annotations

import json

import pytest

from chartqa_dt.data.records import ChartRecord
from chartqa_dt.plans.roundtrip import check_record
from chartqa_dt.prompting.parsing import parse_record, schema_ok
from chartqa_dt.train.targets import (
    TargetError,
    build_answer_only_target,
    build_record,
    build_target,
    plan_labels,
)


def rec(**kw):
    base = {"record_id": "r1", "source": "chartqa", "split": "train",
            "image_path": "a.png", "image_sha256": "ab" * 32, "question": "q",
            "answer": "35", "question_kind": "human", "boxes": None, "plan": None,
            "table": None, "meta": {}}
    base.update(kw)
    return ChartRecord(**base)


def elements(*items):
    return [{"label": lab, "value": val, "unit": None, "bbox": [10, 20, 30, 40]}
            for lab, val in items]


# ------------------------------------------------------------------ the invariant


def test_a_target_must_round_trip_or_it_is_refused():
    """The invariant. A target whose plan does not reproduce its answer is not emitted."""
    good = rec(plan={"op": "difference", "args": ["2019", "2018"]}, answer="35",
               boxes=[[1, 2, 3, 4], [5, 6, 7, 8]],
               meta={"elements": elements(("2019", 245), ("2018", 210))})
    text = build_target(good)
    parsed = parse_record(text)
    assert parsed.ok and schema_ok(parsed.record)[0]
    assert check_record(parsed.record).ok

    bad = rec(plan={"op": "difference", "args": ["2019", "2018"]}, answer="999",
              boxes=[[1, 2, 3, 4], [5, 6, 7, 8]],
              meta={"elements": elements(("2019", 245), ("2018", 210))})
    with pytest.raises(TargetError, match="does not reproduce"):
        build_target(bad)


def test_evidence_is_selected_by_what_the_plan_needs():
    """Not the first `MAX_EVIDENCE` boxes — that yielded 1 usable target in 636.

    A chart with twelve bars whose plan references the tenth must include the tenth. The
    naive version took the first eight and the executor refused with "lookup of unknown
    evidence label".
    """
    twelve = elements(*[(f"c{i}", float(i)) for i in range(12)])
    r = rec(plan={"op": "difference", "args": ["c10", "c2"]}, answer="8",
            boxes=[e["bbox"] for e in twelve], meta={"elements": twelve})
    record = build_record(r)
    labels = [e["label"] for e in record["evidence"]]
    assert labels == ["c10", "c2"], "only the referenced elements, in plan order"
    assert check_record(record).ok


def test_values_come_from_the_table_because_the_plan_was_verified_against_it():
    """Annotation values and table values differ in rounding; the plan was mined on the
    table, so reading values from the annotation made 35 of 105 planned records disagree
    with their own answer."""
    els = elements(("a", 10.04), ("b", 4.98))       # annotation, rounded differently
    r = rec(plan={"op": "difference", "args": ["a", "b"]}, answer="5",
            boxes=[e["bbox"] for e in els], meta={"elements": els},
            table={"columns": ["x", "y"], "rows": [["a", "10"], ["b", "5"]]})
    record = build_record(r)
    assert [e["value"] for e in record["evidence"]] == [10.0, 5.0]
    assert check_record(record).ok


def test_a_plan_referencing_a_label_with_no_box_is_refused():
    els = elements(("a", 10.0))
    r = rec(plan={"op": "difference", "args": ["a", "missing"]}, answer="5",
            boxes=[els[0]["bbox"]], meta={"elements": els})
    with pytest.raises(TargetError, match="references 'missing'"):
        build_target(r)


# ------------------------------------------------- records without element metadata


def test_one_box_and_a_numeric_answer_yields_an_executable_lookup():
    """RefChartQA: boxes but no values. With a single box the value IS the answer."""
    r = rec(source="refchartqa", answer="3.98", boxes=[[850, 226, 896, 786]])
    record = build_record(r)
    assert record["evidence"][0]["value"] == pytest.approx(3.98)
    assert record["plan"] == {"op": "lookup", "args": ["item1"]}
    assert check_record(record).ok


@pytest.mark.parametrize(("boxes", "answer"), [
    ([[1, 2, 3, 4], [5, 6, 7, 8]], "12"),      # two boxes: values unknowable
    ([[1, 2, 3, 4]], "Yes"),                    # non-numeric: value unknowable
])
def test_a_value_that_cannot_be_recovered_is_refused_not_invented(boxes, answer):
    """`PLAN.md` 3.6's "never given an invented plan", applied to values too.

    The first version filled these with null and a `lookup`, and **100% of 800 sampled
    RefChartQA targets failed the round-trip** — training the model to emit
    non-executable plans on the very metric the project exists to move.
    """
    r = rec(source="refchartqa", answer=answer, boxes=boxes)
    with pytest.raises(TargetError, match="cannot be derived"):
        build_target(r)


def test_a_record_with_no_boxes_or_no_answer_is_refused():
    with pytest.raises(TargetError, match="no evidence"):
        build_target(rec(boxes=None))
    with pytest.raises(TargetError, match="no answer"):
        build_target(rec(answer=None, boxes=[[1, 2, 3, 4]]))


# ------------------------------------------------------------------- the format


def test_the_target_is_compact_and_uses_integer_boxes():
    r = rec(source="refchartqa", answer="3.98", boxes=[[850.4, 226.6, 896.0, 786.2]])
    text = build_target(r)
    assert ", " not in text and "\n" not in text, "compact, as the prompt demonstrates"
    boxes = json.loads(text)["evidence"][0]["bbox"]
    assert boxes == [850, 227, 896, 786]
    assert all(isinstance(v, int) and 0 <= v <= 999 for v in boxes)


def test_out_of_range_boxes_are_clamped_like_the_inference_path():
    r = rec(source="refchartqa", answer="1", boxes=[[-5.0, 0.0, 2000.0, 1500.0]])
    assert json.loads(build_target(r))["evidence"][0]["bbox"] == [0, 0, 999, 999]


def test_plan_labels_walks_nested_plans():
    assert plan_labels({"op": "difference",
                        "args": ["a", {"op": "mean", "args": ["b", "c"]}]}) == \
        ["a", "b", "c"]
    assert plan_labels({"op": "sum", "args": []}) == []
    assert plan_labels(None) == []


def test_the_direct_answer_control_emits_only_the_answer():
    """`PLAN.md` 6.4 — required, not optional. Same records, different target."""
    r = rec(answer="35", boxes=[[1, 2, 3, 4]])
    assert build_answer_only_target(r) == "35"
    with pytest.raises(TargetError):
        build_answer_only_target(rec(answer=None))


class TestValueBoxAgreement:
    """`DECISIONS.md` 0075. An evidence entry takes its value from the gold table and its
    box from the annotation, joined only by a label string. Measured on 1,893 entries, 110
    disagreed — in swapped pairs, so the target boxed one mark and stated another's number.
    """

    def test_matching_values_agree(self) -> None:
        from chartqa_dt.train.targets import values_agree

        assert values_agree(9.9, 9.9)
        assert values_agree("9.9", "9.9")

    def test_rounding_between_the_two_sources_is_tolerated(self) -> None:
        from chartqa_dt.train.targets import values_agree

        assert values_agree(9.90, 9.91)

    def test_the_percent_convention_is_not_a_disagreement(self) -> None:
        """`to_float` divides a "%" cell by 100 because the OFFICIAL METRIC does:
        relaxed_correctness(gold="81.9%", pred="0.819") is True and pred="81.9" is False.
        A 100x relation is therefore required, not an error."""
        from chartqa_dt.eval.metrics import relaxed_correctness
        from chartqa_dt.train.targets import values_agree

        assert values_agree(0.819, 81.9)
        assert relaxed_correctness("81.9%", "0.819") is True
        assert relaxed_correctness("81.9%", "81.9") is False

    def test_a_swapped_pair_is_refused(self) -> None:
        """The real failure: the table says Finland is 9.4 while the annotation's Finland
        bar is 9.9, and Hungary is the other way round."""
        from chartqa_dt.train.targets import values_agree

        assert not values_agree(9.4, 9.9)
        assert not values_agree(12.5, 14.2)

    def test_an_unparseable_value_does_not_trigger_a_refusal(self) -> None:
        """Other guards handle those; this one must not double-refuse."""
        from chartqa_dt.train.targets import values_agree

        assert values_agree(None, 9.9)
        assert values_agree("n/a", 9.9)

    def test_a_record_whose_sources_disagree_is_refused_with_the_reason(self) -> None:
        from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord
        from chartqa_dt.train.targets import TargetError, build_target

        record = ChartRecord(
            record_id="r1", source="chartqa", split="train", image_path="x.png",
            image_sha256="0" * 64, question="What is Finland?", answer="9.4",
            question_kind="human",
            table={"columns": ["c", "v"], "rows": [["Finland", "9.4"]]},
            plan={"op": "lookup", "args": ["Finland"]},
            meta={ELEMENTS_KEY: [{"label": "Finland", "value": 9.9, "unit": None,
                                  "bbox": [10, 10, 20, 90]}]})
        with pytest.raises(TargetError, match="disagree about which mark"):
            build_target(record)

    def test_an_agreeing_record_still_builds(self) -> None:
        import json

        from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord
        from chartqa_dt.train.targets import build_target

        record = ChartRecord(
            record_id="r2", source="chartqa", split="train", image_path="x.png",
            image_sha256="0" * 64, question="What is Finland?", answer="9.9",
            question_kind="human",
            table={"columns": ["c", "v"], "rows": [["Finland", "9.9"]]},
            plan={"op": "lookup", "args": ["Finland"]},
            meta={ELEMENTS_KEY: [{"label": "Finland", "value": 9.9, "unit": None,
                                  "bbox": [10, 10, 20, 90]}]})
        assert json.loads(build_target(record))["model_answer"] == "9.9"
