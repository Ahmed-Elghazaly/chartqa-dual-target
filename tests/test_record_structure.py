"""`ChartRecord.elements` / `.evidence` — `Prompt.md` Ideas 1 and 2, `DECISIONS.md` 0124.

The representation these replace produced four defects (0067, 0071, 0098, 0116), the last
one three weeks after 0108 named the assumption in writing. The distinction they carry:

* **ELEMENTS** — every semantic object the chart draws.
* **EVIDENCE** — which of those answer *this* question, or `None` for "unknown".

`None` is a claim, not an omission, and it fails safe.
"""

from __future__ import annotations

import pytest

from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord

ELS = [{"label": "A", "value": 1, "unit": None, "bbox": [0, 0, 10, 10]},
       {"label": "B", "value": 2, "unit": None, "bbox": [10, 0, 20, 10]},
       {"label": "C", "value": 3, "unit": None, "bbox": [20, 0, 30, 10]}]


def rec(**kw):
    base = {"record_id": "r", "source": "synthetic", "split": "train", "image_path": "x.png",
                "image_sha256": "d", "question": "q?", "answer": "1", "question_kind": "synthetic"}
    base.update(kw)
    return ChartRecord(**base)


# --- the two fields -------------------------------------------------------------------

def test_a_record_with_no_evidence_says_it_does_not_know():
    r = rec(elements=ELS, evidence=None)
    assert not r.has_question_evidence
    assert r.evidence_elements is None


def test_a_record_with_evidence_selects_the_marked_subset():
    r = rec(elements=ELS, evidence=[0, 2])
    assert r.has_question_evidence
    assert [e["label"] for e in r.evidence_elements] == ["A", "C"]


def test_evidence_may_be_every_element():
    """RefChartQA's case: every marked region is evidence."""
    r = rec(elements=ELS, evidence=[0, 1, 2])
    assert len(r.evidence_elements) == 3


def test_evidence_may_be_empty_without_being_unknown():
    """An empty list is "none of them"; `None` is "we do not know". Different claims."""
    r = rec(elements=ELS, evidence=[])
    assert r.has_question_evidence and r.evidence_elements == []


def test_an_out_of_range_index_is_dropped_rather_than_raising():
    """A malformed cache should cost the record, not the run — the shape of the TypeError
    that could kill a training loop in 0116."""
    assert rec(elements=ELS, evidence=[0, 99]).evidence_elements == [ELS[0]]


def test_evidence_without_elements_is_unknown():
    assert rec(elements=None, evidence=[0]).evidence_elements is None


# --- serialisation and migration --------------------------------------------------------

def test_the_fields_survive_a_round_trip():
    r = rec(elements=ELS, evidence=[1])
    back = ChartRecord.from_dict(r.to_dict())
    assert back.elements == ELS and back.evidence == [1]


def test_a_legacy_record_lifts_its_elements_out_of_meta():
    """Records cached before `elements` was a field carry it under `meta`. Reading them
    as-is would produce records with no elements at all — the shape of defect 0071."""
    d = {"record_id": "r", "source": "chartqa", "split": "train", "image_path": "x",
             "image_sha256": "d", "question": "q", "answer": "1", "question_kind": "human",
             "meta": {ELEMENTS_KEY: ELS}}
    assert ChartRecord.from_dict(d).elements == ELS


def test_a_legacy_record_does_not_gain_evidence_it_never_had():
    """Lifting elements must not invent the claim that they answer the question."""
    d = {"record_id": "r", "source": "chartqa", "split": "train", "image_path": "x",
             "image_sha256": "d", "question": "q", "answer": "1", "question_kind": "human",
             "meta": {ELEMENTS_KEY: ELS}}
    assert ChartRecord.from_dict(d).evidence is None


def test_an_explicit_field_wins_over_the_legacy_meta_key():
    d = {"record_id": "r", "source": "chartqa", "split": "train", "image_path": "x",
             "image_sha256": "d", "question": "q", "answer": "1", "question_kind": "human",
             "elements": ELS[:1], "meta": {ELEMENTS_KEY: ELS}}
    assert ChartRecord.from_dict(d).elements == ELS[:1]


def test_unknown_keys_are_still_ignored():
    d = {"record_id": "r", "source": "chartqa", "split": "train", "image_path": "x",
             "image_sha256": "d", "question": "q", "answer": "1", "question_kind": "human",
             "something_new": 1}
    assert ChartRecord.from_dict(d).record_id == "r"


# --- the contract each source keeps -----------------------------------------------------

def test_chartqa_is_the_source_that_cannot_know():
    """It annotates the chart: the same elements for every question about that image."""
    from chartqa_dt.train.targets import has_question_specific_boxes

    assert not has_question_specific_boxes(rec(elements=ELS, evidence=None))


def test_refchartqa_and_synthetic_are_the_sources_that_can():
    from chartqa_dt.train.targets import has_question_specific_boxes

    assert has_question_specific_boxes(rec(elements=ELS, evidence=[0]))


def test_the_helper_is_now_just_the_field():
    """It used to infer from a meta flag with `refchartqa_id` as a fallback (0119). If it
    ever starts inferring again, a source can forget to declare and be guessed at."""
    import inspect

    from chartqa_dt.train.targets import has_question_specific_boxes

    body = inspect.getsource(has_question_specific_boxes)
    assert "refchartqa_id" not in body.split('"""')[-1]
    assert "record.has_question_evidence" in body


@pytest.mark.parametrize("evidence", [None, [], [0], [0, 1, 2]])
def test_every_evidence_shape_is_representable(evidence):
    r = rec(elements=ELS, evidence=evidence)
    assert ChartRecord.from_dict(r.to_dict()).evidence == evidence
