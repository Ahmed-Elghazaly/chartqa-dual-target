"""Mixture construction, and the leakage assertion `PLAN.md` 3.7 requires in code.

The leakage test is the one that matters. Rule 1 says validation and test data are never
looked at, and a mixture that silently *filtered out* a leaked record would satisfy the
letter of that while hiding the fact that something upstream produced one.
"""

from __future__ import annotations

import json

import pytest

from chartqa_dt.data.mixture import (
    LEVEL_ORDER,
    STAGE1_CAP,
    LeakageError,
    assert_train_only,
    build_stage1,
    build_stage2,
    is_compositional,
    write_mixture,
)
from chartqa_dt.data.records import ChartRecord


def _rec(source="synthetic", split="train", q="q", sha=None, level=None, boxes=None,
         plan=None, kind="synthetic", i=0):
    sha = sha or f"{i:064d}"
    return ChartRecord(
        record_id=f"{source}-{split}-{i}", source=source, split=split,
        image_path=f"{i}.png", image_sha256=sha, question=f"{q}-{i}", answer="1",
        question_kind=kind, boxes=boxes, plan=plan,
        meta={"level": level} if level else {})


# ------------------------------------------------------------------ leakage (3.7)


def test_zero_validation_or_test_records_appear_in_either_mixture():
    """The assertion `PLAN.md` 3.7 requires in code."""
    train = [_rec(i=i, boxes=[[1, 2, 3, 4]], level="L1") for i in range(5)]
    leaked = _rec(source="chartqa", split="test", i=99, boxes=[[1, 2, 3, 4]])
    with pytest.raises(LeakageError, match="not from the training split"):
        build_stage1(train, [leaked])
    with pytest.raises(LeakageError, match="not from the training split"):
        build_stage2([*train, leaked], [])


def test_a_leaked_record_cannot_hide_by_being_filtered_first():
    """The leak check runs on the inputs, not the survivors.

    A test-split record with no curriculum level would be dropped by stage 1's level
    grouping. If the check ran afterwards it would see a clean mixture and report
    nothing, which is the worst possible outcome: a real leak, silently absorbed.
    """
    train = [_rec(i=i, boxes=[[1, 2, 3, 4]], level="L1") for i in range(5)]
    leaked_unlevelled = _rec(source="chartqa", split="test", i=99, level=None,
                             boxes=[[1, 2, 3, 4]])
    with pytest.raises(LeakageError):
        build_stage1([*train, leaked_unlevelled], [])


def test_an_unlevelled_synthetic_record_is_refused_not_dropped():
    train = [_rec(i=i, boxes=[[1, 2, 3, 4]], level="L1") for i in range(3)]
    with pytest.raises(ValueError, match="no curriculum level"):
        build_stage1([*train, _rec(i=9, boxes=[[1, 2, 3, 4]])], [])


def test_a_leak_is_reported_rather_than_filtered():
    """Filtering would satisfy the letter of rule 1 while hiding the cause."""
    records = [_rec(i=1), _rec(split="val", i=2), _rec(split="test", i=3)]
    with pytest.raises(LeakageError) as exc:
        assert_train_only(records, "stage1")
    message = str(exc.value)
    assert "'val': 1" in message and "'test': 1" in message
    assert "needs fixing, not hiding" in message


def test_a_clean_training_set_passes():
    assert assert_train_only([_rec(i=i) for i in range(4)], "stage1") is None


# ------------------------------------------------------------------ stage 1


def test_stage_one_is_ordered_l1_to_l4_and_not_shuffled():
    """The order is the curriculum. Shuffling it would remove the point of stage 1."""
    synth = [_rec(i=i, level=lvl, boxes=[[1, 2, 3, 4]])
             for i, lvl in enumerate(["L4", "L2", "L1", "L3", "L1"])]
    out, comp = build_stage1(synth, [])
    levels = [r.meta["level"] for r in out]
    assert levels == sorted(levels, key=LEVEL_ORDER.index)
    assert levels[0] == "L1" and levels[-1] == "L4"
    assert comp.by_level["L1"] == 2


def test_stage_one_takes_only_records_that_have_boxes():
    real = [_rec(source="refchartqa", kind="human", i=10, boxes=[[1, 2, 3, 4]]),
            _rec(source="refchartqa", kind="human", i=11, boxes=None)]  # no boxes
    out, comp = build_stage1([], real)
    assert len(out) == 1 and comp.with_boxes == 1


def test_stage_one_respects_the_cap():
    synth = [_rec(i=i, level="L1", boxes=[[1, 2, 3, 4]]) for i in range(50)]
    out, comp = build_stage1(synth, [], cap=10)
    assert len(out) == 10 == comp.total
    assert build_stage1(synth, [])[1].total == min(50, STAGE1_CAP)


# ------------------------------------------------------------------ stage 2


def test_stage_two_is_shuffled_and_includes_replay():
    records = [_rec(source="chartqa", kind="human", i=i, plan={"op": "sum", "args": []})
               for i in range(30)]
    replay = [_rec(i=100 + i, plan={"op": "mean", "args": []}, boxes=[[1, 2, 3, 4]])
              for i in range(10)]
    out, comp = build_stage2(records, replay, replay=5, seed=0)
    assert comp.by_source["synthetic"] == 5
    assert comp.by_source["chartqa"] == 30
    assert [r.record_id for r in out] != [r.record_id for r in records + replay[:5]], \
        "stage 2 must be shuffled"


def test_stage_two_is_deterministic_for_a_seed():
    records = [_rec(source="chartqa", kind="human", i=i) for i in range(20)]
    a, _ = build_stage2(records, [], seed=3)
    b, _ = build_stage2(records, [], seed=3)
    assert [r.record_id for r in a] == [r.record_id for r in b]


# ------------------------------------------------- compositional vs lookup (0046)


def test_a_bare_lookup_is_not_compositional():
    """73.6% of mined plans are bare lookups; a mixture must not look plan-rich on them."""
    assert not is_compositional({"op": "lookup", "args": ["2019"]})
    assert not is_compositional(None)
    assert is_compositional({"op": "difference", "args": ["a", "b"]})
    assert is_compositional({"op": "lookup", "args": [{"op": "argmax", "args": []}]})


def test_composition_separates_plans_from_compositional_plans():
    records = [_rec(source="chartqa", kind="human", i=1, plan={"op": "lookup", "args": ["a"]}),
               _rec(source="chartqa", kind="human", i=2, plan={"op": "sum", "args": []}),
               _rec(source="chartqa", kind="human", i=3, plan=None)]
    _, comp = build_stage2(records, [])
    assert comp.with_plan == 2
    assert comp.with_compositional_plan == 1


# ------------------------------------------------------------------ output file


def test_the_mixture_file_records_composition_and_ids_only(tmp_path):
    """Rule 7: no dataset content in a committed file — ids and hashes only."""
    records = [_rec(source="chartqa", kind="human", i=i, boxes=[[1, 2, 3, 4]])
               for i in range(4)]
    out, comp = build_stage2(records, [])
    path = tmp_path / "mixture_stage2.json"
    write_mixture(out, comp, path)
    data = json.loads(path.read_text())
    assert data["composition"]["total"] == 4
    assert len(data["record_ids"]) == 4 and len(data["keys"]) == 4
    text = path.read_text()
    for r in records:
        assert r.question not in text, "question text must not be written to disk"
        assert r.image_path not in text, "image paths must not be written to disk"


def test_deduplication_runs_before_the_cap():
    """Otherwise the cap counts duplicates and the mixture is smaller than it reports."""
    dup = [_rec(source="chartqa", kind="human", i=0, sha="aa" * 32, q="same"),
           _rec(source="refchartqa", kind="human", i=1, sha="aa" * 32, q="same",
                boxes=[[1, 2, 3, 4]])]
    others = [_rec(source="chartqa", kind="human", i=i + 2) for i in range(3)]
    _out, comp = build_stage2([*dup, *others], [], cap=4)
    assert comp.total == 4
    assert "2 in -> 1 out" in comp.dedup_summary or "merged" in comp.dedup_summary
