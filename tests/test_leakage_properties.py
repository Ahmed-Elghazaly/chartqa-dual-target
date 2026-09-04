"""Leakage and deduplication properties, over seeded random record sets.

A leak does not fail anything. It produces a number that is too good, and every conclusion
drawn from it is wrong in the same direction. This project has already found two real ones:
ChartQA's own train split contains 15 images pixel-identical to val/test charts, and
RefChartQA ships rows labelled "train" that use ChartQA test charts (`DECISIONS.md` 0049).

Both were caught by hashing **decoded pixels** rather than trusting a split label, and both
would have been invisible to a test that only checked the labels. These properties are
written the same way: they assume the labels lie.
"""
from __future__ import annotations

import random

import pytest

from chartqa_dt.data.dedup import deduplicate, find_cross_split_leaks, merge_pair, union_boxes
from chartqa_dt.data.mixture import LeakageError, assert_train_only
from chartqa_dt.data.records import ChartRecord, dedup_key


def rec(rid, *, digest="d", question="q?", split="train", source="chartqa", **kw):
    base = {"record_id": rid, "source": source, "split": split, "image_path": f"{rid}.png",
            "image_sha256": digest, "question": question, "answer": "1",
            "question_kind": "human"}
    return ChartRecord(**{**base, **kw})


# ============================================================== the key that finds duplicates


@pytest.mark.parametrize("seed", range(12))
def test_the_dedup_key_depends_on_the_image_and_the_question_only(seed):
    """Not the record id, not the source, not the split — because the whole point is to catch
    the same example arriving twice **under different labels**."""
    rng = random.Random(seed)
    for _ in range(80):
        digest = f"sha{rng.randint(0, 999)}"
        question = rng.choice(["how many?", "which is largest?", "  How Many?  "])
        a = rec("a", digest=digest, question=question, source="chartqa", split="train")
        b = rec("b", digest=digest, question=question, source="refchartqa", split="val")
        assert a.key == b.key


def test_the_key_normalises_a_question_rather_than_matching_it_literally():
    assert rec("a", question="How many?").key == rec("b", question="  how   many? ").key
    assert rec("a", question="How many?").key != rec("b", question="how few?").key


def test_a_different_image_is_never_the_same_example():
    assert rec("a", digest="one").key != rec("b", digest="two").key


# ================================================================== deduplication properties


@pytest.mark.parametrize("seed", range(12))
def test_deduplication_never_loses_a_distinct_example(seed):
    """Merging may reduce the count only by the number of collisions, never by more."""
    rng = random.Random(100 + seed)
    for _ in range(40):
        records = [rec(f"r{i}", digest=f"d{rng.randint(0, 5)}",
                       question=rng.choice(["a?", "b?", "c?"])) for i in range(rng.randint(1, 20))]
        kept, _ = deduplicate(records)
        distinct = {r.key for r in records}
        assert len(kept) == len(distinct), (len(records), len(kept), len(distinct))
        assert {r.key for r in kept} == distinct, "a distinct example vanished"


@pytest.mark.parametrize("seed", range(10))
def test_deduplication_is_order_independent_in_what_it_keeps(seed):
    """Shuffling the input must not change *which examples* survive, only which copy."""
    rng = random.Random(200 + seed)
    records = [rec(f"r{i}", digest=f"d{i % 4}", question=f"q{i % 3}?") for i in range(24)]
    first, _ = deduplicate(records)
    shuffled = records[:]
    rng.shuffle(shuffled)
    second, _ = deduplicate(shuffled)
    assert {r.key for r in first} == {r.key for r in second}


def test_deduplication_reports_what_it_merged():
    records = [rec("a", digest="x"), rec("b", digest="x"), rec("c", digest="y")]
    kept, report = deduplicate(records)
    assert len(kept) == 2
    assert report.summary(), "a merge that reports nothing is a merge nobody can audit"


@pytest.mark.parametrize("seed", range(10))
def test_merging_never_invents_a_box(seed):
    """`union_boxes` may only combine what it was given."""
    rng = random.Random(300 + seed)
    for _ in range(60):
        a = [[rng.uniform(0, 100) for _ in range(4)] for _ in range(rng.randint(0, 3))]
        b = [[rng.uniform(0, 100) for _ in range(4)] for _ in range(rng.randint(0, 3))]
        got = union_boxes(a, b) or []
        for box in got:
            assert box in a or box in b, "a box appeared that neither side had"
        assert len(got) <= len(a) + len(b)


def test_merging_two_records_keeps_the_plan_that_exists():
    a = rec("a", digest="x")
    b = rec("b", digest="x", plan={"op": "count", "args": []})
    merged = merge_pair(a, b)
    assert merged.plan == {"op": "count", "args": []}


# ========================================================================= split guards


def test_a_record_from_the_wrong_split_is_refused_loudly():
    with pytest.raises(LeakageError):
        assert_train_only([rec("a"), rec("b", split="test")], "stage1")


def test_the_guard_checks_the_inputs_not_the_survivors():
    """Checking what survives lets a leaked record slip through by being dropped for some
    other reason first. The guard runs on the inputs, before any filtering."""
    with pytest.raises(LeakageError):
        assert_train_only([rec("leak", split="val")], "stage1")


@pytest.mark.parametrize("split", ["val", "test", "validation", "TEST", ""])
def test_no_non_train_label_is_ever_accepted(split):
    with pytest.raises(LeakageError):
        assert_train_only([rec("a"), rec("x", split=split)], "stage2")


def test_an_empty_set_of_records_is_not_a_leak():
    assert_train_only([], "stage1")


# ==================================================== the leak the labels do not show


def test_the_same_image_in_two_splits_is_found_even_when_both_say_train():
    """The real defect: RefChartQA ships rows labelled "train" that use ChartQA test charts.
    A label-only check sees nothing; hashing the decoded pixels sees it."""
    records = [rec("a", digest="same", split="train", source="chartqa"),
               rec("b", digest="same", split="test", source="chartqa")]
    leaks = find_cross_split_leaks(records)
    assert leaks, "an image appearing in two splits must be reported"


@pytest.mark.parametrize("seed", range(10))
def test_no_leak_is_reported_when_every_image_stays_in_one_split(seed):
    rng = random.Random(400 + seed)
    records = [rec(f"r{i}", digest=f"d{i}", split=rng.choice(["train", "val", "test"]))
               for i in range(30)]
    assert not find_cross_split_leaks(records)


def test_the_same_image_twice_in_one_split_is_not_a_leak():
    """Duplication within a split is dedup's job, not leakage's."""
    records = [rec("a", digest="same"), rec("b", digest="same")]
    assert not find_cross_split_leaks(records)


# ============================================================ the key itself is well behaved


@pytest.mark.parametrize("seed", range(8))
def test_dedup_key_is_deterministic_and_total(seed):
    rng = random.Random(500 + seed)
    for _ in range(200):
        digest = "".join(rng.choice("0123456789abcdef") for _ in range(8))
        question = "".join(rng.choice(" ?abcXYZé—") for _ in range(rng.randint(0, 40)))
        assert dedup_key(digest, question) == dedup_key(digest, question)
        assert isinstance(dedup_key(digest, question), str)
