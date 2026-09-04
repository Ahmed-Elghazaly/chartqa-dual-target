"""Stage 1 keeping a record that has gold boxes and no plan — `DECISIONS.md` 0116.

The risk in this change is not that it fails to recover records. It is that it recovers
**too much**: a fallback that catches every refusal would turn "this plan is wrong" into
"train on it anyway with the plan removed", which is repair by another name and exactly
what non-negotiable rule 6 forbids. So most of what follows checks the door is narrow.
"""

from __future__ import annotations

import json

import pytest

from chartqa_dt.data.records import ChartRecord
from chartqa_dt.train.feed import MixtureFeed
from chartqa_dt.train.targets import (
    NoPlanAvailable,
    TargetError,
    build_grounding_only_target,
    build_target,
)


def rec(i=0, *, answer="1", boxes=None, plan=None, meta=None):
    return ChartRecord(record_id=f"r{i}", source="refchartqa", split="train",
                       image_path="x.png", image_sha256=f"{i:064d}", question=f"q{i}",
                       answer=answer, question_kind="human",
                       boxes=boxes if boxes is not None else [[1, 2, 3, 4], [5, 6, 7, 8]],
                       plan=plan, meta=meta or {})


class FakeFeed(MixtureFeed):
    def _image(self, record):
        return f"IMG:{record.record_id}"


def drain(feed, n=None):
    """Offer every record exactly once.

    Not `feed.batches()`: that yields forever and rolls epochs, so a feed that refuses
    everything — which several of these tests deliberately build — would spin. This
    mirrors what `batches` does per record, which is what is under test here.
    """
    out = []
    for record in feed.records:
        feed.stats.offered += 1
        example = feed._example(record)
        if example is not None:
            out.append(example)
    return out


# --- the typed refusal ------------------------------------------------------------

def test_no_plan_available_is_a_target_error():
    """Every existing `except TargetError` must keep working, or this change silently
    turns refusals into crashes somewhere else."""
    assert issubclass(NoPlanAvailable, TargetError)


def test_a_record_with_boxes_and_no_plan_raises_the_typed_refusal():
    with pytest.raises(NoPlanAvailable):
        build_target(rec())


def test_a_record_whose_plan_is_wrong_raises_the_untyped_one():
    """The distinction the whole change rests on: inconsistent is not incomplete."""
    r = rec(plan={"op": "lookup", "args": ["nope"]})
    with pytest.raises(TargetError) as exc:
        build_target(r)
    assert not isinstance(exc.value, NoPlanAvailable)


# --- the fallback is off unless asked for -----------------------------------------

def test_the_fallback_is_off_by_default():
    feed = FakeFeed([rec()], shuffle=False)
    assert drain(feed) == []
    assert feed.stats.usable == 0 and feed.stats.recovered_grounding_only == 0


def test_with_the_fallback_the_record_is_kept():
    feed = FakeFeed([rec()], shuffle=False, grounding_only_fallback=True)
    assert len(drain(feed)) == 1
    assert feed.stats.usable == 1 and feed.stats.recovered_grounding_only == 1


def test_the_control_arm_never_takes_the_fallback():
    """`answer_only` is the control for stage 2. Feeding it grounding-only targets would
    change what the control controls for."""
    feed = FakeFeed([rec()], shuffle=False, grounding_only_fallback=True,
                    answer_only=True)
    assert feed.stats.recovered_grounding_only == 0


# --- the door is narrow -----------------------------------------------------------

def test_a_wrong_plan_is_still_dropped_even_with_the_fallback_on():
    feed = FakeFeed([rec(plan={"op": "lookup", "args": ["nope"]})], shuffle=False,
                    grounding_only_fallback=True)
    assert drain(feed) == []
    assert feed.stats.recovered_grounding_only == 0
    assert feed.stats.refused, "the refusal was not recorded"


def test_a_record_with_no_boxes_is_still_dropped():
    feed = FakeFeed([rec(boxes=[])], shuffle=False, grounding_only_fallback=True)
    assert drain(feed) == []
    assert feed.stats.recovered_grounding_only == 0


def test_a_record_with_no_answer_is_still_dropped():
    feed = FakeFeed([rec(answer=None)], shuffle=False, grounding_only_fallback=True)
    assert drain(feed) == []


@pytest.mark.parametrize("box", [[3, 2, 1, 4], [1, 2, 1, 4], [1, 2, 3, 2], "nope", None])
def test_an_unusable_box_is_still_dropped(box):
    """A grounding-only target is nothing but its boxes, so a bad one makes the record
    worthless rather than merely incomplete."""
    feed = FakeFeed([rec(boxes=[box])], shuffle=False, grounding_only_fallback=True)
    assert drain(feed) == []
    assert feed.stats.recovered_grounding_only == 0


def test_a_record_that_builds_a_real_plan_is_not_counted_as_recovered():
    # One box and a numeric answer: `build_record` derives `lookup` and the record needs
    # no fallback. Two boxes (the default fixture) is the case that has no derivation.
    r = rec(boxes=[[1, 2, 3, 4]], answer="7")
    feed = FakeFeed([r], shuffle=False, grounding_only_fallback=True)
    got = drain(feed)
    assert got, "the fixture no longer builds a plan target; the test is not testing"
    assert feed.stats.recovered_grounding_only == 0
    assert "plan" in json.loads(got[0].target)


# --- what the recovered target contains -------------------------------------------

def test_the_recovered_target_omits_the_plan_rather_than_inventing_one():
    target = build_grounding_only_target(rec())
    obj = json.loads(target)
    assert "plan" not in obj, "a plan was invented for a record that has none"
    assert obj["answerable"] is True
    assert obj["model_answer"] == "1"
    assert len(obj["evidence"]) == 2


def test_the_recovered_target_is_a_strict_prefix_of_the_full_schema():
    """Stage 1 must teach a subset of what stage 2 completes, not a different format."""
    obj = json.loads(build_grounding_only_target(rec()))
    full = {"answerable", "evidence", "plan", "model_answer"}
    assert set(obj) < full, f"unexpected keys: {set(obj) - full}"


def test_the_recovered_target_keeps_every_box():
    obj = json.loads(build_grounding_only_target(rec(boxes=[[1, 2, 3, 4], [5, 6, 7, 8],
                                                           [9, 10, 11, 12]])))
    assert len(obj["evidence"]) == 3


# --- the counter is honest --------------------------------------------------------

def test_recovered_records_are_counted_separately_from_plan_records():
    """A stage-1 run that is mostly box-only is a different run from one that is mostly
    plans, and the difference must be visible without re-deriving it."""
    records = [rec(i) for i in range(5)] + [
        rec(99, boxes=[[1, 2, 3, 4]], answer="7")]
    feed = FakeFeed(records, shuffle=False, grounding_only_fallback=True)
    drain(feed)
    assert feed.stats.recovered_grounding_only == 5
    assert feed.stats.usable == 6, "the plan record should be usable and not recovered"


def test_the_fallback_does_not_change_the_offered_count():
    records = [rec(i) for i in range(4)]
    off = FakeFeed(list(records), shuffle=False)
    on = FakeFeed(list(records), shuffle=False, grounding_only_fallback=True)
    drain(off)
    drain(on)
    assert off.stats.offered == on.stats.offered == 4


# --- the malformed box that used to kill a run -------------------------------------

@pytest.mark.parametrize("box", [None, "nope", 5, [1, 2, 3], [1, 2, 3, 4, 5], [],
                                 [1, 2, 3, "x"], [1, 2, 3, None], [True, 2, 3, 4],
                                 {"x": 1}])
def test_a_malformed_box_is_refused_and_does_not_crash(box):
    """`ChartRecord.from_dict` does not validate `boxes` — it takes whatever the cache
    holds. A malformed one used to reach `tuple(box)` and raise `TypeError`, which the
    feed does not catch, so it killed the run instead of costing one record.
    """
    r = rec(boxes=[box], answer="7")
    with pytest.raises(TargetError):
        build_target(r)


@pytest.mark.parametrize("box", [None, "nope", [1, 2, 3], [1, 2, 3, "x"]])
def test_the_feed_survives_a_malformed_box(box):
    feed = FakeFeed([rec(boxes=[box], answer="7")], shuffle=False,
                    grounding_only_fallback=True)
    assert drain(feed) == []          # would previously have raised TypeError
    assert feed.stats.refused


def test_a_well_formed_box_still_builds():
    """The guard must not refuse valid boxes — including floats and zeros."""
    for box in ([0, 0, 1, 1], [1.5, 2.5, 300.0, 400.0], (10, 20, 30, 40)):
        obj = json.loads(build_grounding_only_target(rec(boxes=[list(box)])))
        assert len(obj["evidence"]) == 1


def test_one_bad_box_among_good_ones_still_refuses_the_record():
    """Half a target is not a target: the record points at something that is not there."""
    r = rec(boxes=[[1, 2, 3, 4], None], answer="7")
    with pytest.raises(TargetError):
        build_target(r)
