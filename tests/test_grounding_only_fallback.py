"""Stage 1 keeping a record that has gold boxes and no plan — `DECISIONS.md` 0116.

The risk in this change is not that it fails to recover records. It is that it recovers
**too much**: a fallback that catches every refusal would turn "this plan is wrong" into
"train on it anyway with the plan removed", which is repair by another name and exactly
what non-negotiable rule 6 forbids. So most of what follows checks the door is narrow.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from chartqa_dt.data.records import ChartRecord
from chartqa_dt.train.feed import MixtureFeed
from chartqa_dt.train.targets import (
    NoPlanAvailable,
    TargetError,
    build_grounding_only_target,
    build_target,
    has_question_specific_boxes,
)


def rec(i=0, *, answer="1", boxes=None, plan=None, meta=None):
    return ChartRecord(record_id=f"r{i}", source="refchartqa", split="train",
                       image_path="x.png", image_sha256=f"{i:064d}", question=f"q{i}",
                       answer=answer, question_kind="human",
                       boxes=boxes if boxes is not None else [[1, 2, 3, 4], [5, 6, 7, 8]],
                       plan=plan,
                       # RefChartQA-shaped: its boxes mark the evidence for THIS question.
                       # `chartqa_shaped()` below is the other kind, which must be refused.
                       meta=meta if meta is not None else {"refchartqa_id": f"rc{i}"})


def chartqa_shaped(i=0, **kw):
    """A record whose boxes describe the whole chart, as ChartQA's annotation does.

    `ChartRecord` is frozen, so this builds one rather than mutating a RefChartQA fixture.
    """
    kw.setdefault("meta", {"n_elements": 6})
    return rec(i, **kw)


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


# --- the mixture builder must agree with the feed ----------------------------------

def _split(records):
    import sys
    sys.path.insert(0, ".")
    from scripts.build_mixtures import split_by_usability
    return split_by_usability(records, "test")


def test_the_mixture_builder_keeps_grounding_only_records_separately():
    """The first version of 0116 wired the feed and not this, and the mixture still
    reported *"refchartqa dropped 1,735 of 4,000"* — the fallback was dead code, because
    `usable_only` runs before the feed and the records never reached it."""
    plans, grounding = _split([rec(1), rec(2, boxes=[[1, 2, 3, 4]], answer="7")])
    assert len(grounding) == 1, "the no-plan record was dropped rather than kept"
    assert len(plans) == 1


def test_the_two_halves_are_disjoint():
    records = [rec(i) for i in range(4)] + [
        rec(90 + i, boxes=[[1, 2, 3, 4]], answer="7") for i in range(3)]
    plans, grounding = _split(records)
    assert not ({r.record_id for r in plans} & {r.record_id for r in grounding})
    assert len(plans) + len(grounding) == len(records)


def test_a_record_that_is_neither_is_in_neither_half():
    plans, grounding = _split([rec(1, boxes=[]), rec(2, answer=None),
                               rec(3, plan={"op": "lookup", "args": ["nope"]})])
    assert plans == [] and grounding == []


def test_stage_two_never_receives_a_grounding_only_record():
    """A grounding-only record in stage 2 is supervision with the answer taken out."""
    from chartqa_dt.data.mixture import build_stage2

    plans, grounding = _split([rec(1), rec(2, boxes=[[1, 2, 3, 4]], answer="7")])
    # The builder passes only the plan half to stage 2; assert that half is plan-bearing.
    s2, _ = build_stage2(plans, [], cap=100, replay=0, seed=0)
    for r in s2:
        assert r.plan or r.boxes
    assert all(r.record_id not in {g.record_id for g in grounding} for r in s2)


def test_stage_one_receives_both_halves():
    from chartqa_dt.data.mixture import build_stage1

    plans, grounding = _split([rec(1), rec(2), rec(3, boxes=[[1, 2, 3, 4]], answer="7")])
    s1, _ = build_stage1([], [*plans, *grounding], cap=100)
    assert len(s1) == len(plans) + len(grounding) == 3


def test_plan_bearing_records_come_first_so_the_cap_keeps_the_richer_ones():
    from chartqa_dt.data.mixture import build_stage1

    plans, grounding = _split([rec(1), rec(2),
                               rec(3, boxes=[[1, 2, 3, 4]], answer="7")])
    s1, _ = build_stage1([], [*plans, *grounding], cap=1)
    assert s1[0].record_id in {r.record_id for r in plans}


# --- boxes that describe the chart rather than answering the question ---------------

def test_a_record_whose_boxes_describe_the_whole_chart_is_refused():
    """Measured on real data: the ChartQA target for *"Which year has the most crime?"*
    (answer 2014) came out pointing at all six years. ChartQA annotates the chart, not
    the question — its `boxes` ARE its `elements`, the same ones for every question asked
    about that image — so a grounding-only target from it teaches "point at everything".
    """
    with pytest.raises(TargetError, match="whole chart"):
        build_grounding_only_target(chartqa_shaped())


def test_the_feed_does_not_recover_a_whole_chart_record():
    feed = FakeFeed([chartqa_shaped()], shuffle=False, grounding_only_fallback=True)
    assert drain(feed) == []
    assert feed.stats.recovered_grounding_only == 0


def test_the_mixture_builder_does_not_offer_a_whole_chart_record_to_stage_one():
    plans, grounding = _split([chartqa_shaped(1), chartqa_shaped(2)])
    assert plans == [] and grounding == []


def test_a_source_may_declare_the_property_directly():
    """The flag is the contract; `refchartqa_id` is only the fallback evidence for it."""
    assert has_question_specific_boxes(rec(meta={"question_specific_boxes": True}))
    assert not has_question_specific_boxes(rec(meta={"question_specific_boxes": False}))


def test_the_declared_flag_overrides_the_inferred_one():
    r = rec(meta={"refchartqa_id": "x", "question_specific_boxes": False})
    assert not has_question_specific_boxes(r)
    with pytest.raises(TargetError, match="whole chart"):
        build_grounding_only_target(r)


def test_refchartqa_records_are_recognised_without_a_flag():
    assert has_question_specific_boxes(rec())


def test_the_precondition_does_not_touch_plan_targets():
    """A ChartQA record with a real plan is unaffected: the plan selects the evidence."""
    r = chartqa_shaped(boxes=[[1, 2, 3, 4]], answer="7")
    build_target(r)          # must not raise


# --- against the real cache --------------------------------------------------------

CACHE = pathlib.Path.home() / ".cache/chartqa_dt/data/refchartqa_train.jsonl"
needs_cache = pytest.mark.skipif(not CACHE.exists(), reason="RefChartQA cache not present")


def _cached(cap):
    import sys
    sys.path.insert(0, ".")
    from scripts.build_mixtures import refchartqa_records
    return list(refchartqa_records(cap=cap, cache=CACHE))


@needs_cache
def test_refchartqa_elements_are_the_marked_boxes_and_not_the_whole_chart():
    """The property that makes a grounding-only target legitimate for this source.

    If alignment had enriched each record with *every* ChartQA element instead of only
    the marked ones, RefChartQA would carry the same defect ChartQA does and the
    precondition would not catch it — the boxes would still be per-question by
    provenance and whole-chart in fact. Measured: 0 of 55,486 records disagree.
    """
    from chartqa_dt.data.records import ELEMENTS_KEY

    records = _cached(60_000)
    assert len(records) > 10_000, "cache too small for this test to mean anything"
    bad = [r.record_id for r in records
           if (els := r.meta.get(ELEMENTS_KEY)) and r.boxes and len(els) != len(r.boxes)]
    assert not bad, (f"{len(bad)} records carry more elements than marked boxes "
                     f"(first: {bad[0]}) — a grounding-only target from one of these "
                     f"would point at parts of the chart nobody marked")


@needs_cache
def test_a_grounding_only_target_emits_exactly_the_marked_boxes():
    """Nothing added, and nothing dropped except by the declared cap."""
    from chartqa_dt.prompting.prompts import MAX_EVIDENCE

    built = wrong = 0
    for record in _cached(4_000):
        try:
            build_target(record)
            continue
        except NoPlanAvailable:
            pass
        except TargetError:
            continue
        try:
            obj = json.loads(build_grounding_only_target(record))
        except TargetError:
            continue
        built += 1
        if len(obj["evidence"]) != min(len(record.boxes or []), MAX_EVIDENCE):
            wrong += 1
    assert built > 500, f"only {built} grounding-only targets built; test is too weak"
    assert wrong == 0, f"{wrong} of {built} targets did not emit exactly the marked boxes"


# --- every source declares what its boxes mean --------------------------------------

def test_chartqa_records_declare_whole_chart_boxes():
    """Declared at ingestion, not inferred by each consumer (`DECISIONS.md` 0119)."""
    import sys
    sys.path.insert(0, ".")
    from scripts.build_mixtures import archive_path, chartqa_records

    from chartqa_dt.data.chartqa import ArchiveReader

    # CI has no dataset archive (rule 7 keeps it out of git), so this skips there.
    # It cost a red run once: the first version asserted unconditionally and every
    # CI job failed on a missing zip.
    if not pathlib.Path(archive_path()).exists():
        pytest.skip("ChartQA archive not present")
    records = chartqa_records(ArchiveReader(archive_path()), limit=8, seed=0)
    assert records, "no ChartQA records to check"
    for r in records:
        assert r.meta.get("question_specific_boxes") is False
        assert not has_question_specific_boxes(r)


def test_refchartqa_records_declare_question_specific_boxes():
    import inspect

    from chartqa_dt.data.refchartqa import row_to_record

    src = inspect.getsource(row_to_record)
    assert '"question_specific_boxes": True' in src, (
        "row_to_record no longer declares box semantics; cached records fall back to "
        "inferring from refchartqa_id, and a new cache would carry nothing")


def test_a_source_that_declares_nothing_is_treated_as_whole_chart():
    """The safe direction: lose grounding-only targets rather than emit wrong ones."""
    r = rec(meta={"n_elements": 3})
    assert not has_question_specific_boxes(r)
    with pytest.raises(TargetError, match="whole chart"):
        build_grounding_only_target(r)
