"""The training feed — `PLAN.md` 6.1, 6.2, 6.3.

Two properties carry real risk and are tested hardest: stage 1 must **not** shuffle
(`shuffle=True` is the default almost everywhere and would silently destroy the
curriculum), and the feed must resume at the exact position it left (a resume that
restarts the epoch trains on the first examples twice and never reaches the last, and the
loss curve looks entirely normal).
"""

from __future__ import annotations

import json

import pytest

from chartqa_dt.data.records import ChartRecord
from chartqa_dt.train.feed import FeedStats, MixtureFeed, load_mixture_records


def rec(i, *, answer="1", boxes=None, plan=None, path="x.png"):
    return ChartRecord(record_id=f"r{i}", source="synthetic", split="train",
                       image_path=path, image_sha256=f"{i:064d}", question=f"q{i}",
                       answer=answer, question_kind="synthetic",
                       boxes=boxes or [[1, 2, 3, 4]], plan=plan, meta={})


class FakeFeed(MixtureFeed):
    """Skips real image loading; the tests are about ordering and resume."""

    def _image(self, record):
        return f"IMG:{record.record_id}"


def test_stage_one_does_not_shuffle():
    """The curriculum is the point of stage 1; shuffling it silently removes it."""
    records = [rec(i) for i in range(10)]
    feed = FakeFeed(records, shuffle=False)
    order = [b[0].question for b in _take(feed, 10)]
    assert order == [f"q{i}" for i in range(10)]


def test_stage_two_shuffles_and_is_seeded():
    records = [rec(i) for i in range(20)]
    a = [b[0].question for b in _take(FakeFeed(records, shuffle=True, seed=7), 20)]
    b = [b[0].question for b in _take(FakeFeed(records, shuffle=True, seed=7), 20)]
    c = [b[0].question for b in _take(FakeFeed(records, shuffle=True, seed=8), 20)]
    assert a == b, "same seed, same order"
    assert a != c, "different seed, different order"
    assert sorted(a) == sorted([f"q{i}" for i in range(20)]), "every record appears once"


def test_the_feed_resumes_at_the_exact_position():
    """`PLAN.md` 6.3 requires the dataloader position in the checkpoint."""
    records = [rec(i) for i in range(12)]
    feed = FakeFeed(records, shuffle=True, seed=3)
    first = [b[0].question for b in _take(feed, 5)]
    state = feed.state_dict()

    resumed = FakeFeed(records, shuffle=True, seed=3)
    resumed.load_state_dict(state)
    rest = [b[0].question for b in _take(resumed, 5)]

    straight = [b[0].question for b in _take(FakeFeed(records, shuffle=True, seed=3), 10)]
    assert first + rest == straight, "resume must continue, not restart"


def test_resuming_onto_a_different_mixture_is_refused():
    """Silently training on a different set than the checkpoint recorded is worse than
    stopping."""
    feed = FakeFeed([rec(i) for i in range(10)], shuffle=False)
    state = feed.state_dict()
    smaller = FakeFeed([rec(i) for i in range(4)], shuffle=False)
    with pytest.raises(ValueError, match="different mixture"):
        smaller.load_state_dict(state)


def test_epochs_roll_and_reshuffle():
    records = [rec(i) for i in range(4)]
    feed = FakeFeed(records, shuffle=True, seed=1)
    seen = [b[0].question for b in _take(feed, 8)]
    assert feed.epoch == 1
    assert sorted(seen[:4]) == sorted(seen[4:]), "each epoch covers every record"


def test_records_that_cannot_produce_a_target_are_skipped_and_counted():
    """A target that cannot be built is not silently dropped — `DECISIONS.md` 0067."""
    good = rec(1, answer="5")
    unbuildable = rec(2, answer="Yes", boxes=[[1, 2, 3, 4], [5, 6, 7, 8]])
    feed = FakeFeed([good, unbuildable], shuffle=False)
    batches = _take(feed, 2)
    assert all(len(b) == 1 for b in batches)
    assert all(b[0].question == "q1" for b in batches), "only the buildable record recurs"
    assert feed.stats.usable < feed.stats.offered
    assert feed.stats.refused, "the reason must be recorded"
    assert "usable examples" in feed.stats.describe()


def test_the_answer_only_control_uses_a_bare_answer():
    """`PLAN.md` 6.4 — same records, different target."""
    feed = FakeFeed([rec(1, answer="42")], shuffle=False, answer_only=True)
    assert _take(feed, 1)[0][0].target == "42"


def test_a_mixture_whose_ids_have_drifted_is_refused(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"composition": {}, "record_ids": ["r1", "r-missing"],
                                "keys": []}))
    with pytest.raises(ValueError, match="drifted"):
        load_mixture_records(path, {"r1": rec(1)})


def test_a_mixture_rehydrates_in_its_recorded_order(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"composition": {}, "record_ids": ["r2", "r1"], "keys": []}))
    got = load_mixture_records(path, {"r1": rec(1), "r2": rec(2)})
    assert [r.record_id for r in got] == ["r2", "r1"]


def test_empty_stats_do_not_divide_by_zero():
    assert "0/0" in FeedStats().describe()


def _take(feed, n):
    out = []
    for batch in feed.batches(1):
        out.append(batch)
        if len(out) == n:
            break
    return out
