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


class TestImagesFromTheArchive:
    """`DECISIONS.md` 0073. ChartQA ships as one zip and this project never extracts it.

    A record's `image_path` is the zip member name, which looks exactly like a relative
    disk path — so opening it directly succeeds on a host that happens to have extracted
    the archive and fails everywhere else. The failure is an `OSError`, which `_example`
    catches and counts as a refusal, so it costs records silently rather than raising.
    """

    class _Archive:
        def __init__(self, members: dict[str, bytes]) -> None:
            self.members = members
            self.reads: list[str] = []

        def exists(self, name: str) -> bool:
            return name in self.members

        def read(self, name: str) -> bytes:
            self.reads.append(name)
            return self.members[name]

    @staticmethod
    def _png_bytes() -> bytes:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (8, 6), (1, 2, 3)).save(buf, format="PNG")
        return buf.getvalue()

    def _record(self, path: str):
        from chartqa_dt.data.records import ChartRecord

        return ChartRecord(
            record_id="r1", source="chartqa", split="train", image_path=path,
            image_sha256="0" * 64, question="q?", answer="1",
            question_kind="human")

    def test_an_image_only_in_the_archive_is_still_read(self, tmp_path) -> None:
        from chartqa_dt.train.feed import MixtureFeed

        name = "ChartQA Dataset/train/png/two_col_81790.png"
        archive = self._Archive({name: self._png_bytes()})
        feed = MixtureFeed([], shuffle=False, image_root=tmp_path, archive=archive)
        image = feed._image(self._record(name))
        assert image.size == (8, 6) and image.mode == "RGB"
        assert archive.reads == [name]

    def test_disk_wins_when_the_file_is_actually_there(self, tmp_path) -> None:
        """The archive is a fallback, not a replacement: reading the zip for every image
        would be slower for no benefit where the file exists."""
        from chartqa_dt.train.feed import MixtureFeed

        (tmp_path / "chart.png").write_bytes(self._png_bytes())
        archive = self._Archive({})
        feed = MixtureFeed([], shuffle=False, image_root=tmp_path, archive=archive)
        assert feed._image(self._record("chart.png")).size == (8, 6)
        assert archive.reads == []

    def test_a_missing_image_says_the_archive_was_not_supplied(self, tmp_path) -> None:
        import pytest

        from chartqa_dt.train.feed import MixtureFeed

        feed = MixtureFeed([], shuffle=False, image_root=tmp_path)
        with pytest.raises(FileNotFoundError, match="no archive was supplied"):
            feed._image(self._record("absent.png"))

    def test_the_training_cli_supplies_an_archive_to_the_feed(self) -> None:
        """Without this the feed silently loses every ChartQA record."""
        import inspect

        from chartqa_dt.cli import train

        assert "archive=_chartqa_archive()" in inspect.getsource(train._run_stage)


class TestRefusalRateGate:
    """`DECISIONS.md` 0074. Counting a refusal is not enough — it has to stop the run.

    Four defects in this project had the same shape: something the feed cannot turn into
    an example, caught, counted, skipped. Each produced a smaller training set rather than
    an error, so the run finished on schedule and reported its step count truthfully.
    """

    @staticmethod
    def _records(n: int):
        from chartqa_dt.data.records import ChartRecord

        return [ChartRecord(record_id=f"r{i}", source="synthetic", split="train",
                            image_path=f"{i}.png", image_sha256="0" * 64,
                            question="q?", answer="1", question_kind="synthetic")
                for i in range(n)]

    def _feed(self, *, usable_every: int):
        """A feed whose `_example` succeeds one record in `usable_every`."""
        from chartqa_dt.train.feed import Example, MixtureFeed

        feed = MixtureFeed(self._records(2000), shuffle=False)
        calls = {"n": 0}

        def fake(record):
            calls["n"] += 1
            if calls["n"] % usable_every:
                feed.stats.note_refusal(ValueError("no plan derivable"))
                return None
            feed.stats.usable += 1
            return Example(image=object(), question="q?", target="{}")

        feed._example = fake
        return feed

    def test_a_feed_refusing_most_records_stops_the_run(self) -> None:
        import pytest

        from chartqa_dt.train.feed import FeedRefusedTooMuch

        feed = self._feed(usable_every=3)          # 33% usable
        with pytest.raises(FeedRefusedTooMuch, match="below the 90% floor"):
            # Consumed the way training consumes it. The gate waits for 200 offered
            # records, which at 33% usable is about eight optimizer steps -- under two
            # minutes of GPU, against the ten hours it used to cost to find out.
            for _ in feed.batches(2):
                pass

    def test_the_error_names_the_reasons_and_how_to_reproduce_it_without_a_gpu(self) -> None:
        import pytest

        from chartqa_dt.train.feed import FeedRefusedTooMuch

        with pytest.raises(FeedRefusedTooMuch) as exc:
            for _ in self._feed(usable_every=5).batches(2):
                pass
        message = str(exc.value)
        assert "no plan derivable" in message
        assert "measure_target_yield.py" in message

    def test_a_healthy_feed_is_not_interrupted(self) -> None:
        """Measured yield after the 0071-0073 fixes is 99.5%; the floor must not fire."""
        import itertools

        feed = self._feed(usable_every=1)
        batches = list(itertools.islice(feed.batches(2), 150))
        assert len(batches) == 150 and all(len(b) == 2 for b in batches)
        assert feed.stats.offered > 200, "the gate was passed, not skipped"

    def test_the_check_waits_for_enough_records_to_be_meaningful(self) -> None:
        """Two refusals in the first three records is noise, not a broken pipeline."""
        from chartqa_dt.train.feed import MixtureFeed

        feed = MixtureFeed(self._records(10), shuffle=False)
        feed.stats.offered, feed.stats.usable = 3, 1
        feed.check_refusal_rate()          # does not raise

    def test_the_floor_leaves_room_for_the_known_over_length_loss(self) -> None:
        from chartqa_dt.train.feed import MIN_USABLE_FRACTION

        assert MIN_USABLE_FRACTION <= 0.995, "must tolerate the measured 0.5% over-length"
        assert MIN_USABLE_FRACTION > 0.62, "must catch the 38% loss that actually occurred"
