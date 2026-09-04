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


# --------------------------------------------- image contamination (DECISIONS.md 0049)


def test_a_held_out_image_is_refused_even_when_the_record_says_train(tmp_path):
    """The failure that no split check can see.

    RefChartQA labels rows "train" that use ChartQA *test* charts — 4 in 4,000 measured —
    and ChartQA's own train split holds 15 images pixel-identical to held-out charts.
    Every `split` field says "train". The contamination is at the image level.
    """
    import json as _json

    from chartqa_dt.splits import ImageContaminationError, assert_no_held_out_images

    sealed = tmp_path / "sealed.json"
    held_out_hash = "cc" * 32
    sealed.write_text(_json.dumps({"hashes": {"test": [held_out_hash], "val": []}}))

    clean = [_rec(source="chartqa", kind="human", i=i) for i in range(3)]
    contaminated = ChartRecord(
        record_id="refchartqa-train-x", source="refchartqa", split="train",
        image_path="x.png", image_sha256=held_out_hash, question="q",
        answer="1", question_kind="human", boxes=[[1, 2, 3, 4]], meta={})
    assert contaminated.split == "train", "the point is that the label looks fine"

    assert assert_no_held_out_images(clean, "stage1", path=str(sealed)) is None
    with pytest.raises(ImageContaminationError, match="validation or test IMAGE"):
        assert_no_held_out_images([*clean, contaminated], "stage1",
                                  path=str(sealed))


def test_the_guard_raises_rather_than_filtering(tmp_path):
    """A silent filter would hide that a source is handing us contaminated rows."""
    import json as _json

    from chartqa_dt.splits import ImageContaminationError, assert_no_held_out_images

    sealed = tmp_path / "sealed.json"
    sealed.write_text(_json.dumps({"hashes": {"test": ["dd" * 32], "val": []}}))
    bad = ChartRecord(record_id="r", source="chartqa", split="train", image_path="x.png",
                      image_sha256="dd" * 32, question="q", answer="1",
                      question_kind="human", meta={})
    with pytest.raises(ImageContaminationError) as exc:
        assert_no_held_out_images([bad], "stage2", path=str(sealed))
    assert "Fix the source, do not filter here" in str(exc.value)


def test_an_absent_sealed_file_does_not_silently_pass_everything(tmp_path):
    """It returns empty, so callers that need it must check — as the cache script does."""
    from chartqa_dt.splits import sealed_image_hashes

    assert sealed_image_hashes(str(tmp_path / "missing.json")) == frozenset()


def test_the_committed_sealed_set_covers_both_held_out_splits():
    import json as _json
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    path = root / "data/sealed_images.json"
    if not path.exists():
        pytest.skip("sealed_images.json is built from the archive")
    data = _json.loads(path.read_text())
    assert set(data["hashes"]) == {"val", "test"}
    assert len(data["hashes"]["test"]) > 1000 and len(data["hashes"]["val"]) > 1000
    assert all(len(h) == 64 for split in data["hashes"].values() for h in split)


def test_the_written_mixtures_contain_no_held_out_chart():
    """End-to-end check on the committed artefacts, not on the builder that made them.

    `dedup_key` is `sha256(image)[:16] + ":" + sha256(question)[:16]`, so the image half
    of every key in a mixture file can be matched against the sealed hashes directly —
    without the archive, and without trusting the code that wrote the file.
    """
    import json as _json
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    sealed_path = root / "data/sealed_images.json"
    if not sealed_path.exists():
        pytest.skip("sealed_images.json is built from the archive")
    sealed = {h[:16] for split in _json.loads(sealed_path.read_text())["hashes"].values()
              for h in split}

    checked = 0
    for name in ("mixture_stage1.json", "mixture_stage2.json"):
        path = root / "data" / name
        if not path.exists():
            continue
        keys = _json.loads(path.read_text())["keys"]
        assert keys, f"{name} is empty"
        offenders = [k for k in keys if k.split(":")[0] in sealed]
        assert not offenders, (
            f"{name}: {len(offenders)} records use a held-out ChartQA chart "
            f"(first key {offenders[0]})")
        checked += 1
    if not checked:
        pytest.skip("no mixture files built yet")


# ------------------------------------------- chart types the evaluation never shows


def _synthetic(chart_type: str, level: str = "L1", n: int = 1):
    from chartqa_dt.data.records import ChartRecord
    return [ChartRecord(record_id=f"{chart_type}-{level}-{i}", source="synthetic",
                        split="train", image_path=f"{chart_type}{i}.png",
                        image_sha256=f"d{chart_type}{i}", question="q?", answer="1",
                        question_kind="synthetic",
                        meta={"level": level, "chart_type": chart_type})
            for i in range(n)]


def test_chart_types_the_evaluation_corpus_lacks_are_dropped():
    """Measured over 3,000 real ChartQA charts: area and scatter are 0.0% of them, and 25%
    of the synthetic corpus. Stage 1 was spending a quarter of its budget teaching the model
    to ground chart types it will never be asked about (`DECISIONS.md` 0091)."""
    from chartqa_dt.data.mixture import drop_absent_chart_types
    records = _synthetic("vbar", n=3) + _synthetic("area", n=2) + _synthetic("scatter", n=1)
    kept, dropped = drop_absent_chart_types(records)
    assert dropped == 3
    assert {r.meta["chart_type"] for r in kept} == {"vbar"}


def test_the_drop_is_counted_not_silent():
    """A filter that shrinks a mixture without saying so is how 12,000 stage-1 records were
    lost once already (0071)."""
    from chartqa_dt.data.mixture import drop_absent_chart_types
    kept, dropped = drop_absent_chart_types(_synthetic("line", n=4))
    assert (len(kept), dropped) == (4, 0)


def test_chart_types_the_evaluation_does_show_are_kept():
    from chartqa_dt.data.mixture import ABSENT_FROM_EVALUATION, drop_absent_chart_types
    for kind in ("vbar", "hbar", "grouped_bar", "line", "multi_line", "pie"):
        assert kind not in ABSENT_FROM_EVALUATION
        kept, dropped = drop_absent_chart_types(_synthetic(kind, n=2))
        assert (len(kept), dropped) == (2, 0), kind


def test_a_record_with_no_chart_type_survives():
    """Real ChartQA records carry no `chart_type` in meta; the filter must not eat them."""
    from chartqa_dt.data.mixture import drop_absent_chart_types
    from chartqa_dt.data.records import ChartRecord
    real = ChartRecord(record_id="r", source="chartqa", split="train", image_path="i.png",
                       image_sha256="d", question="q?", answer="1", question_kind="human")
    kept, dropped = drop_absent_chart_types([real])
    assert (len(kept), dropped) == (1, 0)


def test_dropping_leaves_the_curriculum_balanced():
    """Each level loses the same two chart types, so L1-L4 stay equal -- otherwise the
    curriculum would silently tilt."""
    import collections

    from chartqa_dt.data.mixture import drop_absent_chart_types
    records = [r for lvl in ("L1", "L2", "L3", "L4")
               for kind in ("vbar", "line", "area", "scatter")
               for r in _synthetic(kind, lvl, n=5)]
    kept, _ = drop_absent_chart_types(records)
    counts = collections.Counter(r.meta["level"] for r in kept)
    assert set(counts.values()) == {10}, counts
