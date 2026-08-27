"""Loaders, box conversion and deduplication.

The box conversion gets the most attention here because it is the highest-consequence
silent failure in the project: RefChartQA ships absolute-pixel ``{x, y, w, h}`` while
everything downstream expects normalised ``[x1, y1, x2, y2]``, and reading a width as an
x2 produces boxes that are merely *wrong* rather than obviously broken. Every grounding
number would drop with nothing pointing at the cause.
"""

from __future__ import annotations

import random

import pytest

from chartqa_dt.data.chartqa import (
    ChartQAError,
    annotation_boxes,
    annotation_path,
    axis_labels,
    image_path,
    parse_table,
    qa_path,
    table_path,
)
from chartqa_dt.data.dedup import (
    DedupReport,
    deduplicate,
    find_cross_split_leaks,
    merge_pair,
    union_boxes,
)
from chartqa_dt.data.records import ChartRecord, dedup_key
from chartqa_dt.data.refchartqa import (
    RefChartQAError,
    boxes_to_norm1000,
    normalise_split,
    row_to_record,
    xywh_to_norm1000,
)

# --------------------------------------------------------------------------- boxes


def test_xywh_conversion_against_hand_computed_values():
    """A 640x386 chart, the size of the first real RefChartQA validation row."""
    got = xywh_to_norm1000({"x": 276.0, "y": 277.0, "w": 60.0, "h": 23.0}, 640, 386)
    assert got == pytest.approx([1000 * 276 / 640, 1000 * 277 / 386,
                                 1000 * 336 / 640, 1000 * 300 / 386])
    assert got[2] > got[0] and got[3] > got[1]


def test_a_full_image_box_spans_the_whole_range():
    assert xywh_to_norm1000({"x": 0, "y": 0, "w": 800, "h": 557}, 800, 557) == \
        [0.0, 0.0, 1000.0, 1000.0]


def test_width_is_not_read_as_a_second_corner():
    """The failure this module exists to prevent, asserted directly."""
    box = {"x": 400, "y": 300, "w": 100, "h": 50}
    x1, y1, x2, y2 = xywh_to_norm1000(box, 1000, 1000)
    assert (x2, y2) == (500.0, 350.0), "w/h must be added to x/y, not substituted"
    assert (x2 - x1, y2 - y1) == (100.0, 50.0)


def test_a_bare_sequence_is_refused_rather_than_guessed():
    """Our own boxes are corner form; silently reading one as xywh would be invisible."""
    with pytest.raises(RefChartQAError, match="corner form"):
        xywh_to_norm1000([400, 300, 100, 50], 1000, 1000)


@pytest.mark.parametrize("bad", [{"x": 1, "y": 2, "w": 3}, {}, None, "1,2,3,4", 42])
def test_unreadable_boxes_raise(bad):
    with pytest.raises(RefChartQAError):
        xywh_to_norm1000(bad, 100, 100)


def test_boxes_outside_the_image_are_dropped_not_clamped_to_a_sliver():
    """A clamped-to-zero box would be a guaranteed false positive in every AP score."""
    boxes = [{"x": 10, "y": 10, "w": 20, "h": 20}, {"x": 900, "y": 900, "w": 50, "h": 50}]
    assert len(boxes_to_norm1000(boxes, 100, 100)) == 1


def test_zero_sized_images_raise():
    with pytest.raises(RefChartQAError, match="positive"):
        xywh_to_norm1000({"x": 0, "y": 0, "w": 1, "h": 1}, 0, 100)


def test_split_aliases_are_applied_once():
    assert normalise_split("validation") == "val"
    assert normalise_split("val") == "val"
    assert normalise_split("train") == "train"


def test_refchartqa_row_becomes_a_record():
    row = {"id": "RefChartQA_human_val_0", "query": "What is the value for Nigeria?",
           "label": "43.54", "response": "...", "type": "human",
           "grounding_bboxes": [{"x": 276.0, "y": 277.0, "w": 60.0, "h": 23.0}]}
    rec = row_to_record(row, split="validation", image_path="a.png",
                        image_sha256="ab" * 32, image_size=(640, 386))
    assert rec.split == "val" and rec.source == "refchartqa"
    assert rec.question_kind == "human"
    assert rec.answer == "43.54"
    assert rec.boxes and len(rec.boxes) == 1
    assert rec.meta["n_boxes"] == 1


def test_an_unknown_question_type_raises():
    row = {"query": "q", "label": "a", "type": "synthetic-ish", "grounding_bboxes": []}
    with pytest.raises(RefChartQAError, match="unexpected type"):
        row_to_record(row, split="train", image_path="a.png", image_sha256="a" * 64,
                      image_size=(10, 10))


# ------------------------------------------------------------------------- chartqa


def test_archive_paths_match_the_layout_read_from_the_zip():
    assert qa_path("train", "human") == "ChartQA Dataset/train/train_human.json"
    assert qa_path("val", "machine") == "ChartQA Dataset/val/val_augmented.json"
    assert image_path("test", "10095.png") == "ChartQA Dataset/test/png/10095.png"
    assert table_path("train", "10095.png") == "ChartQA Dataset/train/tables/10095.csv"
    assert annotation_path("train", "10095.png") == \
        "ChartQA Dataset/train/annotations/10095.json"


def test_unknown_split_or_kind_raises():
    with pytest.raises(ChartQAError, match="unknown split"):
        qa_path("holdout", "human")
    with pytest.raises(ChartQAError, match="unknown question kind"):
        qa_path("train", "pot")


def test_gold_tables_parse_without_coercing_values():
    text = ('Country,"Projected share of the population in extreme poverty, 2023"\n'
            "Nigeria,43.54\nIndia,0.76\n\n")
    t = parse_table(text)
    assert t["columns"][0] == "Country"
    assert t["rows"] == [["Nigeria", "43.54"], ["India", "0.76"]]
    assert isinstance(t["rows"][0][1], str), "values must stay as written"


def _annotation():
    """Shaped exactly like a real ChartQA annotation (read from the archive)."""
    return {
        "type": "h_bar",
        "general_figure_info": {
            "x_axis": {"major_labels": {
                "bboxes": [{"x": 70, "y": 142, "w": 46, "h": 15},
                           {"x": 10, "y": 211, "w": 106, "h": 15}],
                "values": ["Nigeria", "Extreme fragility"]}},
        },
        "models": [{"name": "bars",
                    "x": ["Nigeria", "Extreme fragility"],
                    "y": [43.54, 31.44],
                    "bboxes": [{"x": 121, "y": 123, "w": 657.9, "h": 54.9},
                               {"x": 121, "y": 192, "w": 475.0, "h": 54.9}]}],
    }


def test_element_boxes_pair_with_their_values():
    els = annotation_boxes(_annotation(), 800, 600)
    assert [e["label"] for e in els] == ["Nigeria", "Extreme fragility"]
    assert [e["value"] for e in els] == [43.54, 31.44]
    assert els[0]["bbox"][0] == pytest.approx(1000 * 121 / 800)
    assert els[0]["bbox"][2] == pytest.approx(1000 * (121 + 657.9) / 800)


def test_a_model_with_mismatched_array_lengths_is_skipped_not_zipped_short():
    """Zipping short would attach boxes to the wrong values — silently."""
    ann = _annotation()
    ann["models"][0]["y"] = [43.54]
    assert annotation_boxes(ann, 800, 600) == []


def test_axis_labels_carry_their_boxes():
    got = axis_labels(_annotation(), 800, 600)
    assert [i["text"] for i in got["x_axis"]] == ["Nigeria", "Extreme fragility"]
    assert got["x_axis"][0]["bbox"][0] == pytest.approx(1000 * 70 / 800)


# --------------------------------------------------------------------------- dedup


def _rec(source, split="train", q="What is the median value?", sha="ab" * 32,
         boxes=None, plan=None, answer="7", table=None, rid=None):
    return ChartRecord(
        record_id=rid or f"{source}-{split}-{abs(hash((source, split, q, sha))) % 10 ** 8}",
        source=source, split=split, image_path=f"{sha[:8]}.png", image_sha256=sha,
        question=q, answer=answer, question_kind="human", table=table, boxes=boxes,
        plan=plan, meta={})


def test_the_same_question_on_two_charts_is_not_a_duplicate():
    """DECISIONS.md 0028: generic questions recur across charts."""
    a, b = _rec("chartqa", sha="11" * 32), _rec("chartqa", sha="22" * 32)
    out, report = deduplicate([a, b])
    assert len(out) == 2 and report.merges == 0


def test_duplicates_merge_rather_than_drop():
    a = _rec("chartqa", boxes=None, plan={"op": "lookup", "args": ["x"]})
    b = _rec("refchartqa", boxes=[[1, 2, 3, 4]], plan=None)
    out, report = deduplicate([a, b])
    assert len(out) == 1
    merged = out[0]
    assert merged.plan == {"op": "lookup", "args": ["x"]}, "the exact plan survives"
    assert merged.boxes == [[1, 2, 3, 4]], "the boxes survive"
    assert report.merges == 1 and report.boxes_gained == 1
    assert report.merged_pairs["chartqa + refchartqa"] == 1


def test_merging_is_order_independent():
    a = _rec("chartqa", boxes=[[1, 2, 3, 4]], plan=None, answer="7")
    b = _rec("refchartqa", boxes=[[5, 6, 7, 8]], plan={"op": "max", "args": []}, answer="7")
    forward = merge_pair(a, b)
    backward = merge_pair(b, a)
    assert forward.answer == backward.answer
    assert forward.plan == backward.plan
    assert sorted(map(tuple, forward.boxes)) == sorted(map(tuple, backward.boxes))
    assert forward.source == backward.source


def test_shuffling_the_input_does_not_change_the_merged_result():
    rng = random.Random(0)
    records = [_rec("chartqa", boxes=[[1, 2, 3, 4]]), _rec("refchartqa", boxes=[[5, 6, 7, 8]]),
               _rec("synthetic", plan={"op": "count", "args": []}),
               _rec("chartqa", sha="33" * 32)]
    base, _ = deduplicate(records)
    for _ in range(8):
        shuffled = records[:]
        rng.shuffle(shuffled)
        got, _ = deduplicate(shuffled)
        assert {r.key for r in got} == {r.key for r in base}
        by_key = {r.key: r for r in got}
        for r in base:
            assert sorted(map(tuple, by_key[r.key].boxes or [])) == \
                sorted(map(tuple, r.boxes or []))


def test_chartqa_answers_win_over_derived_datasets():
    a = _rec("chartqa", answer="43.54")
    b = _rec("refchartqa", answer="43.5")
    merged = merge_pair(a, b)
    assert merged.answer == "43.54" and merged.source == "chartqa"


def test_answer_conflicts_are_counted_not_hidden():
    report = DedupReport()
    merge_pair(_rec("chartqa", answer="7"), _rec("refchartqa", answer="9"), report)
    assert report.answer_conflicts == 1


def test_cross_split_collisions_are_refused_not_merged():
    """A shared key across splits is a leak. Merging it would hide rule 1's whole point."""
    with pytest.raises(ValueError, match="leak"):
        merge_pair(_rec("chartqa", split="train"), _rec("refchartqa", split="test"))
    out, report = deduplicate([_rec("chartqa", split="train"),
                               _rec("refchartqa", split="test")])
    assert len(out) == 2
    assert len(report.cross_split_collisions) == 1
    leaks = find_cross_split_leaks([_rec("chartqa", split="train"),
                                    _rec("refchartqa", split="test")])
    assert len(leaks) == 1
    assert next(iter(leaks.values()))["splits"] == ["test", "train"]


def test_union_boxes_deduplicates_within_tolerance_and_keeps_order():
    assert union_boxes([[1, 2, 3, 4]], [[1.4, 2.0, 3.0, 4.0]]) == [[1, 2, 3, 4]]
    assert union_boxes([[1, 2, 3, 4]], [[9, 9, 9, 9]]) == [[1, 2, 3, 4], [9, 9, 9, 9]]
    assert union_boxes(None, None) is None


def test_a_three_way_duplicate_merges_into_one():
    a = _rec("chartqa", boxes=[[1, 2, 3, 4]])
    b = _rec("refchartqa", boxes=[[5, 6, 7, 8]])
    c = _rec("synthetic", plan={"op": "count", "args": []})
    out, report = deduplicate([a, b, c])
    assert len(out) == 1 and report.merges == 2
    assert len(out[0].boxes) == 2 and out[0].plan is not None
    assert out[0].meta["merged_from"] == ["chartqa", "refchartqa", "synthetic"]


def test_the_report_reads_as_a_sentence():
    a, b = _rec("chartqa"), _rec("refchartqa", boxes=[[1, 2, 3, 4]])
    _, report = deduplicate([a, b])
    text = report.summary()
    assert "2 in -> 1 out" in text and "chartqa + refchartqa=1" in text


def test_dedup_key_is_stable_across_loaders():
    """The same chart read from the zip or from disk must produce the same key."""
    sha = "ab" * 32
    assert dedup_key(sha, "What is the median value?") == \
        dedup_key(sha, "  WHAT  IS   THE MEDIAN VALUE  ")
