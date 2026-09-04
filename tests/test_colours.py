"""Naming a chart's colours the way the people asking the questions did.

21.8% of human-written ChartQA questions mention a colour against 0.5% of machine ones, and
human questions are half the headline metric. Every annotation carries the colour and nothing
read it. These tests pin the decisions that took matching from 27.5% to 61.8%.
"""
from __future__ import annotations

import pytest

from chartqa_dt.data.colours import distinct_palette, mentioned_in, names_for, series_colour


@pytest.mark.parametrize("colour,expected", [
    ("#2876dd", "blue"),
    ("#bababa", "grey"),
    ("#4a9b4f", "green"),
    ("#f5c542", "yellow"),
])
def test_a_hex_gets_the_obvious_word(colour, expected):
    assert expected in names_for(colour)


def test_a_very_dark_colour_answers_to_black():
    """Measured against real questions: `#0f283e` is lightness 0.15 and people call it black
    -- *"the black line"*, *"the highest black casual fan"*. Naming it only navy lost those."""
    words = names_for("#0f283e")
    assert {"black", "navy", "dark blue", "blue"} <= words


def test_a_colour_already_written_as_words_is_kept():
    """ChartQA stores `color` as plain English on line, h_bar and pie charts. Running those
    through a hex parser returned nothing, which is where 58.3% of the misses came from."""
    assert "dark blue" in names_for("dark blue")
    assert next(iter(names_for("orange"))) == "orange"


def test_a_qualified_name_still_answers_to_its_bare_hue():
    """*"the blue bar"* must find a series the annotation calls 'dark blue'."""
    assert "blue" in names_for("dark blue")
    assert "green" in names_for("light green")


def test_both_annotation_shapes_are_read():
    """`colors` (a list of hex, on v_bar) and `color` (singular, on line/h_bar/pie). Reading
    only the first discarded most of the corpus."""
    listed = {"colors": ["#2876dd", "#2876dd", "#0f283e"], "name": "A"}
    singular = {"color": "dark blue", "name": "B"}
    assert distinct_palette([listed]) == ["#2876dd", "#0f283e"]
    assert distinct_palette([singular]) == ["dark blue"]
    assert distinct_palette([listed, singular]) == ["#2876dd", "#0f283e", "dark blue"]


def test_series_colour_reads_the_singular_field_too():
    assert series_colour({"color": "orange"}) == "orange"
    assert series_colour({"colors": ["#111111", "#111111"]}) == "#111111"
    assert series_colour({"colors": ["#111111", "#222222"]}) is None, \
        "a series drawn in two colours has no single colour"


# ------------------------------------------------- matching a question to a colour


def test_a_longer_name_wins_over_a_shorter_one():
    """*"dark blue"* must not be read as *"blue"*, which would select every blue on the
    chart instead of the dark one."""
    palette = ["#2876dd", "#0f283e"]
    assert mentioned_in("What is the dark blue bar value?", palette) == {"#0f283e"}


def test_a_colour_word_inside_a_label_is_not_a_colour_reference():
    """*"the difference between the highest value of lemon oil and lowest of orange oil"* is
    not asking about anything orange."""
    palette = ["#e8811a"]           # a genuine orange
    assert mentioned_in("difference between lemon oil and orange oil", palette,
                        labels=["lemon oil", "orange oil"]) == set()
    assert mentioned_in("what is the orange bar worth", palette,
                        labels=["lemon oil", "orange oil"]) == {"#e8811a"}


def test_a_question_with_no_colour_matches_nothing():
    assert mentioned_in("How many users did Nigeria have?", ["#2876dd"]) == set()


def test_an_unusable_colour_is_ignored_rather_than_guessed():
    """Some annotations carry the literal string 'unk'."""
    assert names_for("unk") == {"unk"} or "unk" in names_for("unk")
    assert mentioned_in("what is the blue bar", ["unk"]) == set()


# ------------------------------------------------- colour reaching the elements themselves


def _chart(models, kind="v_bar"):
    from chartqa_dt.data.chartqa import annotation_boxes
    return annotation_boxes({"type": kind, "models": models}, 100, 100)


BOXES = [{"x": 0, "y": 0, "w": 10, "h": 50}, {"x": 20, "y": 0, "w": 10, "h": 40}]


def test_a_per_datapoint_colour_list_lands_on_each_element():
    """`colors` is one entry per datapoint on v_bar, because colour often distinguishes
    categories WITHIN a series -- which is the chart *"the blue bar"* is asked about."""
    els = _chart([{"name": "S", "bboxes": BOXES, "x": ["a", "b"], "y": [1, 2],
                   "colors": ["#2876dd", "#0f283e"]}])
    assert [e["colour"] for e in els] == ["#2876dd", "#0f283e"]


def test_a_single_series_colour_is_given_to_every_element():
    els = _chart([{"name": "S", "bboxes": BOXES, "x": ["a", "b"], "y": [1, 2],
                   "color": "dark blue"}])
    assert [e["colour"] for e in els] == ["dark blue", "dark blue"]


def test_a_short_colour_list_pads_rather_than_zipping_short():
    """A wrong colour points at the wrong mark; a missing one only declines to answer."""
    els = _chart([{"name": "S", "bboxes": BOXES, "x": ["a", "b"], "y": [1, 2],
                   "colors": ["#2876dd"]}])
    assert [e["colour"] for e in els] == ["#2876dd", None]


def test_the_literal_unk_is_not_treated_as_a_colour():
    """Some annotations carry the string 'unk'. It is not a colour."""
    els = _chart([{"name": "S", "bboxes": BOXES, "x": ["a", "b"], "y": [1, 2],
                   "color": "unk"}])
    assert [e["colour"] for e in els] == [None, None]


def test_a_chart_with_no_colour_still_produces_elements():
    """Colour is an addition; its absence must not cost us the boxes."""
    els = _chart([{"name": "S", "bboxes": BOXES, "x": ["a", "b"], "y": [1, 2]}])
    assert len(els) == 2
    assert all(e["colour"] is None for e in els)
    assert all(e["bbox"] for e in els)


def test_a_pie_wedge_carries_its_colour():
    els = _chart([{"text_label": "Pension funds", "value": "29.0", "color": "#2876dd",
                   "bbox": {"x": 0, "y": 0, "w": 10, "h": 10}}], kind="pie")
    assert els and els[0]["colour"] == "#2876dd"
