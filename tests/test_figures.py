"""`PLAN.md` 9.7. Boxes drawn in the right space, and no dataset derivative committed."""

from __future__ import annotations

import pytest
from PIL import Image

from chartqa_dt.eval.figures import (
    FAILURE_MODES,
    Figure,
    LicenceRefusal,
    classify_failure,
    coverage_report,
    draw_boxes,
    select_figures,
    to_pixels,
    write_figure,
)


class TestCoordinateConversion:
    def test_a_full_frame_box_covers_the_whole_image(self) -> None:
        assert to_pixels([0, 0, 1000, 1000], 800, 600) == (0, 0, 800, 600)

    def test_conversion_is_anisotropic(self) -> None:
        """x scales by width and y by height. Scaling both by one dimension would draw
        every box in the wrong place on a non-square chart — and a grounding figure that
        lies is worse than no figure."""
        assert to_pixels([500, 500, 1000, 1000], 800, 600) == (400, 300, 800, 600)

    def test_a_quarter_box_lands_where_it_should(self) -> None:
        assert to_pixels([250, 250, 750, 750], 400, 200) == (100, 50, 300, 150)


class TestDrawing:
    @staticmethod
    def _chart() -> Image.Image:
        return Image.new("RGB", (100, 80), (255, 255, 255))

    def test_drawing_changes_the_image_and_leaves_the_original_alone(self) -> None:
        original = self._chart()
        before = list(original.get_flattened_data())
        out = draw_boxes(original, [[100, 100, 500, 500]])
        assert list(out.get_flattened_data()) != before
        assert list(original.get_flattened_data()) == before, "the input must not be edited"

    def test_gold_and_predicted_boxes_are_distinguishable(self) -> None:
        from chartqa_dt.eval.figures import GOLD_COLOUR, PRED_COLOUR

        out = draw_boxes(self._chart(), [[100, 100, 300, 300]], [[600, 600, 900, 900]])
        pixels = set(out.get_flattened_data())
        assert GOLD_COLOUR in pixels and PRED_COLOUR in pixels

    def test_an_image_with_no_boxes_is_returned_unmarked(self) -> None:
        original = self._chart()
        assert list(draw_boxes(original, []).get_flattened_data()) == \
            list(original.get_flattened_data())


class TestFailureModes:
    @pytest.mark.parametrize(("item", "expected"), [
        ({"parsed": False}, "invalid_record"),
        ({"parsed": True, "correct": True}, "correct"),
        ({"parsed": True, "executes": True, "agrees": False}, "executor_disagrees"),
        ({"parsed": True, "answerable_wrong": True}, "unanswerable_missed"),
        ({"parsed": True, "operands_exact": False}, "wrong_evidence"),
        ({"parsed": True}, "wrong_operation"),
    ])
    def test_each_mode_is_recognised(self, item: dict, expected: str) -> None:
        assert classify_failure(item) == expected

    def test_a_correct_record_is_never_called_a_failure(self) -> None:
        assert classify_failure({"parsed": True, "correct": True,
                                 "operands_exact": False}) == "correct"

    def test_every_mode_has_a_human_readable_caption(self) -> None:
        assert set(FAILURE_MODES) >= {"wrong_evidence", "wrong_operation",
                                      "executor_disagrees", "invalid_record", "correct"}


class TestLicenceGuard:
    """Rule 7. A chart with boxes drawn on it is a derivative of the chart."""

    def test_a_real_chart_cannot_be_written_into_the_repository(self, tmp_path) -> None:
        figure = Figure(record_id="cq1", source="chartqa", mode="wrong_evidence")
        with pytest.raises(LicenceRefusal, match=r"GPL-3\.0"):
            write_figure(figure, Image.new("RGB", (8, 8)), tmp_path)

    def test_refchartqa_is_refused_too(self, tmp_path) -> None:
        figure = Figure(record_id="rc1", source="refchartqa", mode="correct")
        with pytest.raises(LicenceRefusal, match=r"AGPL-3\.0"):
            write_figure(figure, Image.new("RGB", (8, 8)), tmp_path)

    def test_a_synthetic_chart_is_ours_and_may_be_written(self, tmp_path) -> None:
        figure = Figure(record_id="s1", source="synthetic", mode="correct")
        path = write_figure(figure, Image.new("RGB", (8, 8)), tmp_path)
        assert path.is_file() and figure.path == path

    def test_a_real_chart_may_be_written_outside_the_repository(self, tmp_path) -> None:
        """The check is about version control, not about rendering."""
        figure = Figure(record_id="cq1", source="chartqa", mode="correct")
        assert write_figure(figure, Image.new("RGB", (8, 8)), tmp_path,
                            inside_repo=False).is_file()


class TestSelection:
    def _items(self) -> list[dict]:
        return [
            {"id": "a", "source": "chartqa", "parsed": True, "correct": True},
            {"id": "b", "source": "synthetic", "parsed": True, "correct": True},
            {"id": "c", "source": "synthetic", "parsed": False},
            {"id": "d", "source": "synthetic", "parsed": True, "executes": True,
             "agrees": False},
            {"id": "e", "source": "synthetic", "parsed": True, "operands_exact": False},
        ]

    def test_committable_sources_are_preferred(self) -> None:
        """Eight figures that cannot go in the repository are not eight figures."""
        chosen = select_figures(self._items(), per_mode=1)
        correct = [f for f in chosen if f.mode == "correct"]
        assert correct[0].source == "synthetic"

    def test_each_mode_is_represented_at_most_per_mode_times(self) -> None:
        chosen = select_figures(self._items(), per_mode=1)
        assert len({f.mode for f in chosen}) == len(chosen)

    def test_the_coverage_report_says_when_the_criterion_is_not_met(self) -> None:
        report = coverage_report(select_figures(self._items(), per_mode=1))
        assert report["meets_criterion"] is False
        assert "wrong_operation" in report["missing_modes"]

    def test_eight_figures_over_four_modes_meets_the_criterion(self) -> None:
        figures = [Figure(record_id=f"r{i}", source="synthetic",
                          mode=list(FAILURE_MODES)[i % 4]) for i in range(8)]
        assert coverage_report(figures)["meets_criterion"] is True
