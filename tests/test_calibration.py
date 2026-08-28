"""`PLAN.md` 9.6. The headline stays at 100% coverage; selection is a diagnostic."""

from __future__ import annotations

import math

import pytest

from chartqa_dt.eval.calibration import (
    agreement_confidence,
    assert_headline_is_full_coverage,
    brier_score,
    calibrate,
    coverage_curve,
    expected_calibration_error,
)


def _item(level: str, correct: bool, **kw) -> dict:
    base = {"agrees, clean": {"parsed": True, "schema_valid": True, "executes": True,
                              "agrees": True, "repairs": []},
            "agrees, repaired": {"parsed": True, "schema_valid": True, "executes": True,
                                 "agrees": True, "repairs": ["unwrapped"]},
            "executes but disagrees": {"parsed": True, "schema_valid": True,
                                       "executes": True, "agrees": False},
            "schema-valid only": {"parsed": True, "schema_valid": True, "executes": False},
            "parsed only": {"parsed": True, "schema_valid": False},
            "unusable": {"parsed": False}}[level]
    return {**base, "correct": correct, **kw}


class TestConfidenceLevels:
    @pytest.mark.parametrize("level", ["agrees, clean", "agrees, repaired",
                                       "executes but disagrees", "schema-valid only",
                                       "parsed only", "unusable"])
    def test_each_level_round_trips(self, level: str) -> None:
        assert agreement_confidence(_item(level, True)) == level

    def test_a_truncated_record_is_demoted_however_good_it_looks(self) -> None:
        """Truncation at the token cap means the record is cut off, not confident."""
        item = _item("agrees, clean", True, hit_token_cap=True)
        assert agreement_confidence(item) == "parsed only"


class TestCoverageCurve:
    def test_the_curve_always_ends_at_full_coverage(self) -> None:
        points = coverage_curve([_item("agrees, clean", True),
                                 _item("unusable", False)])
        assert points[-1].coverage == pytest.approx(1.0)
        assert points[-1].accuracy == pytest.approx(0.5)

    def test_accuracy_falls_as_coverage_grows_when_confidence_is_informative(self) -> None:
        items = ([_item("agrees, clean", True)] * 8
                 + [_item("unusable", False)] * 8)
        points = coverage_curve(items)
        assert points[0].accuracy == pytest.approx(1.0)
        assert points[0].coverage == pytest.approx(0.5)
        assert points[-1].accuracy == pytest.approx(0.5)

    def test_an_empty_set_produces_no_points_rather_than_dividing_by_zero(self) -> None:
        assert coverage_curve([]) == []

    def test_levels_with_no_members_are_skipped_not_repeated(self) -> None:
        points = coverage_curve([_item("agrees, clean", True)] * 3)
        assert len(points) == 1 and points[0].coverage == pytest.approx(1.0)


class TestProbabilisticScores:
    def test_brier_is_zero_for_a_perfect_confident_forecast(self) -> None:
        assert brier_score([1.0, 0.0], [True, False]) == pytest.approx(0.0)

    def test_brier_is_one_for_a_confident_wrong_forecast(self) -> None:
        assert brier_score([0.0, 1.0], [True, False]) == pytest.approx(1.0)

    def test_ece_is_zero_when_confidence_matches_accuracy(self) -> None:
        probs = [0.5] * 10
        outcomes = [True] * 5 + [False] * 5
        assert expected_calibration_error(probs, outcomes) == pytest.approx(0.0)

    def test_ece_ignores_empty_bins_rather_than_counting_them_as_perfect(self) -> None:
        """Averaging over all bins would drag ECE towards zero for a model that only ever
        emits two confidences."""
        probs = [0.95] * 10
        outcomes = [False] * 10
        assert expected_calibration_error(probs, outcomes) == pytest.approx(0.95)

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            brier_score([0.5], [True, False])

    def test_empty_input_is_nan_not_zero(self) -> None:
        assert math.isnan(brier_score([], []))
        assert math.isnan(expected_calibration_error([], []))


class TestCalibrate:
    def test_ece_and_brier_are_absent_when_no_probability_was_supplied(self) -> None:
        """This system emits a record, not a probability. Manufacturing one would measure
        the proxy rather than the model."""
        report = calibrate([_item("agrees, clean", True), _item("unusable", False)])
        assert report["ece"] is None and report["brier"] is None
        assert "not computed" in report["probability_note"]

    def test_they_are_computed_when_every_item_carries_one(self) -> None:
        items = [_item("agrees, clean", True, probability=0.9),
                 _item("unusable", False, probability=0.1)]
        report = calibrate(items)
        assert report["ece"] is not None and report["brier"] == pytest.approx(0.01)

    def test_a_partially_probabilistic_set_is_refused_rather_than_averaged(self) -> None:
        items = [_item("agrees, clean", True, probability=0.9),
                 _item("unusable", False)]
        assert calibrate(items)["ece"] is None

    def test_the_headline_is_the_full_coverage_accuracy(self) -> None:
        report = calibrate([_item("agrees, clean", True)] * 3 + [_item("unusable", False)])
        assert report["headline_accuracy"] == pytest.approx(0.75)


class TestTheHeadlineGuard:
    def test_the_full_coverage_number_is_accepted(self) -> None:
        report = calibrate([_item("agrees, clean", True)] * 3 + [_item("unusable", False)])
        assert_headline_is_full_coverage(report, 0.75)

    def test_a_selective_number_is_refused_and_told_where_it_came_from(self) -> None:
        report = calibrate([_item("agrees, clean", True)] * 3 + [_item("unusable", False)])
        with pytest.raises(AssertionError, match=r"75\.0% coverage"):
            assert_headline_is_full_coverage(report, 1.0)

    def test_an_unrelated_number_is_still_refused(self) -> None:
        report = calibrate([_item("agrees, clean", True)])
        with pytest.raises(AssertionError, match="full-coverage accuracy"):
            assert_headline_is_full_coverage(report, 0.42)
