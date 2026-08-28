"""`PLAN.md` 8.2. The calibrator uses what can be observed, never what the model claims."""

from __future__ import annotations

import json
import math
import random

import pytest

from chartqa_dt.eval.calibrator import (
    DEFAULT_FEATURES,
    SelfReportedFeature,
    assert_features_are_observable,
    auc,
    evaluate,
    fit,
)


def _rows(n: int, *, separable: bool, seed: int = 0):
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n):
        correct = rng.random() < 0.5
        if separable:
            mean_lp = rng.gauss(-0.3 if correct else -1.4, 0.35)
            agrees = float(correct or rng.random() < 0.15)
        else:
            mean_lp = rng.gauss(-0.8, 0.35)
            agrees = float(rng.random() < 0.5)
        rows.append({"mean_logprob": mean_lp, "min_logprob": mean_lp * 3,
                     "schema_valid": agrees, "executor_agrees": agrees,
                     "evidence_area": rng.gauss(40000, 12000),
                     "unit_ok": 1.0, "range_ok": 1.0, "n_evidence": rng.randint(1, 4)})
        labels.append(correct)
    return rows, labels


class TestForbiddenFeatures:
    def test_self_declared_confidence_is_refused(self) -> None:
        """A calibrator fitted on the model's own claim measures its self-image."""
        with pytest.raises(SelfReportedFeature, match="self-image"):
            assert_features_are_observable([*DEFAULT_FEATURES, "confidence"])

    @pytest.mark.parametrize("name", ["certainty", "model_confidence", "sureness",
                                      "stated_confidence"])
    def test_every_spelling_of_it_is_refused(self, name: str) -> None:
        with pytest.raises(SelfReportedFeature):
            assert_features_are_observable([name])

    def test_the_check_is_case_insensitive(self) -> None:
        with pytest.raises(SelfReportedFeature):
            assert_features_are_observable(["Confidence"])

    def test_the_default_feature_set_passes(self) -> None:
        assert_features_are_observable(DEFAULT_FEATURES)

    def test_fitting_with_a_forbidden_feature_is_refused(self) -> None:
        rows, labels = _rows(20, separable=True)
        with pytest.raises(SelfReportedFeature):
            fit(rows, labels, features=("confidence",))


class TestFitting:
    def test_it_refuses_to_fit_on_anything_but_validation(self) -> None:
        """Fitting on test would be fitting to the answer."""
        rows, labels = _rows(20, separable=True)
        with pytest.raises(ValueError, match="fitting to the answer"):
            fit(rows, labels, split="test")

    def test_mismatched_lengths_are_refused(self) -> None:
        rows, labels = _rows(20, separable=True)
        with pytest.raises(ValueError, match="same length"):
            fit(rows, labels[:5])

    def test_an_empty_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nothing to fit"):
            fit([], [])

    def test_a_constant_feature_does_not_produce_nan_weights(self) -> None:
        """Dividing by a zero standard deviation would give a calibrator that silently
        predicts nothing at all."""
        rows = [{"unit_ok": 1.0, "mean_logprob": v} for v in (-1.0, -0.2, -0.7, -0.4)]
        cal = fit(rows, [False, True, False, True],
                  features=("unit_ok", "mean_logprob"))
        assert all(math.isfinite(w) for w in cal.weights)
        assert all(math.isfinite(p) for p in cal.predict(rows))

    def test_predictions_are_probabilities(self) -> None:
        rows, labels = _rows(200, separable=True)
        cal = fit(rows, labels)
        assert all(0.0 <= p <= 1.0 for p in cal.predict(rows))

    def test_an_unfitted_calibrator_refuses_to_predict(self) -> None:
        from chartqa_dt.eval.calibrator import Calibrator

        with pytest.raises(RuntimeError, match="not fitted"):
            Calibrator(feature_names=["a"]).predict([{"a": 1.0}])

    def test_the_fitted_model_is_json_serialisable(self) -> None:
        """A numpy scalar survives arithmetic but not json.dumps, and this goes in a
        results file."""
        rows, labels = _rows(60, separable=True)
        assert json.dumps(fit(rows, labels).to_dict())

    def test_the_fit_is_deterministic(self) -> None:
        rows, labels = _rows(120, separable=True)
        assert fit(rows, labels).weights == fit(rows, labels).weights


class TestAuc:
    def test_a_perfect_ranking_scores_one(self) -> None:
        assert auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0

    def test_a_reversed_ranking_scores_zero(self) -> None:
        assert auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0

    def test_a_constant_predictor_scores_exactly_a_half(self) -> None:
        """Ties count a half; without that a constant predictor lands at 0 or 1."""
        assert auc([0.5] * 4, [True, True, False, False]) == 0.5

    def test_one_sided_labels_are_undefined_rather_than_zero(self) -> None:
        assert math.isnan(auc([0.9, 0.1], [True, True]))


class TestSkipTrigger:
    def test_a_separating_calibrator_does_not_trigger_the_skip(self) -> None:
        rows, labels = _rows(400, separable=True)
        report = evaluate(fit(rows[:300], labels[:300]), rows[300:], labels[300:])
        assert report["auc"] > 0.8
        assert report["separates"] and not report["skip_crop"]

    def test_an_uninformative_calibrator_skips_the_crop_and_says_why(self) -> None:
        """8.2's skip trigger: gating on a signal that carries no information is worse
        than not gating."""
        rows, labels = _rows(400, separable=False, seed=3)
        report = evaluate(fit(rows[:300], labels[:300]), rows[300:], labels[300:])
        assert report["skip_crop"]
        assert "does not separate" in report["skip_reason"]

    def test_the_report_supplies_the_ece_that_9_6_could_not_compute(self) -> None:
        """calibration.py reports ECE as not computed because the system emits a record,
        not a probability. A fitted calibrator emits one."""
        rows, labels = _rows(300, separable=True)
        report = evaluate(fit(rows[:200], labels[:200]), rows[200:], labels[200:])
        assert 0.0 <= report["ece"] <= 1.0 and 0.0 <= report["brier"] <= 1.0

    def test_the_report_is_json_serialisable(self) -> None:
        rows, labels = _rows(200, separable=True)
        assert json.dumps(evaluate(fit(rows, labels), rows, labels))
