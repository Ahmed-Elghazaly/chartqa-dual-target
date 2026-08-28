"""Monitoring must produce comparable curves and must never take the run down."""

from __future__ import annotations

import json

import pytest

from chartqa_dt.train.monitor import (
    MetricOutcome,
    MetricSample,
    evaluate_slice,
    make_metric_fn,
    score_generation,
)
from chartqa_dt.train.validate import make_evaluator

GOOD = json.dumps({
    "answerable": True,
    "evidence": [{"label": "2019", "value": 245, "unit": None, "bbox": [412, 180, 468, 640]},
                 {"label": "2018", "value": 210, "unit": None, "bbox": [330, 240, 386, 640]}],
    "plan": {"op": "difference", "args": ["2019", "2018"]},
    "model_answer": "35",
}, separators=(",", ":"))

BOXES = [[412, 180, 468, 640], [330, 240, 386, 640]]


class _Model:
    """Minimal stand-in that records the mode it was left in."""

    def __init__(self) -> None:
        self.training = True
        self.modes: list[bool] = []

    def eval(self) -> None:
        self.training = False

    def train(self, flag: bool = True) -> None:
        self.training = flag
        self.modes.append(flag)


class _Loaded:
    def __init__(self) -> None:
        self.model = _Model()
        self.processor = object()


def _items(n: int = 3) -> list[dict]:
    return [{"record_id": f"r{i}", "question": "q", "image": object(),
             "answer": "35", "boxes": BOXES} for i in range(n)]


@pytest.fixture
def gen(monkeypatch):
    """Replace generation. `calls` records what the monitor asked for."""
    calls: list[dict] = []

    def install(behaviour):
        def fake(loaded, question, image, *, mode="structured", max_new_tokens=None):
            calls.append({"mode": mode, "question": question})
            return behaviour(len(calls) - 1), 0.01, 10, False
        monkeypatch.setattr("chartqa_dt.eval.generate.generate_one", fake)
        return calls

    return install


class TestScoring:
    def test_a_correct_record_passes_all_four_metrics(self) -> None:
        s = score_generation("r", GOOD, "35", BOXES)
        assert (s.parsed, s.schema, s.roundtrip, s.answer_correct) == (True,) * 4

    def test_unparseable_output_fails_every_metric_rather_than_abstaining(self) -> None:
        s = score_generation("r", "I think it is 35.", "35", BOXES)
        assert not any((s.parsed, s.schema, s.roundtrip, s.answer_correct))
        assert s.pred_boxes == []

    def test_a_wrong_answer_still_counts_as_parsed(self) -> None:
        raw = GOOD.replace('"model_answer":"35"', '"model_answer":"999"')
        s = score_generation("r", raw, "35", BOXES)
        assert s.parsed and not s.answer_correct


class TestOutcome:
    def test_ap_ignores_items_that_have_no_ground_truth_boxes(self) -> None:
        """ChartQA items carry no boxes; counting them would sink AP for no reason."""
        grounded = MetricSample("a", pred_boxes=BOXES, gt_boxes=BOXES)
        ungrounded = MetricSample("b", pred_boxes=[[0, 0, 5, 5]], gt_boxes=[])
        assert MetricOutcome([grounded, ungrounded]).ap50() == pytest.approx(1.0)

    def test_ap_is_none_when_nothing_in_the_slice_is_grounded(self) -> None:
        assert MetricOutcome([MetricSample("b")]).ap50() is None

    def test_empty_outcome_reports_no_metrics_instead_of_dividing_by_zero(self) -> None:
        assert MetricOutcome().to_metrics() == {"metric_n": 0, "metric_seconds": 0.0,
                                                "metric_grounded_n": 0}

    def test_fractions_are_over_the_whole_slice(self) -> None:
        out = MetricOutcome([MetricSample("a", schema=True), MetricSample("b")])
        assert out.to_metrics()["schema_valid"] == pytest.approx(0.5)


class TestEvaluateSlice:
    def test_uses_the_training_prompt_not_the_zero_shot_one(self, gen) -> None:
        calls = gen(lambda i: GOOD)
        evaluate_slice(_Loaded(), _items(2))
        assert {c["mode"] for c in calls} == {"training"}

    def test_generation_failure_does_not_propagate(self, gen) -> None:
        def blow_up(i):
            if i == 1:
                raise RuntimeError("CUDA out of memory")
            return GOOD
        gen(blow_up)
        outcome = evaluate_slice(_Loaded(), _items(4))
        assert outcome.n == 1                       # the one that succeeded is kept
        assert "CUDA out of memory" in outcome.stopped_early

    def test_the_reason_reaches_the_reported_metrics(self, gen) -> None:
        gen(lambda i: (_ for _ in ()).throw(RuntimeError("boom")))
        assert "boom" in evaluate_slice(_Loaded(), _items(2)).to_metrics()[
            "metric_stopped_early"]

    def test_time_budget_stops_before_borrowing_from_training(self, gen) -> None:
        gen(lambda i: GOOD)
        outcome = evaluate_slice(_Loaded(), _items(50), time_budget_s=1e-9)
        assert outcome.n == 0 and "time budget" in outcome.stopped_early

    def test_training_mode_is_restored_even_when_generation_raises(self, gen) -> None:
        gen(lambda i: (_ for _ in ()).throw(RuntimeError("boom")))
        loaded = _Loaded()
        evaluate_slice(loaded, _items(2))
        assert loaded.model.training is True

    def test_an_eval_mode_model_is_left_in_eval_mode(self, gen) -> None:
        gen(lambda i: GOOD)
        loaded = _Loaded()
        loaded.model.training = False
        evaluate_slice(loaded, _items(1))
        assert loaded.model.training is False


class TestMetricFn:
    def test_the_slice_is_frozen_so_curves_are_comparable(self, gen) -> None:
        gen(lambda i: GOOD)
        items = _items(2)
        fn = make_metric_fn(items)
        items.append({"record_id": "late", "question": "q", "image": object(),
                      "answer": "35", "boxes": BOXES})
        assert fn(_Loaded(), 1000)["metric_n"] == 2

    def test_metrics_flow_through_the_validation_report(self, gen, monkeypatch) -> None:
        gen(lambda i: GOOD)
        monkeypatch.setattr("chartqa_dt.train.validate.validation_loss",
                            lambda *a, **k: (0.5, 8))
        loaded = _Loaded()
        evaluate = make_evaluator(loaded, [], max_len=1024,
                                  metric_fn=make_metric_fn(_items(2)), metric_every=1000)
        assert evaluate(loaded.model, 1000) == pytest.approx(-0.5)
        report = evaluate.reports[-1]
        assert report.ap50 == pytest.approx(1.0)
        assert report.answer_accuracy == pytest.approx(1.0)
        assert report.extra["metric_n"] == 2

    def test_metrics_are_not_computed_on_off_schedule_steps(self, gen, monkeypatch) -> None:
        calls = gen(lambda i: GOOD)
        monkeypatch.setattr("chartqa_dt.train.validate.validation_loss",
                            lambda *a, **k: (0.5, 8))
        loaded = _Loaded()
        evaluate = make_evaluator(loaded, [], max_len=1024,
                                  metric_fn=make_metric_fn(_items(2)), metric_every=1000)
        evaluate(loaded.model, 500)
        assert calls == [] and evaluate.reports[-1].ap50 is None
