"""Validation during training — `PLAN.md` 6.5, and the signal 6.6 stops on.

The sign convention is the thing most worth guarding. `EarlyStopping` maximises its
metric, and loss falls when a model improves, so the evaluator must return **negative**
loss. Getting that backwards stops the run at the first evaluation and looks exactly like
immediate convergence.
"""

from __future__ import annotations

import pytest

from chartqa_dt.train.checkpoint import EarlyStopping, TrainState
from chartqa_dt.train.validate import (
    LOSS_SLICE,
    METRIC_EVERY_STEPS,
    ValidationReport,
    make_evaluator,
    validation_loss,
)

torch = pytest.importorskip("torch")


class Stub(torch.nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.losses = iter(losses)
        self.eval_calls = 0
        self.mode_when_called = []

    def forward(self, **batch):
        self.mode_when_called.append(self.training)
        return type("Out", (), {"loss": torch.tensor(next(self.losses))})()


class Holder:
    def __init__(self, losses):
        self.model = Stub(losses)
        self.processor = object()


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr("chartqa_dt.train.collate.build_batch",
                        lambda p, ex, ml, **kw: {"input_ids": torch.ones(len(ex), 2,
                                                                        dtype=torch.long),
                                                 "_supervised_positions": 2})


def test_validation_loss_is_the_mean_over_examples(patched):
    holder = Holder([1.0, 3.0])
    loss, n = validation_loss(holder, [object()] * 4, max_len=64, batch_size=2)
    assert n == 4
    assert loss == pytest.approx(2.0), "two batches of two, weighted by size"


def test_the_model_is_put_in_eval_mode_and_restored(patched):
    """Leaving it in eval disables dropout for the rest of training — silently."""
    holder = Holder([1.0])
    holder.model.train(True)
    validation_loss(holder, [object()] * 2, max_len=64, batch_size=2)
    assert holder.model.mode_when_called == [False], "eval mode during validation"
    assert holder.model.training is True, "the caller's mode is restored"


def test_the_evaluator_returns_negative_loss(patched):
    """`EarlyStopping` maximises; loss falls when the model improves.

    The wrong sign stops the run at the first evaluation and looks like convergence.
    """
    holder = Holder([2.0, 1.0])
    evaluate = make_evaluator(holder, [object()], max_len=64)
    assert evaluate(holder.model, 100) == pytest.approx(-2.0)
    assert evaluate(holder.model, 200) == pytest.approx(-1.0)


def test_a_falling_loss_reads_as_improvement_to_early_stopping(patched):
    """The two components together, which is where a sign error would actually bite."""
    holder = Holder([3.0, 2.0, 1.0])
    evaluate = make_evaluator(holder, [object()], max_len=64)
    stopper, state = EarlyStopping(patience=2), TrainState()
    for step in (100, 200, 300):
        assert stopper.update(state, evaluate(holder.model, step), step) is False
    assert state.best_step == 300, "the lowest loss is the best checkpoint"


def test_a_rising_loss_triggers_the_stop(patched):
    holder = Holder([1.0, 2.0, 3.0])
    evaluate = make_evaluator(holder, [object()], max_len=64)
    stopper, state = EarlyStopping(patience=2), TrainState()
    results = [stopper.update(state, evaluate(holder.model, s), s)
               for s in (100, 200, 300)]
    assert results == [False, False, True]
    assert state.best_step == 100


def test_expensive_metrics_run_only_on_their_own_interval(patched):
    """AP costs generation; loss does not. They must not share a cadence."""
    holder = Holder([1.0] * 4)
    calls = []

    def metric_fn(loaded, step):
        calls.append(step)
        return {"ap50": 0.42, "answer_accuracy": 0.5, "unknown_key": 1}

    evaluate = make_evaluator(holder, [object()], max_len=64, metric_fn=metric_fn,
                              metric_every=200)
    for step in (100, 200, 300, 400):
        evaluate(holder.model, step)
    assert calls == [200, 400]
    reports = evaluate.reports
    assert reports[0].ap50 is None and reports[1].ap50 == pytest.approx(0.42)
    assert reports[1].extra["unknown_key"] == 1, "unrecognised keys are kept, not dropped"


def test_the_report_describes_what_it_has():
    bare = ValidationReport(step=10, loss=1.5, n_loss=100)
    assert "val loss 1.5000" in bare.describe() and "AP@0.5" not in bare.describe()
    full = ValidationReport(step=10, loss=1.5, n_loss=100, ap50=0.4,
                            answer_accuracy=0.6, roundtrip=0.7)
    text = full.describe()
    assert "AP@0.5 40.00%" in text and "round-trip 70.00%" in text


def test_the_slice_sizes_reflect_the_cost_measurement():
    """`DECISIONS.md` 0069: loss is nearly free so its slice is generous; AP needs
    generation so it is small and infrequent."""
    assert LOSS_SLICE >= 256
    assert METRIC_EVERY_STEPS >= 1000


def test_an_empty_validation_set_does_not_divide_by_zero(patched):
    loss, n = validation_loss(Holder([]), [], max_len=64)
    assert n == 0 and loss != loss, "NaN, not a crash and not a fake zero"
