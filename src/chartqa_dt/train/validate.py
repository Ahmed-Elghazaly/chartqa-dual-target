"""Validation during training — `PLAN.md` 6.5, and the signal 6.6 stops on.

**Why the early-stopping signal is validation loss and not AP.** 6.6 says *"stop if
validation AP has not improved for N evaluations"*. Measured against the cost of getting
AP at all, that is not affordable as a *stopping* signal:

| slice | every | evals in 3,000 steps | generation cost | AP 95% CI |
|---:|---:|---:|---:|---:|
| 64 | 500 | 6 | 0.9 h | **±12.2 pts** |
| 128 | 500 | 6 | 1.7 h | **±8.7 pts** |
| 400 | 500 | 6 | 5.3 h | ±4.9 pts |

Against roughly 10 h of training, anything under ±5 points costs more than half the run
again — and an AP with a ±8.7 interval cannot detect "has not improved". Stopping on it
would mean stopping on noise, which is the mistake `DECISIONS.md` 0062 was written about.

So the two roles are separated:

* **Early stopping uses validation loss.** It needs no generation — one forward pass per
  batch, the same computation training already does — so it is nearly free, and it is a
  low-variance signal over hundreds of tokens per example rather than one binary outcome.
  Because the target *contains the boxes*, it responds directly to grounding quality.
* **AP and answer accuracy are still measured**, at wider intervals, for the curves
  `PLAN.md` 6.5 asks for and for the report. They inform; they do not gate.

The deviation is recorded in `DECISIONS.md` 0069 with the numbers above.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: Held-out examples used for the loss signal. Cheap, so it can be generous.
LOSS_SLICE = 256
#: Generated examples for the monitoring metrics. Expensive, so it is small and infrequent.
METRIC_SLICE = 200
#: How often the generation-based metric runs. Loss is evaluated far more often and is what
#: early stopping uses (0069); this exists to watch the metric we actually care about drift
#: against the loss, not to decide anything. Every 1,000 steps is three times in a 3,000-step
#: run, which is enough to see a divergence and cheap enough not to matter.
METRIC_EVERY_STEPS = 1000


@dataclass
class ValidationReport:
    step: int
    loss: float
    n_loss: int
    ap50: float | None = None
    answer_accuracy: float | None = None
    schema_valid: float | None = None
    roundtrip: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "loss": self.loss, "n_loss": self.n_loss,
                "ap50": self.ap50, "answer_accuracy": self.answer_accuracy,
                "schema_valid": self.schema_valid, "roundtrip": self.roundtrip,
                **self.extra}

    def describe(self) -> str:
        parts = [f"step {self.step:>5}  val loss {self.loss:.4f} (n={self.n_loss})"]
        if self.ap50 is not None:
            parts.append(f"AP@0.5 {100 * self.ap50:.2f}%")
        if self.answer_accuracy is not None:
            parts.append(f"answer {100 * self.answer_accuracy:.2f}%")
        if self.roundtrip is not None:
            parts.append(f"round-trip {100 * self.roundtrip:.2f}%")
        return "  " + "   ".join(parts)


def validation_loss(loaded: Any, examples: Sequence[Any], *, max_len: int,
                    batch_size: int = 2) -> tuple[float, int]:
    """Mean loss over held-out examples. No generation, so it is nearly free.

    The model is put in eval mode and restored afterwards — leaving it in eval would
    disable dropout for the rest of training, which does not crash and does change the
    run.
    """
    import torch

    from chartqa_dt.train.collate import build_batch

    model = loaded.model
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total, counted = 0.0, 0
    try:
        with torch.inference_mode():
            for start in range(0, len(examples), batch_size):
                chunk = list(examples[start:start + batch_size])
                if not chunk:
                    continue
                batch = build_batch(loaded.processor, chunk, max_len, strict=False)
                batch.pop("_supervised_positions", None)
                batch = {k: (v.to(device) if hasattr(v, "to") else v)
                         for k, v in batch.items()}
                total += float(model(**batch).loss) * len(chunk)
                counted += len(chunk)
    finally:
        model.train(was_training)
    return (total / counted if counted else float("nan")), counted


def make_evaluator(loaded: Any, loss_examples: Sequence[Any], *, max_len: int,
                   metric_fn: Any = None, metric_every: int = METRIC_EVERY_STEPS,
                   on_report: Any = None):
    """Build the `evaluate` callback the training loop expects.

    Returns **negative** loss, because the loop's `EarlyStopping` maximises its metric and
    a falling loss is an improving model. Getting that sign wrong would stop the run at
    the first evaluation and look like immediate convergence.
    """
    reports: list[ValidationReport] = []

    def evaluate(model: Any, step: int) -> float:
        loss, n = validation_loss(loaded, loss_examples, max_len=max_len)
        report = ValidationReport(step=step, loss=loss, n_loss=n)
        if metric_fn is not None and metric_every and step % metric_every == 0:
            for key, value in (metric_fn(loaded, step) or {}).items():
                if hasattr(report, key):
                    setattr(report, key, value)
                else:
                    report.extra[key] = value
        reports.append(report)
        print(report.describe(), flush=True)
        if on_report is not None:
            on_report(report)
        return -loss

    evaluate.reports = reports          # type: ignore[attr-defined]
    return evaluate


__all__ = ["LOSS_SLICE", "METRIC_EVERY_STEPS", "METRIC_SLICE", "ValidationReport",
           "make_evaluator", "validation_loss"]
