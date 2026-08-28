"""Generation-based validation metrics during training — the curves `PLAN.md` 6.5 asks for.

`validate.py` explains why the *stopping* signal is validation loss and not AP. This
module supplies the other half: AP@0.5, answer accuracy, schema validity and round-trip
agreement, measured by actually generating, so the report can show what the model was
doing at each point in the run rather than only that its loss fell.

Three properties matter more here than accuracy of any single number.

**It must never kill the run.** This is monitoring attached to a ten-hour job that costs
real GPU quota. Generation during training is the most likely thing in the loop to hit
CUDA out-of-memory — the optimizer state and activations are already resident and
`generate` then asks for a KV cache on top. So every failure is caught, the partial result
is kept, and the reason is reported. A metric that takes the run down with it is worse
than no metric.

**It must be comparable across steps.** The slice is fixed and consumed in order, never
reshuffled, so a change in the curve is a change in the model.

**It must cost what it says.** Measured zero-shot latency is 6.91–6.97 s/item
(`verification/measured_facts.json`, `prompt_iteration`), so the default slice of 200 is
about 23 minutes per evaluation, three times in a 3,000-step run — roughly 1.2 h on top of
about 10 h. `time_budget_s` caps that: when it is spent the evaluation stops early and
reports the `n` it actually reached, rather than quietly borrowing from training.

**Boxes are scored in 0–1000 space**, which is where both sides already live: records
carry `0-1000 normalised [x1,y1,x2,y2]`, the prompt asks for "four integers 0-999", and
the official evaluator bins to 0–999 itself. No conversion happens here, and none should:
converting one side only is how AP silently reads zero.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from chartqa_dt.eval.metrics import average_precision_coco, relaxed_correctness
from chartqa_dt.plans.roundtrip import check_record
from chartqa_dt.prompting.parsing import coerce_boxes, parse_record, schema_ok

#: The prompt the model is *trained* on. Monitoring with the long zero-shot prompt would
#: measure a distribution the fine-tuned model never sees.
MONITOR_MODE = "training"
#: Stop an evaluation that is running long instead of borrowing from the training budget.
DEFAULT_TIME_BUDGET_S = 30 * 60.0


@dataclass
class MetricSample:
    """What one generated item contributed, kept so a curve point can be explained."""

    record_id: str
    parsed: bool = False
    schema: bool = False
    roundtrip: bool = False
    answer_correct: bool = False
    pred_boxes: list[list[float]] = field(default_factory=list)
    gt_boxes: list[list[float]] = field(default_factory=list)


@dataclass
class MetricOutcome:
    samples: list[MetricSample] = field(default_factory=list)
    seconds: float = 0.0
    stopped_early: str = ""

    @property
    def n(self) -> int:
        return len(self.samples)

    def _fraction(self, attr: str) -> float | None:
        if not self.samples:
            return None
        return sum(bool(getattr(s, attr)) for s in self.samples) / len(self.samples)

    def ap50(self) -> float | None:
        """AP over the items that actually carry ground-truth boxes.

        ChartQA items have no boxes. Including them would put every prediction on them
        into the false-positive pile and drag AP down for a reason that has nothing to do
        with grounding quality, so they are excluded from this metric and kept in the
        answer metric where they belong.
        """
        grounded = [s for s in self.samples if s.gt_boxes]
        if not grounded:
            return None
        gts: dict[str, list[list[float]]] = {}
        preds: list[tuple[str, float, list[float]]] = []
        for i, s in enumerate(grounded):
            key = f"{i}:{s.record_id}"
            gts[key] = s.gt_boxes
            preds.extend((key, 1.0, b) for b in s.pred_boxes)
        return average_precision_coco(preds, gts, 0.5)

    def to_metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ap50": self.ap50(),
            "answer_accuracy": self._fraction("answer_correct"),
            "schema_valid": self._fraction("schema"),
            "roundtrip": self._fraction("roundtrip"),
            "metric_n": self.n,
            "metric_seconds": round(self.seconds, 1),
            "metric_grounded_n": sum(1 for s in self.samples if s.gt_boxes),
        }
        if self.stopped_early:
            out["metric_stopped_early"] = self.stopped_early
        return {k: v for k, v in out.items() if v is not None}


def score_generation(record_id: str, raw: str, gold_answer: str,
                     gt_boxes: Sequence[Sequence[float]] | None) -> MetricSample:
    """Turn one raw generation into its four monitoring outcomes.

    An unparseable generation is a *failure of every metric*, not a missing value —
    `PLAN.md`'s rule 3 counts an invalid output as a failure, and averaging only over the
    records that parsed would make the curve rise as the model emitted fewer of them.
    """
    sample = MetricSample(record_id=record_id,
                          gt_boxes=[list(b) for b in (gt_boxes or [])])
    result = parse_record(raw)
    if not result.ok or result.record is None:
        return sample
    record = result.record
    sample.parsed = True
    sample.schema = schema_ok(record)[0]
    sample.pred_boxes = coerce_boxes(record)
    sample.roundtrip = check_record(record).outcome == "agrees"
    sample.answer_correct = relaxed_correctness(gold_answer,
                                                str(record.get("model_answer", "")))
    return sample


def evaluate_slice(loaded: Any, items: Sequence[dict[str, Any]], *,
                   mode: str = MONITOR_MODE,
                   time_budget_s: float = DEFAULT_TIME_BUDGET_S,
                   max_new_tokens: int | None = None) -> MetricOutcome:
    """Generate over a fixed slice and score it, surviving anything that goes wrong.

    Each item supplies `record_id`, `question`, `image`, `answer`, and optionally `boxes`.
    """
    from chartqa_dt.eval.generate import generate_one

    outcome = MetricOutcome()
    model = loaded.model
    was_training = model.training
    model.eval()
    started = time.perf_counter()
    try:
        for item in items:
            elapsed = time.perf_counter() - started
            if time_budget_s and elapsed >= time_budget_s:
                outcome.stopped_early = (
                    f"time budget {time_budget_s:.0f}s spent after {outcome.n} items")
                break
            try:
                raw, _, _, _ = generate_one(loaded, item["question"], item["image"],
                                            mode=mode, max_new_tokens=max_new_tokens)
            except Exception as exc:                      # noqa: BLE001 — see docstring
                outcome.stopped_early = f"{type(exc).__name__} after {outcome.n} items: {exc}"
                break
            outcome.samples.append(score_generation(
                item["record_id"], raw, str(item.get("answer", "")), item.get("boxes")))
    finally:
        model.train(was_training)
        outcome.seconds = time.perf_counter() - started
    return outcome


def make_metric_fn(items: Sequence[dict[str, Any]], *, mode: str = MONITOR_MODE,
                   time_budget_s: float = DEFAULT_TIME_BUDGET_S,
                   on_outcome: Any = None):
    """The `metric_fn` `validate.make_evaluator` expects: `(loaded, step) -> dict`.

    The slice is bound once here rather than re-derived per call, which is what makes the
    curve comparable across steps.
    """
    frozen = list(items)

    def metric_fn(loaded: Any, step: int) -> dict[str, Any]:
        outcome = evaluate_slice(loaded, frozen, mode=mode, time_budget_s=time_budget_s)
        if outcome.stopped_early:
            print(f"    monitoring stopped early at step {step}: {outcome.stopped_early}",
                  flush=True)
        if on_outcome is not None:
            on_outcome(step, outcome)
        return outcome.to_metrics()

    metric_fn.items = frozen                              # type: ignore[attr-defined]
    return metric_fn


__all__ = ["DEFAULT_TIME_BUDGET_S", "MONITOR_MODE", "MetricOutcome", "MetricSample",
           "evaluate_slice", "make_metric_fn", "score_generation"]
