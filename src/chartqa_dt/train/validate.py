from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from chartqa_dt.eval.metrics import average_precision_coco, relaxed_correctness
from chartqa_dt.model.parsing import coerce_boxes, parse_record, schema_ok
from chartqa_dt.plans.roundtrip import check_record

LOSS_SLICE = 256
METRIC_SLICE = 200
TIME_BUDGET_S = 30 * 60.0


@dataclass
class MetricSample:
    record_id: str
    parsed: bool = False
    schema: bool = False
    roundtrip: bool = False
    answer_correct: bool = False
    pred_boxes: list[list[float]] = field(default_factory=list)
    gt_boxes: list[list[float]] = field(default_factory=list)


def score_generation(record_id: str, raw: str, gold_answer: str,
                     gt_boxes: Sequence[Sequence[float]] | None) -> MetricSample:
    sample = MetricSample(record_id=record_id, gt_boxes=[list(b) for b in (gt_boxes or [])])
    result = parse_record(raw)
    if not result.ok or result.record is None:
        return sample
    record = result.record
    sample.parsed = True
    sample.schema = schema_ok(record)[0]
    sample.pred_boxes = coerce_boxes(record)
    sample.roundtrip = check_record(record).outcome == "agrees"
    sample.answer_correct = relaxed_correctness(gold_answer, str(record.get("model_answer", "")))
    return sample


def generated_metrics(loaded: Any, items: Sequence[dict[str, Any]], *, mode: str,
                      batch_size: int) -> dict[str, Any]:
    from chartqa_dt.eval.generate import generate_batch

    model = loaded.model
    was_training = model.training
    model.eval()
    samples: list[MetricSample] = []
    started = time.perf_counter()
    stopped = ""
    try:
        for start in range(0, len(items), batch_size):
            if time.perf_counter() - started >= TIME_BUDGET_S:
                stopped = f"time budget spent after {len(samples)} items"
                break
            chunk = list(items[start:start + batch_size])
            outputs = generate_batch(loaded, [i["question"] for i in chunk],
                                     [i["image"] for i in chunk], mode=mode)
            for item, (raw, _tokens, _capped) in zip(chunk, outputs):
                samples.append(score_generation(item["record_id"], raw, str(item["answer"]), item["boxes"]))
    finally:
        model.train(was_training)

    grounded = [s for s in samples if s.gt_boxes]
    gts: dict[str, list[list[float]]] = {}
    preds: list[tuple[str, float, list[float]]] = []
    for i, s in enumerate(grounded):
        key = f"{i}:{s.record_id}"
        gts[key] = s.gt_boxes
        preds.extend((key, 1.0, b) for b in s.pred_boxes)
    n = len(samples) or 1
    metrics: dict[str, Any] = {
        "answer_accuracy": sum(s.answer_correct for s in samples) / n,
        "ap50": average_precision_coco(preds, gts, 0.5) if grounded else None,
        "schema_valid": sum(s.schema for s in samples) / n,
        "roundtrip": sum(s.roundtrip for s in samples) / n,
        "metric_n": len(samples),
        "metric_seconds": round(time.perf_counter() - started, 1),
    }
    if stopped or len(samples) != len(items):
        metrics["metric_stopped_early"] = stopped or "incomplete"
        metrics["answer_accuracy"] = metrics["ap50"] = None
    return metrics


def validation_loss(loaded: Any, examples: Sequence[Any], *, max_len: int, batch_size: int,
                    prompt_builder: Callable[[str], str]) -> float:
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
                batch, _ = build_batch(loaded.processor, chunk, max_len, prompt_builder=prompt_builder)
                batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
                total += float(model(**batch).loss) * len(chunk)
                counted += len(chunk)
    finally:
        model.train(was_training)
    return total / counted


def make_evaluator(loaded: Any, loss_examples: Sequence[Any], metric_items: Sequence[dict[str, Any]], *,
                   max_len: int, loss_batch_size: int, metric_batch_size: int, metric_every: int,
                   final_step: int, mode: str, prompt_builder: Callable[[str], str],
                   on_report: Callable[[dict[str, Any]], None]) -> Callable[[Any, int], float | None]:
    def evaluate(model: Any, step: int) -> float | None:
        report: dict[str, Any] = {
            "step": step,
            "loss": validation_loss(loaded, loss_examples, max_len=max_len, batch_size=loss_batch_size,
                                    prompt_builder=prompt_builder),
        }
        selection = None
        if step % metric_every == 0 or step >= final_step:
            report.update(generated_metrics(loaded, metric_items, mode=mode, batch_size=metric_batch_size))
            if report["ap50"] is None or report["answer_accuracy"] is None:
                on_report(report)
                raise RuntimeError("checkpoint selection needs complete answer accuracy and AP@0.5")
            selection = (report["ap50"] + report["answer_accuracy"]) / 2.0
            report["selection_metric"] = selection
        parts = [f"step {step:>5}  val loss {report['loss']:.4f}"]
        if selection is not None:
            parts.append(f"AP@0.5 {100 * report['ap50']:.2f}%  answer {100 * report['answer_accuracy']:.2f}%  "
                         f"selection {100 * selection:.2f}%")
        print("  " + "   ".join(parts), flush=True)
        on_report(report)
        if not math.isfinite(report["loss"]):
            raise FloatingPointError("validation produced a non-finite loss")
        return selection

    return evaluate
