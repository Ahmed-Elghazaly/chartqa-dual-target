from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Box = Sequence[float]


def to_float(text: Any) -> float | None:
    try:
        text = str(text)
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0
        return float(text)
    except (ValueError, AttributeError):
        return None


def relaxed_correctness(target: str, prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    prediction_float = to_float(prediction)
    target_float = to_float(target)
    if prediction_float is not None and target_float:
        return abs(prediction_float - target_float) / abs(target_float) <= max_relative_change
    return str(prediction).lower() == str(target).lower()


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


COCO_RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
COCO_MAX_DETECTIONS = 100


def average_precision_coco(predictions: Iterable[tuple[Any, float, Box]],
                           ground_truths: Mapping[Any, Sequence[Box]],
                           iou_threshold: float = 0.5) -> float:
    n_gt = sum(len(v) for v in ground_truths.values())
    if n_gt == 0:
        return 0.0

    by_image: dict[Any, list[tuple[float, Box]]] = {}
    for img, score, box in predictions:
        by_image.setdefault(img, []).append((float(score), box))
    if not by_image:
        return 0.0

    flat: list[tuple[float, Any, Box]] = []
    for img, dets in by_image.items():
        ordered = sorted(dets, key=lambda d: -d[0])[:COCO_MAX_DETECTIONS]
        flat.extend((score, img, box) for score, box in ordered)
    flat.sort(key=lambda d: -d[0])

    matched = {k: np.zeros(len(v), dtype=bool) for k, v in ground_truths.items()}
    tp = np.zeros(len(flat))
    fp = np.zeros(len(flat))
    for i, (_score, img, box) in enumerate(flat):
        gts = ground_truths.get(img, [])
        best_j, best_iou = -1, iou_threshold
        for j, g in enumerate(gts):
            if matched[img][j]:
                continue
            v = iou(box, g)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            matched[img][best_j] = True
            tp[i] = 1
        else:
            fp[i] = 1

    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, np.finfo(np.float64).eps)
    for i in range(len(precision) - 1, 0, -1):
        if precision[i] > precision[i - 1]:
            precision[i - 1] = precision[i]
    idx = np.searchsorted(recall, COCO_RECALL_THRESHOLDS, side="left")
    out = np.zeros(len(COCO_RECALL_THRESHOLDS))
    valid = idx < len(precision)
    out[valid] = precision[idx[valid]]
    return float(out.mean())


@dataclass(frozen=True)
class Interval:
    mean: float
    lo: float
    hi: float
    n: int

    @property
    def as_percent(self) -> str:
        return (f"{100 * self.mean:.2f}% [{100 * self.lo:.2f}, {100 * self.hi:.2f}] "
                f"(n={self.n})")


def bootstrap_ci(per_item_scores: Sequence[float], n_resamples: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> Interval:
    arr = np.asarray(per_item_scores, dtype=float)
    if arr.size == 0:
        return Interval(0.0, 0.0, 0.0, 0)
    rng = np.random.default_rng(seed)
    rows_per_chunk = max(1, 1_000_000 // int(arr.size))
    means = np.empty(n_resamples, dtype=float)
    for start in range(0, n_resamples, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_resamples)
        idx = rng.integers(0, arr.size, size=(stop - start, arr.size))
        means[start:stop] = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(float(arr.mean()), float(lo), float(hi), int(arr.size))
