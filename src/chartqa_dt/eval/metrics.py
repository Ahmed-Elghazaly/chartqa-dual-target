"""Appendix D metrics: relaxed accuracy, exact match, IoU, AP@0.5, P@F1, bootstrap CIs.

`PLAN.md` 4.1, and `PLAN.md` 4.2 governs what these are *for*:

    Cross-check against the official evaluators. If they disagree, **the official one
    wins** and you fix yours.

So nothing here is the scorer of record. `DECISIONS.md` 0003 gives that to the vendored
official evaluator, and these exist for stratified analysis, per-item scores, confidence
intervals and fast iteration — things the official evaluator cannot do, because it returns
one number for a whole split.

Two places where the official implementation and the obvious implementation differ, both
verified against the vendored source rather than assumed:

* **The zero guard.** The official `relaxed_accuracy` tests ``target_float`` for
  *truthiness*, not for ``is not None``. A gold answer of ``"0"`` is therefore falsy and
  the comparison silently falls through to the string branch — which is why ``"0"`` vs
  ``"0"`` is correct and ``"0"`` vs ``"0.0"`` is **not** (`DECISIONS.md` 0015). Appendix D
  writes the guard explicitly and gets the same answers for the same reason made visible.
* **Commas.** Appendix D strips thousands separators; the vendored official evaluator does
  not, so ``"1,234"`` parses as a number for us and as a string for it. That divergence is
  measured in `scripts/crosscheck_evaluators.py` rather than assumed away.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Box = Sequence[float]

# --------------------------------------------------------------------------- ChartQA


def to_float(text: Any) -> float | None:
    """ChartQA's numeric parser, byte-faithful to the official implementation.

    Deliberately **not** Appendix D's version. Appendix D strips thousands separators and
    calls `.strip()`; the canonical implementation
    (`google-research/pix2struct/pix2struct/metrics.py`, which the RefChartQA evaluator
    vendors verbatim) does neither:

        def _to_float(text):
          try:
            if text.endswith("%"):
              return float(text.rstrip("%")) / 100.0
            else:
              return float(text)
          except ValueError:
            return None

    `PLAN.md` 4.2 settles the conflict — *"If they disagree, the official one wins and you
    fix yours"* — and the stakes are concrete: every published ChartQA number means the
    official metric, so a more generous parser would make our results incomparable with
    the literature while looking better. Measured: Appendix D's version disagreed with the
    official on 61 of 423 cases, all of them in our favour.

    The one addition is coercing to `str` first, so a non-string input returns None
    instead of raising `AttributeError`. That changes no result the official can produce —
    it would have crashed.
    """
    try:
        text = str(text)
        if text.endswith("%"):
            return float(text.rstrip("%")) / 100.0
        return float(text)
    except (ValueError, AttributeError):
        return None


def relaxed_correctness(target: str, prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """The official ChartQA metric. Numeric within 5%, otherwise exact and lowercased.

    Two behaviours look like bugs and are load-bearing, because published numbers were
    produced with them:

    * **A gold answer of "0" never takes the numeric path.** The guard is
      ``if prediction_float is not None and target_float`` — a *truthiness* test on the
      target, so 0.0 is falsy and the comparison falls through to string equality. Hence
      "0" vs "0" is correct while "0" vs "0.0" is not (`DECISIONS.md` 0015). Appendix D
      writes an explicit ``t == 0`` guard, which is more principled and gives a *different
      answer*.
    * **No whitespace stripping on the string branch.** ``" Yes "`` does not match
      ``"Yes"``. Normalising a prediction is the pipeline's job, not the metric's — see
      `normalise_prediction`, which is applied before scoring so we never rely on the
      metric to be lenient.
    """
    prediction_float = to_float(prediction)
    target_float = to_float(target)
    if prediction_float is not None and target_float:
        return abs(prediction_float - target_float) / abs(target_float) <= max_relative_change
    return str(prediction).lower() == str(target).lower()


def relaxed_correctness_appendix_d(target: str, prediction: str,
                                   max_relative_change: float = 0.05) -> bool:
    """Appendix D's variant, kept **only** so the divergence stays measurable.

    Never used for a reported number. `WORKING_AGREEMENT.md` forbids averaging two
    evaluator variants; this exists so `scripts/crosscheck_evaluators.py` can quantify how
    far the plan's text sits from the metric it describes, rather than the difference being
    an argument.
    """
    def parse(text: Any) -> float | None:
        try:
            text = str(text).strip()
            if text.endswith("%"):
                return float(text.rstrip("%").replace(",", "")) / 100.0
            return float(text.replace(",", ""))
        except (ValueError, AttributeError):
            return None

    p, t = parse(prediction), parse(target)
    if p is not None and t is not None:
        if t == 0:
            return p == 0
        return abs(p - t) / abs(t) <= max_relative_change
    return str(prediction).strip().lower() == str(target).strip().lower()


def normalise_prediction(text: Any) -> str:
    """Tidy a model's output *before* scoring, so the metric stays the official one.

    The official metric does not strip whitespace, and it should not be changed to. But a
    trailing newline from generation is not a wrong answer, so it is removed here — in the
    pipeline, in one place, where it is visible — rather than by quietly loosening the
    metric everyone else's numbers were produced with.
    """
    return str(text).strip()


def exact_match(target: str, prediction: str) -> bool:
    return str(prediction).strip().lower() == str(target).strip().lower()


# ------------------------------------------------------------------------- grounding


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


def average_precision_at_iou(predictions: Iterable[tuple[Any, float, Box]],
                             ground_truths: Mapping[Any, Sequence[Box]],
                             iou_threshold: float = 0.5) -> float:
    """Greedy matching by descending score, then all-point interpolated AP.

    Each ground-truth box may be matched at most once; a second prediction on the same
    box is a false positive, which is what makes emitting extra boxes costly
    (`DECISIONS.md` 0014).
    """
    n_gt = sum(len(v) for v in ground_truths.values())
    if n_gt == 0:
        return 0.0
    preds = sorted(predictions, key=lambda p: -p[1])
    if not preds:
        return 0.0
    matched = {k: np.zeros(len(v), dtype=bool) for k, v in ground_truths.items()}
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    for i, (img, _score, box) in enumerate(preds):
        gts = ground_truths.get(img, [])
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gts):
            if matched[img][j]:
                continue
            v = iou(box, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_threshold:
            matched[img][best_j] = True
            tp[i] = 1
        else:
            fp[i] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


#: COCO's 101 recall thresholds, which is what `torchmetrics.MeanAveragePrecision` uses
#: by default and therefore what the official RefChartQA evaluator computes. Verified by
#: reading the instantiated metric, not assumed: `len(m.rec_thresholds) == 101`.
#: COCO's 101 recall thresholds, which is what `torchmetrics.MeanAveragePrecision` uses
#: by default and therefore what the official RefChartQA evaluator computes. Verified by
#: reading the instantiated metric — `len(m.rec_thresholds) == 101` — not assumed.
#:
#: Kept in float64. `torchmetrics` stores them as float32, and matching that was tried:
#: measured over 120 randomised scenarios it was *worse*, agreeing exactly on 112 rather
#: than 119. See `average_precision_coco` for what remains.
COCO_RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)

#: COCO reports AP at maxDets=100. The official evaluator never approaches it — a chart
#: has a handful of boxes — but truncating is part of the definition.
COCO_MAX_DETECTIONS = 100


def average_precision_coco(predictions: Iterable[tuple[Any, float, Box]],
                           ground_truths: Mapping[Any, Sequence[Box]],
                           iou_threshold: float = 0.5) -> float:
    """AP@IoU computed the way the **official** evaluator computes it.

    `PLAN.md` 4.2: where our metric and the official one disagree, the official one wins.
    They did disagree, and this function is the fix.

    Two differences from `average_precision_at_iou` (Appendix D), both measured:

    * **101-point interpolation, not all-point.** COCO averages the precision envelope at
      101 evenly spaced recall thresholds. On five images each emitting one spurious extra
      box, all-point gave 0.6787 and the official gave 0.6815.
    * **Deterministic ordering under tied scores.** The official evaluator assigns every
      prediction a score of 1.0, so the ranking is entirely decided by sort stability.
      Appendix D's implementation sorted the caller's list, which made the result depend on
      whether predictions arrived grouped by image or interleaved — 1.0 versus 0.6787 for
      the *same* predictions. Here detections are grouped by image in first-seen order and
      then stably sorted, matching pycocotools, so the input order cannot change the score.

    The cost of an extra box is real and large: one spurious box per image took AP from
    1.0 to 0.68. That is the measurement behind `DECISIONS.md` 0014's "emit few boxes,
    best first".

    **Residual disagreement, stated rather than hidden.** Across 120 randomised scenarios
    of 24 images this reproduces the official value exactly (< 1e-6) on 119; on the
    remaining one it differs by 0.0019. The difference is confined to whether the single
    highest recall threshold is included, which is decided by floating-point storage
    inside `torchmetrics`. Reproducing that bit-for-bit was attempted and abandoned: the
    obvious float32 hypothesis made agreement *worse* (112 of 120), and fitting the
    remaining case without understanding it would be worse than reporting it.

    This does not affect any reported number. `DECISIONS.md` 0003 makes the official
    evaluator the scorer of record; this function exists for stratified analysis and
    confidence intervals, which the official cannot produce because it returns one number
    for a whole split.
    """
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
        # Per image: sort by score, keep at most maxDets, exactly as COCO does.
        ordered = sorted(dets, key=lambda d: -d[0])[:COCO_MAX_DETECTIONS]
        flat.extend((score, img, box) for score, box in ordered)
    # Stable sort across images, so tied scores keep image order — pycocotools uses
    # mergesort here for the same reason.
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

    # Monotone-decreasing precision envelope, right to left.
    for i in range(len(precision) - 1, 0, -1):
        if precision[i] > precision[i - 1]:
            precision[i - 1] = precision[i]

    idx = np.searchsorted(recall, COCO_RECALL_THRESHOLDS, side="left")
    out = np.zeros(len(COCO_RECALL_THRESHOLDS))
    valid = idx < len(precision)
    out[valid] = precision[idx[valid]]
    return float(out.mean())


def grounding_is_perfect(pred_boxes: Sequence[Box], gt_boxes: Sequence[Box],
                         iou_threshold: float = 0.5) -> bool:
    """The official P@F1 predicate, reproduced exactly. **Not** an F1 of 1.0.

    The official helper is named `is_image_grounding_correct` and its docstring says
    "IoU-based precision at 0.5 threshold is perfect (F_1 score = 1.0)". It does not
    compute F1. It computes COCO AP on the single image and tests ``map == 1.0``, and
    those are not the same predicate. Characterised against the vendored evaluator:

    | predictions (in emitted order)      | official |
    |-------------------------------------|----------|
    | the true box only                   | correct  |
    | true box, then a spurious one       | correct  |
    | true box, then **two** spurious ones| correct  |
    | spurious box, then the true one     | wrong    |
    | true, spurious, true (2 GT)         | wrong    |
    | one of two ground truths found      | wrong    |

    So the rule is: **every ground truth must be matched, and every false positive must
    come after every true positive.** Trailing false positives are free — AP's precision
    envelope is computed right-to-left, so a false positive after recall has already
    reached 1.0 cannot lower it.

    This matters for how the model should emit. `DECISIONS.md` 0014 says "few boxes,
    best first"; these measurements say the *ordering* half is what P@F1 rewards, while
    the *count* half is what AP punishes — one spurious box per image took AP from 1.00 to
    0.68. Both point the same way, for different reasons.
    """
    if not pred_boxes or not gt_boxes:
        return False
    used: set[int] = set()
    seen_false_positive = False
    for pred in pred_boxes:
        best_j, best_iou = -1, iou_threshold
        for j, gt in enumerate(gt_boxes):
            if j in used:
                continue
            v = iou(pred, gt)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j < 0:
            seen_false_positive = True
            continue
        if seen_false_positive:
            return False          # a true positive ranked below a false one
        used.add(best_j)
    return len(used) == len(gt_boxes)


def f1_of_boxes(pred_boxes: Sequence[Box], gt_boxes: Sequence[Box],
                iou_threshold: float = 0.5) -> float:
    """Actual F1 over greedily matched boxes — a diagnostic, never a reported number.

    This is what `PLAN.md` Appendix D calls `p_at_f1`, and it is a genuinely different
    quantity from the official predicate above: on one measured case, one true box plus
    one spurious box scored F1 = 0.667 here and *correct* officially. Kept because it is
    the more informative number when reading a failure, and because keeping it visible is
    how the divergence stays honest rather than becoming folklore.
    """
    if not pred_boxes and not gt_boxes:
        return 1.0
    if not pred_boxes or not gt_boxes:
        return 0.0
    used: set[int] = set()
    matches = 0
    for pred in pred_boxes:
        best_j, best_iou = -1, iou_threshold
        for j, gt in enumerate(gt_boxes):
            if j in used:
                continue
            v = iou(pred, gt)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            used.add(best_j)
            matches += 1
    if matches == 0:
        return 0.0
    precision = matches / len(pred_boxes)
    recall = matches / len(gt_boxes)
    return 2 * precision * recall / (precision + recall)


def p_at_f1(items: Iterable[tuple[Sequence[Box], Sequence[Box]]],
            iou_threshold: float = 0.5) -> float:
    """Fraction of items whose grounding is perfect, by the official predicate.

    Items the official evaluator *skips* — a malformed template, or no predicted boxes —
    stay in the denominator, so they score zero rather than being excluded. Reproduced
    here, because excluding them would flatter a model that declines to answer.
    """
    items = list(items)
    if not items:
        return 0.0
    return sum(1 for pred, gt in items
               if grounding_is_perfect(pred, gt, iou_threshold)) / len(items)


# ------------------------------------------------------------------------ statistics


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile bootstrap interval."""

    mean: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.lo:.4f}, {self.hi:.4f}] (n={self.n})"

    @property
    def as_percent(self) -> str:
        return (f"{100 * self.mean:.2f}% [{100 * self.lo:.2f}, {100 * self.hi:.2f}] "
                f"(n={self.n})")


def bootstrap_ci(per_item_scores: Sequence[float], n_resamples: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> Interval:
    """Resample per-item scores with replacement; report the percentile interval.

    Requires *per-item* scores. A metric that only exists at dataset level — AP is one,
    because it depends on the ranking across items — cannot be bootstrapped this way; use
    `bootstrap_ci_of` with a callable instead.
    """
    arr = np.asarray(per_item_scores, dtype=float)
    if arr.size == 0:
        return Interval(0.0, 0.0, 0.0, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Interval(float(arr.mean()), float(lo), float(hi), int(arr.size))


def bootstrap_ci_of(items: Sequence[Any], statistic, n_resamples: int = 1_000,
                    alpha: float = 0.05, seed: int = 0) -> Interval:
    """Bootstrap a statistic that needs the whole resampled set, such as AP.

    Far more expensive than `bootstrap_ci` — `statistic` runs once per resample — so the
    default resample count is lower. AP cannot be averaged over items, so resampling the
    items and recomputing is the only honest way to put an interval on it.
    """
    n = len(items)
    if n == 0:
        return Interval(0.0, 0.0, 0.0, 0)
    rng = np.random.default_rng(seed)
    point = float(statistic(list(items)))
    draws = np.empty(n_resamples, dtype=float)
    for k in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        draws[k] = float(statistic([items[i] for i in idx]))
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(point, float(lo), float(hi), n)


__all__ = [
    "COCO_MAX_DETECTIONS",
    "COCO_RECALL_THRESHOLDS",
    "Box",
    "Interval",
    "average_precision_at_iou",
    "average_precision_coco",
    "bootstrap_ci",
    "bootstrap_ci_of",
    "exact_match",
    "iou",
    "normalise_prediction",
    "p_at_f1",
    "relaxed_correctness",
    "relaxed_correctness_appendix_d",
    "to_float",
]
