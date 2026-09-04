"""Scoring a prediction set end to end, with seeds and confidence intervals.

`PLAN.md` 4.6: *"Seeded evaluation with three seeds and bootstrap CIs on every headline
number."* And the Phase 4 acceptance criterion: *"`cdt-eval` runs end to end on `--dev`
data and writes a structured results JSON."*

**What a seed means here, and what it does not.** Scoring a fixed prediction file is
deterministic — running it three times gives the same number three times, and reporting
that as "three seeds" would be theatre. Seeds vary two things that are genuinely random:

* **the bootstrap resampling**, which is what produces the interval; and
* **generation**, when predictions are produced rather than supplied — that is where seed
  variance actually lives, and `cdt-eval --adapter` will pass its seed through in Phase 5.

So `evaluate_predictions` takes a list of seeds and reports the spread across them
alongside the interval. On a fixed file the spread is the bootstrap's own variability,
which is the honest thing to show: it says how much of the reported precision is real.

**Answer accuracy is bootstrapped per item; AP is not.** AP depends on the ranking across
the whole set, so it has no per-item score to resample — `bootstrap_ci_of` recomputes it
on each resampled set instead, which is slower and is why its resample count is lower.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chartqa_dt.eval.metrics import (
    Box,
    Interval,
    average_precision_coco,
    bootstrap_ci,
    bootstrap_ci_of,
    exact_match,
    grounding_is_perfect,
    normalise_prediction,
    relaxed_correctness,
)

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
#: Bootstrap resamples for the AP interval. Far fewer than the 10,000 `bootstrap_ci` uses for
#: per-item metrics, and deliberately: AP cannot be bootstrapped from per-item scores because
#: it depends on the ranking across the whole set, so each resample recomputes AP over every
#: prediction. 400 keeps a full evaluation to seconds at the cost of a slightly coarser
#: interval, which is the right trade for a number reported to two decimals.
AP_RESAMPLES = 400


@dataclass
class ScoredItem:
    """One prediction, with everything needed to re-derive every headline number."""

    item_id: str
    correct: bool
    exact: bool
    grounding_perfect: bool
    pred_boxes: list[list[float]] = field(default_factory=list)
    gt_boxes: list[list[float]] = field(default_factory=list)
    subset: str = ""
    gold: str = ""
    prediction: str = ""


@dataclass
class EvalResult:
    n_items: int
    relaxed_accuracy: Interval
    exact_match: Interval
    p_at_f1: Interval
    ap50: Interval
    seeds: list[int]
    seed_spread: dict[str, float]
    by_subset: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        def interval(i: Interval) -> dict[str, float]:
            return {"mean": i.mean, "ci_lo": i.lo, "ci_hi": i.hi, "n": i.n}

        return {
            "n_items": self.n_items,
            "relaxed_accuracy": interval(self.relaxed_accuracy),
            "exact_match": interval(self.exact_match),
            "p_at_f1": interval(self.p_at_f1),
            "ap50": interval(self.ap50),
            "seeds": self.seeds,
            "seed_spread": self.seed_spread,
            "by_subset": self.by_subset,
        }

    def describe(self) -> str:
        rows = [
            ("relaxed accuracy", self.relaxed_accuracy),
            ("exact match", self.exact_match),
            ("P@F1 (IoU>=0.5)", self.p_at_f1),
            ("AP@0.5", self.ap50),
        ]
        out = [f"{'metric':<20}{'value':>9}  {'95% CI':>20}"]
        for name, iv in rows:
            out.append(f"  {name:<18}{100 * iv.mean:>8.2f}%  "
                       f"[{100 * iv.lo:>6.2f}, {100 * iv.hi:>6.2f}]")
        out.append(f"\n  n = {self.n_items:,}   seeds {self.seeds}   "
                   f"max spread across seeds: "
                   f"{max(self.seed_spread.values(), default=0.0) * 100:.3f} pp")
        if self.by_subset:
            out.append(f"\n{'subset':<12}{'n':>8}{'accuracy':>11}{'AP@0.5':>10}{'P@F1':>9}")
            for name, m in sorted(self.by_subset.items()):
                out.append(f"  {name:<10}{m['n']:>8,.0f}{100 * m['relaxed_accuracy']:>10.2f}%"
                           f"{100 * m['ap50']:>9.2f}%{100 * m['p_at_f1']:>8.2f}%")
        return "\n".join(out)


def score_item(item_id: str, gold: str, prediction: str,
               pred_boxes: Sequence[Box] | None = None,
               gt_boxes: Sequence[Box] | None = None,
               subset: str = "") -> ScoredItem:
    """Score one prediction. `prediction` is normalised first, the metric is not loosened."""
    pred = normalise_prediction(prediction)
    pred_boxes = [list(b) for b in (pred_boxes or [])]
    gt_boxes = [list(b) for b in (gt_boxes or [])]
    return ScoredItem(
        item_id=item_id,
        correct=relaxed_correctness(gold, pred),
        exact=exact_match(gold, pred),
        grounding_perfect=grounding_is_perfect(pred_boxes, gt_boxes),
        pred_boxes=pred_boxes, gt_boxes=gt_boxes, subset=subset,
        gold=str(gold), prediction=pred,
    )


def _ap_over(items: Sequence[ScoredItem]) -> float:
    gts: dict[str, list[Box]] = {}
    preds: list[tuple[str, float, Box]] = []
    for i, item in enumerate(items):
        key = f"{i}:{item.item_id}"          # unique even if a resample repeats an item
        if item.gt_boxes:
            gts[key] = item.gt_boxes
        preds.extend((key, 1.0, b) for b in item.pred_boxes)
    return average_precision_coco(preds, gts, 0.5)


def evaluate_predictions(items: Sequence[ScoredItem], *,
                         seeds: Sequence[int] = DEFAULT_SEEDS,
                         ap_resamples: int = AP_RESAMPLES) -> EvalResult:
    """Headline numbers with bootstrap intervals, plus the spread across seeds."""
    items = list(items)
    seeds = list(seeds)
    if not items:
        empty = Interval(0.0, 0.0, 0.0, 0)
        return EvalResult(0, empty, empty, empty, empty, seeds, {})

    accuracy = [float(i.correct) for i in items]
    exact = [float(i.exact) for i in items]
    perfect = [float(i.grounding_perfect) for i in items]

    primary = seeds[0]
    result = EvalResult(
        n_items=len(items),
        relaxed_accuracy=bootstrap_ci(accuracy, seed=primary),
        exact_match=bootstrap_ci(exact, seed=primary),
        p_at_f1=bootstrap_ci(perfect, seed=primary),
        ap50=bootstrap_ci_of(items, _ap_over, n_resamples=ap_resamples, seed=primary),
        seeds=seeds,
        seed_spread={},
    )

    # How much of the reported precision survives changing the seed. On a fixed
    # prediction file the point estimates cannot move, so any spread here is the
    # bootstrap's own variability — worth showing rather than hiding behind one seed.
    spread: dict[str, list[float]] = {"relaxed_accuracy": [], "ap50_ci_width": []}
    for seed in seeds:
        spread["relaxed_accuracy"].append(bootstrap_ci(accuracy, seed=seed).lo)
        interval = bootstrap_ci_of(items, _ap_over, n_resamples=ap_resamples, seed=seed)
        spread["ap50_ci_width"].append(interval.hi - interval.lo)
    result.seed_spread = {k: (max(v) - min(v)) for k, v in spread.items()}

    subsets = {i.subset for i in items if i.subset}
    for name in sorted(subsets):
        rows = [i for i in items if i.subset == name]
        result.by_subset[name] = {
            "n": len(rows),
            "relaxed_accuracy": sum(i.correct for i in rows) / len(rows),
            "exact_match": sum(i.exact for i in rows) / len(rows),
            "p_at_f1": sum(i.grounding_perfect for i in rows) / len(rows),
            "ap50": _ap_over(rows),
        }
    return result


def write_results(result: EvalResult, path: str | Path, *,
                  meta: dict[str, Any] | None = None,
                  stratified: dict[str, Any] | None = None) -> Path:
    """The structured results JSON the Phase 4 acceptance criterion asks for."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"meta": meta or {}, "results": result.to_dict()}
    if stratified is not None:
        payload["stratified"] = stratified
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return p


def items_to_records(items: Sequence[ScoredItem]) -> list[dict[str, Any]]:
    return [asdict(i) for i in items]


__all__ = ["AP_RESAMPLES", "DEFAULT_SEEDS", "EvalResult", "ScoredItem",
           "evaluate_predictions", "items_to_records", "score_item", "write_results"]
