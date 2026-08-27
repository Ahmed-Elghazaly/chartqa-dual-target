"""Stratified grounding reporting — `PLAN.md` 4.5.

    AP@0.5 split by target-box area, with the bucket boundary at **one visual token**
    (see Appendix C) — expect roughly 23.9% of targets below it.

**Why this split and not a percentile.** A visual token is the smallest thing the model
can point at: after `smart_resize`, one token covers a 32×32 px patch of the resized image
(`DECISIONS.md` 0008 — factor 32 for Qwen3-VL, derived from the processor rather than
copied from Qwen2.5-VL's 28). A target smaller than that cannot be localised precisely no
matter how good the model is, because the representation has no finer unit. Splitting
there separates "the model is bad at this" from "the architecture cannot express this",
and only the first is worth training against.

Targets are measured **in resized-image space**, not source pixels. A box is one token
wide on a 1000-px-wide chart and sub-token on a 400-px one, at the same normalised size;
using source pixels would put those in the same bucket and the split would mean nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from chartqa_dt.eval.metrics import Box, average_precision_coco, grounding_is_perfect

#: `PLAN.md` 4.5's expected sub-token fraction, for the boundary at one visual token.
EXPECTED_SUBTOKEN_FRACTION = 0.239


@dataclass
class Bucket:
    """One area stratum, with enough detail to explain its number."""

    name: str
    n_targets: int = 0
    n_items: int = 0
    ap50: float = 0.0
    p_at_f1: float = 0.0
    median_area_tokens: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "n_targets": self.n_targets, "n_items": self.n_items,
                "ap50": self.ap50, "p_at_f1": self.p_at_f1,
                "median_area_tokens": self.median_area_tokens}


@dataclass
class StratifiedReport:
    boundary_tokens: float
    buckets: list[Bucket] = field(default_factory=list)
    subtoken_fraction: float = 0.0
    overall_ap50: float = 0.0
    overall_p_at_f1: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"boundary_tokens": self.boundary_tokens,
                "subtoken_fraction": self.subtoken_fraction,
                "overall_ap50": self.overall_ap50,
                "overall_p_at_f1": self.overall_p_at_f1,
                "buckets": [b.to_dict() for b in self.buckets]}

    def describe(self) -> str:
        lines = [f"{'bucket':<22}{'targets':>9}{'items':>8}{'AP@0.5':>10}{'P@F1':>9}"
                 f"{'median area':>13}"]
        for b in self.buckets:
            lines.append(f"  {b.name:<20}{b.n_targets:>9,}{b.n_items:>8,}"
                         f"{100 * b.ap50:>9.2f}%{100 * b.p_at_f1:>8.2f}%"
                         f"{b.median_area_tokens:>12.2f}t")
        lines.append(f"  {'ALL':<20}{sum(b.n_targets for b in self.buckets):>9,}"
                     f"{'':>8}{100 * self.overall_ap50:>9.2f}%"
                     f"{100 * self.overall_p_at_f1:>8.2f}%")
        lines.append(f"\n  sub-token targets: {100 * self.subtoken_fraction:.1f}% "
                     f"(boundary = {self.boundary_tokens:g} visual token)")
        return "\n".join(lines)


def box_area_in_tokens(box: Box, image_w: float, image_h: float,
                       resized_w: float, resized_h: float, token_px: float) -> float:
    """Area of a normalised-1000 box, in units of one visual token."""
    if not (image_w and image_h and resized_w and resized_h and token_px):
        return 0.0
    width = (box[2] - box[0]) / 1000.0 * resized_w
    height = (box[3] - box[1]) / 1000.0 * resized_h
    return max(0.0, width) * max(0.0, height) / (token_px * token_px)


def is_subtoken(box: Box, resized_w: float, resized_h: float, token_px: float) -> bool:
    """Narrower than one token on **at least one axis** — the Phase 0 definition.

    Area alone would call a 4×256 px sliver "two tokens" and hide it, when it is exactly
    the target a token grid cannot localise. Kept identical to the definition used for the
    Phase 0 sub-token measurements so the numbers stay comparable.
    """
    width = (box[2] - box[0]) / 1000.0 * resized_w
    height = (box[3] - box[1]) / 1000.0 * resized_h
    return width < token_px or height < token_px


def stratify(items: Iterable[Mapping[str, Any]], *, token_px: float = 32.0,
             boundary_tokens: float = 1.0) -> StratifiedReport:
    """Bucket by target area and score each bucket independently.

    Each item is a mapping with ``pred_boxes``, ``gt_boxes`` (both normalised 0–1000),
    and ``resized_size`` as ``(w, h)``.

    AP is computed **within** a bucket, over only the targets in it, because AP is a
    dataset-level quantity: a per-item AP averaged across items is a different and less
    meaningful number.
    """
    import numpy as np

    items = list(items)
    buckets = {"sub-token": [], f"≥ {boundary_tokens:g} token": []}
    areas: dict[str, list[float]] = {k: [] for k in buckets}
    n_sub = n_total = 0

    for idx, item in enumerate(items):
        rw, rh = item.get("resized_size", (0, 0))
        gts: Sequence[Box] = item.get("gt_boxes") or []
        preds: Sequence[Box] = item.get("pred_boxes") or []
        for gt in gts:
            n_total += 1
            sub = is_subtoken(gt, rw, rh, token_px)
            n_sub += sub
            key = "sub-token" if sub else f"≥ {boundary_tokens:g} token"
            buckets[key].append((f"i{idx}", gt, preds))
            areas[key].append(box_area_in_tokens(gt, rw, rh, rw, rh, token_px))

    report = StratifiedReport(boundary_tokens=boundary_tokens)
    for name, entries in buckets.items():
        gt_map: dict[str, list[Box]] = {}
        pred_list: list[tuple[str, float, Box]] = []
        seen: set[str] = set()
        pairs = []
        for key, gt, preds in entries:
            gt_map.setdefault(key, []).append(gt)
            if key not in seen:
                seen.add(key)
                pred_list.extend((key, 1.0, b) for b in preds)
        for key in gt_map:
            preds = next(p for k, _g, p in entries if k == key)
            pairs.append((list(preds), gt_map[key]))
        bucket = Bucket(
            name=name, n_targets=len(entries), n_items=len(gt_map),
            ap50=average_precision_coco(pred_list, gt_map, 0.5) if entries else 0.0,
            p_at_f1=(sum(grounding_is_perfect(p, g) for p, g in pairs) / len(pairs)
                     if pairs else 0.0),
            median_area_tokens=float(np.median(areas[name])) if areas[name] else 0.0,
        )
        report.buckets.append(bucket)

    all_gt: dict[str, list[Box]] = {}
    all_pred: list[tuple[str, float, Box]] = []
    all_pairs = []
    for idx, item in enumerate(items):
        key = f"i{idx}"
        gts = list(item.get("gt_boxes") or [])
        preds = list(item.get("pred_boxes") or [])
        if gts:
            all_gt[key] = gts
        all_pred.extend((key, 1.0, b) for b in preds)
        all_pairs.append((preds, gts))
    report.overall_ap50 = average_precision_coco(all_pred, all_gt, 0.5)
    report.overall_p_at_f1 = (sum(grounding_is_perfect(p, g) for p, g in all_pairs)
                              / len(all_pairs)) if all_pairs else 0.0
    report.subtoken_fraction = n_sub / n_total if n_total else 0.0
    return report


__all__ = ["EXPECTED_SUBTOKEN_FRACTION", "Bucket", "StratifiedReport",
           "box_area_in_tokens", "is_subtoken", "stratify"]
