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
    subtoken_fraction: float = 0.0          # by area — the PLAN 4.5 bucketing rule
    subtoken_fraction_by_axis: float = 0.0  # the Phase 0 definition, for comparability
    overall_ap50: float = 0.0
    overall_p_at_f1: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"boundary_tokens": self.boundary_tokens,
                "subtoken_fraction": self.subtoken_fraction,
                "subtoken_fraction_by_axis": self.subtoken_fraction_by_axis,
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
        lines.append(f"\n  sub-token targets: {100 * self.subtoken_fraction:.1f}% by area "
                     f"(boundary = {self.boundary_tokens:g} visual token); "
                     f"{100 * self.subtoken_fraction_by_axis:.1f}% narrower than one "
                     f"token on an axis")
        return "\n".join(lines)


def box_area_in_tokens(box: Box, image_w: float, image_h: float,
                       resized_w: float, resized_h: float, token_px: float) -> float:
    """Area of a normalised-1000 box, in units of one visual token."""
    if not (image_w and image_h and resized_w and resized_h and token_px):
        return 0.0
    width = (box[2] - box[0]) / 1000.0 * resized_w
    height = (box[3] - box[1]) / 1000.0 * resized_h
    return max(0.0, width) * max(0.0, height) / (token_px * token_px)


def is_subtoken_by_area(box: Box, resized_w: float, resized_h: float,
                        token_px: float) -> bool:
    """Area smaller than one visual token — the bucketing rule `PLAN.md` 4.5 specifies.

    4.5 says "split by target-box **area**" and expects "roughly 23.9% of targets below
    it". Measured on 7,158 RefChartQA training boxes at 512 px: **24.8%**. That match is
    what confirms this is the intended definition.
    """
    return box_area_in_tokens(box, resized_w, resized_h, resized_w, resized_h,
                              token_px) < 1.0


def is_subtoken_by_axis(box: Box, resized_w: float, resized_h: float,
                        token_px: float) -> bool:
    """Narrower than one token on **at least one axis** — the Phase 0 definition.

    A different and much larger population: **66.7%** of the same 7,158 boxes, against
    24.8% by area. Both are meaningful and they answer different questions. Area asks "is
    there enough of this target for a token to be mostly inside it"; axis asks "can a
    token grid resolve this target's narrow dimension at all" — a 4×256 px sliver has the
    area of two tokens and still cannot be localised across its short side.

    Kept because the Phase 0 sub-token analysis used it, and mixing the two definitions
    would make those numbers incomparable. `PLAN.md` 4.5's buckets use area.
    """
    width = (box[2] - box[0]) / 1000.0 * resized_w
    height = (box[3] - box[1]) / 1000.0 * resized_h
    return width < token_px or height < token_px


def _match(preds: Sequence[Box], gts: Sequence[Box],
           iou_threshold: float = 0.5) -> list[int | None]:
    """Greedy match in emitted order; returns the GT index each prediction took."""
    from chartqa_dt.eval.metrics import iou

    used: set[int] = set()
    out: list[int | None] = []
    for pred in preds:
        best_j, best_iou = None, iou_threshold
        for j, gt in enumerate(gts):
            if j in used:
                continue
            v = iou(pred, gt)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j is not None:
            used.add(best_j)
        out.append(best_j)
    return out


def stratify(items: Iterable[Mapping[str, Any]], *, token_px: float = 32.0,
             boundary_tokens: float = 1.0) -> StratifiedReport:
    """Bucket targets by area and score each bucket independently.

    Each item is a mapping with ``pred_boxes``, ``gt_boxes`` (both normalised 0–1000) and
    ``resized_size`` as ``(w, h)``.

    **Predictions are filtered with the targets, following COCO's area-range semantics.**
    The obvious implementation — restrict the ground truths to a bucket but score every
    prediction against them — is wrong, and visibly so: with *perfect* predictions it
    reported 78% and 94% for the two buckets while the overall score was 100%, because a
    prediction matching a large target became a false positive in the small-target bucket.

    So for each bucket: keep the targets in it, keep the predictions that matched those
    targets, and keep an unmatched prediction only if its **own** area falls in the
    bucket. Targets outside the bucket are ignored rather than missed, exactly as COCO
    ignores ground truths outside an area range.

    AP is computed *within* a bucket over that bucket's targets, because AP is a
    dataset-level quantity — averaging a per-item AP would be a different, less meaningful
    number.
    """
    import numpy as np

    items = list(items)
    names = ("sub-token", f"≥ {boundary_tokens:g} token")
    gt_maps: dict[str, dict[str, list[Box]]] = {n: {} for n in names}
    pred_lists: dict[str, list[tuple[str, float, Box]]] = {n: [] for n in names}
    pair_lists: dict[str, list[tuple[list, list]]] = {n: [] for n in names}
    areas: dict[str, list[float]] = {n: [] for n in names}
    n_sub = n_total = n_sub_axis = 0

    all_gt: dict[str, list[Box]] = {}
    all_pred: list[tuple[str, float, Box]] = []
    all_pairs: list[tuple[list, list]] = []

    for idx, item in enumerate(items):
        key = f"i{idx}"
        rw, rh = item.get("resized_size", (0, 0))
        gts: list[Box] = list(item.get("gt_boxes") or [])
        preds: list[Box] = list(item.get("pred_boxes") or [])

        def bucket_of(box: Box, _w: float = rw, _h: float = rh) -> str:
            return names[0] if is_subtoken_by_area(box, _w, _h, token_px) else names[1]

        for gt in gts:
            n_total += 1
            sub = is_subtoken_by_area(gt, rw, rh, token_px)
            n_sub += sub
            n_sub_axis += is_subtoken_by_axis(gt, rw, rh, token_px)
            name = names[0] if sub else names[1]
            gt_maps[name].setdefault(key, []).append(gt)
            areas[name].append(box_area_in_tokens(gt, rw, rh, rw, rh, token_px))

        matched = _match(preds, gts)
        for name in names:
            kept = [p for p, m in zip(preds, matched)
                    if (bucket_of(gts[m]) == name) if m is not None]
            kept += [p for p, m in zip(preds, matched)
                     if m is None and bucket_of(p) == name]
            if kept or key in gt_maps[name]:
                pred_lists[name].extend((key, 1.0, b) for b in kept)
                pair_lists[name].append((kept, gt_maps[name].get(key, [])))

        if gts:
            all_gt[key] = gts
        all_pred.extend((key, 1.0, b) for b in preds)
        all_pairs.append((preds, gts))

    report = StratifiedReport(boundary_tokens=boundary_tokens)
    for name in names:
        pairs = pair_lists[name]
        report.buckets.append(Bucket(
            name=name,
            n_targets=sum(len(v) for v in gt_maps[name].values()),
            n_items=len(gt_maps[name]),
            ap50=average_precision_coco(pred_lists[name], gt_maps[name], 0.5),
            p_at_f1=(sum(grounding_is_perfect(p, g) for p, g in pairs) / len(pairs)
                     if pairs else 0.0),
            median_area_tokens=float(np.median(areas[name])) if areas[name] else 0.0,
        ))

    report.overall_ap50 = average_precision_coco(all_pred, all_gt, 0.5)
    report.overall_p_at_f1 = (sum(grounding_is_perfect(p, g) for p, g in all_pairs)
                              / len(all_pairs)) if all_pairs else 0.0
    report.subtoken_fraction = n_sub / n_total if n_total else 0.0
    report.subtoken_fraction_by_axis = n_sub_axis / n_total if n_total else 0.0
    return report


