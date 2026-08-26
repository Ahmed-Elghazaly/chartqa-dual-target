"""Coordinate and resize mathematics.

This module is `PLAN.md` Appendix C, **corrected**. Appendix C hard-codes
``FACTOR = 28`` with the comment "Qwen: patch 14 x spatial merge 2". That is the
Qwen2-VL / Qwen2.5-VL geometry. Qwen3-VL uses ``patch_size = 16`` and
``spatial_merge_size = 2``, so its factor is **32** and one visual token covers
**32x32 pixels**, not 28x28. Appendix C's pixel bounds are wrong for this model
too, and it places the ``max(factor, ...)`` guard on the initial rounding rather
than inside the downscale branch, where ``transformers`` puts it.

See `DECISIONS.md` 0008. Because the deviation is deliberate, Appendix C's exact
formulation is kept below as :func:`smart_resize_appendix_c` and the two are
compared in `tests/test_coords.py`, so the divergence is demonstrable rather
than asserted.

Nothing here hard-codes the geometry. :class:`VisualGeometry` is built from the
processor of the model that is actually loaded, which is what stops a backbone
switch from silently reintroducing the bug.

Why any of this matters
-----------------------
The model cannot resolve a target smaller than one visual token. Chart evidence
regions frequently are. Measuring a box's size *in visual tokens* is therefore
the single most informative thing you can know about whether the model has any
chance of localising it, and it is the axis the grounding results are stratified
along.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Qwen3-VL geometry, recorded for reference and for tests. Never imported as a
# default by anything that touches a real model.
QWEN3VL_PATCH_SIZE = 16
QWEN3VL_MERGE_SIZE = 2
QWEN3VL_FACTOR = QWEN3VL_PATCH_SIZE * QWEN3VL_MERGE_SIZE  # 32

# Qwen2-VL / Qwen2.5-VL geometry, i.e. what PLAN.md Appendix C assumed.
QWEN2VL_FACTOR = 14 * 2  # 28


# --------------------------------------------------------------------------- #
# smart_resize
# --------------------------------------------------------------------------- #


def smart_resize(
    height: int | float,
    width: int | float,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Port of ``transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize``.

    Resizes so that both dimensions are divisible by ``factor``, the total pixel
    count lands within ``[min_pixels, max_pixels]``, and the aspect ratio is
    preserved as closely as possible.

    Ported from what actually runs at inference, not from the plan's transcription.
    """
    if min(height, width) <= 0:
        raise ValueError(f"image dimensions must be positive, got {height}x{width}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def smart_resize_appendix_c(
    height: int | float,
    width: int | float,
    factor: int = QWEN2VL_FACTOR,
    min_pixels: int = 4 * 28 * 28,
    max_pixels: int = 16384 * 28 * 28,
) -> tuple[int, int]:
    """`PLAN.md` Appendix C's formulation, kept verbatim for comparison only.

    Do not use for anything that touches a real model. It exists so that
    `tests/test_coords.py` can demonstrate exactly where and why it diverges from
    the implementation that actually runs, rather than the deviation being an
    unverifiable claim in a decision log.
    """

    def round_by_factor(n: float, f: int) -> int:
        return round(n / f) * f

    def ceil_by_factor(n: float, f: int) -> int:
        return math.ceil(n / f) * f

    def floor_by_factor(n: float, f: int) -> int:
        return math.floor(n / f) * f

    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio too extreme")
    h = max(factor, round_by_factor(height, factor))
    w = max(factor, round_by_factor(width, factor))
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = floor_by_factor(height / beta, factor)
        w = floor_by_factor(width / beta, factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = ceil_by_factor(height * beta, factor)
        w = ceil_by_factor(width * beta, factor)
    return int(h), int(w)


# --------------------------------------------------------------------------- #
# Geometry, derived from the loaded model rather than assumed
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VisualGeometry:
    """How one model turns pixels into visual tokens.

    ``factor`` is ``patch_size * merge_size``: the side length in pixels of the
    smallest region the model can represent separately. ``min_pixels`` and
    ``max_pixels`` are the processor's ``size.shortest_edge`` / ``size.longest_edge``.
    """

    factor: int
    min_pixels: int
    max_pixels: int
    patch_size: int
    merge_size: int
    source: str = "explicit"

    @classmethod
    def from_processor(cls, processor: Any) -> VisualGeometry:
        """Read the geometry off a loaded processor. Never guesses.

        Accepts a processor or an image processor; ``AutoProcessor`` returns the
        former and keeps the latter at ``.image_processor``.
        """
        ip = getattr(processor, "image_processor", processor)

        patch = getattr(ip, "patch_size", None)
        merge = getattr(ip, "merge_size", None)
        if patch is None or merge is None:
            raise ValueError(
                "processor exposes no patch_size/merge_size; refusing to guess the "
                "visual-token factor (DECISIONS.md 0008 exists because it was guessed once)"
            )

        size = getattr(ip, "size", None) or {}
        if hasattr(size, "get"):
            min_px = size.get("shortest_edge")
            max_px = size.get("longest_edge")
        else:  # SizeDict-style object
            min_px = getattr(size, "shortest_edge", None)
            max_px = getattr(size, "longest_edge", None)
        # Older processors expose these directly instead.
        min_px = min_px if min_px is not None else getattr(ip, "min_pixels", None)
        max_px = max_px if max_px is not None else getattr(ip, "max_pixels", None)
        if min_px is None or max_px is None:
            raise ValueError(f"processor exposes no pixel bounds: size={size!r}")

        return cls(
            factor=int(patch) * int(merge),
            min_pixels=int(min_px),
            max_pixels=int(max_px),
            patch_size=int(patch),
            merge_size=int(merge),
            source=type(ip).__name__,
        )

    def with_max_pixels(self, max_pixels: int) -> VisualGeometry:
        """Same geometry at a different input budget (decision 0010's two arms)."""
        return VisualGeometry(
            factor=self.factor,
            min_pixels=min(self.min_pixels, max_pixels),
            max_pixels=int(max_pixels),
            patch_size=self.patch_size,
            merge_size=self.merge_size,
            source=f"{self.source}@max_pixels={max_pixels}",
        )

    def resize(self, height: int | float, width: int | float) -> tuple[int, int]:
        return smart_resize(height, width, self.factor, self.min_pixels, self.max_pixels)

    def n_visual_tokens(self, height: int | float, width: int | float) -> int:
        h, w = self.resize(height, width)
        return (h // self.factor) * (w // self.factor)

    def box_in_tokens(
        self, bbox_px: tuple[float, float, float, float], img_h: int, img_w: int
    ) -> tuple[float, float]:
        """Size of a pixel box measured in visual tokens after preprocessing.

        A value below 1.0 on either axis means the target is **sub-token**: the
        model physically cannot resolve it at this input size.
        """
        h, w = self.resize(img_h, img_w)
        sx, sy = w / img_w, h / img_h
        x1, y1, x2, y2 = bbox_px
        return ((x2 - x1) * sx / self.factor, (y2 - y1) * sy / self.factor)

    def is_sub_token(
        self, bbox_px: tuple[float, float, float, float], img_h: int, img_w: int, *, rule: str = "axis"
    ) -> bool:
        """``rule='axis'``: smaller than one token on either axis (the strict,
        mechanistically meaningful test). ``rule='area'``: total area below one token."""
        tw, th = self.box_in_tokens(bbox_px, img_h, img_w)
        if rule == "axis":
            return min(tw, th) < 1.0
        if rule == "area":
            return tw * th < 1.0
        raise ValueError(f"unknown rule {rule!r}; use 'axis' or 'area'")

    def describe(self) -> str:
        return (
            f"patch={self.patch_size} merge={self.merge_size} -> factor={self.factor} "
            f"(one visual token = {self.factor}x{self.factor} px), "
            f"min_pixels={self.min_pixels:,} max_pixels={self.max_pixels:,} [{self.source}]"
        )


# --------------------------------------------------------------------------- #
# Normalised 0-1000 coordinates
# --------------------------------------------------------------------------- #

# The official RefChartQA evaluator accepts a box only when every coordinate is
# in 0..999 and SILENTLY DISCARDS the whole box otherwise. Qwen3-VL emits 0..1000
# inclusive, and edge-touching boxes are common in charts. See DECISIONS.md 0004.
OFFICIAL_MAX_COORD = 999


def px_to_norm1000(
    bbox_px: tuple[float, float, float, float], img_w: int, img_h: int
) -> list[float]:
    x1, y1, x2, y2 = bbox_px
    return [
        1000.0 * x1 / img_w,
        1000.0 * y1 / img_h,
        1000.0 * x2 / img_w,
        1000.0 * y2 / img_h,
    ]


def norm1000_to_px(
    bbox: tuple[float, float, float, float], img_w: int, img_h: int
) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [
        x1 * img_w / 1000.0,
        y1 * img_h / 1000.0,
        x2 * img_w / 1000.0,
        y2 * img_h / 1000.0,
    ]


def clamp_for_official_evaluator(bbox: tuple[float, float, float, float]) -> list[int]:
    """Integer box in 0..999, which is the only range the official evaluator accepts.

    ``extract_bounding_boxes()`` in the released evaluator does::

        if all(0 <= elem <= bins - 1 for elem in bbox_floats):
            bboxes.append(bbox_floats)

    with ``bins = 1000`` and **no else branch**. A coordinate of exactly 1000 —
    which is what the model emits for a box touching the right or bottom edge —
    makes the entire box vanish with no error and no warning. Clamping costs one
    part in a thousand of resolution and removes a whole class of silent AP loss.
    """
    return [int(max(0, min(OFFICIAL_MAX_COORD, round(v)))) for v in bbox]


def remap_crop_box_to_original(
    bbox_norm_in_crop: tuple[float, float, float, float],
    crop_px: tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
) -> list[float]:
    """Map a box predicted *inside a crop* back to original-image coordinates.

    Used only by the crop ablation (Phase 8.2). Getting this wrong silently
    destroys the ablation rather than failing, so it is unit-tested on a case
    with a hand-computed answer.
    """
    cx1, cy1, cx2, cy2 = crop_px
    cw, ch = cx2 - cx1, cy2 - cy1
    x1, y1, x2, y2 = bbox_norm_in_crop
    px = (
        cx1 + x1 * cw / 1000.0,
        cy1 + y1 * ch / 1000.0,
        cx1 + x2 * cw / 1000.0,
        cy1 + y2 * ch / 1000.0,
    )
    return px_to_norm1000(px, orig_w, orig_h)
