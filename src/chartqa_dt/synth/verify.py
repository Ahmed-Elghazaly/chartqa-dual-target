from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

RGB = tuple[int, int, int]


def render_rgb(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3]


def _crop(img: np.ndarray, box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = (round(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=img.dtype)
    return img[y1:y2, x1:x2]


def colour_fraction(img: np.ndarray, box: Sequence[float], colour: RGB, tol: int = 12) -> float:
    crop = _crop(img, box)
    if crop.size == 0:
        return 0.0
    return float((np.abs(crop.astype(int) - np.array(colour)) <= tol).all(axis=-1).mean())


def containment(img: np.ndarray, box: Sequence[float], colour: RGB, tol: int = 12) -> float:
    mask = (np.abs(img.astype(int) - np.array(colour)) <= tol).all(axis=-1)
    total = int(mask.sum())
    if total == 0:
        return 0.0
    x1, y1 = math.floor(box[0]), math.floor(box[1])
    x2, y2 = math.ceil(box[2]), math.ceil(box[3])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    return float(mask[y1:y2, x1:x2].sum()) / total


@dataclass
class BoxCheck:
    label: str
    fill: float
    expanded_fill: float
    ok: bool
    reason: str = ""


EXPAND_FACTOR = 0.35


def ink_bbox_iou(
    img: np.ndarray,
    box: Sequence[float],
    colour: RGB,
    tol: int = 12,
    slack_px: float = 1.0,
) -> float:
    region = expand(box)
    crop = _crop(img, region)
    if crop.size == 0:
        return 0.0
    mask = (np.abs(crop.astype(int) - np.array(colour)) <= tol).all(axis=-1)
    if not mask.any():
        return 0.0
    ys, xs = np.nonzero(mask)
    ox, oy = max(0.0, region[0]), max(0.0, region[1])
    ink = (
        ox + xs.min() - slack_px,
        oy + ys.min() - slack_px,
        ox + xs.max() + 1 + slack_px,
        oy + ys.max() + 1 + slack_px,
    )

    ix1, iy1 = max(box[0], ink[0]), max(box[1], ink[1])
    ix2, iy2 = min(box[2], ink[2]), min(box[3], ink[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_ink = (ink[2] - ink[0]) * (ink[3] - ink[1])
    union = area_box + area_ink - inter
    return 0.0 if union <= 0 else inter / union


def expand(box: Sequence[float], factor: float = EXPAND_FACTOR) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * factor, (y2 - y1) * factor
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


GEOMETRY_THRESHOLDS: dict[str, dict[str, float | None]] = {
    "rect": {"min_containment": 0.98, "min_ink_iou": 0.70},
    "wedge": {"min_containment": 0.98, "min_ink_iou": 0.70},
    "disc_unique": {"min_containment": 0.98, "min_ink_iou": 0.70},
    "disc_shared": {"min_containment": None, "min_ink_iou": 0.70},
}


def check_box_for(
    img: np.ndarray,
    box: Sequence[float],
    colour: RGB,
    label: str,
    geometry: str,
) -> BoxCheck:
    try:
        t = GEOMETRY_THRESHOLDS[geometry]
    except KeyError:
        raise ValueError(
            f"unknown geometry {geometry!r}; expected one of {sorted(GEOMETRY_THRESHOLDS)}"
        ) from None

    fill = colour_fraction(img, box, colour)
    expanded_fill = colour_fraction(img, expand(box), colour)

    min_c = t["min_containment"]
    if min_c is not None:
        got = containment(img, box, colour)
        if got < min_c:
            return BoxCheck(
                label,
                fill,
                expanded_fill,
                False,
                f"only {100 * got:.1f}% of the element's ink is inside the box (min {100 * min_c:.0f}%)",
            )

    min_iou = t["min_ink_iou"]
    if min_iou is not None:
        iou = ink_bbox_iou(img, box, colour)
        if iou < min_iou:
            return BoxCheck(
                label,
                fill,
                expanded_fill,
                False,
                f"box vs the element's actual ink extent: IoU {iou:.3f} "
                f"(min {min_iou:.2f}) — the box is offset or the wrong size",
            )
    return BoxCheck(label, fill, expanded_fill, True)
