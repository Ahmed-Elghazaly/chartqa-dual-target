"""The mandatory box-correctness self-test for generated charts.

`PLAN.md` 3.5 requires it: *"For a sample of generated charts, re-render the box
onto the image and assert the pixels inside it actually contain the intended
element."* A generator whose boxes are subtly wrong is very hard to detect later,
so this runs on every generation batch rather than on request.

The check is deliberately two-sided. "Does the box contain the element" passes
trivially for a box that is far too large, so the element must also **fill** the
box, and a box displaced by most of its own width must **fail**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

RGB = tuple[int, int, int]


def render_rgb(fig: Any) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3]


def _crop(img: np.ndarray, box) -> np.ndarray:
    x1, y1, x2, y2 = (round(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=img.dtype)
    return img[y1:y2, x1:x2]


def colour_fraction(img: np.ndarray, box, colour: RGB, tol: int = 12) -> float:
    """Fraction of pixels inside `box` within `tol` of `colour`."""
    crop = _crop(img, box)
    if crop.size == 0:
        return 0.0
    return float((np.abs(crop.astype(int) - np.array(colour)) <= tol).all(axis=-1).mean())


def containment(img: np.ndarray, box, colour: RGB, tol: int = 12) -> float:
    """Fraction of ALL pixels of `colour` in the image that fall inside `box`.

    The right check for a non-rectangular artist such as a pie wedge, where
    exclusion of other colours is impossible by construction.
    """
    mask = (np.abs(img.astype(int) - np.array(colour)) <= tol).all(axis=-1)
    total = int(mask.sum())
    if total == 0:
        return 0.0
    # Floor the near edges and ceil the far ones: a pixel the box only partly covers
    # is still inside it. Rounding both ways instead drops a boundary row or column,
    # which on a thin bar is a few percent of its ink and read as a bad box.
    x1, y1 = math.floor(box[0]), math.floor(box[1])
    x2, y2 = math.ceil(box[2]), math.ceil(box[3])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    return float(mask[y1:y2, x1:x2].sum()) / total


@dataclass
class BoxCheck:
    """The outcome of verifying one box, with the numbers that produced it.

    `fill` and `expanded_fill` are reported even though neither gates acceptance any
    more: when a box is rejected they are what makes the failure readable.
    """

    label: str
    fill: float
    expanded_fill: float
    ok: bool
    reason: str = ""


#: How far `expand` grows a box. Expanding by f multiplies the area by (1 + 2f)^2,
#: which is what makes the relative-tightness floor below have a closed form.
EXPAND_FACTOR = 0.35


def ink_bbox_iou(img: np.ndarray, box, colour: RGB, tol: int = 12,
                 slack_px: float = 1.0) -> float:
    """IoU between `box` and the tight bounding box of the element's ink near it.

    The decisive test, and the only one that is genuinely shape-independent: rather
    than compare the fill to a constant that depends on the shape, it compares the box
    to where the ink actually *is*.

    Ink is collected from ``expand(box)`` rather than the whole image so that the test
    still works when several elements share a colour — line markers, for instance,
    where `containment` is meaningless. Neighbours sit far outside that window.

    `slack_px` grows the measured ink box by a pixel on each side: an antialiased edge
    blends into the background and falls outside the match tolerance, so raw ink reads
    about a pixel small in every direction.
    """
    region = expand(box)
    crop = _crop(img, region)
    if crop.size == 0:
        return 0.0
    mask = (np.abs(crop.astype(int) - np.array(colour)) <= tol).all(axis=-1)
    if not mask.any():
        return 0.0
    ys, xs = np.nonzero(mask)
    ox, oy = max(0.0, region[0]), max(0.0, region[1])
    ink = (ox + xs.min() - slack_px, oy + ys.min() - slack_px,
           ox + xs.max() + 1 + slack_px, oy + ys.max() + 1 + slack_px)

    ix1, iy1 = max(box[0], ink[0]), max(box[1], ink[1])
    ix2, iy2 = min(box[2], ink[2]), min(box[3], ink[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_ink = (ink[2] - ink[0]) * (ink[3] - ink[1])
    union = area_box + area_ink - inter
    return 0.0 if union <= 0 else inter / union


def expand(box, factor: float = EXPAND_FACTOR):
    """Grow a box outward by `factor` of its own size on every side."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * factor, (y2 - y1) * factor
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


# ======================================================================================
# Per-geometry acceptance — every threshold below was MEASURED on rendered pixels.
#
# Two tests do the work, and neither depends on the element being a rectangle:
#
#   ink_bbox_iou  the box against the tight extent of the element's own ink nearby.
#                 Decisive and shape-independent. Measured across all eight chart types:
#                 exact boxes 0.841-0.990; shifted by half a width 0.365-0.377; shrunk
#                 to 0.6x 0.315-0.359; grown to 1.8x 0.312-0.352. The floor of 0.70 sits
#                 with roughly 2x margin on either side.
#   containment   fraction of ALL the element's ink that lands inside the box. Exact for
#                 any shape, but only meaningful when the colour identifies the element
#                 uniquely. Measured 100% for exact boxes. Kept as a second, independent
#                 signal wherever it applies: it catches ink of the element living
#                 somewhere the local window would never look.
#
# Two earlier designs were rejected against measurement rather than argued about:
#
#   displacement  sliding the box sideways and requiring the fill to collapse. Fails on
#                 a bar chart, where the displaced box lands on the *next bar* — it
#                 reported false failures on exact boxes.
#   tightness     the fill drop on expansion, relative to the original fill. Elegant —
#                 an exact box loses 1 - 1/(1 + 2f)^2 = 65.4% at f = 0.35 regardless of
#                 shape, and measurement agreed (60.6-65.6%). But it is scale-invariant
#                 for exactly the same reason, so it cannot see an oversized box: a pie
#                 wedge box grown 1.8x passed it. `ink_bbox_iou` scores that 0.312.
#
# Absolute `fill` is not gated at all. It is shape-dependent in a way no single number
# survives: a bar reaches ~100%, a disc inscribed in its square reaches pi/4 = 78.5%,
# and a circular sector's tight bbox reaches only 21-78% depending on span — a
# 10-degree pie sliver measured 21.0%, which is CORRECT geometry, not a bad box.
# `BoxCheck` still reports fill because it is informative when reading a failure.
#
# Geometry classes:
#   rect         bars.               unique colour -> containment applies.
#   wedge        pie slices.         unique colour -> containment applies.
#   disc_unique  scatter points.     unique colour -> containment applies.
#   disc_shared  line/area markers.  colour is SHARED with the other markers on the
#                                    line, so containment reads ~12.5% for a perfect
#                                    box and must not be used; `ink_bbox_iou` measures
#                                    ink only within `expand(box)`, where neighbouring
#                                    markers do not reach, so it still applies.
# ======================================================================================

GEOMETRY_THRESHOLDS: dict[str, dict[str, float | None]] = {
    "rect":        {"min_containment": 0.98, "min_ink_iou": 0.70},
    "wedge":       {"min_containment": 0.98, "min_ink_iou": 0.70},
    "disc_unique": {"min_containment": 0.98, "min_ink_iou": 0.70},
    "disc_shared": {"min_containment": None, "min_ink_iou": 0.70},
}


def check_box_for(img: np.ndarray, box, colour: RGB, label: str, geometry: str) -> BoxCheck:
    """Verify `box` using the tests that are valid for `geometry`.

    Raises on an unknown geometry rather than falling back to a default: a silent
    default would apply a rectangle's assumptions to a wedge, which is exactly the
    mistake this table exists to prevent.
    """
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
            return BoxCheck(label, fill, expanded_fill, False,
                            f"only {100 * got:.1f}% of the element's ink is inside the box "
                            f"(min {100 * min_c:.0f}%)")

    min_iou = t["min_ink_iou"]
    if min_iou is not None:
        iou = ink_bbox_iou(img, box, colour)
        if iou < min_iou:
            return BoxCheck(label, fill, expanded_fill, False,
                            f"box vs the element's actual ink extent: IoU {iou:.3f} "
                            f"(min {min_iou:.2f}) — the box is offset or the wrong size")
    return BoxCheck(label, fill, expanded_fill, True)
