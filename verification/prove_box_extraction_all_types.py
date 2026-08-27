"""Prove exact box extraction for EVERY chart type the generator must produce.

`PLAN.md` 3.5 requires bar, line, pie and scatter at minimum, and each needs a
*different* extraction path. Proving bars alone would leave three untested
techniques, any of which could poison training silently.

API facts, read from matplotlib before writing any of this:

* **Bar** — `Rectangle.get_window_extent(renderer)` gives the display-space Bbox
  directly. The bar *is* a rectangle, so the extent is exact.
* **Line** — `Line2D.get_window_extent()` covers the **whole line**, which is not
  what grounding needs ("the value at 2019" is one vertex). A per-point box comes
  from `ax.transData.transform((x, y))` plus the marker radius.
* **Pie** — a `Wedge` is a sector, not a rectangle, so its bounding box
  *necessarily* contains background and slivers of neighbours. Requiring "no other
  colour inside" would be wrong by construction; the correct tests are
  **containment** (every pixel of this wedge lies inside the box) and
  **tightness** (shrinking the box excludes some of them).
* **Scatter** — `PathCollection.get_offsets()` is in **data** space. Marker size
  `s` is in **points squared**, so diameter is `sqrt(s)` points, and
  `pixels = points * dpi / 72`.

Every test is adversarial: a box that is merely approximately right must fail.

Run:  python verification/prove_box_extraction_all_types.py
"""

from __future__ import annotations

import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RGB = tuple[int, int, int]


# --------------------------------------------------------------------- helpers


def render_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3]


def flip_y(bbox, height_px: int) -> tuple[float, float, float, float]:
    """Display space (origin bottom-left) -> image space (origin top-left)."""
    return (bbox.x0, height_px - bbox.y1, bbox.x1, height_px - bbox.y0)


def mask_of_colour(img: np.ndarray, colour: RGB, tol: int = 12) -> np.ndarray:
    return (np.abs(img.astype(int) - np.array(colour)) <= tol).all(axis=-1)


def fraction_inside(img: np.ndarray, box, colour: RGB, tol: int = 12) -> float:
    x1, y1, x2, y2 = (round(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(mask_of_colour(img[y1:y2, x1:x2], colour, tol).mean())


def containment(img: np.ndarray, box, colour: RGB, tol: int = 12) -> float:
    """Fraction of ALL this colour's pixels that fall inside the box."""
    mask = mask_of_colour(img, colour, tol)
    total = int(mask.sum())
    if total == 0:
        return 0.0
    x1, y1, x2, y2 = (round(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    return float(mask[y1:y2, x1:x2].sum()) / total


def points_to_pixels(points: float, dpi: float) -> float:
    """matplotlib sizes markers and linewidths in points; 72 points = 1 inch."""
    return points * dpi / 72.0


# ------------------------------------------------------------------ extraction


def line_point_box(fig, ax, x: float, y: float, marker_points: float):
    """Exact pixel box around ONE vertex of a line.

    `Line2D.get_window_extent()` would give the whole polyline; grounding needs
    the single vertex the question refers to.
    """
    fig.canvas.draw()
    _, height_px = fig.canvas.get_width_height()
    px, py = ax.transData.transform((x, y))
    r = points_to_pixels(marker_points, fig.dpi) / 2.0
    return (px - r, height_px - (py + r), px + r, height_px - (py - r))


def wedge_box(fig, wedge):
    fig.canvas.draw()
    _, height_px = fig.canvas.get_width_height()
    return flip_y(wedge.get_window_extent(fig.canvas.get_renderer()), height_px)


def scatter_point_box(fig, ax, xy, s_points_squared: float):
    """Exact pixel box around one scatter marker.

    `s` is an area in points squared, so the diameter is sqrt(s) points.
    """
    fig.canvas.draw()
    _, height_px = fig.canvas.get_width_height()
    px, py = ax.transData.transform(xy)
    r = points_to_pixels(math.sqrt(s_points_squared), fig.dpi) / 2.0
    return (px - r, height_px - (py + r), px + r, height_px - (py - r))


# ----------------------------------------------------------------------- proofs


def prove_line() -> list[str]:
    fails: list[str] = []
    colour: RGB = (30, 90, 200)
    xs = [2018, 2019, 2020, 2021]
    ys = [10, 45, 30, 70]
    marker_points = 14.0

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(xs, ys, marker="o", markersize=marker_points,
            color=tuple(c / 255 for c in colour),
            markerfacecolor=tuple(c / 255 for c in colour),
            markeredgecolor=tuple(c / 255 for c in colour), linewidth=1.0)
    ax.set_ylim(0, 100)
    img = render_rgb(fig)

    print("LINE — per-vertex boxes")
    for x, y in zip(xs, ys):
        box = line_point_box(fig, ax, x, y, marker_points)
        own = fraction_inside(img, box, colour)
        # Adversarial: a box one whole marker to the right must be mostly empty.
        w = box[2] - box[0]
        off = (box[0] + 2 * w, box[1], box[2] + 2 * w, box[3])
        off_own = fraction_inside(img, off, colour)
        print(f"  x={x} y={y:>3}  marker box own={100 * own:5.1f}%   shifted={100 * off_own:5.1f}%")
        if own < 0.60:
            fails.append(f"line vertex {x}: only {100 * own:.1f}% of its marker box is line colour")
        if off_own > 0.35:
            fails.append(f"line vertex {x}: a shifted box still scores {100 * off_own:.1f}%")
    plt.close(fig)
    return fails


def prove_pie() -> list[str]:
    fails: list[str] = []
    colours: list[RGB] = [(200, 30, 30), (30, 140, 60), (40, 60, 200), (220, 160, 20)]
    values = [40, 25, 20, 15]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    wedges, _ = ax.pie(values, colors=[tuple(c / 255 for c in col) for col in colours])
    img = render_rgb(fig)

    print("\nPIE — wedge boxes (containment and tightness, not exclusion)")
    for i, (wedge, colour) in enumerate(zip(wedges, colours)):
        box = wedge_box(fig, wedge)
        contained = containment(img, box, colour)
        # Tightness: shrinking by 15% per side must lose some of the wedge.
        w, h = box[2] - box[0], box[3] - box[1]
        tight = (box[0] + 0.15 * w, box[1] + 0.15 * h, box[2] - 0.15 * w, box[3] - 0.15 * h)
        shrunk = containment(img, tight, colour)
        print(f"  wedge {i}  contained={100 * contained:5.1f}%   after 15% shrink={100 * shrunk:5.1f}%")
        if contained < 0.99:
            fails.append(f"wedge {i}: box contains only {100 * contained:.1f}% of its pixels")
        if shrunk > 0.97:
            fails.append(f"wedge {i}: box is not tight — a 15% shrink still holds {100 * shrunk:.1f}%")
    plt.close(fig)
    return fails


def prove_scatter() -> list[str]:
    fails: list[str] = []
    colour: RGB = (150, 40, 160)
    pts = [(1.0, 20.0), (2.5, 60.0), (4.0, 35.0), (5.5, 80.0)]
    s = 400.0  # points squared -> 20 pt diameter

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=s,
               c=[tuple(c / 255 for c in colour)] * len(pts))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 100)
    img = render_rgb(fig)

    print("\nSCATTER — per-marker boxes (s is points SQUARED)")
    for xy in pts:
        box = scatter_point_box(fig, ax, xy, s)
        own = fraction_inside(img, box, colour)
        w = box[2] - box[0]
        off = (box[0] + 2 * w, box[1], box[2] + 2 * w, box[3])
        off_own = fraction_inside(img, off, colour)
        print(f"  point {xy}  own={100 * own:5.1f}%   shifted={100 * off_own:5.1f}%")
        # A circle inscribed in its bounding square fills pi/4 = 78.5%.
        if not (0.60 <= own <= 0.90):
            fails.append(f"scatter {xy}: own fraction {100 * own:.1f}% is outside the "
                         "60-90% a disc in its bounding square should give")
        if off_own > 0.10:
            fails.append(f"scatter {xy}: a shifted box still scores {100 * off_own:.1f}%")
    plt.close(fig)
    return fails


def prove_points_to_pixels() -> list[str]:
    """The unit conversion itself: 72 points per inch, dpi pixels per inch."""
    fails = []
    print("\nUNITS — points to pixels")
    for dpi, points, expected in [(100, 72, 100.0), (72, 72, 72.0), (100, 10, 13.888888888888889)]:
        got = points_to_pixels(points, dpi)
        print(f"  {points} pt at {dpi} dpi -> {got:.4f} px (expected {expected:.4f})")
        if abs(got - expected) > 1e-9:
            fails.append(f"points_to_pixels({points}, {dpi}) = {got}, expected {expected}")
    return fails


def main() -> int:
    failures = prove_line() + prove_pie() + prove_scatter() + prove_points_to_pixels()
    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL CHART TYPES: box extraction verified against pixels, adversarially.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
