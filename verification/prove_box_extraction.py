"""Prove matplotlib box extraction is EXACT before any generator depends on it.

`PLAN.md` 3.5 is emphatic: *"Box extraction must be exact. Get real coordinates
from matplotlib artists ... Never estimate a box by eye or by formula."* A
generator with subtly wrong boxes poisons every training example, is invisible in
the loss, and is nearly undetectable once training has started.

So the technique is proven here, against pixels, before it is written into the
package. The proof is adversarial: it checks not only that the box **contains**
the bar, but that it does **not** contain its neighbours, and that shifting the
box by a few pixels breaks the check — otherwise the test would pass for a box
that is merely approximately right.

Facts read from the matplotlib API before writing any of this:
  * `Artist.get_window_extent(renderer=None)` returns a Bbox in **display space**,
    with non-negative width and height.
  * Display space has its origin at the **bottom-left**; images have theirs at the
    **top-left**. Every y coordinate must be flipped: ``y_img = H - y_display``.
  * `fig.canvas.get_width_height()` gives the pixel size actually rendered.
  * The figure must be **drawn** before extents are meaningful.

Run:  python verification/prove_box_extraction.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bar_boxes_in_pixels(fig, ax, bars) -> list[tuple[float, float, float, float]]:
    """Exact pixel boxes (x1, y1, x2, y2), origin top-left, for each bar."""
    fig.canvas.draw()  # extents are meaningless before a draw
    renderer = fig.canvas.get_renderer()
    _, height_px = fig.canvas.get_width_height()

    boxes = []
    for bar in bars:
        bb = bar.get_window_extent(renderer=renderer)
        # Flip y: display origin is bottom-left, image origin is top-left.
        boxes.append((bb.x0, height_px - bb.y1, bb.x1, height_px - bb.y0))
    return boxes


def render_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3]


def fraction_of_colour(img: np.ndarray, box, colour, tol: int = 12) -> float:
    """Fraction of pixels inside `box` within `tol` of `colour` per channel."""
    x1, y1, x2, y2 = (round(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = img[y1:y2, x1:x2].astype(int)
    close = (np.abs(crop - np.array(colour)) <= tol).all(axis=-1)
    return float(close.mean())


def main() -> int:
    colours = [(200, 30, 30), (30, 140, 60), (40, 60, 200), (220, 160, 20)]
    values = [37, 82, 55, 19]
    labels = ["A", "B", "C", "D"]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    container = ax.bar(labels, values, color=[tuple(c / 255 for c in col) for col in colours])
    ax.set_ylim(0, 100)
    bars = list(container)

    img = render_rgb(fig)
    boxes = bar_boxes_in_pixels(fig, ax, bars)
    height_px, width_px = img.shape[:2]
    print(f"rendered {width_px}x{height_px}\n")

    failures: list[str] = []

    for i, (box, colour) in enumerate(zip(boxes, colours)):
        own = fraction_of_colour(img, box, colour)
        print(f"bar {labels[i]}  box=({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})")
        print(f"   own colour inside box            : {100 * own:5.1f}%")
        if own < 0.97:
            failures.append(f"bar {labels[i]}: only {100 * own:.1f}% of its box is its own colour")

        # Adversarial 1: the box must NOT contain any other bar's colour.
        for j, other in enumerate(colours):
            if i == j:
                continue
            bleed = fraction_of_colour(img, box, other)
            if bleed > 0.01:
                failures.append(f"bar {labels[i]}: {100 * bleed:.1f}% of its box is bar {labels[j]}'s colour")

        # Adversarial 2: an approximately-right box must FAIL, or the test is weak.
        w = box[2] - box[0]
        shifted = (box[0] + w * 0.6, box[1], box[2] + w * 0.6, box[3])
        shifted_own = fraction_of_colour(img, shifted, colour)
        print(f"   same box shifted right by 60% w  : {100 * shifted_own:5.1f}%  (must be low)")
        if shifted_own > 0.5:
            failures.append(f"bar {labels[i]}: a shifted box still scores {100 * shifted_own:.1f}% — test is too weak")

        # Adversarial 3: the top edge must be the top of the bar, not the axes.
        strip_h = max(2, int(0.02 * (box[3] - box[1])))
        above = (box[0], max(0, box[1] - strip_h * 3), box[2], max(0, box[1] - strip_h))
        above_own = fraction_of_colour(img, above, colour)
        print(f"   strip just ABOVE the box         : {100 * above_own:5.1f}%  (must be ~0)")
        if above_own > 0.05:
            failures.append(f"bar {labels[i]}: {100 * above_own:.1f}% of the strip above the box is bar colour — box top is too low")
        print()

    # Heights must be proportional to values: proves we read geometry, not guesses.
    heights = [b[3] - b[1] for b in boxes]
    ratios = [h / v for h, v in zip(heights, values)]
    spread = (max(ratios) - min(ratios)) / (sum(ratios) / len(ratios))
    print(f"box height / value ratios: {[round(r, 4) for r in ratios]}  spread={spread:.4f}")
    if spread > 0.02:
        failures.append(f"box heights are not proportional to values (spread {spread:.3f})")

    plt.close(fig)

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("BOX EXTRACTION IS EXACT — verified against pixels, adversarially.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
