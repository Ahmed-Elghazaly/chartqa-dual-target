"""An automated pre-screen for the RefChartQA box audit, proven on known ground truth.

`PLAN.md` 3.4 gates RefChartQA training data on a manual audit of 200 boxes:
"a box is acceptable if it plausibly contains evidence a person would use to
answer that question". Two hundred human judgements is an afternoon, and human
judgement drifts over an afternoon.

This does not replace the judgement. It provides one objective signal that can be
computed for **every** box, not just the 200: **how much chart ink does the box
actually contain?** A box sitting on blank canvas contains no evidence by
definition, whatever the question was.

The technique is proven here against synthetic charts whose correct boxes are
known by construction — from the artist geometry proven in
`prove_box_extraction.py` — so the detector is validated before it is pointed at
data whose labels are the thing in question.

Run:  python verification/prove_ink_prescreen.py
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ink_fraction(img: np.ndarray, box, background_tol: int = 18) -> float:
    """Fraction of pixels in `box` that are not background.

    Background is taken as the image's modal colour, which for a chart is the
    canvas. Using the modal colour rather than assuming white makes this work on
    dark or tinted themes, which RefChartQA does contain.
    """
    x1, y1, x2, y2 = (round(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    flat = img.reshape(-1, 3)
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    background = colours[counts.argmax()]

    crop = img[y1:y2, x1:x2].astype(int)
    is_background = (np.abs(crop - background) <= background_tol).all(axis=-1)
    return float(1.0 - is_background.mean())


def render(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3]


def bar_boxes(fig, bars):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    _, h = fig.canvas.get_width_height()
    out = []
    for b in bars:
        bb = b.get_window_extent(renderer=r)
        out.append((bb.x0, h - bb.y1, bb.x1, h - bb.y0))
    return out


def main() -> int:
    failures: list[str] = []

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    bars = list(ax.bar(list("ABCD"), [37, 82, 55, 19], color="#3060c8"))
    ax.set_ylim(0, 100)
    img = render(fig)
    boxes = bar_boxes(fig, bars)
    h, w = img.shape[:2]
    plt.close(fig)

    print(f"canvas {w}x{h}\n")
    print("1. TRUE boxes (known by construction) must be ink-rich")
    for i, box in enumerate(boxes):
        frac = ink_fraction(img, box)
        print(f"   bar {'ABCD'[i]}  ink = {100 * frac:5.1f}%")
        if frac < 0.90:
            failures.append(f"true box for bar {'ABCD'[i]} has only {100 * frac:.1f}% ink")

    print("\n2. EMPTY regions must be ink-poor (the failure the pre-screen catches)")
    # Derive the empty band from the geometry rather than choosing coordinates by
    # eye. The first attempt hard-coded a box that clipped the top of the tallest
    # bar and reported 6% ink -- the detector was right and the test case was wrong.
    highest_top = min(b[1] for b in boxes)          # smallest y = tallest bar
    left = min(b[0] for b in boxes)
    right = max(b[2] for b in boxes)
    band_bottom = highest_top - 10                  # strictly above every bar
    mid = (left + right) / 2
    empties = {
        "blank band, left half": (left, band_bottom - 45, mid, band_bottom),
        "blank band, right half": (mid, band_bottom - 45, right, band_bottom),
    }
    print(f"   (band derived from geometry: y < {band_bottom:.0f}, above the tallest bar)")
    for name, box in empties.items():
        frac = ink_fraction(img, box)
        print(f"   {name:<30} ink = {100 * frac:5.1f}%")
        if frac > 0.05:
            failures.append(f"{name} shows {100 * frac:.1f}% ink; the pre-screen would not flag it")

    print("\n3. The signal must SEPARATE the two populations")
    true_ink = [ink_fraction(img, b) for b in boxes]
    empty_ink = [ink_fraction(img, b) for b in empties.values()]
    margin = min(true_ink) - max(empty_ink)
    print(f"   worst true box = {100 * min(true_ink):.1f}%   best empty box = {100 * max(empty_ink):.1f}%")
    print(f"   separation margin = {100 * margin:.1f} percentage points")
    if margin < 0.50:
        failures.append(f"separation is only {100 * margin:.1f} points; the threshold would be fragile")

    print("\n4. A box half on the bar and half off must fall in between")
    box = boxes[1]
    bw = box[2] - box[0]
    half = (box[0] + bw * 0.5, box[1], box[2] + bw * 0.5, box[3])
    frac = ink_fraction(img, half)
    print(f"   half-on-half-off ink = {100 * frac:5.1f}%  (expected between the two populations)")
    if not (0.10 < frac < 0.95):
        failures.append(f"a half-overlapping box scores {100 * frac:.1f}%, which is not intermediate")

    print("\n5. Background detection must not assume white")
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100, facecolor="#20242b")
    ax.set_facecolor("#20242b")
    dark_bars = list(ax.bar(list("XY"), [60, 30], color="#e0a020"))
    ax.set_ylim(0, 100)
    dimg = render(fig)
    dboxes = bar_boxes(fig, dark_bars)
    plt.close(fig)
    dark_true = ink_fraction(dimg, dboxes[0])
    print(f"   dark-theme chart, true box ink = {100 * dark_true:5.1f}%")
    if dark_true < 0.90:
        failures.append(f"dark theme: true box shows only {100 * dark_true:.1f}% ink — "
                        "background detection is assuming white")

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("INK PRE-SCREEN VALIDATED on ground truth known by construction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
