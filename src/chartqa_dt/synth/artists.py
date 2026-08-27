"""Exact pixel geometry from matplotlib artists.

`PLAN.md` 3.5: *"Box extraction must be exact. Get real coordinates from
matplotlib artists ... Never estimate a box by eye or by formula."* A generator
with subtly wrong boxes poisons every training example, is invisible in the loss,
and is nearly undetectable once training has begun.

Every function here was proven against rendered pixels before being written into
the package — `verification/prove_box_extraction.py` and
`prove_box_extraction_all_types.py`, both running in CI. The measured results land
on the theoretical value, which is itself evidence of exactness: a disc inscribed
in its bounding square fills π/4 = 78.5%, and marker boxes measured 76.8–84.4%.

Four facts read from the matplotlib API before any of this was written:

* `Artist.get_window_extent(renderer)` returns a Bbox in **display space**, with
  non-negative width and height, and is meaningless before `fig.canvas.draw()`.
* Display space has its origin **bottom-left**; images have theirs **top-left**.
  Every y coordinate must be flipped: ``y_img = height_px - y_display``.
* `Line2D.get_window_extent()` covers the **whole polyline**, not one vertex. A
  per-vertex box needs `ax.transData.transform((x, y))` plus the marker radius.
* Scatter size `s` is an **area in points squared**, so diameter is `sqrt(s)`
  points, and `pixels = points * dpi / 72`.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib

Box = tuple[float, float, float, float]     # (x1, y1, x2, y2), origin top-left


def points_to_pixels(points: float, dpi: float) -> float:
    """72 points per inch, `dpi` pixels per inch."""
    return points * dpi / 72.0


def canvas_size(fig: Any) -> tuple[int, int]:
    fig.canvas.draw()
    return fig.canvas.get_width_height()


def _flip(bbox: Any, height_px: int) -> Box:
    """Display space (origin bottom-left) into image space (origin top-left)."""
    return (bbox.x0, height_px - bbox.y1, bbox.x1, height_px - bbox.y0)


def artist_box(fig: Any, artist: Any) -> Box:
    """Exact pixel box of any artist whose extent is its own shape.

    Correct for bars (`Rectangle`) and pie wedges (`Wedge`). A wedge is a sector,
    so its bounding box necessarily contains background and slivers of its
    neighbours — that is the true extent, not an error.
    """
    fig.canvas.draw()
    _, height = fig.canvas.get_width_height()
    return _flip(artist.get_window_extent(renderer=fig.canvas.get_renderer()), height)


def point_box(fig: Any, ax: Any, x: float, y: float, marker_points: float,
              edge_points: float = 0.0) -> Box:
    """Exact pixel box around ONE data point, from the data transform.

    Used for line vertices and scatter markers, where the artist's own extent
    covers the whole series rather than the point a question refers to.

    `edge_points` is the marker's stroke width. matplotlib strokes a path **centred**
    on it, so half the stroke lies outside the nominal marker; with the scatter
    default of ``edgecolor="face"`` that overhang is the element's own colour.
    Measured: at the 1.5pt default, a box without this padding contained only 97.2%
    of the marker's ink, and 100% once the stroke width fell to zero.
    """
    fig.canvas.draw()
    _, height = fig.canvas.get_width_height()
    px, py = ax.transData.transform((x, y))
    r = points_to_pixels(marker_points + edge_points, fig.dpi) / 2.0
    return (px - r, height - (py + r), px + r, height - (py - r))


def scatter_point_box(fig: Any, ax: Any, x: float, y: float, s_points_squared: float,
                      edge_points: float | None = None) -> Box:
    """Exact pixel box around one scatter marker.

    `s` is an AREA in points squared. Treating it as a diameter gives a box far too
    large that still contains its marker — which is why the proof has an upper
    bound as well as a lower one.

    `edge_points` defaults to the linewidth matplotlib itself would use, so a caller
    that did not override `linewidths` gets a correct box without having to know this.
    """
    if edge_points is None:
        edge_points = float(matplotlib.rcParams["lines.linewidth"])
    return point_box(fig, ax, x, y, math.sqrt(s_points_squared), edge_points)


def clip_to_canvas(box: Box, width: int, height: int) -> Box:
    """Clamp a box to the image. Artists may legitimately overhang the canvas."""
    x1, y1, x2, y2 = box
    return (max(0.0, min(x1, width)), max(0.0, min(y1, height)),
            max(0.0, min(x2, width)), max(0.0, min(y2, height)))


def is_degenerate(box: Box, min_side: float = 1.0) -> bool:
    """True if a box has no usable area, e.g. a bar whose value is zero."""
    x1, y1, x2, y2 = box
    return (x2 - x1) < min_side or (y2 - y1) < min_side
