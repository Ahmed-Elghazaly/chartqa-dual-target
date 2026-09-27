from __future__ import annotations

import math
from typing import Any

import matplotlib

Box = tuple[float, float, float, float]


def points_to_pixels(points: float, dpi: float) -> float:
    return points * dpi / 72.0


def _flip(bbox: Any, height_px: int) -> Box:
    return (bbox.x0, height_px - bbox.y1, bbox.x1, height_px - bbox.y0)


def artist_box(fig: Any, artist: Any) -> Box:
    fig.canvas.draw()
    _, height = fig.canvas.get_width_height()
    return _flip(artist.get_window_extent(renderer=fig.canvas.get_renderer()), height)


def point_box(fig: Any, ax: Any, x: float, y: float, marker_points: float, edge_points: float = 0.0) -> Box:
    fig.canvas.draw()
    _, height = fig.canvas.get_width_height()
    px, py = ax.transData.transform((x, y))
    r = points_to_pixels(marker_points + edge_points, fig.dpi) / 2.0
    return (px - r, height - (py + r), px + r, height - (py - r))


def scatter_point_box(
    fig: Any, ax: Any, x: float, y: float, s_points_squared: float, edge_points: float | None = None
) -> Box:
    if edge_points is None:
        edge_points = float(matplotlib.rcParams["lines.linewidth"])
    return point_box(fig, ax, x, y, math.sqrt(s_points_squared), edge_points)


def clip_to_canvas(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(x1, width)),
        max(0.0, min(y1, height)),
        max(0.0, min(x2, width)),
        max(0.0, min(y2, height)),
    )


def is_degenerate(box: Box, min_side: float = 1.0) -> bool:
    x1, y1, x2, y2 = box
    return (x2 - x1) < min_side or (y2 - y1) < min_side
