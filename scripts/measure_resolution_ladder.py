"""Visual-token cost and sub-token fraction as a function of input resolution.

Evidence for `DECISIONS.md` 0008: `IDEA.md` warns that raising resolution
quadruples the token budget (~580 -> ~2280). That does not hold for these
images. RefChartQA charts are around 800 px wide, so any `max_pixels` cap at or
above 768^2 is already native.

Samples the **validation** split only (rule 1).

Run:  python scripts/measure_resolution_ladder.py
"""

from __future__ import annotations

import statistics

from measure_subtoken import fetch_rows, smart_resize

FACTOR = 32  # Qwen3-VL: patch_size 16 * spatial_merge_size 2
MIN_PIXELS = 65536  # the model's own preprocessor size.shortest_edge
UNBOUNDED = 10**9


def measure(rows: list[dict], *, max_pixels: int | None = None, longest_edge: int | None = None):
    """Return (median tokens, % sub-token by axis, % sub-token by area).

    Two interpretations of "a 512-pixel image":
      * ``max_pixels``   - cap the total pixel count at R*R (what the processor does)
      * ``longest_edge`` - scale so the longest side is R, then let smart_resize round
    """
    tokens: list[int] = []
    sub_axis = sub_area = total = 0
    for row in rows:
        width, height = row["W"], row["H"]
        if longest_edge is not None:
            scale = longest_edge / max(width, height)
            rh, rw = smart_resize(height * scale, width * scale, FACTOR, MIN_PIXELS, UNBOUNDED)
        else:
            rh, rw = smart_resize(height, width, FACTOR, MIN_PIXELS, max_pixels or UNBOUNDED)
        tokens.append((rh // FACTOR) * (rw // FACTOR))
        scale_x, scale_y = rw / width, rh / height
        for box in row["boxes"]:
            tw = box["w"] * scale_x / FACTOR
            th = box["h"] * scale_y / FACTOR
            total += 1
            if min(tw, th) < 1.0:
                sub_axis += 1
            if tw * th < 1.0:
                sub_area += 1
    return statistics.median(tokens), 100 * sub_axis / total, 100 * sub_area / total


def main() -> None:
    rows = fetch_rows()
    header = f"{'setting':<34}{'medTok':>8}{'subtok(axis)':>14}{'subtok(area)':>14}"
    print(header)
    print("-" * len(header))
    for r in (448, 512, 640, 768, 896, 1024):
        med, axis, area = measure(rows, max_pixels=r * r)
        print(f"{'max_pixels = ' + str(r) + '^2':<34}{med:>8.0f}{axis:>13.1f}%{area:>13.1f}%")
    print("-" * len(header))
    for r in (448, 512, 640, 768, 896, 1024):
        med, axis, area = measure(rows, longest_edge=r)
        print(f"{'longest edge = ' + str(r):<34}{med:>8.0f}{axis:>13.1f}%{area:>13.1f}%")
    print("-" * len(header))
    med, axis, area = measure(rows, max_pixels=16777216)
    print(f"{'native (no downscale)':<34}{med:>8.0f}{axis:>13.1f}%{area:>13.1f}%")


if __name__ == "__main__":
    main()
