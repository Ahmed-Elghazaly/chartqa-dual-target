"""Measure what fraction of RefChartQA grounding targets are sub-token.

This is the script behind the corrected figure in `verification/phase0.md` F11
and `DECISIONS.md` 0008. It re-derives `IDEA.md` 5.2, which was computed with
factor 28 (Qwen2/2.5-VL) rather than the factor 32 that Qwen3-VL actually uses.

It samples the **validation** split. Non-negotiable rule 1 seals test.

Run:  python scripts/measure_subtoken.py
"""

from __future__ import annotations

import collections
import json
import math
import statistics
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import chartqa_dt.net  # noqa: F401  repairs the TLS trust store on import

ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=omoured/RefChartQA&config=default&split=validation&offset={off}&length=100"
)
N_ROWS = 800


def fetch_rows(n: int = N_ROWS) -> list[dict]:
    """Image dimensions and grounding boxes, straight from the dataset server."""
    rows: list[dict] = []
    for off in range(0, n, 100):
        with urllib.request.urlopen(ROWS_URL.format(off=off), timeout=90) as resp:
            payload = json.loads(resp.read())
        for item in payload["rows"]:
            row = item["row"]
            rows.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "W": row["image"]["width"],
                    "H": row["image"]["height"],
                    "boxes": row["grounding_bboxes"],
                }
            )
    return rows


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Port of transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize.

    Ported from what actually runs, not from PLAN.md Appendix C, which places the
    `max(factor, ...)` guard on the initial rounding rather than in the downscale
    branch and uses the wrong pixel bounds for this model.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
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
    return h_bar, w_bar


# (factor, min_pixels, max_pixels)
CONFIGS: dict[str, tuple[int, int, int]] = {
    "A  PLAN.md AppC assumption (f=28, min=3136,  max=12845056)": (28, 4 * 28 * 28, 16384 * 28 * 28),
    "B  REAL Qwen3-VL-2B      (f=32, min=65536, max=16777216)": (32, 65536, 16777216),
    "C  f=28 with real bounds (isolates the factor change)": (28, 65536, 16777216),
    "D  REAL f=32 @ 512px budget (planned train/infer setting)": (32, 65536, 512 * 512),
}


def measure(rows: list[dict], factor: int, min_px: int, max_px: int) -> tuple[float, float, float]:
    """Return (median visual tokens, % sub-token by axis, % sub-token by area)."""
    tokens: list[int] = []
    sub_axis = sub_area = total = 0
    for row in rows:
        rh, rw = smart_resize(row["H"], row["W"], factor, min_px, max_px)
        tokens.append((rh // factor) * (rw // factor))
        scale_x, scale_y = rw / row["W"], rh / row["H"]
        for box in row["boxes"]:
            tw = box["w"] * scale_x / factor
            th = box["h"] * scale_y / factor
            total += 1
            if min(tw, th) < 1.0:
                sub_axis += 1
            if tw * th < 1.0:
                sub_area += 1
    return statistics.median(tokens), 100 * sub_axis / total, 100 * sub_area / total


def main() -> None:
    rows = fetch_rows()
    n_boxes = sum(len(r["boxes"]) for r in rows)
    sizes = collections.Counter((r["W"], r["H"]) for r in rows)
    print(f"images: {len(rows)}   boxes: {n_boxes}")
    print(f"most common image sizes: {sizes.most_common(4)}\n")

    header = f"{'configuration':<58}{'medTok':>8}{'subtok(axis)':>14}{'subtok(area)':>14}"
    print(header)
    print("-" * len(header))
    for name, (factor, min_px, max_px) in CONFIGS.items():
        med, axis, area = measure(rows, factor, min_px, max_px)
        print(f"{name:<58}{med:>8.0f}{axis:>13.1f}%{area:>13.1f}%")

    print("\nIDEA.md 5.2 claims 23.9% of targets are smaller than one 28x28 visual token.")
    print("IDEA.md also quotes 'median visual tokens about 580', which configuration A reproduces exactly.")


if __name__ == "__main__":
    main()
