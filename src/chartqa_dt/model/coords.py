from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

OFFICIAL_MAX_COORD = 999


def smart_resize(
    height: int | float,
    width: int | float,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
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


@dataclass(frozen=True)
class VisualGeometry:
    factor: int
    min_pixels: int
    max_pixels: int
    patch_size: int
    merge_size: int
    source: str = "explicit"

    @classmethod
    def from_processor(cls, processor: Any) -> VisualGeometry:
        ip = getattr(processor, "image_processor", processor)
        patch = getattr(ip, "patch_size", None)
        merge = getattr(ip, "merge_size", None)
        if patch is None or merge is None:
            raise ValueError("processor exposes no patch_size/merge_size")
        size = getattr(ip, "size", None) or {}
        if hasattr(size, "get"):
            min_px = size.get("shortest_edge")
            max_px = size.get("longest_edge")
        else:
            min_px = getattr(size, "shortest_edge", None)
            max_px = getattr(size, "longest_edge", None)
        min_px = min_px if min_px is not None else getattr(ip, "min_pixels", None)
        max_px = max_px if max_px is not None else getattr(ip, "max_pixels", None)
        if min_px is None or max_px is None:
            raise ValueError(f"processor exposes no pixel bounds: size={size!r}")
        return cls(
            factor=patch * merge,
            min_pixels=min_px,
            max_pixels=max_px,
            patch_size=patch,
            merge_size=merge,
            source=type(ip).__name__,
        )

    def with_pixel_budget(
        self,
        *,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> VisualGeometry:
        resolved_max = self.max_pixels if max_pixels is None else max_pixels
        resolved_min = min(self.min_pixels, resolved_max) if min_pixels is None else min_pixels
        if resolved_min <= 0 or resolved_max <= 0 or resolved_min > resolved_max:
            raise ValueError(f"invalid processor pixel budget [{resolved_min}, {resolved_max}]")
        return VisualGeometry(
            factor=self.factor,
            min_pixels=resolved_min,
            max_pixels=resolved_max,
            patch_size=self.patch_size,
            merge_size=self.merge_size,
            source=f"{self.source}@min_pixels={resolved_min},max_pixels={resolved_max}",
        )

    def resize(self, height: int | float, width: int | float) -> tuple[int, int]:
        return smart_resize(height, width, self.factor, self.min_pixels, self.max_pixels)

    def describe(self) -> str:
        return (
            f"patch={self.patch_size} merge={self.merge_size} -> factor={self.factor} "
            f"(one visual token = {self.factor}x{self.factor} px), "
            f"min_pixels={self.min_pixels:,} max_pixels={self.max_pixels:,} [{self.source}]"
        )


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


def clamp_for_official_evaluator(bbox: tuple[float, float, float, float]) -> list[int]:
    return [int(max(0, min(OFFICIAL_MAX_COORD, round(v)))) for v in bbox]
