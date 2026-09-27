from __future__ import annotations

import colorsys
import re
from typing import Any

_HUES: tuple[tuple[float, float, str], ...] = (
    (345.0, 360.0, "red"),
    (0.0, 12.0, "red"),
    (12.0, 40.0, "orange"),
    (40.0, 68.0, "yellow"),
    (68.0, 160.0, "green"),
    (160.0, 195.0, "teal"),
    (195.0, 255.0, "blue"),
    (255.0, 290.0, "purple"),
    (290.0, 345.0, "pink"),
)

_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


def _hsl(hex_colour: str) -> tuple[float, float, float] | None:
    m = _HEX.match(str(hex_colour).strip())
    if not m:
        return None
    raw = m.group(1)
    r, g, b = (int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4))
    h, ligh, sat = colorsys.rgb_to_hls(r, g, b)
    return h * 360.0, sat, ligh


_SYNONYMS: dict[str, set[str]] = {
    "dark blue": {"navy", "blue", "black"},
    "navy": {"dark blue", "blue", "black"},
    "light blue": {"blue", "sky blue"},
    "sky blue": {"light blue", "blue"},
    "grey": {"gray"},
    "gray": {"grey"},
    "dark grey": {"dark gray", "grey", "gray"},
    "dark gray": {"dark grey", "grey", "gray"},
    "light grey": {"light gray", "grey", "gray"},
    "light gray": {"light grey", "grey", "gray"},
    "dark red": {"maroon", "red"},
    "maroon": {"dark red", "red"},
    "dark green": {"green"},
    "light green": {"green"},
}


def names_for(colour: str) -> set[str]:
    raw = str(colour).strip().lower()
    if raw and not _HEX.match(raw):
        out = {raw} | _SYNONYMS.get(raw, set())
        parts = raw.split()
        if len(parts) == 2 and parts[0] in {"dark", "light", "deep", "pale"}:
            out.add(parts[1])
        return out
    parsed = _hsl(colour)
    if parsed is None:
        return set()
    hue, sat, light = parsed

    if light <= 0.10:
        return {"black", "dark"}
    if light <= 0.22:
        base = next((n for lo, hi, n in _HUES if lo <= hue < hi), "blue") if sat > 0.12 else "grey"
        return {"black", "dark", f"dark {base}", base} | ({"navy"} if base == "blue" else set())
    if light >= 0.92:
        return {"white", "light"}
    if sat <= 0.12:
        return {"grey", "gray", "silver"} | (
            {"dark grey", "dark gray"} if light < 0.4 else {"light grey", "light gray"}
        )

    base = next((name for lo, hi, name in _HUES if lo <= hue < hi), "blue")
    out = {base}
    if light < 0.38:
        out |= {f"dark {base}"}
        if base == "blue":
            out |= {"navy"}
        if base == "red":
            out |= {"maroon"}
    elif light > 0.62:
        out |= {f"light {base}"}
        if base == "blue":
            out |= {"sky blue"}
    if base == "grey":
        out |= {"gray"}
    return out


def describe_colour(colour: Any) -> str:
    words = names_for(colour)
    return min(words, key=lambda w: (-len(w.split()), -len(w), w)) if words else ""
