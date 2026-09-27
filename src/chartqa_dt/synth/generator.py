from __future__ import annotations

import itertools
import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chartqa_dt.data.records import image_content_sha256
from chartqa_dt.model.coords import px_to_norm1000
from chartqa_dt.plans.executor import SERIES_SEPARATOR, EvidenceItem, execute, resolve_label_plan
from chartqa_dt.plans.roundtrip import answers_agree_at_gold_precision
from chartqa_dt.synth.artists import (
    Box,
    artist_box,
    clip_to_canvas,
    is_degenerate,
    point_box,
    scatter_point_box,
)
from chartqa_dt.synth.curriculum import LEVELS, Level, build_question
from chartqa_dt.synth.verify import check_box_for, render_rgb

ChartType = Literal["vbar", "hbar", "grouped_bar", "line", "multi_line", "pie", "scatter", "area"]
BoxFn = Callable[[], Box]
RecolourFn = Callable[[list[str]], None]
DrawResult = tuple[Any, Any, dict[str, BoxFn], RecolourFn, bool]
CHART_TYPES: tuple[ChartType, ...] = (
    "vbar",
    "hbar",
    "grouped_bar",
    "line",
    "multi_line",
    "pie",
    "scatter",
    "area",
)
HOLDOUT_STYLE_SEEDS: frozenset[int] = frozenset({7, 13, 23})
HOLDOUT_SEED_START = 900_000

PALETTES: list[list[str]] = [
    ["#3060c8", "#c83030", "#30a050", "#e0a020", "#8040c0"],
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
    ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"],
    ["#22223b", "#4a4e69", "#9a8c98", "#c9ada7", "#f2e9e4"],
    ["#006d77", "#83c5be", "#ffddd2", "#e29578", "#b5838d"],
]
FONT_SIZES = (8, 9, 10, 11, 12)
CATEGORY_POOLS: list[list[str]] = [
    [str(y) for y in range(2014, 2024)],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"],
    ["North", "South", "East", "West", "Central"],
    ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"],
    ["Q1", "Q2", "Q3", "Q4"],
    [str(y) for y in range(1980, 2024)],
    [f"{q} {y}" for y in range(2012, 2024) for q in ("Q1", "Q2", "Q3", "Q4")],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    [
        "Argentina",
        "Australia",
        "Austria",
        "Belgium",
        "Brazil",
        "Canada",
        "Chile",
        "China",
        "Colombia",
        "Czechia",
        "Denmark",
        "Egypt",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Hungary",
        "India",
        "Indonesia",
        "Ireland",
        "Israel",
        "Italy",
        "Japan",
        "Kenya",
        "Malaysia",
        "Mexico",
        "Morocco",
        "Netherlands",
        "Nigeria",
        "Norway",
        "Pakistan",
        "Peru",
        "Philippines",
        "Poland",
        "Portugal",
        "Romania",
        "Russia",
        "Saudi Arabia",
        "Singapore",
        "South Africa",
        "South Korea",
        "Spain",
        "Sweden",
        "Switzerland",
        "Thailand",
        "Turkey",
        "Ukraine",
        "United Kingdom",
        "United States",
        "Vietnam",
    ],
    [
        "Alabama",
        "Alaska",
        "Arizona",
        "Arkansas",
        "California",
        "Colorado",
        "Connecticut",
        "Delaware",
        "Florida",
        "Georgia",
        "Hawaii",
        "Idaho",
        "Illinois",
        "Indiana",
        "Iowa",
        "Kansas",
        "Kentucky",
        "Louisiana",
        "Maine",
        "Maryland",
        "Massachusetts",
        "Michigan",
        "Minnesota",
        "Mississippi",
        "Missouri",
        "Montana",
        "Nebraska",
        "Nevada",
        "New Jersey",
        "New Mexico",
        "New York",
        "Ohio",
        "Oklahoma",
        "Oregon",
        "Pennsylvania",
        "Tennessee",
        "Texas",
        "Utah",
        "Vermont",
        "Virginia",
        "Washington",
        "Wisconsin",
        "Wyoming",
    ],
    [
        "18-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65-69",
        "70-74",
        "75-79",
        "80+",
    ],
]

_MONTHS_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_LEVEL_WORDS = ("Very low", "Low", "Medium-low", "Medium", "Medium-high", "High", "Very high")


def _age_bands(rng: random.Random, n: int) -> list[str]:
    width = rng.choice((5, 10))
    start = rng.choice((0, 15, 18, 20))
    joiner, suffix = rng.choice(
        ((" to ", " years old"), ("-", " years"), ("-", " years old"), (" to ", " years"))
    )
    return [f"{start + i * width}{joiner}{start + i * width + width - 1}{suffix}" for i in range(n)]


def _quarters(rng: random.Random, n: int) -> list[str]:
    year, style = (
        rng.randint(2008, 2021),
        rng.choice(("Q{q} '{yy}", "Q{q} {yyyy}", "{q}Q '{yy}", "Q{q} FY{yy}")),
    )
    return [
        style.format(q=(i % 4) + 1, yy=f"{(year + i // 4) % 100:02d}", yyyy=year + i // 4) for i in range(n)
    ]


def _half_years(rng: random.Random, n: int) -> list[str]:
    year = rng.randint(2008, 2021)
    return [f"H{(i % 2) + 1} {year + i // 2}" for i in range(n)]


def _month_year(rng: random.Random, n: int) -> list[str]:
    year, start = rng.randint(2010, 2021), rng.randrange(12)
    style = rng.choice(("{m} '{yy}", "{m} {yy}"))
    return [
        style.format(m=_MONTHS_ABBR[(start + i) % 12], yy=f"{(year + (start + i) // 12) % 100:02d}")
        for i in range(n)
    ]


def _fiscal_years(rng: random.Random, n: int) -> list[str]:
    year = rng.randint(1990, 2020)
    tail = rng.choice(("", "", f" ({rng.choice(('Group', 'Total', 'Consolidated'))})"))
    return [f"{year + i}/{(year + i + 1) % 100:02d}{tail}" for i in range(n)]


def _frequency_bands(rng: random.Random, n: int) -> list[str]:
    period = rng.choice(("a week", "a month", "a day", "per week"))
    return [f"{1 + i * 2}-{2 + i * 2} times {period}" for i in range(n)]


def _ordinal_levels(rng: random.Random, n: int) -> list[str]:
    noun = rng.choice(("implementation", "satisfaction", "engagement", "confidence", "awareness", "adoption"))
    start = rng.randint(0, len(_LEVEL_WORDS) - n)
    return [f"{w} level of {noun}" for w in _LEVEL_WORDS[start : start + n]]


LABEL_TEMPLATES: tuple[tuple[Callable[[random.Random, int], list[str]], int | None], ...] = (
    (_age_bands, None),
    (_quarters, None),
    (_half_years, None),
    (_month_year, None),
    (_fiscal_years, None),
    (_frequency_bands, None),
    (_ordinal_levels, len(_LEVEL_WORDS)),
)

TEMPLATED_LABEL_SHARE = 0.04


CHARTQA_DENSITY_QUANTILES: tuple[tuple[float, int], ...] = (
    (0.00, 2),
    (0.10, 4),
    (0.25, 6),
    (0.50, 10),
    (0.75, 15),
    (0.90, 24),
    (0.99, 45),
    (1.00, 60),
)

MAX_MARKS = 40

DENSITY_BY_LEVEL: dict[str, tuple[int, int] | None] = {
    "L1": (3, 6),
    "L2": (4, 9),
    "L3": None,
    "L4": None,
}


def sample_density(rng: random.Random, level: str) -> int:
    span = DENSITY_BY_LEVEL.get(level, (3, 7))
    if span is not None:
        return rng.randint(*span)
    u = rng.random()
    qs = CHARTQA_DENSITY_QUANTILES
    for (p0, v0), (p1, v1) in itertools.pairwise(qs):
        if u <= p1:
            frac = 0.0 if p1 == p0 else (u - p0) / (p1 - p0)
            return max(2, min(MAX_MARKS, round(v0 + frac * (v1 - v0))))
    return min(MAX_MARKS, qs[-1][1])


UNITS = (None, "millions", "%", "thousands", "units", "USD")

LINEWIDTH = 1.5

TEXT_LAYOUT_MIN_GAP_PX = 1.0

SENTINELS: tuple[str, ...] = (
    "#ff00ff",
    "#00ff00",
    "#ff0000",
    "#0000ff",
    "#ffff00",
    "#00ffff",
    "#ff8000",
    "#8000ff",
    "#00ff80",
    "#ff0080",
    "#80ff00",
    "#0080ff",
)
_SENTINEL_GRID = tuple((h / 48.0, s, v) for h in range(48) for s in (1.0, 0.8) for v in (1.0, 0.75, 0.5))


def sentinel_colours(n: int) -> list[str]:
    if n <= len(SENTINELS):
        return list(SENTINELS[: max(n, 0)])
    return _extend_by_farthest_point(list(SENTINELS), n, _SENTINEL_GRID)


SENTINEL_LINE = "#404040"
MAX_VALUE_RATIO = 8.0

MIN_BOX_SIDE_PX = 4.0

QUANTITIES = ("value", "share", "count", "revenue", "score")


_HSV_GRID = tuple((h / 36.0, s, v) for h in range(36) for s in (0.55, 0.75, 0.95) for v in (0.45, 0.65, 0.85))


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def element_colours(palette: list[str], n: int) -> list[str]:
    if n <= 0:
        return []
    out = [palette[i % len(palette)] for i in range(min(n, len(palette)))]
    if n <= len(palette):
        return out

    return _extend_by_farthest_point(out, n, _HSV_GRID)


def _extend_by_farthest_point(
    seed: list[str], n: int, grid: tuple[tuple[float, float, float], ...]
) -> list[str]:
    import colorsys

    out = list(seed)
    chosen = [_rgb(c) for c in out]
    candidates: list[tuple[int, int, int]] = []
    for hsv in grid:
        red, green, blue = colorsys.hsv_to_rgb(*hsv)
        candidates.append((round(255 * red), round(255 * green), round(255 * blue)))
    candidates = [c for c in candidates if c not in chosen]
    while len(out) < n and candidates:
        best = max(candidates, key=lambda c: min(_distance(c, x) for x in chosen))
        chosen.append(best)
        candidates.remove(best)
        out.append("#{:02x}{:02x}{:02x}".format(*best))
    return out


def is_holdout(style_seed: int, data_seed: int) -> bool:
    return style_seed in HOLDOUT_STYLE_SEEDS or data_seed >= HOLDOUT_SEED_START


@dataclass
class Style:
    palette: list[str]
    font_size: int
    grid: bool
    tick_rotation: int
    legend: bool
    title: bool
    value_labels: bool
    figsize: tuple[float, float]
    dpi: int
    background: str
    style_seed: int

    @classmethod
    def sample(cls, style_seed: int) -> Style:
        r = random.Random(style_seed)
        return cls(
            palette=r.choice(PALETTES),
            font_size=r.choice(FONT_SIZES),
            grid=r.random() < 0.5,
            tick_rotation=r.choice([0, 0, 15, 30, 45]),
            legend=r.random() < 0.4,
            title=r.random() < 0.7,
            value_labels=r.random() < 0.3,
            figsize=(r.choice([5.0, 6.0, 7.0, 8.0]), r.choice([3.5, 4.0, 4.5, 5.0])),
            dpi=r.choice([90, 100, 110]),
            background=r.choice(["white", "white", "white", "#f7f7f7", "#20242b"]),
            style_seed=style_seed,
        )

    @property
    def dark(self) -> bool:
        return self.background.startswith("#2")


@dataclass
class SynthExample:
    example_id: str
    chart_type: str
    level: str
    question: str
    answer: str
    plan: dict[str, Any]
    evidence: list[dict[str, Any]]
    table: dict[str, Any]
    image_path: str
    image_sha256: str
    image_size: tuple[int, int]
    style_seed: int
    data_seed: int
    holdout: bool
    elements: list[dict[str, Any]] = field(default_factory=list)
    evidence_index: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        facts = [
            {
                "id": f"f{source_index + 1}",
                "label": e["label"],
                "value": e["value"],
                "unit": e["unit"],
                "bbox": e["bbox"],
            }
            for source_index, e in zip(self.evidence_index, self.evidence)
        ]
        items = [EvidenceItem(fact["label"], fact["value"], fact["unit"], fact["id"]) for fact in facts]
        return {
            "answerable": True,
            "facts": facts,
            "evidence_refs": [fact["id"] for fact in facts],
            "plan": resolve_label_plan(self.plan, items),
            "model_answer": self.answer,
        }


def sample_series(
    rng: random.Random, n_min: int = 3, n_max: int = 7, n: int | None = None, chart_type: str | None = None
) -> tuple[list[tuple[str, float]], str, str | None]:
    n = rng.randint(n_min, n_max) if n is None else n
    usable = [f for f, cap in LABEL_TEMPLATES if cap is None or cap >= n]
    if usable and rng.random() < TEMPLATED_LABEL_SHARE:
        labels = rng.choice(usable)(rng, n)
        values = sample_values(rng, len(labels), allow_negative=chart_type not in NON_NEGATIVE_CHART_TYPES)
        return list(zip(labels, values)), rng.choice(QUANTITIES), rng.choice(UNITS)
    eligible = [p for p in CATEGORY_POOLS if len(p) >= n]
    if not eligible:
        eligible = [max(CATEGORY_POOLS, key=len)]
    pool = rng.choice(eligible)
    n = min(n, len(pool))
    start = rng.randint(0, max(0, len(pool) - n))
    labels = pool[start : start + n]
    values = sample_values(rng, len(labels), allow_negative=chart_type not in NON_NEGATIVE_CHART_TYPES)
    return list(zip(labels, values)), rng.choice(QUANTITIES), rng.choice(UNITS)


PERCENTAGE_SHARE = 0.074

NEGATIVE_SHARE = 0.017

MAGNITUDE_LOG10_MEAN = 1.45
MAGNITUDE_LOG10_SD = 1.33
MAGNITUDE_LOG10_RANGE = (-0.7, 8.4)


MARK_WORD = {
    "vbar": "bar",
    "hbar": "bar",
    "grouped_bar": "bar",
    "pie": "slice",
    "line": "line",
    "multi_line": "line",
    "area": "area",
    "scatter": "point",
}

NON_NEGATIVE_CHART_TYPES = frozenset({"pie"})


def sample_values(rng: random.Random, n: int, *, allow_negative: bool = True) -> list[float]:
    if n <= 0:
        return []
    if rng.random() < PERCENTAGE_SHARE and n >= 3:
        raw = [rng.uniform(1.0, 10.0) for _ in range(n)]
        total = sum(raw)
        vals = [round(100.0 * v / total, 1) for v in raw]
        vals[vals.index(max(vals))] = round(max(vals) + 100.0 - sum(vals), 1)
        return vals

    exponent = min(
        MAGNITUDE_LOG10_RANGE[1],
        max(MAGNITUDE_LOG10_RANGE[0], rng.gauss(MAGNITUDE_LOG10_MEAN, MAGNITUDE_LOG10_SD)),
    )
    lo = 10.0**exponent
    hi = lo * rng.uniform(1.5, MAX_VALUE_RATIO)
    places = max(0, 1 - math.floor(math.log10(lo))) if lo > 0 else 2
    places = min(places, 3)
    if places == 0 and hi < 1000 and rng.random() < 0.25:
        places = 1
    values = [round(rng.uniform(lo, hi), places) for _ in labels_range(n)]
    if all(v == 0 for v in values):
        raise ValueError(
            f"sample_values produced an all-zero chart at lo={lo!r}, places={places}. "
            f"A chart of zeros renders as no marks and can never be verified."
        )
    if allow_negative and rng.random() < NEGATIVE_SHARE:
        k = rng.randint(1, max(1, n // 2))
        for i in rng.sample(range(n), k):
            values[i] = -values[i]
    return values


def labels_range(n: int) -> range:
    return range(n)


def value_axis_limits(values: list[float]) -> tuple[float, float]:
    top = max(values) * 1.25 if max(values) > 0 else 0.0
    bottom = min(values) * 1.25 if min(values) < 0 else 0.0
    if top == bottom:
        return 0.0, 1.0
    return bottom, top


def _apply_style(fig: Any, ax: Any, st: Style, title: str | None) -> None:
    fig.patch.set_facecolor(st.background)
    if ax is not None:
        ax.set_facecolor(st.background)
        if st.grid:
            ax.grid(True, alpha=0.3)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(st.tick_rotation)
        colour = "white" if st.dark else "black"
        ax.tick_params(colors=colour, labelsize=st.font_size)
        for spine in ax.spines.values():
            spine.set_color(colour)
        for axis in (ax.yaxis, ax.xaxis):
            if isinstance(axis.get_major_formatter(), matplotlib.ticker.ScalarFormatter):
                axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _pos: f"{v:,.10g}"))
    if st.title and title:
        fig.suptitle(title, fontsize=st.font_size + 2, color="white" if st.dark else "black")


def _texts_are_readable(fig: Any, texts: list[Any]) -> bool:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    boxes = [
        text.get_window_extent(renderer=renderer)
        for text in texts
        if text.get_visible() and str(text.get_text()).strip()
    ]
    for box in boxes:
        if (
            box.x0 < figure_box.x0 - TEXT_LAYOUT_MIN_GAP_PX
            or box.y0 < figure_box.y0 - TEXT_LAYOUT_MIN_GAP_PX
            or box.x1 > figure_box.x1 + TEXT_LAYOUT_MIN_GAP_PX
            or box.y1 > figure_box.y1 + TEXT_LAYOUT_MIN_GAP_PX
        ):
            return False
    for index, first in enumerate(boxes):
        for second in boxes[index + 1 :]:
            overlap_x = min(first.x1, second.x1) - max(first.x0, second.x0)
            overlap_y = min(first.y1, second.y1) - max(first.y0, second.y0)
            if overlap_x > -TEXT_LAYOUT_MIN_GAP_PX and overlap_y > -TEXT_LAYOUT_MIN_GAP_PX:
                return False
    return True


def _finalise_text_layout(fig: Any, ax: Any, st: Style, chart_type: ChartType) -> bool:
    if chart_type == "pie":
        texts = list(ax.texts)
        side = min(12.0, max(*fig.get_size_inches(), 4.0 + 0.20 * len(texts)))
        fig.set_size_inches(side, side, forward=True)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95) if st.title else None, pad=1.0)
        return _texts_are_readable(fig, texts)

    if chart_type == "hbar":
        texts = list(ax.get_yticklabels())
        longest = max((len(str(text.get_text())) for text in texts), default=0)
        width, height = fig.get_size_inches()
        width = min(12.0, max(width, 3.0 + longest * st.font_size / 50.0))
        height = min(12.0, max(height, 1.5 + len(texts) * st.font_size / 55.0))
        fig.set_size_inches(width, height, forward=True)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95) if st.title else None, pad=1.0)
        return _texts_are_readable(fig, texts)

    texts = list(ax.get_xticklabels())
    width, height = fig.get_size_inches()
    longest = max((len(str(text.get_text())) for text in texts), default=0)
    width = min(14.0, max(width, 2.0 + 0.35 * len(texts)))
    height = min(9.0, max(height, 3.0 + longest * st.font_size / 100.0))
    fig.set_size_inches(width, height, forward=True)
    rotations = dict.fromkeys((st.tick_rotation, 30, 45, 60, 75, 90))
    for rotation in rotations:
        for text in texts:
            text.set_rotation(rotation)
            text.set_rotation_mode("anchor")
            text.set_horizontalalignment("center" if rotation == 0 else "right")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95) if st.title else None, pad=1.0)
        if _texts_are_readable(fig, texts):
            return True
    return False


def _patch_recolour(patches: Sequence[Any]) -> RecolourFn:
    def apply(colours: list[str]) -> None:
        for patch, colour in zip(patches, colours):
            patch.set_facecolor(colour)

    return apply


def _collection_recolour(coll: Any) -> RecolourFn:
    def apply(colours: list[str]) -> None:
        coll.set_facecolor(list(colours))
        coll.set_edgecolor(list(colours))

    return apply


def _line_recolour(line: Any) -> RecolourFn:
    def apply(colours: list[str]) -> None:
        line.set_color(SENTINEL_LINE)
        line.set_markerfacecolor(colours[0])
        line.set_markeredgecolor(colours[0])

    return apply


def _multi_line_recolour(lines: Sequence[Any], first_indices: list[int]) -> RecolourFn:
    def apply(colours: list[str]) -> None:
        for line, index in zip(lines, first_indices):
            colour = colours[index]
            line.set_color(SENTINEL_LINE)
            line.set_markerfacecolor(colour)
            line.set_markeredgecolor(colour)

    return apply


def _qualified_parts(label: str) -> tuple[str, str]:
    if SERIES_SEPARATOR not in label:
        raise ValueError(f"multi-series mark {label!r} is not series-qualified with {SERIES_SEPARATOR!r}")
    series_name, category = label.split(SERIES_SEPARATOR, 1)
    if not series_name or not category:
        raise ValueError(f"malformed series-qualified mark {label!r}")
    return series_name, category


def _display_colours(style: Style, series: list[tuple[str, float]], chart_type: str) -> list[str]:
    if chart_type in {"line", "area"}:
        return [style.palette[0]] * len(series)
    if chart_type in {"grouped_bar", "multi_line"}:
        names = list(dict.fromkeys(_qualified_parts(label)[0] for label, _ in series))
        by_name = {name: style.palette[index % len(style.palette)] for index, name in enumerate(names)}
        return [by_name[_qualified_parts(label)[0]] for label, _ in series]
    return element_colours(style.palette, len(series))


def _verification_colours(series: list[tuple[str, float]], chart_type: str) -> list[str]:
    colours = sentinel_colours(len(series))
    if chart_type in {"line", "area"}:
        return [colours[0]] * len(series)
    if chart_type == "multi_line":
        names = list(dict.fromkeys(_qualified_parts(label)[0] for label, _ in series))
        by_name = {name: colours[index] for index, name in enumerate(names)}
        return [by_name[_qualified_parts(label)[0]] for label, _ in series]
    return colours


GEOMETRY_OF: dict[str, str] = {
    "vbar": "rect",
    "hbar": "rect",
    "grouped_bar": "rect",
    "line": "disc_shared",
    "multi_line": "disc_shared",
    "area": "disc_shared",
    "scatter": "disc_unique",
    "pie": "wedge",
}


def _artist_box_fn(fig: Any, artist: Any) -> BoxFn:
    def box() -> Box:
        return artist_box(fig, artist)

    return box


def _point_box_fn(
    fig: Any,
    ax: Any,
    x: float,
    y: float,
    marker_points: float,
    edge_points: float,
) -> BoxFn:
    def box() -> Box:
        return point_box(fig, ax, x, y, marker_points, edge_points)

    return box


def _scatter_box_fn(
    fig: Any,
    ax: Any,
    x: float,
    y: float,
    area_points: float,
    edge_points: float,
) -> BoxFn:
    def box() -> Box:
        return scatter_point_box(fig, ax, x, y, area_points, edge_points)

    return box


def _draw(
    chart_type: ChartType, series: list[tuple[str, float]], st: Style, rng: random.Random, title: str | None
) -> DrawResult:
    labels = [lab for lab, _ in series]
    values = [v for _, v in series]
    colours = _display_colours(st, series, chart_type)
    fig, ax = plt.subplots(figsize=st.figsize, dpi=st.dpi)
    annotations: list[tuple[Any, float]] = []
    boxes: dict[str, BoxFn]
    recolour: RecolourFn

    if chart_type == "vbar":
        bars = list(ax.bar(labels, values, color=colours))
        ax.set_ylim(*value_axis_limits(values))
        boxes = {lab: _artist_box_fn(fig, bar) for lab, bar in zip(labels, bars)}
        recolour = _patch_recolour(bars)
        annotations = list(series)
    elif chart_type == "grouped_bar":
        names = list(dict.fromkeys(_qualified_parts(label)[0] for label in labels))
        categories = list(dict.fromkeys(_qualified_parts(label)[1] for label in labels))
        by_key = {_qualified_parts(label): value for label, value in series}
        if set(by_key) != {(name, category) for name in names for category in categories}:
            plt.close(fig)
            raise ValueError("grouped bars require a complete series-by-category grid")
        width = 0.8 / len(names)
        bars = []
        boxes = {}
        for series_index, name in enumerate(names):
            xs = [index - 0.4 + width / 2 + series_index * width for index in range(len(categories))]
            vals = [by_key[(name, category)] for category in categories]
            group = list(ax.bar(xs, vals, width=width, color=st.palette[series_index], label=name))
            bars.extend(group)
            for category, x, value, bar in zip(categories, xs, vals, group):
                label = f"{name}{SERIES_SEPARATOR}{category}"
                boxes[label] = _artist_box_fn(fig, bar)
                annotations.append((x, value))
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_ylim(*value_axis_limits(values))
        fig.subplots_adjust(right=0.78)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=st.font_size - 1)
        recolour = _patch_recolour(bars)
    elif chart_type == "hbar":
        bars = list(ax.barh(labels, values, color=colours))
        ax.set_xlim(*value_axis_limits(values))
        boxes = {lab: _artist_box_fn(fig, bar) for lab, bar in zip(labels, bars)}
        recolour = _patch_recolour(bars)
    elif chart_type in ("line", "area"):
        marker_pts = rng.choice([10.0, 12.0, 14.0])
        if chart_type == "area":
            ax.fill_between(range(len(values)), values, alpha=0.3, color=st.palette[0], zorder=1)
        (line,) = ax.plot(
            range(len(values)),
            values,
            marker="o",
            markersize=marker_pts,
            color=st.palette[0],
            markerfacecolor=st.palette[0],
            markeredgecolor=st.palette[0],
            markeredgewidth=LINEWIDTH,
            zorder=3,
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylim(*value_axis_limits(values))
        boxes = {
            lab: _point_box_fn(fig, ax, float(i), v, marker_pts, LINEWIDTH)
            for i, (lab, v) in enumerate(series)
        }
        recolour = _line_recolour(line)
    elif chart_type == "multi_line":
        marker_pts = rng.choice([10.0, 12.0, 14.0])
        names = list(dict.fromkeys(_qualified_parts(label)[0] for label in labels))
        categories = list(dict.fromkeys(_qualified_parts(label)[1] for label in labels))
        by_key = {_qualified_parts(label): value for label, value in series}
        if set(by_key) != {(name, category) for name in names for category in categories}:
            plt.close(fig)
            raise ValueError("multi-line charts require a complete series-by-category grid")
        lines = []
        first_indices = []
        boxes = {}
        for series_index, name in enumerate(names):
            vals = [by_key[(name, category)] for category in categories]
            (line,) = ax.plot(
                range(len(categories)),
                vals,
                marker="o",
                markersize=marker_pts,
                color=st.palette[series_index],
                markerfacecolor=st.palette[series_index],
                markeredgecolor=st.palette[series_index],
                markeredgewidth=LINEWIDTH,
                linestyle=("-" if series_index == 0 else "--"),
                label=name,
                zorder=3 - series_index,
            )
            lines.append(line)
            first_indices.append(labels.index(f"{name}{SERIES_SEPARATOR}{categories[0]}"))
            for index, (category, value) in enumerate(zip(categories, vals)):
                label = f"{name}{SERIES_SEPARATOR}{category}"
                boxes[label] = _point_box_fn(fig, ax, float(index), value, marker_pts, LINEWIDTH)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories)
        ax.set_ylim(*value_axis_limits(values))
        fig.subplots_adjust(right=0.78)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=st.font_size - 1)
        recolour = _multi_line_recolour(lines, first_indices)
    elif chart_type == "scatter":
        s = rng.choice([200.0, 300.0, 400.0])
        coll = ax.scatter(range(len(values)), values, s=s, c=colours, linewidths=LINEWIDTH)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.6, len(values) - 0.4)
        ax.set_ylim(*value_axis_limits(values))
        boxes = {
            lab: _scatter_box_fn(fig, ax, float(i), v, s, LINEWIDTH) for i, (lab, v) in enumerate(series)
        }
        recolour = _collection_recolour(coll)
    elif chart_type == "pie":
        visible_values = iter(values)
        wedges, _, _ = ax.pie(
            values,
            labels=labels,
            colors=colours,
            autopct=lambda _computed: f"{next(visible_values):g}%",
            textprops={"fontsize": st.font_size, "color": "white" if st.dark else "black"},
        )
        boxes = {lab: _artist_box_fn(fig, wedge) for lab, wedge in zip(labels, wedges)}
        recolour = _patch_recolour(wedges)
    else:
        plt.close(fig)
        raise ValueError(f"unknown chart type: {chart_type!r}")

    _apply_style(fig, ax if chart_type != "pie" else None, st, title)
    if st.value_labels and chart_type in ("vbar", "grouped_bar"):
        for annotation_x, annotation_value in annotations:
            ax.annotate(
                f"{annotation_value:g}",
                (annotation_x, annotation_value),
                ha="center",
                va="bottom",
                fontsize=st.font_size - 1,
                color="white" if st.dark else "black",
            )
    if st.legend and chart_type in ("line", "area"):
        ax.legend(["series"], fontsize=st.font_size - 1)
    text_layout_ok = _finalise_text_layout(fig, ax, st, chart_type)
    return fig, ax, boxes, recolour, text_layout_ok


def _normalise_percentages(series: list[tuple[str, float]]) -> list[tuple[str, float]]:
    raw = [(label, max(abs(float(value)), 1e-6)) for label, value in series]
    total = sum(value for _, value in raw)
    values = [round(100 * value / total, 1) for _, value in raw]
    largest = max(range(len(values)), key=values.__getitem__)
    values[largest] = round(values[largest] + 100.0 - sum(values), 1)
    return [(raw[index][0], value) for index, value in enumerate(values)]


def _sample_series_for_chart(
    rng: random.Random,
    chart_type: ChartType,
    mark_count: int,
) -> tuple[list[tuple[str, float]], str, str | None]:
    if chart_type not in {"grouped_bar", "multi_line"}:
        series, quantity, unit = sample_series(rng, n=mark_count, chart_type=chart_type)
        if chart_type == "pie":
            return _normalise_percentages(series), "share", "%"
        return series, quantity, unit

    category_count = max(2, min(MAX_MARKS // 2, math.ceil(mark_count / 2)))
    base, quantity, unit = sample_series(rng, n=category_count, chart_type=chart_type)
    categories = [label for label, _ in base]
    first = [value for _, value in base]
    span = max(first) - min(first)
    offset = max(span, max(abs(value) for value in first), 1.0) * rng.uniform(0.30, 0.45)
    second = [round(value + offset, 3) for value in first]
    if all(value == 0 for value in second):
        second[0] = 1.0
    names = ("Series A", "Series B")
    flattened = [
        (f"{name}{SERIES_SEPARATOR}{category}", value)
        for name, values in zip(names, (first, second))
        for category, value in zip(categories, values)
    ]
    return flattened, quantity, unit


def generate_example(
    *,
    chart_type: ChartType,
    level: Level,
    style_seed: int,
    data_seed: int,
    out_dir: Path,
    verify: bool = True,
    l2_style: str | None = None,
    comparison_outcome: str | None = None,
) -> SynthExample | None:
    data_rng = random.Random(data_seed)
    style = Style.sample(style_seed)
    series, quantity, unit = _sample_series_for_chart(data_rng, chart_type, sample_density(data_rng, level))
    if level == "L2" and l2_style == "compare" and comparison_outcome == "equal":
        series = [series[0], (series[1][0], series[0][1]), *series[2:]]

    element_colour_list = _display_colours(style, series, chart_type)
    question = build_question(
        level,
        series,
        data_rng,
        unit=unit,
        quantity=quantity,
        colours=element_colour_list,
        mark=MARK_WORD.get(chart_type, "bar"),
        chart_type=chart_type,
        l2_style=l2_style,
        comparison_outcome=comparison_outcome,
    )
    if question is None:
        return None

    title = f"{quantity.title()} by category" if style.title else None
    fig, _ax, box_fns, recolour, text_layout_ok = _draw(chart_type, series, style, data_rng, title)

    try:
        if verify and not text_layout_ok:
            return None
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        by_label = dict(series)

        wanted = set(question.evidence_labels)
        elements: list[dict[str, Any]] = []
        evidence_index: list[int] = []
        for label, _value in series:
            box_px = clip_to_canvas(box_fns[label](), width, height)
            if is_degenerate(box_px, MIN_BOX_SIDE_PX):
                return None
            if label in wanted:
                evidence_index.append(len(elements))
            element = {
                "label": label,
                "value": by_label[label],
                "unit": unit,
                "bbox": [round(v, 2) for v in px_to_norm1000(box_px, width, height)],
                "bbox_px": [round(v, 2) for v in box_px],
            }
            if chart_type in {"grouped_bar", "multi_line"}:
                element["series"], element["category"] = _qualified_parts(label)
            elements.append(element)
        if len(evidence_index) != len(wanted):
            return None
        evidence = [elements[i] for i in evidence_index]

        if verify:
            real = _display_colours(style, series, chart_type)
            sentinels = _verification_colours(series, chart_type)
            recolour(sentinels)
            try:
                if not _verify_boxes(render_rgb(fig), elements, series, sentinels, chart_type):
                    return None
            finally:
                recolour(real)

        fact_items = [
            EvidenceItem(
                elements[index]["label"],
                elements[index]["value"],
                elements[index]["unit"],
                f"f{index + 1}",
            )
            for index in evidence_index
        ]
        try:
            typed_plan = resolve_label_plan(question.plan, fact_items)
            executed = execute(typed_plan, fact_items)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"synthetic invariant failed for {chart_type}/{level}: {exc}") from exc
        if not answers_agree_at_gold_precision(question.answer, executed):
            raise RuntimeError(f"synthetic plan disagrees exactly: {question.answer!r} != {executed!r}")

        out_dir.mkdir(parents=True, exist_ok=True)
        example_id = f"synth_{chart_type}_{level}_{style_seed}_{data_seed}"
        path = out_dir / f"{example_id}.png"
        fig.savefig(path, facecolor=style.background, bbox_inches=None)
    finally:
        plt.close(fig)

    digest = image_content_sha256(path)
    if chart_type in {"grouped_bar", "multi_line"}:
        names = list(dict.fromkeys(_qualified_parts(label)[0] for label, _ in series))
        categories = list(dict.fromkeys(_qualified_parts(label)[1] for label, _ in series))
        by_key = {_qualified_parts(label): value for label, value in series}
        table = {
            "columns": ["category", *names],
            "rows": [[category, *[by_key[(name, category)] for name in names]] for category in categories],
            "quantity": quantity,
            "unit": unit,
        }
    else:
        table = {
            "labels": [label for label, _ in series],
            "values": [value for _, value in series],
            "quantity": quantity,
            "unit": unit,
        }

    example = SynthExample(
        example_id=example_id,
        chart_type=chart_type,
        level=level,
        question=question.question,
        answer=question.answer,
        plan=typed_plan,
        evidence=evidence,
        elements=elements,
        evidence_index=evidence_index,
        table=table,
        image_path=str(path),
        image_sha256=digest,
        image_size=(width, height),
        style_seed=style_seed,
        data_seed=data_seed,
        holdout=is_holdout(style_seed, data_seed),
        meta={
            "question_meta": question.meta,
            "font_size": style.font_size,
            "dark": style.dark,
            "grid": style.grid,
        },
    )
    from chartqa_dt.plans.schema import validate_record

    validation = validate_record(example.to_record())
    if not validation.ok:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"synthetic target failed dual-target validation: {validation.errors}")
    return example


def _verify_boxes(
    img: np.ndarray,
    evidence: list[dict[str, Any]],
    series: list[tuple[str, float]],
    colours: list[str],
    chart_type: ChartType,
) -> bool:
    labels = [lab for lab, _ in series]
    for item in evidence:
        idx = labels.index(item["label"])
        hexcol = colours[idx].lstrip("#")
        rgb = (int(hexcol[0:2], 16), int(hexcol[2:4], 16), int(hexcol[4:6], 16))
        if not check_box_for(img, item["bbox_px"], rgb, item["label"], GEOMETRY_OF[chart_type]).ok:
            return False
    return True


def write_manifest(examples: list[SynthExample], path: Path, *, seed: int,
                   outcomes: dict[str, int]) -> dict[str, Any]:
    from collections import Counter

    summary = {
        "count": len(examples),
        "by_chart_type": dict(Counter(e.chart_type for e in examples)),
        "by_level": dict(Counter(e.level for e in examples)),
        "holdout": sum(e.holdout for e in examples),
        "generation_outcomes": dict(sorted(outcomes.items())),
    }
    root = path.parent.resolve()
    serialized = []
    for example in examples:
        row = example.to_dict()
        row["image_path"] = Path(example.image_path).resolve().relative_to(root).as_posix()
        serialized.append(row)
    payload = {"generation_seed": seed, "summary": summary, "examples": serialized}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def generate_batch(
    n: int,
    out_dir: Path,
    *,
    seed: int = 0,
    chart_types: tuple[ChartType, ...] = CHART_TYPES,
    levels: tuple[Level, ...] = LEVELS,
    holdout: bool = False,
    verify: bool = True,
    outcomes: dict[str, int] | None = None,
) -> list[SynthExample]:
    rng = random.Random(seed)
    out: list[SynthExample] = []
    attempts = 0
    refused = 0
    l2_schedule = (
        ("signed_difference", None),
        ("absolute_distance", None),
        ("ratio", None),
        ("winner", None),
        ("compare", "greater"),
        ("compare", "less"),
        ("compare", "equal"),
    )
    while len(out) < n and attempts < n * 12:
        attempts += 1
        ct = chart_types[len(out) % len(chart_types)]
        lv = levels[(len(out) // max(1, len(chart_types))) % len(levels)]
        style_seed = (
            rng.choice(sorted(HOLDOUT_STYLE_SEEDS))
            if holdout
            else rng.choice([s for s in range(60) if s not in HOLDOUT_STYLE_SEEDS])
        )
        data_seed = (
            HOLDOUT_SEED_START + rng.randrange(50_000) if holdout else rng.randrange(HOLDOUT_SEED_START)
        )
        style_name = comparison = None
        if lv == "L2":
            l2_count = sum(example.level == "L2" for example in out)
            style_name, comparison = l2_schedule[l2_count % len(l2_schedule)]
        ex = generate_example(
            chart_type=ct,
            level=lv,
            style_seed=style_seed,
            data_seed=data_seed,
            out_dir=out_dir,
            verify=verify,
            l2_style=style_name,
            comparison_outcome=comparison,
        )
        if ex is not None:
            out.append(ex)
        else:
            refused += 1
    if outcomes is not None:
        outcomes.clear()
        outcomes.update({"attempted": attempts, "accepted": len(out), "invariant_refusal": refused})
    return out
