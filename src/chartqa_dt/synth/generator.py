"""Chart generator with exact boxes, exact answers and exact typed plans.

`PLAN.md` 3.5. Because we control the plotting data, everything a training example
needs is known **by construction**: the table, the answer, the evidence boxes and
the typed plan. That is what makes this the primary source of plan supervision,
given that the uniqueness rule admits only ~5.7% of real ChartQA questions.

Boxes come from the matplotlib artists themselves (`synth/artists.py`), never from
a formula — the technique was proven against rendered pixels for every chart type
before this module was written.

**Sealed holdout.** Style seeds in `HOLDOUT_STYLE_SEEDS` and data seeds at or above
`HOLDOUT_SEED_START` are reserved for the Phase 9.5 robustness test and never
appear in training. `is_holdout` is the single place that decides, so training and
holdout cannot drift apart.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chartqa_dt.data.records import image_content_sha256
from chartqa_dt.synth.artists import artist_box, clip_to_canvas, is_degenerate, point_box, scatter_point_box
from chartqa_dt.synth.curriculum import LEVELS, Level, build_question
from chartqa_dt.synth.verify import check_box_for, render_rgb
from chartqa_dt.vision.coords import px_to_norm1000

ChartType = Literal["vbar", "hbar", "grouped_bar", "line", "multi_line", "pie", "scatter", "area"]
CHART_TYPES: tuple[ChartType, ...] = ("vbar", "hbar", "grouped_bar", "line", "multi_line",
                                      "pie", "scatter", "area")

# Reserved for the Phase 9.5 robustness test; never generated for training.
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
#: Label pools. The five short ones are the originals, kept so that sparse charts look
#: exactly as they did; the long ones exist because **the pool was a second density
#: ceiling**. `sample_series` takes `min(n, len(pool))`, and the largest pool held ten
#: entries, so no chart could carry more than ten marks however the count was drawn —
#: against a ChartQA median of 10 and a 90th percentile of 24 (`DECISIONS.md` 0118).
CATEGORY_POOLS: list[list[str]] = [
    [str(y) for y in range(2014, 2024)],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"],
    ["North", "South", "East", "West", "Central"],
    ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"],
    ["Q1", "Q2", "Q3", "Q4"],
    # --- long pools, for the dense half of the distribution ---
    [str(y) for y in range(1980, 2024)],
    [f"{q} {y}" for y in range(2012, 2024) for q in ("Q1", "Q2", "Q3", "Q4")],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    ["Argentina", "Australia", "Austria", "Belgium", "Brazil", "Canada", "Chile",
     "China", "Colombia", "Czechia", "Denmark", "Egypt", "Finland", "France",
     "Germany", "Greece", "Hungary", "India", "Indonesia", "Ireland", "Israel",
     "Italy", "Japan", "Kenya", "Malaysia", "Mexico", "Morocco", "Netherlands",
     "Nigeria", "Norway", "Pakistan", "Peru", "Philippines", "Poland", "Portugal",
     "Romania", "Russia", "Saudi Arabia", "Singapore", "South Africa", "South Korea",
     "Spain", "Sweden", "Switzerland", "Thailand", "Turkey", "Ukraine",
     "United Kingdom", "United States", "Vietnam"],
    ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
     "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois",
     "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
     "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana",
     "Nebraska", "Nevada", "New Jersey", "New Mexico", "New York", "Ohio",
     "Oklahoma", "Oregon", "Pennsylvania", "Tennessee", "Texas", "Utah", "Vermont",
     "Virginia", "Washington", "Wisconsin", "Wyoming"],
    ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
     "60-64", "65-69", "70-74", "75-79", "80+"],
]

#: ChartQA's mark-count distribution, **measured** over 6,264 real training charts with
#: annotation elements. Quantiles rather than a fitted curve: the shape is long-tailed and
#: a two-parameter family would misstate the middle, which is where most charts are.
#:
#: | p10 | p25 | p50 | p75 | p90 | p99 | max | mean |
#: |---:|---:|---:|---:|---:|---:|---:|---:|
#: | 4 | 6 | 10 | 15 | 24 | 45 | 78 | 12.1 |
#:
#: Synthetic before this change: p50 **4**, max **7**, mean 4.6 (`DECISIONS.md` 0098).
CHARTQA_DENSITY_QUANTILES: tuple[tuple[float, int], ...] = (
    (0.00, 2), (0.10, 4), (0.25, 6), (0.50, 10),
    (0.75, 15), (0.90, 24), (0.99, 45), (1.00, 60),
)

#: The largest mark count the box verifier sustains. Measured across chart types after the
#: sentinel fix: bars and pie verify 10 of 10 at 40 marks, and **98.8% of real ChartQA
#: charts have 40 marks or fewer**, so this truncates a 1.2% tail rather than the middle.
MAX_MARKS = 40

#: Marks per chart by curriculum level. `PLAN.md` 6.1 grades stage 1 easy->hard, and 0101
#: says what the grade should be: L1-L2 stay deliberately sparse so the format is learnable
#: in isolation, and **L3-L4 should look like ChartQA** — which is the widest part of the
#: gap, not the narrowest. So the first two levels keep the old range and the last two draw
#: from the measured distribution above.
DENSITY_BY_LEVEL: dict[str, tuple[int, int] | None] = {
    "L1": (3, 6),
    "L2": (4, 9),
    "L3": None,      # None = draw from CHARTQA_DENSITY_QUANTILES
    "L4": None,
}


def sample_density(rng: random.Random, level: str) -> int:
    """How many marks this chart should carry.

    For L3-L4, inverse-transform sampling on the measured quantiles with linear
    interpolation between them, so the synthetic distribution reproduces ChartQA's shape
    rather than a guess at its parameters.
    """
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

# Lightness step applied when a palette wraps; must exceed the verifier's
# colour-matching tolerance (verify.colour_fraction tol=12) by a wide margin.
COLOUR_SHIFT = 60

# Marker stroke width. matplotlib centres a stroke on its path, so half of it falls
# outside the nominal marker; the box functions pad by half of this. Pinned rather
# than left to rcParams so the box and the drawing can never disagree.
LINEWIDTH = 1.5

# Colours used ONLY for the verification render (see `generate_example`). Boxes are
# checked by matching pixels against an element's colour, and a *style* colour may sit
# within the matching tolerance of antialiased text or gridlines: the near-greyscale
# palette produced element colour (94, 94, 119), and 48 of its 686 matched pixels were
# actually text at (102, 105, 110). Verifying on a recoloured render removes the whole
# class of collision, because these colours are fully saturated, mutually distant, and
# nothing else on a chart is drawn in them. Geometry is unaffected by colour, so the
# boxes checked here are the boxes shipped.
SENTINELS: tuple[str, ...] = (
    "#ff00ff", "#00ff00", "#ff0000", "#0000ff", "#ffff00", "#00ffff",
    "#ff8000", "#8000ff", "#00ff80", "#ff0080", "#80ff00", "#0080ff",
)
#: Sentinels beyond the fixed twelve. Kept at high saturation, like the twelve, so they
#: stay unlike anything a chart draws in — text, gridlines and backgrounds are all low
#: saturation or near-grey.
_SENTINEL_GRID = tuple((h / 48.0, s, v)
                       for h in range(48)
                       for s in (1.0, 0.8)
                       for v in (1.0, 0.75, 0.5))


def sentinel_colours(n: int) -> list[str]:
    """`n` verification colours, all mutually distinguishable.

    **This was the ceiling on synthetic chart density.** `generate_example` recoloured
    with `SENTINELS[i % len(SENTINELS)]`, and there are twelve sentinels — so a chart
    with more than twelve elements gave two of them the same verification colour,
    `containment` split its pixels between them, and the example was discarded. Measured
    before the fix: 10 marks verified 10 of 10, **16 verified 3 of 10, 24 verified 0 of
    10** — the partial rate at 16 being the charts whose evidence labels happened to miss
    the collision.

    That is why *"no synthetic chart has more than 7 marks"* (`DECISIONS.md` 0098) while
    ChartQA's median is 10 and its maximum 77, and why 0101 concluded the corpus needed
    regenerating. The limit was never the renderer or the label pool. It was a twelve-item
    tuple consumed with a modulo (`DECISIONS.md` 0118).

    The first twelve are unchanged, so every chart that already verified still does.
    """
    if n <= len(SENTINELS):
        return list(SENTINELS[:max(n, 0)])
    return _extend_by_farthest_point(list(SENTINELS), n, _SENTINEL_GRID)


SENTINEL_LINE = "#404040"      # for a line whose markers must stay distinguishable
#: Largest ratio between the biggest and smallest value in one chart.
MAX_VALUE_RATIO = 8.0

#: Smallest box side, in source pixels, that counts as a usable grounding target.
#: One visual token is 32x32 px (`DECISIONS.md` 0008), so anything a few pixels across
#: is far below what the model can resolve.
MIN_BOX_SIDE_PX = 4.0

QUANTITIES = ("value", "share", "count", "revenue", "score")


#: Minimum RGB distance between any two element colours. The verifier matches a colour
#: within `tol=12` per channel (`synth/verify.py`), so two colours closer than about
#: 12*sqrt(3) = 20.8 are the same colour to it. 60 leaves a wide margin and is comfortably
#: reachable for the densities ChartQA actually contains.
MIN_COLOUR_DISTANCE = 60.0

#: The HSV box element colours are drawn from. Saturation stays high and value avoids both
#: ends, so no element blends into a white background or into the dark text and axes.
_HSV_GRID = tuple((h / 36.0, s, v)
                  for h in range(36)
                  for s in (0.55, 0.75, 0.95)
                  for v in (0.45, 0.65, 0.85))


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def element_colours(palette: list[str], n: int) -> list[str]:
    """`n` colours that are distinct *under the verifier's matching tolerance*.

    The verifier identifies an element by its colour: `containment` counts every pixel of
    that colour in the image, so if two elements share one, a perfectly exact box reports
    ~50% and the example is thrown away.

    **The previous scheme silently ran out of colours.** It wrapped the five-colour
    palette and shifted lightness by `COLOUR_SHIFT * wrap`, which clamps at 0 and 255 —
    so after four wraps every colour saturated to the same black or white. Measured:
    n=24 produced **21 unique colours with a minimum pairwise distance of 0.0**, and n=40
    also produced 21. That is why box verification collapsed above ten marks — 3 of 10
    charts verified at 16 elements and **0 of 10 at 24** — and therefore why no synthetic
    chart has more than 7 marks while ChartQA's median is 10 and its maximum is 77
    (`DECISIONS.md` 0098, 0118).

    So beyond the palette, colours are chosen by farthest-point sampling over an HSV grid:
    each new colour is the candidate furthest from everything already chosen. That
    maximises the minimum separation rather than hoping a fixed shift preserves it, and it
    degrades gracefully — if the grid is ever exhausted the separation shrinks smoothly
    instead of collapsing to zero.

    The first `len(palette)` colours are the palette itself, unchanged, so the charts that
    were already dense enough keep exactly the appearance they had.
    """
    if n <= 0:
        return []
    out = [palette[i % len(palette)] for i in range(min(n, len(palette)))]
    if n <= len(palette):
        return out

    return _extend_by_farthest_point(out, n, _HSV_GRID)


def _extend_by_farthest_point(seed: list[str], n: int,
                              grid: tuple[tuple[float, float, float], ...]) -> list[str]:
    """Extend `seed` to `n` colours, each as far as possible from all chosen so far.

    Farthest-point sampling rather than a fixed shift: it maximises the minimum
    separation instead of hoping a formula preserves it, and when the grid runs low the
    separation shrinks smoothly rather than collapsing to zero.
    """
    import colorsys

    out = list(seed)
    chosen = [_rgb(c) for c in out]
    candidates = [tuple(round(255 * v) for v in colorsys.hsv_to_rgb(*hsv)) for hsv in grid]
    candidates = [c for c in candidates if c not in chosen]
    while len(out) < n and candidates:
        best = max(candidates, key=lambda c: min(_distance(c, x) for x in chosen))
        chosen.append(best)
        candidates.remove(best)
        out.append("#{:02x}{:02x}{:02x}".format(*best))
    return out


def is_holdout(style_seed: int, data_seed: int) -> bool:
    """The single decision point for what is reserved. Never duplicated."""
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
    plan: dict
    evidence: list[dict]              # label, value, unit, bbox (0-1000) — the operands
    table: dict
    image_path: str
    image_sha256: str
    image_size: tuple[int, int]
    style_seed: int
    data_seed: int
    holdout: bool
    #: Every mark on the chart, in chart order. `evidence` is a subset of these, and
    #: `evidence_index` says which (`DECISIONS.md` 0124).
    elements: list[dict] = field(default_factory=list)
    evidence_index: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_record(self) -> dict[str, Any]:
        """The example as the JSON record the model is trained to emit.

        The single place that maps a generated example onto `OUTPUT_SCHEMA`. Keeping it
        here rather than at each call site is what stops consumers disagreeing about
        field names — the schema calls the answer `model_answer`, and `bbox_px` is a
        working value that must never reach a training target.
        """
        return {
            "answerable": True,
            "evidence": [{"label": e["label"], "value": e["value"], "unit": e["unit"],
                          "bbox": e["bbox"]} for e in self.evidence],
            "plan": self.plan,
            "model_answer": self.answer,
        }


def sample_series(rng: random.Random, n_min: int = 3, n_max: int = 7,
                  n: int | None = None, chart_type: str | None = None
                  ) -> tuple[list[tuple[str, float]], str, str | None]:
    """`n` labelled values, or a count drawn from `[n_min, n_max]` when `n` is not given.

    **The pool is chosen to fit the count, not the other way round.** It used to be picked
    first and the count clipped with `min(n, len(pool))`, so asking for 24 marks and
    drawing the four-entry quarters pool silently produced four — a density request that
    looked satisfied and was not (`DECISIONS.md` 0118).
    """
    n = rng.randint(n_min, n_max) if n is None else n
    eligible = [p for p in CATEGORY_POOLS if len(p) >= n]
    if not eligible:
        eligible = [max(CATEGORY_POOLS, key=len)]
    pool = rng.choice(eligible)
    n = min(n, len(pool))
    start = rng.randint(0, max(0, len(pool) - n))
    labels = pool[start:start + n]
    values = sample_values(
        rng, len(labels),
        allow_negative=chart_type not in NON_NEGATIVE_CHART_TYPES)
    return list(zip(labels, values)), rng.choice(QUANTITIES), rng.choice(UNITS)


#: How often a chart's values are percentages that sum to about 100. **Measured** over
#: 4,574 real ChartQA charts: 7.4%. Synthetic produced 0.6% by accident, and a percentage
#: chart is a distinct reasoning setting — the parts are constrained, so "what share does X
#: represent" has an answer the chart states rather than one the reader computes.
PERCENTAGE_SHARE = 0.074

#: How often a chart contains a negative value. **Measured** on the same charts: 1.7%.
#: Synthetic produced none, so the model never saw a minus sign in a value it had to read.
NEGATIVE_SHARE = 0.017

#: The magnitude model, and the largest gap between real and synthetic data.
#: **Measured** |value| over 4,574 real ChartQA charts: p10 3, p50 41, p90 4,447,
#: p99 1.02M, max 272M — and **20.5% of charts carry a value above 1,000**. Synthetic
#: reached p90 198 and a maximum of 457, so **no training chart ever showed a thousands
#: separator** even though `executor.parse_numeric` exists to strip them.
#:
#: Fitted rather than guessed: `log10` of each chart's smallest positive value is close to
#: normal with mean 1.45 and standard deviation 1.33 (p05 -0.20, p50 1.20, p95 4.24), so a
#: chart's scale is drawn from that lognormal. A first attempt used hand-picked decade
#: weights and overshot every quantile — p50 604 against 41 — which is why this is fitted
#: to the data instead (`DECISIONS.md` 0120).
MAGNITUDE_LOG10_MEAN = 1.45
#: The spread of that same fit. 1.33 decades of standard deviation is what makes 20.5%%
#: of charts exceed 1,000 while the median stays near 41; a narrower spread reproduces the
#: median and loses the tail, which is the half that matters here.
MAGNITUDE_LOG10_SD = 1.33
#: Below this a chart's numbers stop looking like data, above it they exceed anything in
#: ChartQA (max 2.7e8). Clamps the tails of the lognormal rather than shaping the middle.
MAGNITUDE_LOG10_RANGE = (-0.7, 8.4)


#: What a reader calls one mark of each chart type, for questions that point at one —
#: *"the blue bar"*, *"the green slice"*. ChartQA's own questions say "bar", "graph",
#: "line" and "slice"; using the wrong noun for the chart would be a giveaway that the
#: question was generated (`DECISIONS.md` 0147).
MARK_WORD = {"vbar": "bar", "hbar": "bar", "grouped_bar": "bar", "pie": "slice",
             "line": "line", "multi_line": "line", "area": "area", "scatter": "point"}

#: Chart families that cannot draw a value below zero. A pie is a part-to-whole chart, so
#: matplotlib refuses outright — `ValueError: Wedge sizes 'x' must be non negative values`
#: — which is the correct behaviour and not something to work around by taking absolutes.
NON_NEGATIVE_CHART_TYPES = frozenset({"pie"})


def sample_values(rng: random.Random, n: int, *,
                  allow_negative: bool = True) -> list[float]:
    """`n` chart values whose distribution resembles ChartQA's rather than a fixed band.

    Three properties are drawn from measurement rather than convenience:

    * **magnitude**, from a lognormal fitted to ChartQA's own chart scales, because real
      chart values span eight orders of magnitude and the old fixed band spanned one;
    * **percentage charts**, which sum to ~100 and are 7.4% of real charts;
    * **negatives**, 1.7%.

    `MAX_VALUE_RATIO` still bounds the *within-chart* spread, and that bound is doing real
    work: it is what keeps the smallest bar tall enough to have a verifiable box. So charts
    move up and down the ladder as a whole rather than mixing a billion with a three.
    """
    if n <= 0:
        return []
    if rng.random() < PERCENTAGE_SHARE and n >= 3:
        raw = [rng.uniform(1.0, 10.0) for _ in range(n)]
        total = sum(raw)
        vals = [round(100.0 * v / total, 1) for v in raw]
        # Put the rounding residue on the largest part, so the chart really sums to 100.
        vals[vals.index(max(vals))] = round(
            max(vals) + 100.0 - sum(vals), 1)
        return vals

    exponent = min(MAGNITUDE_LOG10_RANGE[1],
                   max(MAGNITUDE_LOG10_RANGE[0],
                       rng.gauss(MAGNITUDE_LOG10_MEAN, MAGNITUDE_LOG10_SD)))
    lo = 10.0 ** exponent
    hi = lo * rng.uniform(1.5, MAX_VALUE_RATIO)
    # Precision follows magnitude, in both directions.
    #
    # Upwards, because nobody writes 4,381,220.57 on a chart. Downwards, because rounding
    # is destructive: a chart whose values sit between 0.2 and 1.2, rounded to whole
    # numbers, becomes a chart of zeros — and that is not hypothetical. Widening the
    # magnitude range let `lo` fall below 1 while the old fixed `4.0` floor had made it
    # impossible, and the first run produced `[0.0, 0.0, 0.0]`, which renders as no bars
    # at all and fails verification (`DECISIONS.md` 0120).
    #
    # So: enough decimals that the *smallest* value keeps two significant figures, then
    # optionally one more for variety.
    places = max(0, 1 - math.floor(math.log10(lo))) if lo > 0 else 2
    places = min(places, 3)
    if places == 0 and hi < 1000 and rng.random() < 0.25:
        places = 1
    values = [round(rng.uniform(lo, hi), places) for _ in labels_range(n)]
    if all(v == 0 for v in values):                    # pragma: no cover - guarded above
        raise ValueError(
            f"sample_values produced an all-zero chart at lo={lo!r}, places={places}. "
            f"A chart of zeros renders as no marks and can never be verified.")
    if allow_negative and rng.random() < NEGATIVE_SHARE:
        # A real negative chart is usually a change or a balance: some bars below zero.
        k = rng.randint(1, max(1, n // 2))
        for i in rng.sample(range(n), k):
            values[i] = -values[i]
    return values


def labels_range(n: int) -> range:
    """Trivial, but it keeps `sample_values` readable at the call site."""
    return range(n)


def value_axis_limits(values: list[float]) -> tuple[float, float]:
    """Axis bounds that contain every mark, including negative ones.

    The old bound was `(0, max(values) * 1.25)`, which is correct only while every value
    is positive — and every value was, because `sample_values` could not produce a
    negative one. Adding negatives at ChartQA's measured 1.7% put marks below an axis that
    still started at zero: the bar was clipped out of the figure and its box could not be
    verified, so the example was silently discarded (`DECISIONS.md` 0120).
    """
    top = max(values) * 1.25 if max(values) > 0 else 0.0
    bottom = min(values) * 1.25 if min(values) < 0 else 0.0
    if top == bottom:                                  # a chart of all zeros
        return 0.0, 1.0
    return bottom, top


def _apply_style(fig, ax, st: Style, title: str | None) -> None:
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
        # **Thousands separators, not scientific notation.** Once values reach ChartQA's
        # real magnitudes (p90 4,447 — `sample_values`), matplotlib's default axis
        # formatter switches to an offset like `1e6`, which no Statista chart uses. That
        # would hand the model a chart whose numbers it cannot read while the gold table
        # says 1,234,567. `executor.parse_numeric` exists to strip these separators, and
        # until now no training chart ever contained one (`DECISIONS.md` 0120).
        # The value axis is y for vertical charts and x for horizontal ones; whichever it
        # is, the other is categorical and its formatter must be left alone.
        for axis in (ax.yaxis, ax.xaxis):
            if isinstance(axis.get_major_formatter(), matplotlib.ticker.ScalarFormatter):
                axis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda v, _pos: f"{v:,.10g}"))
    if st.title and title:
        fig.suptitle(title, fontsize=st.font_size + 2,
                     color="white" if st.dark else "black")


def _patch_recolour(patches):
    """Recolour a list of independently coloured patches (bars, pie wedges)."""
    def apply(colours: list[str]) -> None:
        for patch, colour in zip(patches, colours):
            patch.set_facecolor(colour)
    return apply


def _collection_recolour(coll):
    """Recolour a scatter collection, whose points are one artist with a colour array."""
    def apply(colours: list[str]) -> None:
        coll.set_facecolor(list(colours))
        coll.set_edgecolor(list(colours))
    return apply


def _line_recolour(line):
    """Recolour a Line2D's markers, holding the line itself at a separate colour.

    All markers on one line necessarily share a colour, so `containment` cannot
    separate them; the marker check is fill plus tightness (geometry class
    ``disc_shared``). Giving the line its own colour still matters — it stops the
    line's own ink counting toward the marker's fill.
    """
    def apply(colours: list[str]) -> None:
        line.set_color(SENTINEL_LINE)
        line.set_markerfacecolor(colours[0])
        line.set_markeredgecolor(colours[0])
    return apply


# Which *shape* each chart type grounds onto. Drives the verification thresholds.
GEOMETRY_OF: dict[str, str] = {
    "vbar": "rect", "hbar": "rect", "grouped_bar": "rect",
    "line": "disc_shared", "multi_line": "disc_shared", "area": "disc_shared",
    "scatter": "disc_unique",
    "pie": "wedge",
}


def _element_rgb(label: str, series, style: Style, chart_type: str) -> tuple[int, int, int]:
    """The colour the element for `label` was actually drawn in."""
    if chart_type in ("line", "multi_line", "area"):
        hexcol = style.palette[0]           # one series, one colour, markers included
    else:
        idx = [lab for lab, _ in series].index(label)
        hexcol = element_colours(style.palette, len(series))[idx]
    h = hexcol.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _draw(chart_type: ChartType, series: list[tuple[str, float]], st: Style, rng: random.Random,
          title: str | None):
    """Draw the chart and return (fig, ax, per-label pixel-box function)."""
    labels = [lab for lab, _ in series]
    values = [v for _, v in series]
    colours = element_colours(st.palette, len(series))
    fig, ax = plt.subplots(figsize=st.figsize, dpi=st.dpi)

    if chart_type in ("vbar", "grouped_bar"):
        bars = list(ax.bar(labels, values, color=colours))
        ax.set_ylim(*value_axis_limits(values))
        boxes = {lab: (lambda b=b: artist_box(fig, b)) for lab, b in zip(labels, bars)}
        recolour = _patch_recolour(bars)
    elif chart_type == "hbar":
        bars = list(ax.barh(labels, values, color=colours))
        ax.set_xlim(*value_axis_limits(values))
        boxes = {lab: (lambda b=b: artist_box(fig, b)) for lab, b in zip(labels, bars)}
        recolour = _patch_recolour(bars)
    elif chart_type in ("line", "multi_line", "area"):
        marker_pts = rng.choice([10.0, 12.0, 14.0])
        if chart_type == "area":
            # Below the line, or the translucent fill overlays the markers and their
            # colour no longer matches — measured as spurious tightness failures.
            ax.fill_between(range(len(values)), values, alpha=0.3, color=st.palette[0],
                            zorder=1)
        (line,) = ax.plot(range(len(values)), values, marker="o", markersize=marker_pts,
                          color=st.palette[0], markerfacecolor=st.palette[0],
                          markeredgecolor=st.palette[0], markeredgewidth=LINEWIDTH,
                          zorder=3)
        if chart_type == "multi_line":
            distractor = [v * rng.uniform(0.4, 0.7) for v in values]
            ax.plot(range(len(values)), distractor, marker="s", markersize=marker_pts * 0.8,
                    color=st.palette[1], linestyle="--", zorder=2)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylim(*value_axis_limits(values))
        boxes = {lab: (lambda i=i, v=v: point_box(fig, ax, i, v, marker_pts, LINEWIDTH))
                 for i, (lab, v) in enumerate(series)}
        recolour = _line_recolour(line)
    elif chart_type == "scatter":
        s = rng.choice([200.0, 300.0, 400.0])
        coll = ax.scatter(range(len(values)), values, s=s, c=colours, linewidths=LINEWIDTH)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.6, len(values) - 0.4)
        ax.set_ylim(*value_axis_limits(values))
        boxes = {lab: (lambda i=i, v=v: scatter_point_box(fig, ax, i, v, s, LINEWIDTH))
                 for i, (lab, v) in enumerate(series)}
        recolour = _collection_recolour(coll)
    elif chart_type == "pie":
        wedges, _ = ax.pie(values, labels=labels, colors=colours,
                           textprops={"fontsize": st.font_size,
                                      "color": "white" if st.dark else "black"})
        boxes = {lab: (lambda w=w: artist_box(fig, w)) for lab, w in zip(labels, wedges)}
        recolour = _patch_recolour(wedges)
    else:
        plt.close(fig)
        raise ValueError(f"unknown chart type: {chart_type!r}")

    _apply_style(fig, ax if chart_type != "pie" else None, st, title)
    if st.value_labels and chart_type in ("vbar", "grouped_bar"):
        for lab, v in series:
            ax.annotate(f"{v:g}", (lab, v), ha="center", va="bottom", fontsize=st.font_size - 1,
                        color="white" if st.dark else "black")
    if st.legend and chart_type in ("line", "multi_line", "area"):
        ax.legend(["series"], fontsize=st.font_size - 1)
    return fig, ax, boxes, recolour


def generate_example(
    *,
    chart_type: ChartType,
    level: Level,
    style_seed: int,
    data_seed: int,
    out_dir: Path,
    verify: bool = True,
) -> SynthExample | None:
    """One fully specified example, or None if this combination did not yield one."""
    data_rng = random.Random(data_seed)
    style = Style.sample(style_seed)
    # Density is a property of the curriculum level (`DENSITY_BY_LEVEL`), not a fixed
    # range: L1-L2 stay sparse so the format is learnable, L3-L4 reproduce ChartQA's
    # measured distribution (`DECISIONS.md` 0118).
    series, quantity, unit = sample_series(data_rng, n=sample_density(data_rng, level),
                                           chart_type=chart_type)

    # Colours are deterministic from the style and the count, so they can be known before
    # the chart is drawn — which is what lets a question refer to a mark by colour
    # (`DECISIONS.md` 0147).
    element_colour_list = element_colours(style.palette, len(series))
    question = build_question(level, series, data_rng, unit=unit, quantity=quantity,
                              colours=element_colour_list,
                              mark=MARK_WORD.get(chart_type, "bar"),
                              chart_type=chart_type)
    if question is None:
        return None

    title = f"{quantity.title()} by category" if style.title else None
    fig, _ax, box_fns, recolour = _draw(chart_type, series, style, data_rng, title)

    try:
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        by_label = dict(series)

        # **Every mark on the chart**, not only the ones the question needs.
        #
        # `box_fns` has always held a function per label; `generate_example` only ever
        # called the ones in `question.evidence_labels`, so the rest of the chart was
        # discarded before the record was written. That is the information `ChartRecord`
        # now has a place for (`DECISIONS.md` 0124), and it is what a distractor-aware
        # spurious-program check needs: without it, "could another operand pair reach this
        # answer?" cannot even be asked of a synthetic record (0098).
        wanted = set(question.evidence_labels)
        elements: list[dict] = []
        evidence_index: list[int] = []
        for label, _value in series:
            box_px = clip_to_canvas(box_fns[label](), width, height)
            if is_degenerate(box_px, MIN_BOX_SIDE_PX):
                # A mark too small to point at makes the whole chart unusable, not just
                # that mark: an element list with a hole in it would misstate the chart.
                return None
            if label in wanted:
                evidence_index.append(len(elements))
            elements.append({
                "label": label,
                "value": by_label[label],
                "unit": unit,
                "bbox": [round(v, 2) for v in px_to_norm1000(box_px, width, height)],
                "bbox_px": [round(v, 2) for v in box_px],
            })
        if len(evidence_index) != len(wanted):
            return None                    # a plan label that is not a mark on the chart
        evidence = [elements[i] for i in evidence_index]

        if verify:
            # Check the boxes on a recoloured render (see `SENTINELS`), then put the
            # real colours back before saving. Colour does not move any artist, so the
            # geometry verified here is exactly the geometry shipped.
            real = element_colours(style.palette, len(series))
            sentinels = sentinel_colours(len(series))
            recolour(sentinels)
            try:
                if not _verify_boxes(render_rgb(fig), elements, series, sentinels,
                                     chart_type):
                    return None
            finally:
                recolour(real)

        out_dir.mkdir(parents=True, exist_ok=True)
        example_id = f"synth_{chart_type}_{level}_{style_seed}_{data_seed}"
        path = out_dir / f"{example_id}.png"
        fig.savefig(path, facecolor=style.background, bbox_inches=None)
    finally:
        plt.close(fig)

    # Pixel content, matching every other loader, so a synthetic chart and a
    # re-encoded copy of it are one record (`DECISIONS.md` 0048).
    digest = image_content_sha256(path)
    return SynthExample(
        example_id=example_id,
        chart_type=chart_type,
        level=level,
        question=question.question,
        answer=question.answer,
        plan=question.plan,
        evidence=evidence,
        elements=elements,
        evidence_index=evidence_index,
        table={"labels": [lab for lab, _ in series], "values": [v for _, v in series],
               "quantity": quantity, "unit": unit},
        image_path=str(path),
        image_sha256=digest,
        image_size=(width, height),
        style_seed=style_seed,
        data_seed=data_seed,
        holdout=is_holdout(style_seed, data_seed),
        meta={"question_meta": question.meta, "font_size": style.font_size,
              "dark": style.dark, "grid": style.grid},
    )


def _verify_boxes(img, evidence: list[dict], series, colours: list[str],
                  chart_type: str) -> bool:
    """The mandatory self-test: every emitted box must actually contain its element.

    Thresholds come from `GEOMETRY_THRESHOLDS`, which is keyed by *shape*, not chart
    type — a bar's box is ~100% filled while a pie wedge's tight box cannot exceed pi/4.
    """
    labels = [lab for lab, _ in series]
    for item in evidence:
        idx = 0 if chart_type in ("line", "multi_line", "area") \
            else labels.index(item["label"])
        hexcol = colours[idx].lstrip("#")
        rgb = (int(hexcol[0:2], 16), int(hexcol[2:4], 16), int(hexcol[4:6], 16))
        if not check_box_for(img, item["bbox_px"], rgb, item["label"],
                             GEOMETRY_OF[chart_type]).ok:
            return False
    return True


def write_manifest(examples: list[SynthExample], path: Path) -> dict[str, Any]:
    """Record composition, so a mixture is reproducible and auditable."""
    from collections import Counter

    summary = {
        "count": len(examples),
        "by_chart_type": dict(Counter(e.chart_type for e in examples)),
        "by_level": dict(Counter(e.level for e in examples)),
        "holdout": sum(e.holdout for e in examples),
        "style_seeds": sorted({e.style_seed for e in examples}),
        "data_seed_range": [min((e.data_seed for e in examples), default=0),
                            max((e.data_seed for e in examples), default=0)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": summary,
                                "examples": [e.to_dict() for e in examples]},
                               indent=2), encoding="utf-8")
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
) -> list[SynthExample]:
    """Generate up to `n` examples, balanced across chart types and levels."""
    rng = random.Random(seed)
    out: list[SynthExample] = []
    attempts = 0
    while len(out) < n and attempts < n * 12:
        attempts += 1
        ct = chart_types[len(out) % len(chart_types)]
        lv = levels[(len(out) // max(1, len(chart_types))) % len(levels)]
        style_seed = (rng.choice(sorted(HOLDOUT_STYLE_SEEDS)) if holdout
                      else rng.choice([s for s in range(60) if s not in HOLDOUT_STYLE_SEEDS]))
        data_seed = (HOLDOUT_SEED_START + rng.randrange(50_000) if holdout
                     else rng.randrange(HOLDOUT_SEED_START))
        ex = generate_example(chart_type=ct, level=lv, style_seed=style_seed,
                              data_seed=data_seed, out_dir=out_dir, verify=verify)
        if ex is not None:
            out.append(ex)
    return out


__all__ = [
    "CHART_TYPES",
    "HOLDOUT_SEED_START",
    "HOLDOUT_STYLE_SEEDS",
    "MAX_VALUE_RATIO",
    "MIN_BOX_SIDE_PX",
    "NON_NEGATIVE_CHART_TYPES",
    "ChartType",
    "Style",
    "SynthExample",
    "element_colours",
    "generate_batch",
    "generate_example",
    "is_holdout",
    "sample_density",
    "sample_series",
    "sample_values",
    "sentinel_colours",
    "value_axis_limits",
    "write_manifest",
]
