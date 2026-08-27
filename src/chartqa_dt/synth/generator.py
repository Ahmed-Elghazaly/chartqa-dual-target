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

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
CATEGORY_POOLS: list[list[str]] = [
    [str(y) for y in range(2014, 2024)],
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"],
    ["North", "South", "East", "West", "Central"],
    ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"],
    ["Q1", "Q2", "Q3", "Q4"],
]
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
SENTINEL_LINE = "#404040"      # for a line whose markers must stay distinguishable
#: Largest ratio between the biggest and smallest value in one chart.
MAX_VALUE_RATIO = 8.0

#: Smallest box side, in source pixels, that counts as a usable grounding target.
#: One visual token is 32x32 px (`DECISIONS.md` 0008), so anything a few pixels across
#: is far below what the model can resolve.
MIN_BOX_SIDE_PX = 4.0

QUANTITIES = ("value", "share", "count", "revenue", "score")


def element_colours(palette: list[str], n: int) -> list[str]:
    """`n` colours that are distinct *under the verifier's matching tolerance*.

    Palettes hold five colours but a series may have seven categories. Wrapping with
    ``palette[i % len(palette)]`` gives two elements the same colour, and `containment`
    — which counts every pixel of that colour in the image — then splits between them
    and reports ~50% for a perfectly exact box. So on each wrap the base colour is
    shifted in lightness by well beyond the tolerance (`COLOUR_TOL`) instead.
    """
    out: list[str] = []
    for i in range(n):
        base = palette[i % len(palette)]
        wrap = i // len(palette)
        r, g, b = (int(base.lstrip("#")[j:j + 2], 16) for j in (0, 2, 4))
        if wrap:
            shift = COLOUR_SHIFT * wrap * (-1 if (r + g + b) / 3 > 127 else 1)
            r, g, b = (min(255, max(0, c + shift)) for c in (r, g, b))
        out.append(f"#{r:02x}{g:02x}{b:02x}")
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
    evidence: list[dict]              # label, value, unit, bbox (0-1000)
    table: dict
    image_path: str
    image_sha256: str
    image_size: tuple[int, int]
    style_seed: int
    data_seed: int
    holdout: bool
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


def sample_series(rng: random.Random, n_min: int = 3, n_max: int = 7) -> tuple[list[tuple[str, float]], str, str | None]:
    pool = rng.choice(CATEGORY_POOLS)
    n = min(rng.randint(n_min, n_max), len(pool))
    start = rng.randint(0, max(0, len(pool) - n))
    labels = pool[start:start + n]
    decimals = rng.random() < 0.4
    # Values are drawn within a bounded dynamic range. Spanning 1..100 freely produced
    # bars roughly one pixel tall whose boxes cannot be verified — and which would be
    # useless grounding targets regardless. `MAX_VALUE_RATIO` keeps every element large
    # enough to point at while leaving the charts varied.
    lo = rng.uniform(4.0, 60.0)
    hi = lo * rng.uniform(1.5, MAX_VALUE_RATIO)
    values = [round(rng.uniform(lo, hi), 2 if decimals else 0) for _ in labels]
    return list(zip(labels, values)), rng.choice(QUANTITIES), rng.choice(UNITS)


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
        ax.set_ylim(0, max(values) * 1.25)
        boxes = {lab: (lambda b=b: artist_box(fig, b)) for lab, b in zip(labels, bars)}
        recolour = _patch_recolour(bars)
    elif chart_type == "hbar":
        bars = list(ax.barh(labels, values, color=colours))
        ax.set_xlim(0, max(values) * 1.25)
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
        ax.set_ylim(0, max(values) * 1.25)
        boxes = {lab: (lambda i=i, v=v: point_box(fig, ax, i, v, marker_pts, LINEWIDTH))
                 for i, (lab, v) in enumerate(series)}
        recolour = _line_recolour(line)
    elif chart_type == "scatter":
        s = rng.choice([200.0, 300.0, 400.0])
        coll = ax.scatter(range(len(values)), values, s=s, c=colours, linewidths=LINEWIDTH)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.6, len(values) - 0.4)
        ax.set_ylim(0, max(values) * 1.25)
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
    series, quantity, unit = sample_series(data_rng)

    question = build_question(level, series, data_rng, unit=unit, quantity=quantity)
    if question is None:
        return None

    title = f"{quantity.title()} by category" if style.title else None
    fig, _ax, box_fns, recolour = _draw(chart_type, series, style, data_rng, title)

    try:
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        by_label = dict(series)

        evidence: list[dict] = []
        for label in question.evidence_labels:
            box_px = clip_to_canvas(box_fns[label](), width, height)
            if is_degenerate(box_px, MIN_BOX_SIDE_PX):
                return None
            evidence.append({
                "label": label,
                "value": by_label[label],
                "unit": unit,
                "bbox": [round(v, 2) for v in px_to_norm1000(box_px, width, height)],
                "bbox_px": [round(v, 2) for v in box_px],
            })

        if verify:
            # Check the boxes on a recoloured render (see `SENTINELS`), then put the
            # real colours back before saving. Colour does not move any artist, so the
            # geometry verified here is exactly the geometry shipped.
            real = element_colours(style.palette, len(series))
            sentinels = [SENTINELS[i % len(SENTINELS)] for i in range(len(series))]
            recolour(sentinels)
            try:
                if not _verify_boxes(render_rgb(fig), evidence, series, sentinels, chart_type):
                    return None
            finally:
                recolour(real)

        out_dir.mkdir(parents=True, exist_ok=True)
        example_id = f"synth_{chart_type}_{level}_{style_seed}_{data_seed}"
        path = out_dir / f"{example_id}.png"
        fig.savefig(path, facecolor=style.background, bbox_inches=None)
    finally:
        plt.close(fig)

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return SynthExample(
        example_id=example_id,
        chart_type=chart_type,
        level=level,
        question=question.question,
        answer=question.answer,
        plan=question.plan,
        evidence=evidence,
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
    "ChartType",
    "Style",
    "SynthExample",
    "element_colours",
    "generate_batch",
    "generate_example",
    "is_holdout",
    "sample_series",
    "write_manifest",
]
