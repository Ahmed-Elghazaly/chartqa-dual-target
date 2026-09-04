"""Naming a chart's colours the way the people who asked the questions did.

**21.8% of human-written ChartQA questions mention a colour** — *"the highest dark blue
bar"*, *"what colour denotes Rep Party"*, *"the longest light blue section"* — against 0.5%
of machine-generated ones (`DECISIONS.md` 0086). Human questions are half the test split and
half the headline metric, so that is a fifth of the half that decides the score.

The information was there the whole time. Every ChartQA annotation model carries a `colors`
list of hex strings, one per datapoint (`#2876dd`, `#0f283e`, `#bababa`), on 80% of charts,
and nothing in this project has ever read it.

Turning a hex into a word people would use is the whole difficulty. `#0f283e` is *navy* or
*dark blue*, never simply *blue*; `#bababa` is *grey*; a saturated `#2876dd` is *blue* and
also, on a chart that has a darker one, *light blue*. So a colour is named with a **set** of
acceptable words rather than one canonical name, and a question matches if it uses any of
them. Being generous here is the safe direction: the name only ever selects which marks a
question is about, and the plan built from that selection is still checked against the gold
answer.
"""

from __future__ import annotations

import colorsys
import re
from collections.abc import Iterable, Sequence

#: Hue ranges in degrees, in the order a person would reach for the word.
_HUES: tuple[tuple[float, float, str], ...] = (
    (345.0, 360.0, "red"), (0.0, 12.0, "red"),
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
    r, g, b = (int(raw[i:i + 2], 16) / 255 for i in (0, 2, 4))
    h, ligh, sat = colorsys.rgb_to_hls(r, g, b)
    return h * 360.0, sat, ligh


#: A colour already written as words needs no mapping, only its neighbours. ChartQA stores
#: `color` as a plain English name on line, h_bar and pie charts — `'dark blue'`, `'orange'`
#: — and as hex on v_bar. Both shapes are real and both are read.
_SYNONYMS: dict[str, set[str]] = {
    "dark blue": {"navy", "blue", "black"}, "navy": {"dark blue", "blue", "black"},
    "light blue": {"blue", "sky blue"}, "sky blue": {"light blue", "blue"},
    "grey": {"gray"}, "gray": {"grey"},
    "dark grey": {"dark gray", "grey", "gray"}, "dark gray": {"dark grey", "grey", "gray"},
    "light grey": {"light gray", "grey", "gray"},
    "light gray": {"light grey", "grey", "gray"},
    "dark red": {"maroon", "red"}, "maroon": {"dark red", "red"},
    "dark green": {"green"}, "light green": {"green"},
}


def names_for(colour: str) -> set[str]:
    """Every word a person might reasonably use for this colour.

    A set, not a name: `#0f283e` answers to *navy*, *dark blue*, *blue* and *black*, and a
    question saying any of them means that series.
    """
    raw = str(colour).strip().lower()
    if raw and not _HEX.match(raw):
        # Already words. Keep them, add the neighbours, and add the bare hue so *"the blue
        # bar"* still finds a series the annotation calls "dark blue".
        out = {raw} | _SYNONYMS.get(raw, set())
        parts = raw.split()
        if len(parts) == 2 and parts[0] in {"dark", "light", "deep", "pale"}:
            out.add(parts[1])
        return out
    parsed = _hsl(colour)
    if parsed is None:
        return set()
    hue, sat, light = parsed

    # Measured against real questions: `#0f283e` (lightness 0.15) is routinely called
    # "black" -- *"the black line"*, *"the highest black casual fan"* -- and calling it only
    # "navy" lost those. A very dark colour answers to both.
    if light <= 0.10:
        return {"black", "dark"}
    if light <= 0.22:
        base = next((n for lo, hi, n in _HUES if lo <= hue < hi), "blue") if sat > 0.12 \
            else "grey"
        return {"black", "dark", f"dark {base}", base} | ({"navy"} if base == "blue" else set())
    if light >= 0.92:
        return {"white", "light"}
    if sat <= 0.12:
        return {"grey", "gray", "silver"} | ({"dark grey", "dark gray"} if light < 0.4
                                             else {"light grey", "light gray"})

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


def series_colour(model: dict) -> str | None:
    """The one colour a series is drawn in, when it is drawn in only one.

    A series whose datapoints differ in colour has no single answer — that is a chart where
    colour distinguishes *categories* rather than series, and a question about "the blue bar"
    means one bar, not the series. Returns `None` so the caller can tell the two cases apart.
    """
    colours = model.get("colors")
    if colours is None and model.get("color") is not None:
        return str(model["color"])
    if isinstance(colours, str):
        return colours
    if not isinstance(colours, list) or not colours:
        return None
    unique = {str(c).lower() for c in colours if c}
    return next(iter(unique)) if len(unique) == 1 else None


def mentioned_in(question: str, palette: Iterable[str],
                 labels: Iterable[str] = ()) -> set[str]:
    """Which of these hex colours the question is talking about.

    Longer names are tested first so *"dark blue"* is not read as *"blue"*, which would match
    every blue series on the chart instead of the dark one.

    `labels` guards against a colour word that is part of a name rather than a colour:
    *"the difference between highest value of lemon oil and lowest value of orange oil"* is
    not about anything orange. A word appearing in one of the chart's own labels is treated
    as that label's name.
    """
    q = f" {re.sub(r'[^a-z ]+', ' ', str(question).lower())} "
    q = re.sub(r"\s+", " ", q)

    # A colour word is not a colour reference when it belongs to a label the question
    # actually names: *"the highest value of lemon oil and lowest of orange oil"* is about
    # two oils. Suppressing the word whenever it appears in ANY label was too blunt -- it
    # also killed *"what is the orange bar worth"* on the same chart -- so the whole label
    # has to be present in the question, not merely the word.
    spoken: set[str] = set()
    for label in labels:
        norm = re.sub(r"[^a-z ]+", " ", str(label).lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        if norm and " " in norm and f" {norm} " in q:
            spoken |= set(norm.split())

    hits: dict[str, int] = {}
    for hex_colour in palette:
        best = 0
        for word in names_for(hex_colour):
            if word in spoken and " " not in word:
                continue
            if f" {word} " in q:
                best = max(best, len(word.split()))
        if best:
            hits[hex_colour] = best

    if not hits:
        return set()
    # *"dark blue"* must not also select every plain blue on the chart. A colour matched by a
    # qualified name beats one matched only by the bare hue.
    strongest = max(hits.values())
    return {c for c, score in hits.items() if score == strongest}


def distinct_palette(models: Sequence[dict]) -> list[str]:
    """Every colour the chart draws, in order of first appearance.

    Not one colour per series. Taking only series drawn in a *single* colour discarded 58.3%
    of colour-mentioning questions, because on many charts colour distinguishes **categories
    within** a series -- each bar its own colour -- and that is exactly the chart a question
    like *"the blue bar"* is asked about.

    Two shapes in the wild: `colors` (a list, one hex per datapoint, on v_bar) and `color`
    (singular, on line / h_bar / pie -- hex on some charts, a plain English name like
    'dark blue' on others). Reading only the first discarded most of the corpus.
    """
    out: list[str] = []
    seen: set[str] = set()
    for model in models:
        raw = model.get("colors")
        listed = ([raw] if isinstance(raw, str) else list(raw or []))
        if not listed and model.get("color") is not None:
            listed = [model["color"]]
        for colour in listed:
            key = str(colour).strip().lower()
            if key and key not in seen and names_for(key):
                seen.add(key)
                out.append(key)
    return out


__all__ = ["distinct_palette", "mentioned_in", "names_for", "series_colour"]
