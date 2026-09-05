"""Synthetic chart density — `DECISIONS.md` 0118, and the three ceilings behind it.

`DECISIONS.md` 0098 measured that no synthetic chart has more than **7** marks while
ChartQA's median is 10 and its maximum 78, and 0101 concluded the corpus needed
regenerating. Neither found *why* the generator could not make a dense chart. There were
three separate caps, each of which alone would have held the ceiling down:

1. `SENTINELS[i % len(SENTINELS)]` — twelve verification colours, cycled, so a thirteenth
   element silently shared a colour and the box check split its pixels;
2. `element_colours` shifting lightness by `COLOUR_SHIFT * wrap`, which clamps at 0/255,
   so beyond ~20 elements the colours were byte-identical;
3. `CATEGORY_POOLS` — the largest pool held ten labels, and `min(n, len(pool))` clipped
   the count without saying so.

Every test here fails if one of those comes back.
"""

from __future__ import annotations

import itertools
import random
import tempfile
from pathlib import Path

import pytest

from chartqa_dt.synth.generator import (
    CATEGORY_POOLS,
    CHARTQA_DENSITY_QUANTILES,
    DENSITY_BY_LEVEL,
    MAX_MARKS,
    SENTINELS,
    element_colours,
    generate_example,
    sample_density,
    sample_series,
    sentinel_colours,
)

#: The verifier matches within `tol=12` per channel (`synth/verify.py`), so two colours
#: closer than 12*sqrt(3) are one colour to it.
VERIFIER_FLOOR = 12 * 3 ** 0.5


def _rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _min_distance(colours):
    return min(sum((a - b) ** 2 for a, b in zip(_rgb(x), _rgb(y))) ** 0.5
               for x, y in itertools.combinations(colours, 2))


# --- ceiling 1: the sentinels -------------------------------------------------------

@pytest.mark.parametrize("n", [1, 5, 12])
def test_the_first_twelve_sentinels_are_unchanged(n):
    """Every chart that already verified must still verify identically."""
    assert sentinel_colours(n) == list(SENTINELS[:n])


@pytest.mark.parametrize("n", [13, 16, 24, 40, 60, 78, 100])
def test_no_two_elements_ever_share_a_sentinel(n):
    """The regression that held the whole corpus to seven marks."""
    assert len(set(sentinel_colours(n))) == n


@pytest.mark.parametrize("n", [13, 24, 40, 78, 100])
def test_sentinels_stay_distinguishable_to_the_verifier(n):
    assert _min_distance(sentinel_colours(n)) > VERIFIER_FLOOR


def test_sentinel_separation_degrades_smoothly_rather_than_collapsing():
    """The old scheme went to a minimum distance of exactly 0.0 at n=24."""
    distances = [_min_distance(sentinel_colours(n)) for n in (16, 24, 40, 60, 78)]
    assert all(d > VERIFIER_FLOOR for d in distances)
    assert distances == sorted(distances, reverse=True)


def test_sentinel_colours_is_deterministic():
    assert sentinel_colours(40) == sentinel_colours(40)


def test_asking_for_no_sentinels_gives_none():
    assert sentinel_colours(0) == []


# --- ceiling 2: the element colours -------------------------------------------------

@pytest.mark.parametrize("n", [13, 24, 40, 78])
def test_element_colours_are_all_distinct(n):
    """Measured before the fix: n=24 gave 21 unique colours, n=40 also 21."""
    palette = ["#3060c8", "#c83030", "#30a050", "#e0a020", "#8040c0"]
    assert len(set(element_colours(palette, n))) == n


@pytest.mark.parametrize("n", [13, 24, 40, 78])
def test_element_colours_stay_distinguishable_to_the_verifier(n):
    palette = ["#3060c8", "#c83030", "#30a050", "#e0a020", "#8040c0"]
    assert _min_distance(element_colours(palette, n)) > VERIFIER_FLOOR


def test_the_palette_itself_is_kept_for_small_charts():
    palette = ["#3060c8", "#c83030", "#30a050"]
    assert element_colours(palette, 3) == palette


def test_element_colours_is_deterministic():
    palette = ["#3060c8", "#c83030", "#30a050"]
    assert element_colours(palette, 30) == element_colours(palette, 30)


# --- ceiling 3: the label pools -----------------------------------------------------

def test_some_pool_can_supply_the_largest_chart_we_allow():
    assert max(len(p) for p in CATEGORY_POOLS) >= MAX_MARKS


@pytest.mark.parametrize("n", [3, 6, 10, 16, 24, 40])
def test_sample_series_returns_exactly_the_count_asked_for(n):
    """`min(n, len(pool))` used to clip this silently."""
    rng = random.Random(0)
    for _ in range(40):
        series, _, _ = sample_series(rng, n=n)
        assert len(series) == n


@pytest.mark.parametrize("n", [3, 12, 24, 40])
def test_sample_series_labels_are_unique(n):
    rng = random.Random(1)
    for _ in range(30):
        series, _, _ = sample_series(rng, n=n)
        labels = [lab for lab, _ in series]
        assert len(set(labels)) == len(labels)


def test_sample_series_still_honours_the_old_range_arguments():
    rng = random.Random(2)
    counts = {len(sample_series(rng, n_min=3, n_max=7)[0]) for _ in range(200)}
    assert counts <= {3, 4, 5, 6, 7}


# --- the density model --------------------------------------------------------------

def test_the_early_levels_stay_sparse():
    """L1-L2 teach the format; density is not what they are for (`PLAN.md` 6.1)."""
    rng = random.Random(0)
    for level in ("L1", "L2"):
        lo, hi = DENSITY_BY_LEVEL[level]
        assert all(lo <= sample_density(rng, level) <= hi for _ in range(500))


def test_the_late_levels_reproduce_chartqa_density():
    """0101: *"L3-L4 should look like ChartQA"* — median 10, p90 24, not median 4."""
    rng = random.Random(0)
    xs = sorted(sample_density(rng, "L3") for _ in range(20_000))

    def q(p):
        return xs[int(p * len(xs))]

    assert q(0.10) == pytest.approx(4, abs=1)
    assert q(0.50) == pytest.approx(10, abs=1)
    assert q(0.90) == pytest.approx(24, abs=2)


def test_density_never_exceeds_what_the_verifier_sustains():
    rng = random.Random(0)
    assert all(2 <= sample_density(rng, "L4") <= MAX_MARKS for _ in range(5_000))


def test_the_quantiles_are_monotonic_and_span_the_unit_interval():
    ps = [p for p, _ in CHARTQA_DENSITY_QUANTILES]
    vs = [v for _, v in CHARTQA_DENSITY_QUANTILES]
    assert ps == sorted(ps) and vs == sorted(vs)
    assert ps[0] == 0.0 and ps[-1] == 1.0


def test_synthetic_density_is_no_longer_capped_at_seven():
    """The headline number in 0098: *no synthetic chart has more than 7 marks*."""
    rng = random.Random(0)
    assert max(sample_density(rng, "L3") for _ in range(5_000)) > 7


# --- end to end ---------------------------------------------------------------------

@pytest.mark.parametrize("chart_type", ["vbar", "hbar", "grouped_bar", "pie"])
def test_a_dense_chart_verifies_its_own_boxes(chart_type):
    """Before the sentinel fix this was 0 of 10 at 24 marks, for every one of these."""
    out = Path(tempfile.mkdtemp())
    built = 0
    for k in range(6):
        rng_patch = {"n": 24}
        ex = _generate_with_marks(chart_type, out, seed=k, **rng_patch)
        built += ex is not None
    assert built >= 5, f"{built}/6 dense {chart_type} charts verified"


def _generate_with_marks(chart_type, out, *, seed, n):
    from chartqa_dt.synth import generator as G

    real = G.sample_series
    G.sample_series = (lambda rng, n_min=3, n_max=7, n=None, chart_type=None:
                       real(rng, n=24, chart_type=chart_type))
    try:
        return generate_example(chart_type=chart_type, level="L3", style_seed=seed,
                                data_seed=7000 + seed, out_dir=out, verify=True)
    finally:
        G.sample_series = real


# --- the value distribution, fitted to ChartQA rather than to convenience ------------

from chartqa_dt.synth.generator import (  # noqa: E402
    MAGNITUDE_LOG10_MEAN,
    MAGNITUDE_LOG10_RANGE,
    NEGATIVE_SHARE,
    PERCENTAGE_SHARE,
    sample_values,
)


def _corpus(n_charts=6000, marks=8, seed=0):
    rng = random.Random(seed)
    return [sample_values(rng, marks) for _ in range(n_charts)]


def _quantile(charts, p):
    xs = sorted(abs(v) for c in charts for v in c if v)
    return xs[int(p * len(xs))]


def test_values_reach_the_magnitudes_real_charts_use():
    """The largest measured gap: ChartQA |value| p90 is 4,447 and synthetic reached 198,
    so no training chart ever showed a thousands separator (`DECISIONS.md` 0120)."""
    charts = _corpus()
    assert _quantile(charts, 0.90) > 1000
    assert max(abs(v) for c in charts for v in c) > 100_000


def test_the_share_of_charts_above_a_thousand_matches_the_measurement():
    """ChartQA: 20.5% of charts carry a value above 1,000."""
    charts = _corpus()
    share = sum(1 for c in charts if max(abs(v) for v in c) > 1000) / len(charts)
    assert 0.15 < share < 0.28, f"{share:.1%} of charts exceed 1,000; measured 20.5%"


@pytest.mark.parametrize("p,target,tol", [(0.10, 3, 6), (0.50, 41, 40), (0.90, 4447, 4000)])
def test_the_magnitude_quantiles_track_chartqa(p, target, tol):
    assert abs(_quantile(_corpus(), p) - target) <= tol


def test_negative_values_appear_at_the_measured_rate():
    """1.7% of real charts contain one, and synthetic contained none — so the model never
    saw a minus sign in a value it had to read."""
    charts = _corpus()
    share = sum(1 for c in charts if any(v < 0 for v in c)) / len(charts)
    assert 0.005 < share < 0.04, f"{share:.1%}; measured 1.7%"


def test_percentage_charts_actually_sum_to_a_hundred():
    """A percentage chart whose parts do not sum to 100 is not a percentage chart."""
    charts = _corpus()
    # Exactly 100, not merely near it: eight values around twelve sum to ~100 by
    # coincidence often enough that a +/-1 window mostly catches ordinary charts. That
    # coincidence is what an earlier version of this test mistook for a generator bug.
    exact = [c for c in charts if abs(sum(c) - 100.0) < 0.05]
    share = len(exact) / len(charts)
    assert 0.03 < share < 0.13, f"{share:.1%} sum to exactly 100; measured 7.4%"
    for c in exact:
        assert all(v > 0 for v in c), "a percentage part cannot be negative"


def test_a_chart_keeps_one_scale_rather_than_mixing_a_billion_with_a_three():
    """`MAX_VALUE_RATIO` bounds the within-chart spread, and it earns its place: it is
    what keeps the smallest mark tall enough to have a verifiable box."""
    from chartqa_dt.synth.generator import MAX_VALUE_RATIO

    for c in _corpus(1500):
        vs = [abs(v) for v in c if v]
        if len(vs) > 1 and min(vs) > 0 and abs(sum(c) - 100) > 1:
            assert max(vs) / min(vs) <= MAX_VALUE_RATIO * 1.05


def test_values_are_bounded_by_the_declared_range():
    for c in _corpus(2000):
        for v in c:
            assert abs(v) <= 10 ** (MAGNITUDE_LOG10_RANGE[1] + 1)


def test_sample_values_is_deterministic():
    assert sample_values(random.Random(7), 9) == sample_values(random.Random(7), 9)


def test_no_values_asked_gives_none():
    assert sample_values(random.Random(0), 0) == []


def test_decimals_only_appear_where_they_are_plausible():
    """Nobody writes 4,381,220.57 on a chart."""
    for c in _corpus(3000):
        for v in c:
            if abs(v) > 10_000:
                assert v == int(v), f"{v} carries decimals at that magnitude"


def test_the_declared_shares_are_the_measured_ones():
    assert pytest.approx(0.074, abs=0.01) == PERCENTAGE_SHARE
    assert pytest.approx(0.017, abs=0.005) == NEGATIVE_SHARE
    assert pytest.approx(1.45, abs=0.1) == MAGNITUDE_LOG10_MEAN


def test_a_chart_is_never_all_zeros():
    """Rounding is destructive. Widening the magnitude range let `lo` fall below 1 while
    precision was still chosen by a coin flip, and the first run produced [0.0, 0.0, 0.0]
    — a chart that renders as no bars at all and can never be verified."""
    rng = random.Random(0)
    for _ in range(20_000):
        assert any(v != 0 for v in sample_values(rng, 5))


def test_the_smallest_value_keeps_two_significant_figures():
    rng = random.Random(1)
    for _ in range(5_000):
        values = [abs(v) for v in sample_values(rng, 6) if v]
        assert values, "every value rounded away"
        assert min(values) > 0


def test_pie_charts_never_receive_a_negative_value():
    """matplotlib refuses outright — `Wedge sizes must be non negative` — and it is right
    to: a pie is a part-to-whole chart."""
    from chartqa_dt.synth.generator import NON_NEGATIVE_CHART_TYPES

    assert "pie" in NON_NEGATIVE_CHART_TYPES
    rng = random.Random(0)
    for _ in range(4_000):
        series, _, _ = sample_series(rng, n=6, chart_type="pie")
        assert all(v >= 0 for _, v in series)


def test_other_chart_types_still_get_negatives():
    rng = random.Random(0)
    seen = any(v < 0
               for _ in range(4_000)
               for _, v in sample_series(rng, n=6, chart_type="vbar")[0])
    assert seen, "negatives disappeared from the chart types that can draw them"


@pytest.mark.parametrize("values,expected_contains_zero", [
    ([1.0, 2.0, 3.0], True), ([-5.0, 2.0], True), ([-9.0, -1.0], True),
])
def test_value_axis_limits_contain_every_mark(values, expected_contains_zero):
    from chartqa_dt.synth.generator import value_axis_limits

    lo, hi = value_axis_limits(values)
    assert lo <= min(values) and hi >= max(values)
    if expected_contains_zero:
        assert lo <= 0 <= hi


def test_value_axis_limits_survive_an_all_zero_chart():
    from chartqa_dt.synth.generator import value_axis_limits

    lo, hi = value_axis_limits([0.0, 0.0])
    assert lo < hi, "a degenerate axis would make matplotlib pick its own bounds"
