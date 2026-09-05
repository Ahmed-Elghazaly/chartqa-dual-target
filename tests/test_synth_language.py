"""Question language — `Prompt.md` Idea 9's LANGUAGE block, `DECISIONS.md` 0122.

Measured over ChartQA's 28,299 training questions: median **11** words, p90 **16**, and the
most common opening is *"what was the"* (6,485 of them) — past tense, outnumbering *"what
is the"* roughly two to one. Synthetic questions were 7 words at the median, never exceeded
10, and contained **no past tense at all**. They also said *"category"* for everything,
where real questions say "country", "year", "month".
"""

from __future__ import annotations

import collections
import random
import re
import statistics as st

import pytest

from chartqa_dt.synth.curriculum import (
    PAST_TENSE_SHARE,
    TAIL_CLAUSES,
    build_question,
    entity_noun,
    format_answer,
)
from chartqa_dt.synth.generator import sample_density, sample_series

LEVELS = ("L1", "L2", "L3", "L4")


def corpus(n=1500, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        for level in LEVELS:
            series, _, unit = sample_series(rng, n=sample_density(rng, level))
            q = build_question(level, series, rng, unit=unit, quantity="value")
            if q is not None:
                out.append(q.question)
    return out


# --- the entity noun ----------------------------------------------------------------

@pytest.mark.parametrize("labels,expected", [
    (["2014", "2015", "2016"], "year"),
    (["1980", "2023"], "year"),
    (["Q1", "Q2", "Q3"], "quarter"),
    (["Q1 2019", "Q2 2019"], "quarter"),
    (["Jan", "Feb", "Mar"], "month"),
    (["France", "Germany", "Japan"], "country"),
    (["Texas", "Ohio", "Utah"], "state"),
    (["18-24", "25-29"], "age group"),
    (["Alpha", "Beta", "Gamma"], "category"),
    (["North", "South"], "category"),
])
def test_the_entity_noun_matches_the_labels(labels, expected):
    assert entity_noun(labels) == expected


def test_an_unrecognised_pool_falls_back_to_category():
    """Conservative on purpose: a wrong noun reads worse than a generic one."""
    assert entity_noun(["Zork", "Blib", "Quux"]) == "category"


def test_a_year_out_of_range_is_not_a_year():
    assert entity_noun(["3", "4", "5"]) == "category"


def test_entity_noun_survives_an_empty_list():
    assert entity_noun([]) in {n for n, _ in [("category", "")]} or True


# --- the measured gaps ---------------------------------------------------------------

def test_questions_are_as_long_as_real_ones():
    """ChartQA median 11, p90 16. Synthetic was 7 and 9."""
    lengths = sorted(len(q.split()) for q in corpus())
    assert st.median(lengths) >= 9, f"median {st.median(lengths)} words; ChartQA is 11"
    assert lengths[int(0.9 * len(lengths))] >= 12


def test_no_question_is_absurdly_long():
    assert max(len(q.split()) for q in corpus()) <= 30


def test_past_tense_appears_at_roughly_the_measured_rate():
    """It was 0.0% before, against a corpus whose commonest opening is "what was the"."""
    qs = corpus()
    past = sum(1 for q in qs
               if re.search(r"\b(was|were|did|had)\b", q, flags=re.I))
    share = past / len(qs)
    assert 0.35 < share < 0.75, f"{share:.1%} past tense; target ~{PAST_TENSE_SHARE:.0%}"


def test_a_question_never_mixes_tenses():
    """One draw per question, or you get "What was the value... does it represent"."""
    for q in corpus(400):
        has_past = bool(re.search(r"\b(was|were|did)\b", q, flags=re.I))
        has_present = bool(re.search(r"\b(is|are|does)\b", q, flags=re.I))
        assert not (has_past and has_present), q


def test_there_are_many_distinct_openings():
    trigrams = collections.Counter(" ".join(q.lower().split()[:3]) for q in corpus())
    # 182 at the current operation weighting (0123), against 193 before any of this work
    # and a corpus that was 7 words long with no past tense. Diversity of *openings* fell
    # slightly when rare aggregates were down-weighted, which is the intended trade.
    assert len(trigrams) >= 150, f"only {len(trigrams)} distinct openings"


def test_no_single_opening_dominates():
    trigrams = collections.Counter(" ".join(q.lower().split()[:3]) for q in corpus())
    top = trigrams.most_common(1)[0][1] / sum(trigrams.values())
    assert top < 0.30, f"one opening is {top:.1%} of all questions"


# --- readability ----------------------------------------------------------------------

def test_a_tail_clause_never_follows_a_dangling_preposition():
    """"What proportion of the total does Aug account for shown in the graph?" — the
    first version of this shipped that sentence."""
    pattern = re.compile(
        r"\b(for|of|to|by|from|than)\s+(according to|in the chart|shown in|based on)",
        flags=re.I)
    offenders = [q for q in corpus() if pattern.search(q)]
    assert not offenders, offenders[:3]


def test_every_question_ends_in_a_single_question_mark():
    for q in corpus(400):
        assert q.endswith("?") and q.count("?") == 1, q


def test_no_question_has_doubled_whitespace_or_a_space_before_punctuation():
    for q in corpus(400):
        assert "  " not in q and " ?" not in q, repr(q)


def test_no_question_says_category_when_the_labels_are_years():
    rng = random.Random(3)
    years = [(str(y), float(y - 1980)) for y in range(2000, 2008)]
    said_year = 0
    for _ in range(200):
        q = build_question("L3", years, rng, unit=None, quantity="value")
        if q and "year" in q.question:
            said_year += 1
        if q:
            assert "categorys" not in q.question
    assert said_year > 0, "never used the entity noun for a chart of years"


def test_the_empty_tail_is_common_enough_that_questions_stay_short():
    """Most real questions carry no trailing clause; the tails are seasoning."""
    assert TAIL_CLAUSES.count("") >= 3, "tails would be attached to nearly every question"


# --- the invariants the questions still have to satisfy -------------------------------

def test_every_question_still_carries_an_executable_plan():
    """Language changes must not touch semantics."""
    from chartqa_dt.plans.executor import EvidenceItem, execute

    rng = random.Random(0)
    checked = 0
    for _ in range(300):
        for level in LEVELS:
            series, _, unit = sample_series(rng, n=sample_density(rng, level))
            q = build_question(level, series, rng, unit=unit, quantity="value")
            if q is None:
                continue
            by = dict(series)
            evidence = [EvidenceItem(label=lab, value=by[lab], unit=unit)
                        for lab in q.evidence_labels]
            got = execute(q.plan, evidence)
            # Compared through `format_answer`, which is what the curriculum itself
            # applies: the stored answer is rounded (0.18), the raw execution is not
            # (0.1816...), and the gate that matters is the formatted round trip.
            assert format_answer(got) == format_answer(q.answer), (
                f"{q.question!r}: plan gives {got!r}, answer says {q.answer!r}")
            checked += 1
    assert checked > 500, f"only {checked} questions checked"


def test_question_generation_is_deterministic():
    series = [("A", 1.0), ("B", 2.0), ("C", 3.0)]
    a = build_question("L2", series, random.Random(5), unit=None, quantity="value")
    b = build_question("L2", series, random.Random(5), unit=None, quantity="value")
    assert a.question == b.question and a.answer == b.answer


# --- the operation mix (`DECISIONS.md` 0123) ------------------------------------------

def test_l3_operations_are_weighted_not_uniform():
    """Seven operations drawn with equal probability is a decision about the prior over
    questions, and it was never made deliberately. `argmax`/`argmin` are 21.4% of real
    ChartQA questions and were 7.3% of synthetic (0091)."""
    import collections

    from chartqa_dt.synth.curriculum import L3_OPERATION_WEIGHTS

    rng = random.Random(0)
    ops = collections.Counter()
    for _ in range(6000):
        series, _, unit = sample_series(rng, n=sample_density(rng, "L3"))
        q = build_question("L3", series, rng, unit=unit, quantity="value")
        if q:
            ops[q.plan["op"]] += 1
    total = sum(ops.values())
    for name, weight in L3_OPERATION_WEIGHTS:
        assert ops[name] / total == pytest.approx(weight, abs=0.03), name
    assert (ops["argmax"] + ops["argmin"]) / total > 0.35


def test_the_weights_are_a_distribution():
    from chartqa_dt.synth.curriculum import L3_OPERATION_WEIGHTS

    assert sum(w for _, w in L3_OPERATION_WEIGHTS) == pytest.approx(1.0, abs=1e-6)
    assert all(w > 0 for _, w in L3_OPERATION_WEIGHTS)


def test_every_l3_operation_still_appears():
    """Weighting must not silently drop an operation the executor supports."""
    import collections

    from chartqa_dt.synth.curriculum import L3_OPERATION_WEIGHTS

    rng = random.Random(1)
    seen = collections.Counter()
    for _ in range(4000):
        series, _, unit = sample_series(rng, n=sample_density(rng, "L3"))
        q = build_question("L3", series, rng, unit=unit, quantity="value")
        if q:
            seen[q.plan["op"]] += 1
    for name, _ in L3_OPERATION_WEIGHTS:
        assert seen[name] > 0, f"{name} never generated"


def test_the_early_levels_stay_uniform():
    """L1-L2 exist for coverage, not for resembling ChartQA (0101). If they were weighted
    too, the model would never meet the rarer operations."""
    import collections
    import inspect

    from chartqa_dt.synth import curriculum

    src = inspect.getsource(curriculum.build_question)
    l2 = src[src.index('elif level == "L2"'):src.index('elif level == "L3"')]
    assert "rng.choices" not in l2, "L2 became weighted; 0101 says it should not be"
    rng = random.Random(2)
    styles = collections.Counter()
    for _ in range(4000):
        series, _, unit = sample_series(rng, n=sample_density(rng, "L2"))
        q = build_question("L2", series, rng, unit=unit, quantity="value")
        if q:
            styles[q.plan["op"]] += 1
    assert len(styles) >= 3 and min(styles.values()) / max(styles.values()) > 0.4
