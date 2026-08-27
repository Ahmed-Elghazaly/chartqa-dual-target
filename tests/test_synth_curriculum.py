"""Every synthetic question must be answerable from its own plan.

The whole justification for synthetic data is that the answer and the plan are known by
construction. An example whose plan does not reproduce its answer would teach the model
something false with perfect confidence, so `build_question` is required to reject it
rather than emit it — these tests make that requirement enforceable.
"""

from __future__ import annotations

import random
import statistics

import pytest

from chartqa_dt.plans.executor import EvidenceItem, execute
from chartqa_dt.plans.schema import validate_record
from chartqa_dt.synth.curriculum import LEVELS, build_question, format_answer


def _series(rng, n=5):
    labels = [f"c{i}" for i in range(n)]
    return [(lab, round(rng.uniform(5, 90), 2)) for lab in labels]


@pytest.mark.parametrize("level", LEVELS)
def test_plan_reproduces_the_answer(level):
    """Across many seeds, not one sampled question may disagree with its own plan."""
    rng = random.Random(11)
    checked = 0
    for _ in range(300):
        series = _series(rng, rng.randint(2, 7))
        q = build_question(level, series, rng)
        if q is None:
            continue
        checked += 1
        by_label = dict(series)
        got = execute(q.plan, [EvidenceItem(lab, by_label[lab], q.unit)
                               for lab in q.evidence_labels])
        assert format_answer(got) == q.answer, f"{q.question} -> {got} != {q.answer}"
    assert checked > 100, f"{level}: only {checked} questions produced"


@pytest.mark.parametrize("level", LEVELS)
def test_every_label_the_plan_uses_is_in_the_evidence(level):
    rng = random.Random(12)
    for _ in range(200):
        series = _series(rng, rng.randint(2, 7))
        q = build_question(level, series, rng)
        if q is None:
            continue
        _walk(q.plan, set(q.evidence_labels))


def _walk(node, labels):
    for arg in node.get("args", []):
        if isinstance(arg, str):
            assert arg in labels, f"{arg} used but not grounded"
        elif isinstance(arg, dict):
            _walk(arg, labels)


def test_levels_have_the_declared_shapes():
    """L1 is one lookup; L4 nests. If that stops holding, the curriculum is not one."""
    rng = random.Random(13)
    series = _series(rng, 5)

    def depth(node):
        if not isinstance(node, dict):
            return 0
        return 1 + max([depth(a) for a in node.get("args", [])] or [0])

    for _ in range(120):
        for level, want in (("L1", 1), ("L4", 2)):
            q = build_question(level, series, rng)
            if q is None:
                continue
            assert depth(q.plan) == want, f"{level}: {q.plan}"
        q3 = build_question("L3", series, rng)
        if q3 is not None:
            # Aggregates fold over the evidence, so the evidence list *is* the argument.
            assert q3.plan["args"] == [], f"L3 must use the fold-over-evidence form: {q3.plan}"
            assert q3.evidence_labels == [lab for lab, _ in series]


def test_answers_render_the_way_the_official_metric_compares_them():
    """`relaxed_correctness` compares a gold '0' as a string (DECISIONS.md 0015)."""
    assert format_answer(0.0) == "0"
    assert format_answer(-0.0) == "0"
    assert format_answer(35.0) == "35"
    assert format_answer(35.5) == "35.5"
    assert format_answer("greater") == "greater"
    assert format_answer(True) == "Yes"
    assert "." not in format_answer(1e9)


def test_a_two_point_series_is_the_minimum():
    rng = random.Random(14)
    assert build_question("L1", [("a", 1.0)], rng) is None
    assert build_question("L2", [], rng) is None


def test_generated_records_pass_the_output_schema(tmp_path):
    """The generator's own output must satisfy the schema the model is trained to emit."""
    from chartqa_dt.synth.generator import CHART_TYPES, generate_example

    for ct in CHART_TYPES:
        for level in LEVELS:
            ex = generate_example(chart_type=ct, level=level, style_seed=4,
                                  data_seed=3001, out_dir=tmp_path)
            assert ex is not None, f"{ct}/{level}"
            result = validate_record(ex.to_record())
            assert result.ok, f"{ct}/{level}: {result.errors}"


def test_l3_aggregates_match_plain_python():
    """A spot check against the obvious implementation, not against our own executor."""
    rng = random.Random(15)
    series = [("a", 10.0), ("b", 20.0), ("c", 30.0), ("d", 45.0)]
    values = [v for _, v in series]
    seen = {}
    for _ in range(400):
        q = build_question("L3", series, rng)
        if q is not None:
            seen[q.plan["op"]] = q.answer
    assert seen["sum"] == format_answer(sum(values))
    assert seen["mean"] == format_answer(statistics.fmean(values))
    assert seen["max"] == format_answer(max(values))
    assert seen["min"] == format_answer(min(values))
    assert seen["count"] == "4"
    assert seen["argmax"] == "d"
    assert seen["argmin"] == "a"
