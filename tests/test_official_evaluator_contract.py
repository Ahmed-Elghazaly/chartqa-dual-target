"""Pins what the OFFICIAL RefChartQA evaluator actually does.

Decision 0003 makes this evaluator the scorer of record, so its behaviour is part
of the definition of every grounding number this project reports. If it changes —
because the upstream file was edited, or torchmetrics changed its interpolation —
these tests fail and the reported numbers stop meaning what the report says.

Every expectation here was obtained by running the evaluator, not by reading it.
Reading it was how decision 0003 reached a conclusion whose reasoning turned out
to be wrong (see 0014).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

pytestmark = pytest.mark.official

EVALUATOR = pathlib.Path(__file__).resolve().parents[1] / "verification/refchartqa_eval/evaluate.py"
W, H = 800, 386
GT_A = {"x": 276.0, "y": 277.0, "w": 60.0, "h": 23.0}
GT_B = {"x": 500.0, "y": 100.0, "w": 60.0, "h": 23.0}
BAD = [[10, 10, 60, 40], [100, 100, 150, 140], [200, 200, 250, 240]]


@pytest.fixture(scope="module")
def ev():
    pytest.importorskip("torchmetrics")
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("refeval", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def item(ev, boxes, gts, answer="47"):
    body = "".join(f"<box>{','.join(map(str, b))}</box>" for b in boxes)
    return {"model_answer": f"{body}<grounding-sep>{answer}", "label": answer,
            "width": W, "height": H, "grounding_bboxes": [dict(g) for g in gts]}


# ------------------------------------------------- relaxed accuracy


@pytest.mark.parametrize(
    ("target", "pred", "expected"),
    [
        ("10", "10.4", True),        # within 5%
        ("10", "10.6", False),       # outside 5%
        ("0", "0", True),            # zero-division guard
        ("0", "0.1", False),
        ("Yes", "yes", True),        # case-insensitive
        ("10", " 10 ", True),        # float() tolerates whitespace
        ("50%", "0.5", True),        # percent divided by 100 on both sides
    ],
)
def test_relaxed_accuracy_agreed_cases(ev, target, pred, expected):
    assert bool(ev.relaxed_accuracy(pred, target)) is expected


@pytest.mark.parametrize(
    ("target", "pred", "official", "why"),
    [
        ("0", "0.0", False, "target 0.0 is falsy, so the numeric branch is skipped entirely"),
        ("1,234", "1234", False, "official does not strip commas; falls back to string equality"),
        ("Yes", "Yes.", False, "no punctuation normalisation"),
    ],
)
def test_relaxed_accuracy_documented_divergences(ev, target, pred, official, why):
    """These differ from PLAN.md Appendix D. Official wins (rule 5, decision 0003)."""
    assert bool(ev.relaxed_accuracy(pred, target)) is official, why


# ------------------------------------------------- the silent discard


def test_a_coordinate_of_1000_discards_the_whole_box(ev):
    """Decision 0004. Qwen3-VL emits 0..1000; the evaluator accepts only 0..999."""
    assert ev.extract_bounding_boxes("<box>0,0,999,999</box>", bins=1000) == [[0.0, 0.0, 999.0, 999.0]]
    assert ev.extract_bounding_boxes("<box>0,0,1000,1000</box>", bins=1000) == []


def test_ground_truth_is_also_capped_at_999(ev):
    """So clamping predictions to 0..999 matches the GT convention exactly."""
    full_image = {"x": 0.0, "y": 0.0, "w": float(W), "h": float(H)}
    assert ev.transform_bbox_to_quantized(full_image, W, H, 1000) == [0, 0, 999, 999]


def test_malformed_box_is_dropped_not_raised(ev):
    assert ev.extract_bounding_boxes("<box>1,2,3</box>", bins=1000) == []
    assert ev.extract_bounding_boxes("no boxes at all", bins=1000) == []


# ------------------------------------------------- the template


@pytest.mark.parametrize(
    ("answer", "scores"),
    [("<box>1,2,3,4</box><grounding-sep>47", True),
     ("47", False),
     ("<box>1,2,3,4</box><grounding-sep>47<grounding-sep>x", False)],
)
def test_answer_requires_exactly_one_grounding_sep(ev, answer, scores):
    assert bool(ev.eval_is_element_correct(answer, "47")) is scores


# ------------------------------------------------- ordering and extras (0014)


@pytest.mark.parametrize(
    ("rank", "expected_ap"),
    [(0, 1.0), (1, 0.5), (2, 1 / 3), (3, 0.25)],
)
def test_ap_equals_one_over_the_rank_of_the_first_correct_box(ev, rank, expected_ap):
    g = ev.transform_bbox_to_quantized(dict(GT_A), W, H, 1000)
    boxes = [*BAD[:rank], g]
    assert ev.compute_AP_50([item(ev, boxes, [GT_A])]) == pytest.approx(expected_ap, abs=1e-3)


def test_p_at_f1_is_zero_if_any_wrong_box_precedes_a_correct_one(ev):
    g = ev.transform_bbox_to_quantized(dict(GT_A), W, H, 1000)
    assert ev.compute_P_at_FI([item(ev, [g], [GT_A])]) == 1.0
    assert ev.compute_P_at_FI([item(ev, [BAD[0], g], [GT_A])]) == 0.0


def test_p_at_f1_ignores_trailing_false_positives(ev):
    """PLAN.md calls P@F1 'the full predicted set must be correct'. It is not.

    Boxes appended AFTER a correct one do not break it; only a false positive
    before a true one does.
    """
    g = ev.transform_bbox_to_quantized(dict(GT_A), W, H, 1000)
    assert ev.compute_P_at_FI([item(ev, [g, *BAD], [GT_A])]) == 1.0


def test_extra_boxes_are_free_per_image_but_ruinous_in_aggregate(ev):
    """The central finding of decision 0014.

    Dataset AP pools every prediction into one PR curve, and all scores are tied
    at 1.0, so extras from other images interleave with true positives.
    """
    g = ev.transform_bbox_to_quantized(dict(GT_A), W, H, 1000)

    single_clean = ev.compute_AP_50([item(ev, [g], [GT_A])])
    single_extras = ev.compute_AP_50([item(ev, [g, *BAD], [GT_A])])
    assert single_clean == pytest.approx(1.0)
    assert single_extras == pytest.approx(1.0), "per image, extras look free"

    n = 20
    many_clean = ev.compute_AP_50([item(ev, [g], [GT_A]) for _ in range(n)])
    many_extras = ev.compute_AP_50([item(ev, [g, *BAD], [GT_A]) for _ in range(n)])
    assert many_clean == pytest.approx(1.0, abs=1e-3)
    assert many_extras < 0.40, (
        f"in aggregate the same extras collapse AP to {many_extras:.4f}; "
        "this is why the evidence list is filtered before scoring"
    )


def test_missing_one_of_two_ground_truth_boxes_halves_ap(ev):
    g_a = ev.transform_bbox_to_quantized(dict(GT_A), W, H, 1000)
    g_b = ev.transform_bbox_to_quantized(dict(GT_B), W, H, 1000)
    both = ev.compute_AP_50([item(ev, [g_a, g_b], [GT_A, GT_B])])
    one = ev.compute_AP_50([item(ev, [g_a], [GT_A, GT_B])])
    assert both == pytest.approx(1.0)
    assert 0.45 < one < 0.55


def test_empty_prediction_scores_zero_and_does_not_crash(ev):
    assert ev.compute_AP_50([item(ev, [], [GT_A])]) == 0.0
    assert ev.compute_P_at_FI([item(ev, [], [GT_A])]) == 0.0


# ------------------------------- canonical equivalence (decision 0015)

# Verbatim copy of google-research/pix2struct/pix2struct/metrics.py::relaxed_correctness,
# inlined so this test needs no network. RefChartQA's evaluate.py cites it by URL.
def _pix2struct_relaxed_correctness(target: str, prediction: str, max_relative_change: float = 0.05) -> bool:
    def _to_float(text: str):
        try:
            if text.endswith("%"):
                return float(text.rstrip("%")) / 100.0
            return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        return abs(prediction_float - target_float) / abs(target_float) <= max_relative_change
    return prediction.lower() == target.lower()


def test_vendored_metric_equals_the_canonical_pix2struct_one(ev):
    """Decision 0015: one function scores BOTH benchmarks.

    The ChartQA repository ships no answer evaluator, so the implementation that
    every published ChartQA number was produced with is pix2struct's. RefChartQA
    vendors it verbatim. If these ever diverge, our ChartQA and RefChartQA answer
    scores stop being the same metric.
    """
    import itertools

    targets = ["10", "0", "0.0", "50%", "1,234", "Yes", "yes", "", "3.5", "-2", "100%", "abc"]
    preds = ["10", "10.4", "10.6", "0", "0.0", "0.1", "0.5", "50%", "1234", "1,234",
             "Yes", "Yes.", " 10 ", "", "3.6", "-2.05", "1.0", "ABC"]
    mismatches = [
        (t, p) for t, p in itertools.product(targets, preds)
        if bool(ev.relaxed_accuracy(p, t)) != bool(_pix2struct_relaxed_correctness(t, p))
    ]
    assert not mismatches, f"vendored copy diverged from canonical on: {mismatches[:5]}"


def test_the_zero_quirk_is_canonical_not_a_refchartqa_edit():
    """`and target_float` is a truthiness test, so a gold answer of 0 is falsy.

    Every published ChartQA number carries this. Reporting a 'fixed' metric would
    make our numbers unmatched against all of them (rule 6).
    """
    assert _pix2struct_relaxed_correctness("0", "0.0") is False
    assert _pix2struct_relaxed_correctness("0", "0") is True


def test_answers_must_normalise_to_bare_zero():
    """Consequence for the answer normaliser, frozen in PREREGISTRATION.md."""
    assert _pix2struct_relaxed_correctness("0", "0") is True
    for bad in ("0.0", "0.00", "0%"):
        assert _pix2struct_relaxed_correctness("0", bad) is False, f"{bad!r} scores wrong against gold '0'"
