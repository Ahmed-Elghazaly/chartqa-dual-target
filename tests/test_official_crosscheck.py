"""`PLAN.md` 4.2 as a test: our metrics must agree with the official evaluators.

Marked `official` because it imports the vendored evaluator, which needs torch,
torchmetrics and pycocotools — the CI job that installs those runs it, and the fast CPU
job skips it. The agreement itself is not optional: `DECISIONS.md` 0003 makes the official
evaluator the scorer of record, and a divergence here means our stratified analysis is
describing a different metric from the one the headline numbers use.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.official

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "verification/refchartqa_eval"

torch = pytest.importorskip("torch")
pytest.importorskip("torchmetrics")


@pytest.fixture(scope="module")
def official():
    spec = importlib.util.spec_from_file_location("official_evaluate",
                                                  VENDOR / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["official_evaluate"] = module
    spec.loader.exec_module(module)
    return module


def _official_ap(per_image_preds, per_image_gts, threshold=0.5):
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(iou_thresholds=[threshold], class_metrics=False)
    for preds, gts in zip(per_image_preds, per_image_gts):
        metric.update(
            [{"boxes": torch.tensor(preds, dtype=torch.float).reshape(-1, 4),
              "scores": torch.ones(len(preds)),
              "labels": torch.ones(len(preds), dtype=torch.int64)}],
            [{"boxes": torch.tensor(gts, dtype=torch.float).reshape(-1, 4),
              "labels": torch.ones(len(gts), dtype=torch.int64)}])
    return float(metric.compute()["map"])


def test_relaxed_accuracy_matches_the_official_on_every_edge_case(official):
    """Zero disagreements. Each of these was a real divergence before 0053."""
    from chartqa_dt.eval.metrics import relaxed_correctness

    cases = [("10", "10.4"), ("10", "10.6"), ("10", "10.5"), ("0", "0"), ("0", "0.1"),
             ("0", "0.0"), ("50%", "0.5"), ("0.5", "50%"), ("50%", "50"), ("Yes", "yes"),
             ("Yes", "Yes."), ("Yes, No", "yes, no"), ("", ""), ("42", ""), ("", "42"),
             ("1,234", "1234"), ("1234", "1,234"), ("1,234", "1,234"), ("-5", "-5.1"),
             ("2020", "2020.0"), ("abc", "ABC"), (" Yes ", "Yes")]
    rng = random.Random(0)
    for _ in range(200):
        t = rng.choice([0, 1, 5, 42, 100, 1000, 0.5, -7.25])
        cases.append((str(t), str(round(t * (1 + rng.uniform(-0.12, 0.12)), 4))))

    bad = [(t, p) for t, p in cases
           if bool(official.relaxed_accuracy(p, t)) != relaxed_correctness(t, p)]
    assert not bad, f"{len(bad)} disagreement(s) with the official metric: {bad[:5]}"


def test_p_at_f1_matches_the_official_predicate(official):
    """The official helper is COCO AP == 1.0 on one image, not an F1."""
    from chartqa_dt.eval.metrics import grounding_is_perfect

    good = [0.0, 0.0, 100.0, 100.0]
    other = [200.0, 200.0, 300.0, 300.0]
    bad = [500.0, 500.0, 600.0, 600.0]
    cases = [
        ([good], [good]), ([good, bad], [good]), ([bad, good], [good]),
        ([good, other, bad], [good, other]), ([good, bad, other], [good, other]),
        ([good], [good, other]), ([bad], [good]),
    ]
    for preds, gts in cases:
        assert grounding_is_perfect(preds, gts) == \
            bool(official.is_image_grounding_correct(preds, gts)), \
            f"disagreement on {preds} vs {gts}"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_ap_matches_the_official_on_randomised_scenes(seed):
    """Randomised, because hand-picked cases only test what the author suspected."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from crosscheck_evaluators import ours_ap, random_scene

    rng = random.Random(seed)
    preds, gts = random_scene(rng, 12)
    theirs, ours = _official_ap(preds, gts), ours_ap(preds, gts)
    assert abs(theirs - ours) < 2e-3, \
        f"seed {seed}: official {theirs:.6f} vs ours {ours:.6f}"


def test_the_recorded_crosscheck_shows_no_disagreements():
    """The committed record of the full run, so CI notices if it regresses."""
    path = ROOT / "verification/evaluator_crosscheck.json"
    if not path.exists():
        pytest.skip("cross-check has not been run in this checkout")
    data = json.loads(path.read_text())
    assert data["relaxed_accuracy"]["disagreements"] == []
    assert data["p_at_f1"]["disagreements"] == []
    assert data["ap"]["coco_mean_abs_err"] < 1e-3
    assert data["ap"]["coco_max_abs_err"] < 5e-3


def test_the_level_b_investigation_is_recorded():
    """`PLAN.md` 4.4 requires the discrepancy to be documented if it does not reproduce."""
    path = ROOT / "verification/level_b_reproduction.json"
    if not path.exists():
        pytest.skip("Level-B reproduction has not been run in this checkout")
    data = json.loads(path.read_text())
    assert data["reproduces_32_83"] is False
    human = data["subsets"]["human"]
    assert human["published_ap50"] == 32.83
    assert abs(human["ap50_abs_diff_ours_vs_official"]) < 1e-3, \
        "our metric must match the official on the real prediction set"
