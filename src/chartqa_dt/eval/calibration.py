"""Calibration and selective accuracy — `PLAN.md` 9.6.

9.6 asks for *"ECE, Brier score, accuracy-versus-coverage curves"*, and adds the constraint
that matters more than any of them: **"ChartQA accuracy remains reported at 100% coverage;
selective accuracy is a separate diagnostic only."** Selective accuracy always looks better
than the headline — that is what selection does — so the two are kept structurally apart
here, and `assert_headline_is_full_coverage` refuses a report that quotes a selective
number as the result.

**ECE and Brier need a probability, and this system does not emit one.** It emits a JSON
record. So they are computed only when a real per-item probability is supplied — sequence
log-probabilities from `generate`, if a run chooses to pay for them — and are otherwise
reported as *not computed*, rather than manufactured from an ordinal proxy. A calibration
error computed against a made-up confidence is a number about the proxy, not the model.

What the system *does* have is a **checkable** signal, which is unusual and worth using:
the executor either reproduces the model's own stated answer or it does not
(`DECISIONS.md` 0014). `agreement_confidence` turns that, plus repair count, schema
validity and token-cap truncation, into an ordinal score for the coverage curve.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: Ordinal confidence levels, best first. Coarse by construction — there are only so many
#: distinguishable states a record can be in without a probability.
CONFIDENCE_LEVELS = ("agrees, clean", "agrees, repaired", "executes but disagrees",
                     "schema-valid only", "parsed only", "unusable")


@dataclass
class CoveragePoint:
    threshold: str
    coverage: float
    accuracy: float
    n: int
    n_correct: int

    def to_dict(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "coverage": self.coverage,
                "accuracy": self.accuracy, "n": self.n, "n_correct": self.n_correct}


def agreement_confidence(item: dict[str, Any]) -> str:
    """Which ordinal level a record sits in, best first.

    Built from what the record itself shows: whether the executor reproduced the stated
    answer, whether the JSON needed repair, whether it satisfies the schema, and whether
    generation was truncated at the token cap.
    """
    if not item.get("parsed"):
        return "unusable"
    if item.get("hit_token_cap"):
        return "parsed only"
    if not item.get("schema_valid"):
        return "parsed only"
    if not item.get("executes"):
        return "schema-valid only"
    if not item.get("agrees"):
        return "executes but disagrees"
    return "agrees, clean" if not item.get("repairs") else "agrees, repaired"


def coverage_curve(items: Sequence[dict[str, Any]]) -> list[CoveragePoint]:
    """Accuracy as the confidence bar is lowered, ending at 100% coverage.

    The last point is always the whole set, because that is the number `PLAN.md` requires
    to be reported as the result.
    """
    total = len(items)
    if not total:
        return []
    levels = [agreement_confidence(i) for i in items]
    points: list[CoveragePoint] = []
    kept: list[int] = []
    for level in CONFIDENCE_LEVELS:
        added = [i for i, lv in enumerate(levels) if lv == level]
        # A level nobody is in adds no point. Emitting one anyway repeats the previous
        # coverage and accuracy under a new name, which reads as a flat region of the
        # curve that is really an absence of data.
        if not added:
            continue
        kept += added
        correct = sum(bool(items[i].get("correct")) for i in kept)
        points.append(CoveragePoint(threshold=level, coverage=len(kept) / total,
                                    accuracy=correct / len(kept), n=len(kept),
                                    n_correct=correct))
    if not points or points[-1].coverage < 1.0:
        correct = sum(bool(i.get("correct")) for i in items)
        points.append(CoveragePoint(threshold="all", coverage=1.0,
                                    accuracy=correct / total, n=total, n_correct=correct))
    return points


def brier_score(probabilities: Sequence[float], outcomes: Sequence[bool]) -> float:
    """Mean squared error of a probabilistic forecast. Needs a real probability."""
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must be the same length")
    if not probabilities:
        return float("nan")
    return sum((p - bool(o)) ** 2 for p, o in zip(probabilities, outcomes)) / len(outcomes)


def expected_calibration_error(probabilities: Sequence[float], outcomes: Sequence[bool],
                               *, bins: int = 10) -> float:
    """Equal-width binned ECE, the standard formulation.

    Empty bins contribute nothing rather than counting as perfectly calibrated: averaging
    over `bins` instead of over the bins that hold data would drag ECE towards zero for a
    model that only ever emits two confidences.
    """
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must be the same length")
    n = len(probabilities)
    if not n:
        return float("nan")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for p, o in zip(probabilities, outcomes):
        index = min(int(p * bins), bins - 1)
        buckets[index].append((p, bool(o)))
    total = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        confidence = sum(p for p, _ in bucket) / len(bucket)
        accuracy = sum(o for _, o in bucket) / len(bucket)
        total += len(bucket) / n * abs(confidence - accuracy)
    return total


def calibrate(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The full 9.6 report. ECE and Brier appear only if probabilities were supplied."""
    points = coverage_curve(items)
    probabilities = [i["probability"] for i in items if i.get("probability") is not None]
    outcomes = [bool(i.get("correct")) for i in items if i.get("probability") is not None]
    report: dict[str, Any] = {
        "n": len(items),
        "headline_accuracy": points[-1].accuracy if points else 0.0,
        "coverage_curve": [p.to_dict() for p in points],
        "confidence_signal": "executor agreement, repairs, schema, token cap",
    }
    if len(probabilities) == len(items) and probabilities:
        report["ece"] = expected_calibration_error(probabilities, outcomes)
        report["brier"] = brier_score(probabilities, outcomes)
    else:
        report["ece"] = None
        report["brier"] = None
        report["probability_note"] = (
            "not computed: this system emits a JSON record, not a probability, and "
            f"{len(probabilities)} of {len(items)} items carried one. Manufacturing a "
            "confidence from an ordinal proxy would measure the proxy, not the model.")
    return report


def assert_headline_is_full_coverage(report: dict[str, Any], quoted: float,
                                     *, tolerance: float = 1e-9) -> None:
    """Refuse a report that quotes a selective accuracy as the result (`PLAN.md` 9.6)."""
    headline = report.get("headline_accuracy")
    if headline is None or not math.isclose(quoted, headline, abs_tol=tolerance):
        selective = [p for p in report.get("coverage_curve", [])
                     if math.isclose(quoted, p["accuracy"], abs_tol=tolerance)]
        where = (f" — that is the accuracy at {100 * selective[0]['coverage']:.1f}% "
                 f"coverage ('{selective[0]['threshold']}')" if selective else "")
        raise AssertionError(
            f"quoted accuracy {quoted} is not the full-coverage accuracy "
            f"{headline}{where}. `PLAN.md` 9.6 requires the headline at 100% coverage; "
            f"selective accuracy is a separate diagnostic.")


__all__ = ["CONFIDENCE_LEVELS", "CoveragePoint", "agreement_confidence",
           "assert_headline_is_full_coverage", "brier_score", "calibrate",
           "coverage_curve", "expected_calibration_error"]
