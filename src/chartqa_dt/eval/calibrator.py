"""The reliability calibrator — `PLAN.md` 8.2, and the probability 9.6 needs.

8.2 asks for *"a small logistic model fitted on **validation** correctness using measurable
features"*, and states the constraint that gives the whole thing its point: **it must not
use the model's self-declared confidence.** A model that says it is sure is making the same
kind of claim as the answer it is unsure about; a calibrator built on that measures the
model's self-image. Every feature here is something *observed about the output* — whether it
parsed, whether the executor agreed, how large the evidence boxes are — or a token
log-probability, which is a property of the decoding rather than a statement by the model.
`FORBIDDEN_FEATURES` refuses the ones that are self-report.

This also supplies what `eval/calibration.py` declines to invent. That module reports ECE
and Brier as *not computed* because the system emits a record and not a probability. A
fitted calibrator emits a probability, so those numbers become well-defined — computed on a
model whose inputs are auditable, rather than on a proxy.

Implemented with plain NumPy: logistic regression with L2, full-batch gradient descent, no
scikit-learn. The model has a handful of features and a few hundred training rows, the fit
is deterministic, and adding a dependency to a Kaggle image for thirty lines of arithmetic
is a worse trade than writing them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: Features that are the model talking about itself. `PLAN.md` 8.2 forbids them.
FORBIDDEN_FEATURES = frozenset({
    "confidence", "self_confidence", "model_confidence", "certainty",
    "self_reported_confidence", "stated_confidence", "sureness",
})

#: The features 8.2 names. Everything here is observed about the output or the decoding.
DEFAULT_FEATURES = ("mean_logprob", "min_logprob", "schema_valid", "executor_agrees",
                    "evidence_area", "unit_ok", "range_ok", "n_evidence")


class SelfReportedFeature(ValueError):
    """A feature that is the model's own claim about itself."""


def assert_features_are_observable(names: Sequence[str]) -> None:
    """`PLAN.md` 8.2: the calibrator may not use the model's self-declared confidence."""
    offenders = [n for n in names if n.lower() in FORBIDDEN_FEATURES]
    if offenders:
        raise SelfReportedFeature(
            f"{offenders} are the model's own claim about itself. `PLAN.md` 8.2 requires "
            f"the calibrator to use measurable features only — a calibrator fitted on "
            f"self-reported confidence measures the model's self-image, not its accuracy.")


@dataclass
class Calibrator:
    feature_names: list[str]
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    n_train: int = 0
    fitted_on: str = ""

    def predict(self, rows: Sequence[dict[str, Any]]) -> list[float]:
        import numpy as np

        if not self.weights:
            raise RuntimeError("calibrator is not fitted")
        x = _matrix(rows, self.feature_names)
        z = (x - np.array(self.mean)) / np.array(self.scale)
        # Plain floats: a numpy scalar survives arithmetic but not `json.dumps`, and this
        # goes into a results file.
        return [float(v) for v in 1.0 / (1.0 + np.exp(-(z @ np.array(self.weights)
                                                        + self.bias)))]

    def to_dict(self) -> dict[str, Any]:
        return {"feature_names": self.feature_names, "weights": self.weights,
                "bias": self.bias, "mean": self.mean, "scale": self.scale,
                "n_train": self.n_train, "fitted_on": self.fitted_on}


def _matrix(rows: Sequence[dict[str, Any]], names: Sequence[str]):
    import numpy as np

    return np.array([[float(row.get(n, 0.0) or 0.0) for n in names] for row in rows],
                    dtype=float)


def fit(rows: Sequence[dict[str, Any]], labels: Sequence[bool], *,
        features: Sequence[str] = DEFAULT_FEATURES, l2: float = 1.0,
        steps: int = 2000, lr: float = 0.1, split: str = "validation") -> Calibrator:
    """Fit on **validation** correctness. Fitting on test would be fitting to the answer.

    Features are standardised before the fit: a token log-probability lives near -0.5 and
    an evidence area near 40,000, and gradient descent on raw columns of that ratio spends
    its whole budget on the large one.
    """
    import numpy as np

    assert_features_are_observable(features)
    if split != "validation":
        raise ValueError(
            f"the calibrator is fitted on validation correctness (`PLAN.md` 8.2), not on "
            f"{split!r}. Fitting on test would be fitting to the answer.")
    if len(rows) != len(labels):
        raise ValueError("rows and labels must be the same length")
    if not rows:
        raise ValueError("nothing to fit")

    x = _matrix(rows, features)
    y = np.array([float(bool(v)) for v in labels])
    mean = x.mean(axis=0)
    # A constant column has zero spread; dividing by it would produce NaN weights and a
    # calibrator that silently predicts nothing.
    scale = np.where(x.std(axis=0) < 1e-9, 1.0, x.std(axis=0))
    z = (x - mean) / scale

    w = np.zeros(z.shape[1])
    b = 0.0
    n = len(y)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(z @ w + b)))
        error = p - y
        w -= lr * ((z.T @ error) / n + l2 * w / n)
        b -= lr * error.mean()

    return Calibrator(feature_names=list(features), weights=[float(v) for v in w],
                      bias=float(b), mean=[float(v) for v in mean],
                      scale=[float(v) for v in scale], n_train=n, fitted_on=split)


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve — *can this separate correct from incorrect at all?*

    `PLAN.md` 8.2's skip trigger depends on this: if the calibrator cannot separate them on
    validation, the crop is skipped and the reason reported. Ties count a half, which is
    what keeps a constant predictor at exactly 0.5 rather than at 0 or 1 by accident.
    """
    positives = [s for s, y in zip(scores, labels) if y]
    negatives = [s for s, y in zip(scores, labels) if not y]
    if not positives or not negatives:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def evaluate(calibrator: Calibrator, rows: Sequence[dict[str, Any]],
             labels: Sequence[bool]) -> dict[str, Any]:
    """Separability and calibration of a fitted calibrator, and 8.2's skip decision."""
    from chartqa_dt.eval.calibration import brier_score, expected_calibration_error

    scores = calibrator.predict(rows)
    outcomes = [bool(v) for v in labels]
    area = float(auc(scores, outcomes))
    ece = float(expected_calibration_error(scores, outcomes))
    separates = bool(area == area and area >= 0.60)          # NaN-safe
    return {
        "n": len(rows), "auc": area, "ece": ece,
        "brier": float(brier_score(scores, outcomes)),
        "separates": separates,
        "skip_crop": not separates,
        "skip_reason": "" if separates else (
            f"the calibrator reaches AUC {area:.3f} on validation, which does not separate "
            f"correct from incorrect. `PLAN.md` 8.2 says to skip the crop and report why "
            f"rather than gate it on a signal that carries no information."),
    }


__all__ = ["DEFAULT_FEATURES", "FORBIDDEN_FEATURES", "Calibrator", "SelfReportedFeature",
           "assert_features_are_observable", "auc", "evaluate", "fit"]
