"""Robustness perturbations — `PLAN.md` 9.5.

Two kinds, and the difference between them is the whole point.

**Appearance perturbations** — blur, noise, re-encoding — change the pixels and *not* the
data. The gold answer is unchanged, so a drop measures the encoder's fragility.

**Value counterfactuals** change the data. The plan says it plainly: their answers must be
*"recomputed from the source data rather than assumed unchanged"*. Assuming otherwise would
score the model against the *old* answer, so a model that read the new chart correctly would
be marked wrong and a model ignoring the image entirely would be marked right — the metric
would reward exactly the failure the counterfactual exists to detect.

So `counterfactual` re-executes the original plan over the modified series and refuses to
return a pair whose answer did not move. A counterfactual that changes nothing is not a
harder example; it is the original example under a different name, and averaging it in
dilutes the result.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from chartqa_dt.plans.executor import EvidenceItem, execute
from chartqa_dt.plans.roundtrip import answers_agree

#: Appearance perturbations, and what each is meant to probe.
APPEARANCE = {
    "blur": "defocus — does grounding survive soft edges?",
    "noise": "sensor noise — does the encoder key on exact pixel values?",
    "jpeg": "re-encoding artefacts, the most common real-world degradation",
    "greyscale": "colour removal — is the model reading colour or position?",
}
#: How much a counterfactual must move a numeric answer to be worth scoring. Below the
#: relaxed tolerance the metric cannot tell the two answers apart anyway.
MIN_ANSWER_SHIFT = 0.10


class VacuousCounterfactual(ValueError):
    """The perturbed data produced the same answer, so the pair tests nothing."""


@dataclass
class Perturbed:
    kind: str
    image: Any
    answer: str
    series: list[tuple[str, float]] | None = None
    note: str = ""


def perturb_image(image: Any, kind: str, *, severity: float = 1.0, seed: int = 0) -> Any:
    """One appearance perturbation. The data — and so the answer — is untouched."""
    from PIL import Image, ImageFilter

    if kind not in APPEARANCE:
        raise ValueError(f"unknown perturbation {kind!r}; expected {sorted(APPEARANCE)}")
    img = image.convert("RGB")
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=1.5 * severity))
    if kind == "greyscale":
        return img.convert("L").convert("RGB")
    if kind == "jpeg":
        import io

        quality = max(5, int(60 - 40 * severity))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    rng = random.Random(seed)
    pixels = img.load()
    width, height = img.size
    amplitude = int(24 * severity)
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            pixels[x, y] = (
                min(255, max(0, r + rng.randint(-amplitude, amplitude))),
                min(255, max(0, g + rng.randint(-amplitude, amplitude))),
                min(255, max(0, b + rng.randint(-amplitude, amplitude))),
            )
    return img


def recompute_answer(plan: dict, series: list[tuple[str, float]],
                     unit: str | None = None) -> str:
    """The answer this plan gives on this data. The only way a counterfactual is honest."""
    evidence = [EvidenceItem(label, value, unit) for label, value in series]
    got = execute(plan, evidence)
    if isinstance(got, float):
        return f"{got:.2f}".rstrip("0").rstrip(".")
    return str(got)


def counterfactual(series: list[tuple[str, float]], plan: dict, original_answer: str, *,
                   label: str | None = None, factor: float = 1.6,
                   unit: str | None = None, seed: int = 0) -> Perturbed:
    """Change one value, recompute the answer, and refuse a pair that tests nothing.

    `label` picks the bar to change; by default one the plan actually reads, because
    perturbing a value the plan never touches cannot change the answer and would always
    raise `VacuousCounterfactual`.
    """
    if len(series) < 2:
        raise VacuousCounterfactual("a series of fewer than two points cannot be varied")
    by_label = dict(series)
    if label is None:
        # Prefer a label the plan actually reads: perturbing one it never touches cannot
        # change the answer, so it would always be refused as vacuous.
        from chartqa_dt.train.targets import plan_labels

        candidates = [lab for lab in plan_labels(plan) if lab in by_label]
        pool = candidates or [lab for lab, _ in series]
        label = pool[random.Random(seed).randrange(len(pool))]
    if label not in by_label:
        raise VacuousCounterfactual(f"{label!r} is not in the series")

    changed = [(lab, value * factor if lab == label else value) for lab, value in series]
    new_answer = recompute_answer(plan, changed, unit)

    if answers_agree(original_answer, new_answer):
        raise VacuousCounterfactual(
            f"changing {label!r} by {factor}x left the answer at {new_answer!r}, which the "
            f"relaxed metric cannot distinguish from {original_answer!r}. Scoring this pair "
            f"would add a duplicate of the original, not a harder example.")
    return Perturbed(kind=f"counterfactual:{label}x{factor}", image=None,
                     answer=new_answer, series=changed,
                     note=f"{original_answer} -> {new_answer}")


def counterfactual_or_none(series, plan, original_answer, **kw) -> Perturbed | None:
    """`counterfactual`, returning None instead of raising, for batch generation."""
    try:
        return counterfactual(series, plan, original_answer, **kw)
    except Exception:      # noqa: BLE001 — vacuous pairs and unexecutable plans alike
        return None


__all__ = ["APPEARANCE", "MIN_ANSWER_SHIFT", "Perturbed", "VacuousCounterfactual",
           "counterfactual", "counterfactual_or_none", "perturb_image", "recompute_answer"]
