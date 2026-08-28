"""Qualitative figures — `PLAN.md` 9.7: successes and failures with boxes drawn on charts.

Two constraints shape this module, and neither is about drawing.

**Rule 7: never commit chart images or dataset content.** ChartQA is GPL-3.0 and
RefChartQA is AGPL-3.0, and a chart image with boxes drawn on it is a derivative of the
image. Synthetic charts are this project's own work and may be committed; real ones may
not. `write_figure` enforces that by source rather than trusting the caller to remember,
because "which of these eight figures came from ChartQA" is exactly the question that gets
answered wrongly at 2am before a deadline.

**Boxes are drawn from 0–1000 normalised coordinates**, converted with the image's actual
size. That conversion is the one place a grounding figure can lie: draw the boxes in the
wrong space and the figure shows a model that cannot point, or one that can, regardless of
what it did. The conversion is anisotropic — x by width, y by height — matching the space
the records, the prompt and the official evaluator all use.

Failure modes are named rather than numbered so a reader of the report can tell what they
are looking at: `PLAN.md` 9.7 asks for at least eight figures covering *distinct* modes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Sources whose rendered charts may be committed to the repository (rule 7).
COMMITTABLE_SOURCES = frozenset({"synthetic", "synth"})

#: Colours are chosen to stay distinguishable in greyscale print as well as on screen.
GOLD_COLOUR = (0, 160, 0)
PRED_COLOUR = (220, 30, 30)

#: The distinct failure modes `PLAN.md` 9.7 wants covered.
FAILURE_MODES = {
    "wrong_evidence": "boxes point at the wrong element",
    "wrong_operation": "elements read correctly, wrong operation applied",
    "executor_disagrees": "the tree executes to a different number than the model stated",
    "invalid_record": "the output is not a valid record at all",
    "unanswerable_missed": "answerable marked wrongly",
    "correct": "success",
}


class LicenceRefusal(RuntimeError):
    """Refusing to write a derivative of a dataset image into the repository."""


@dataclass
class Figure:
    record_id: str
    source: str
    mode: str
    path: Path | None = None
    caption: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def to_pixels(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    """A 0–1000 normalised box in image pixels.

    Anisotropic by design: x scales by width and y by height, which is the space the
    records, the prompt and the official evaluator all use. Scaling both by one dimension
    would draw every box in the wrong place on a non-square chart.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    return (round(x1 / 1000 * width), round(y1 / 1000 * height),
            round(x2 / 1000 * width), round(y2 / 1000 * height))


def classify_failure(item: dict[str, Any]) -> str:
    """Which distinct mode this record illustrates. Checked in order of severity."""
    if not item.get("parsed"):
        return "invalid_record"
    if item.get("correct"):
        return "correct"
    if item.get("executes") and not item.get("agrees"):
        return "executor_disagrees"
    if item.get("answerable_wrong"):
        return "unanswerable_missed"
    if item.get("operands_exact") is False:
        return "wrong_evidence"
    return "wrong_operation"


def draw_boxes(image: Any, pred_boxes: Sequence[Sequence[float]],
               gt_boxes: Sequence[Sequence[float]] = (), *, width_px: int = 3) -> Any:
    """A copy of the chart with gold boxes in green and predicted boxes in red."""
    from PIL import ImageDraw

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    for box in gt_boxes:
        draw.rectangle(to_pixels(box, w, h), outline=GOLD_COLOUR, width=width_px)
    for box in pred_boxes:
        draw.rectangle(to_pixels(box, w, h), outline=PRED_COLOUR, width=width_px)
    return canvas


def write_figure(figure: Figure, image: Any, out_dir: Path, *,
                 inside_repo: bool = True) -> Path:
    """Save a rendered figure, refusing to put a dataset derivative in the repository.

    `inside_repo=False` says the destination is outside version control — a scratch
    directory, or an artefact store — and the licence check is then not this module's to
    make.
    """
    if inside_repo and figure.source not in COMMITTABLE_SOURCES:
        raise LicenceRefusal(
            f"{figure.record_id} comes from {figure.source!r}. A chart image with boxes "
            f"drawn on it is a derivative of that image, and rule 7 forbids committing "
            f"dataset content (ChartQA is GPL-3.0, RefChartQA is AGPL-3.0). Render it "
            f"outside the repository, or use a synthetic example to make the same point.")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{figure.mode}__{figure.record_id}.png"
    image.save(path)
    figure.path = path
    return path


def select_figures(items: Sequence[dict[str, Any]], *, per_mode: int = 2,
                   committable_only: bool = True) -> list[Figure]:
    """At least one example of each distinct mode, `PLAN.md` 9.7's acceptance criterion.

    Preferring committable sources is not a cosmetic choice: a set of eight figures that
    cannot go in the repository is not a set of eight figures.
    """
    chosen: dict[str, list[Figure]] = {}
    ordered = sorted(items, key=lambda i: str(i.get("source")) not in COMMITTABLE_SOURCES) \
        if committable_only else list(items)
    for item in ordered:
        mode = classify_failure(item)
        bucket = chosen.setdefault(mode, [])
        if len(bucket) >= per_mode:
            continue
        bucket.append(Figure(record_id=str(item.get("id", "?")),
                             source=str(item.get("source", "unknown")), mode=mode,
                             caption=FAILURE_MODES.get(mode, mode),
                             meta={"gold": item.get("gold"),
                                   "prediction": item.get("prediction")}))
    return [f for mode in FAILURE_MODES for f in chosen.get(mode, [])]


def coverage_report(figures: Sequence[Figure]) -> dict[str, Any]:
    """Whether 9.7's criterion is met: at least eight figures over distinct modes."""
    modes = sorted({f.mode for f in figures})
    return {"n": len(figures), "modes": modes, "n_modes": len(modes),
            "meets_criterion": len(figures) >= 8 and len(modes) >= 4,
            "missing_modes": [m for m in FAILURE_MODES if m not in modes]}


__all__ = ["COMMITTABLE_SOURCES", "FAILURE_MODES", "Figure", "LicenceRefusal",
           "classify_failure", "coverage_report", "draw_boxes", "select_figures",
           "to_pixels", "write_figure"]
