"""Emit predictions in the official RefChartQA format, and score them with its evaluator.

`PLAN.md` 5.4 requires the **released evaluator**, and `DECISIONS.md` 0003 makes it the
scorer of record. Our own metrics agree with it to within 0.07 percentage points on 11,690
real predictions (`DECISIONS.md` 0053), but "agrees with" is not "is": a reported number
must come from the official code, with our implementation used for the stratified analysis
and confidence intervals it cannot produce.

The official evaluator reads one string per item:

    <box>x1,y1,x2,y2</box><box>...</box><grounding-sep>ANSWER

with **integer** coordinates in 0–999 (`DECISIONS.md` 0004 — its own
`ensure_xyxy_bbox_within_bounds` clamps to `bins - 1`, so 1000 is out of range and a box
emitted at 1000 is silently dropped by `extract_bounding_boxes`).

Two details in its parser are easy to get wrong and both are handled here:

* it splits on `<grounding-sep>` and requires **exactly two** parts, so an answer
  containing the separator, or a missing separator, scores zero;
* `extract_bounding_boxes` discards any box whose coordinates fall outside
  `[0, bins - 1]`, silently — so clamping must happen before emission, not after.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from chartqa_dt.vision.coords import clamp_for_official_evaluator

GROUNDING_SEPARATOR = "<grounding-sep>"
BOX_START, BOX_END = "<box>", "</box>"


def format_box(box: Sequence[float]) -> str:
    """One box as the official evaluator expects it: integers, clamped to 0–999."""
    x1, y1, x2, y2 = clamp_for_official_evaluator(tuple(box))
    return f"{BOX_START}{x1},{y1},{x2},{y2}{BOX_END}"


def format_prediction(boxes: Iterable[Sequence[float]], answer: str) -> str:
    """The full `model_answer` string for one item.

    The answer is stripped of any separator token. A model that emitted one would split
    the string into three parts and the official evaluator would score the item zero for
    a reason that has nothing to do with its answer.
    """
    text = str(answer).replace(GROUNDING_SEPARATOR, " ").strip()
    return "".join(format_box(b) for b in boxes) + GROUNDING_SEPARATOR + text


def build_rows(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows shaped for `evaluate.analyse_dataset`.

    Each item needs `pred_boxes` (normalised 0–1000), `answer`, `label`, `image_size` and
    the dataset's own `grounding_bboxes` in absolute-pixel `{x, y, w, h}` — the official
    evaluator quantises those itself, so they are passed through untouched rather than
    converted and back.
    """
    rows = []
    for item in items:
        width, height = item["image_size"]
        rows.append({
            "model_answer": format_prediction(item.get("pred_boxes") or [],
                                              item.get("answer", "")),
            "label": item["label"],
            "width": width,
            "height": height,
            "grounding_bboxes": item["grounding_bboxes"],
            "type": item.get("question_kind", ""),
        })
    return rows


def score_with_official(rows: Sequence[dict[str, Any]], *, bins: int = 1000,
                        vendor_dir: str = "verification/refchartqa_eval") -> dict[str, float]:
    """Run the vendored `evaluate.py` verbatim. This produces the reported number."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(vendor_dir) / "evaluate.py"
    spec = importlib.util.spec_from_file_location("official_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["official_evaluate"] = module
    spec.loader.exec_module(module)
    return {k: float(v) for k, v in module.analyse_dataset(list(rows), bins).items()}


__all__ = ["BOX_END", "BOX_START", "GROUNDING_SEPARATOR", "build_rows", "format_box",
           "format_prediction", "score_with_official"]
