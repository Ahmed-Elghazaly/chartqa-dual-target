"""RefChartQA → `ChartRecord`, with the box conversion that everything depends on.

`PLAN.md` 3.2. RefChartQA supplies `grounding_bboxes` as **absolute pixels** in
``{x, y, w, h}`` form (`phase0.md` F7). Two conversions therefore stand between the
dataset and anything we train on or score:

    {x, y, w, h} absolute  ->  [x1, y1, x2, y2] absolute  ->  [x1, y1, x2, y2] / 1000

Getting either wrong is the kind of error that produces a plausible-looking number and
poisons the entire grounding half of the project — a width read as x2 shrinks every box
towards the origin and would simply score badly, with nothing pointing at the cause. The
conversion is one function, used everywhere, and tested against hand-computed values.

The official evaluator's own format is a separate concern handled at emit time
(`clamp_for_official_evaluator`, `DECISIONS.md` 0004); this module stays in the
project's internal 0–1000 float convention.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from chartqa_dt.data.records import ChartRecord, image_content_sha256, make_record_id

#: `PLAN.md` uses "val"; the HF split is "validation". One mapping, applied on entry.
SPLIT_ALIASES = {"validation": "val", "valid": "val", "dev": "val"}

QUESTION_KINDS = ("human", "machine", "pot")


class RefChartQAError(ValueError):
    """A row could not be converted without guessing."""


def xywh_to_norm1000(box: Any, image_w: int, image_h: int) -> list[float]:
    """One absolute-pixel ``{x, y, w, h}`` box to normalised ``[x1, y1, x2, y2]``.

    Only the keyed form is accepted. A bare 4-sequence is refused on purpose: this
    project's own boxes are ``[x1, y1, x2, y2]`` corner form, so silently reading one as
    ``x, y, w, h`` would shrink every box toward the origin and produce a plausible bad
    grounding score with nothing pointing at the cause. The dataset ships structs
    (verified against the live rows endpoint), so there is nothing to lose by refusing.
    """
    if not isinstance(box, dict):
        raise RefChartQAError(
            f"expected a keyed box {{x, y, w, h}}, got {type(box).__name__}: {box!r}. "
            f"A 4-sequence is ambiguous here — this project's own boxes are corner form."
        )
    try:
        x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    except KeyError as exc:
        raise RefChartQAError(f"box is missing {exc.args[0]!r}: {box!r}") from None

    if image_w <= 0 or image_h <= 0:
        raise RefChartQAError(f"image size must be positive, got {image_w}x{image_h}")
    if w < 0 or h < 0:
        raise RefChartQAError(f"negative extent in {box!r}")

    x1 = max(0.0, min(1000.0, 1000.0 * x / image_w))
    y1 = max(0.0, min(1000.0, 1000.0 * y / image_h))
    x2 = max(0.0, min(1000.0, 1000.0 * (x + w) / image_w))
    y2 = max(0.0, min(1000.0, 1000.0 * (y + h) / image_h))
    return [x1, y1, x2, y2]


def boxes_to_norm1000(boxes: Iterable[Any], image_w: int, image_h: int) -> list[list[float]]:
    """Convert every box on a row, dropping ones that clamp away to nothing.

    A box entirely outside the image collapses to zero area under clamping. Keeping it
    would add a guaranteed false positive to every grounding score, so it is dropped and
    the count is reported by the caller rather than silently absorbed.
    """
    out = []
    for b in boxes or ():
        norm = xywh_to_norm1000(b, image_w, image_h)
        if norm[2] > norm[0] and norm[3] > norm[1]:
            out.append(norm)
    return out


def normalise_split(split: str) -> str:
    return SPLIT_ALIASES.get(split, split)


def row_to_record(row: dict[str, Any], *, split: str, image_path: str | Path,
                  image_sha256: str, image_size: tuple[int, int]) -> ChartRecord:
    """One RefChartQA row as a `ChartRecord`.

    `response` is the model-style rationale and `label` is the gold answer; the answer
    is what we train and score against, so `label` is the one that becomes `answer`.
    """
    width, height = image_size
    kind = str(row.get("type", "")).lower()
    if kind not in QUESTION_KINDS:
        raise RefChartQAError(f"unexpected type {row.get('type')!r}; "
                              f"expected one of {QUESTION_KINDS}")
    question = str(row["query"])
    answer = row.get("label")
    boxes = boxes_to_norm1000(row.get("grounding_bboxes"), width, height)
    dropped = len(row.get("grounding_bboxes") or ()) - len(boxes)

    return ChartRecord(
        record_id=make_record_id("refchartqa", normalise_split(split), image_sha256, question),
        source="refchartqa",
        split=normalise_split(split),
        image_path=str(image_path),
        image_sha256=image_sha256,
        question=question,
        answer=None if answer is None else str(answer),
        question_kind=kind,
        table=None,
        boxes=boxes or None,
        plan=None,
        # RefChartQA marks, per question, which regions a person used to answer it, so
        # every box it gives is evidence for *this* question. Identity comes later, from
        # `align_refchartqa.py`; until then an element is a box with no label, which is
        # still enough to say *what* the evidence is (`DECISIONS.md` 0124).
        elements=[{"label": None, "value": None, "unit": None, "bbox": b}
                  for b in (boxes or [])] or None,
        evidence=list(range(len(boxes))) if boxes else None,
        meta={
            "refchartqa_id": row.get("id"),
            "image_size": [width, height],
            "n_boxes": len(boxes),
            "boxes_dropped_outside_image": dropped,
            "response": row.get("response"),
        },
    )


def iter_records(rows: Iterable[dict[str, Any]], *, split: str,
                 image_dir: str | Path,
                 image_name: str = "_image_file") -> Iterator[ChartRecord]:
    """Convert rows that already have their images on disk (the dev-subset layout)."""
    from PIL import Image

    image_dir = Path(image_dir)
    for row in rows:
        path = image_dir / row[image_name]
        with Image.open(path) as im:
            size = im.size
        digest = image_content_sha256(path)
        yield row_to_record(row, split=row.get("_split", split), image_path=path,
                            image_sha256=digest, image_size=size)


__all__ = ["QUESTION_KINDS", "SPLIT_ALIASES", "RefChartQAError", "boxes_to_norm1000",
           "iter_records", "normalise_split", "row_to_record", "xywh_to_norm1000"]
