from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord, make_record_id

QUESTION_KINDS = ("human", "machine", "pot")

MIN_IOU = 0.90
MIN_MARGIN = 0.50
CONTEXT_ELEMENTS_KEY = "context_elements"


def xywh_to_norm1000(box: Any, image_w: int, image_h: int) -> list[float]:
    if not isinstance(box, dict):
        raise ValueError(f"expected a keyed box {{x, y, w, h}}, got {type(box).__name__}: {box!r}")
    try:
        x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    except KeyError as exc:
        raise ValueError(f"box is missing {exc.args[0]!r}: {box!r}") from None
    if image_w <= 0 or image_h <= 0:
        raise ValueError(f"image size must be positive, got {image_w}x{image_h}")
    if w < 0 or h < 0:
        raise ValueError(f"negative extent in {box!r}")
    x1 = max(0.0, min(1000.0, 1000.0 * x / image_w))
    y1 = max(0.0, min(1000.0, 1000.0 * y / image_h))
    x2 = max(0.0, min(1000.0, 1000.0 * (x + w) / image_w))
    y2 = max(0.0, min(1000.0, 1000.0 * (y + h) / image_h))
    return [x1, y1, x2, y2]


def boxes_to_norm1000(boxes: Iterable[Any] | None, image_w: int, image_h: int) -> list[list[float]]:
    out = []
    for b in boxes or ():
        norm = xywh_to_norm1000(b, image_w, image_h)
        if norm[2] > norm[0] and norm[3] > norm[1]:
            out.append(norm)
    return out


def row_to_record(row: dict[str, Any], *, split: str, image_path: str | Path,
                  image_sha256: str, image_size: tuple[int, int],
                  source_row_index: int) -> ChartRecord:
    width, height = image_size
    kind = str(row.get("type", "")).lower()
    if kind not in QUESTION_KINDS:
        raise ValueError(f"unexpected type {row.get('type')!r}; expected one of {QUESTION_KINDS}")
    question = str(row["query"])
    answer = row.get("label")
    raw_boxes = row.get("grounding_bboxes")
    if raw_boxes is None:
        raw_boxes = []
    if not isinstance(raw_boxes, list):
        raise ValueError("grounding_bboxes must be a list of keyed boxes")
    boxes = boxes_to_norm1000(raw_boxes, width, height)
    dropped = len(raw_boxes) - len(boxes)

    return ChartRecord(
        schema_version=2,
        source_row_index=source_row_index,
        record_id=make_record_id("refchartqa", split, image_sha256, question, source_row_index),
        source="refchartqa",
        split=split,
        image_path=str(image_path),
        image_sha256=image_sha256,
        question=question,
        answer=None if answer is None else str(answer),
        question_kind=kind,
        table=None,
        boxes=boxes or None,
        plan=None,
        elements=[{"label": None, "value": None, "unit": None, "bbox": b,
                   "grounding_provenance": "refchartqa_gold",
                   "value_provenance": "unknown"}
                  for b in (boxes or [])] or None,
        evidence=list(range(len(boxes))) if boxes else None,
        source_id=None if row.get("id") is None else str(row["id"]),
        meta={
            "refchartqa_id": row.get("id"),
            "image_size": [width, height],
            "n_boxes": len(boxes),
            "boxes_dropped_outside_image": dropped,
            "response": row.get("response"),
        },
    )


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(canonical_json(row) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def cache_training_split(stream: Iterable[dict[str, Any]], *, data_root: Path, image_dir: Path,
                         heldout: frozenset[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from chartqa_dt.data.records import image_content_sha256

    image_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    outcomes: dict[str, int] = {}

    def note(outcome: str) -> None:
        outcomes[outcome] = outcomes.get(outcome, 0) + 1

    for source_row_index, row in enumerate(stream):
        source_id = None if row.get("id") is None else str(row["id"])
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_id or "no-id")[:80]
        raw_path = image_dir / f"{source_row_index:06d}_{safe_id}.png"
        image = None
        try:
            image = row["image"].convert("RGB")
            image.save(raw_path)
            digest = image_content_sha256(image)
            image_size = image.size
        except (KeyError, OSError, ValueError, TypeError):
            raw_path.unlink(missing_ok=True)
            note("missing_image")
            continue
        finally:
            close = getattr(image, "close", None)
            if callable(close):
                close()
        try:
            record = row_to_record(row, split="train", image_path=raw_path.relative_to(data_root),
                                   image_sha256=digest, image_size=image_size,
                                   source_row_index=source_row_index)
        except (KeyError, ValueError, TypeError):
            raw_path.unlink(missing_ok=True)
            note("malformed")
            continue
        if not record.boxes:
            raw_path.unlink(missing_ok=True)
            note("no_usable_box")
            continue
        if record.image_sha256 in heldout:
            raw_path.unlink(missing_ok=True)
            note("held_out_image")
            continue
        records.append(record.to_dict())
        note("accepted")
        if (source_row_index + 1) % 2500 == 0:
            print(f"  {source_row_index + 1:,} rows processed", flush=True)
    return records, outcomes


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def align_record(record: ChartRecord, elements: list[dict[str, Any]]) -> dict[str, Any] | None:
    from chartqa_dt.data.chartqa import canonicalise_chart_element

    boxes = record.boxes or []
    if not boxes:
        return None
    candidates = [e for e in elements if e.get("bbox")]
    if not candidates:
        return None

    chosen: list[int] = []
    matched: list[dict[str, Any]] = []
    for gb in boxes:
        scored = sorted(((iou(gb, e["bbox"]), i, e) for i, e in enumerate(candidates)),
                        key=lambda t: -t[0])
        if not scored:
            return None
        best_iou, index, element = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best_iou < MIN_IOU or (best_iou - runner_up) < MIN_MARGIN:
            return None
        chosen.append(index)
        matched.append(canonicalise_chart_element({
            **element,
            "match_iou": round(best_iou, 4),
            "match_margin": round(best_iou - runner_up, 4),
        }))
    if len(chosen) != len(set(chosen)):
        return None
    marked = set(chosen)
    taken = [element["bbox"] for element in matched]
    context: list[dict[str, Any]] = []
    for index, element in enumerate(candidates):
        if index in marked:
            continue
        canonical = canonicalise_chart_element(element)
        label = canonical.get("label")
        if not isinstance(label, str) or not label.strip() or canonical["bbox"] in taken:
            continue
        taken.append(canonical["bbox"])
        context.append(canonical)
    return {
        "schema_version": 3,
        "source_row_index": record.source_row_index,
        "source_id": record.source_id,
        "record_id": record.record_id,
        ELEMENTS_KEY: matched,
        CONTEXT_ELEMENTS_KEY: context,
    }


def align_training_cache(records: list[ChartRecord], chartqa: list[ChartRecord]
                         ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    elements_by_image: dict[str, list[dict[str, Any]]] = {}
    table_by_image: dict[str, dict[str, Any]] = {}
    for r in chartqa:
        chart_elements = [e for e in (r.elements or []) if isinstance(e, dict)]
        if chart_elements:
            elements_by_image.setdefault(r.image_sha256, chart_elements)
            if r.table:
                table_by_image.setdefault(r.image_sha256, r.table)

    aligned_rows: list[dict[str, Any]] = []
    stats = {"aligned": 0, "unaligned": 0}
    for r in records:
        matched_elements = elements_by_image.get(r.image_sha256)
        aligned = align_record(r, matched_elements) if matched_elements else None
        if aligned is None:
            stats["unaligned"] += 1
            continue
        aligned["table"] = table_by_image.get(r.image_sha256)
        aligned["n_boxes"] = len(r.boxes or [])
        aligned_rows.append(aligned)
        stats["aligned"] += 1
    return aligned_rows, stats
