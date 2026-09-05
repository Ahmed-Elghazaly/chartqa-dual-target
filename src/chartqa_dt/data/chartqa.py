"""ChartQA → `ChartRecord`, including the gold tables and element annotations.

`PLAN.md` 3.2. The archive layout, established by range-reading the zip's central
directory before downloading anything (`data/remote_zip.py`):

    ChartQA Dataset/{train,val,test}/
        png/          {imgname}.png
        tables/       {stem}.csv          gold data table
        annotations/  {stem}.json         chart type, axis labels, ELEMENT BOXES
        {split}_human.json                [{imgname, query, label}, ...]
        {split}_augmented.json            the machine-generated questions

`train` holds 18,317 images against 28,299 QA rows: several questions per chart, split
across the human and augmented files. `question_kind` follows which file a row came from
— the upstream parquet calls the augmented set ``machine`` (`phase0.md` F7), and that
name is kept so the two sources describe the same rows the same way.

The annotations are the reason this loader reads more than the parquet does. They carry
per-datapoint bounding boxes in absolute-pixel ``{x, y, w, h}``, the same form as
RefChartQA, aligned index-for-index with the series values.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from chartqa_dt.data.records import (
    ELEMENTS_KEY,
    ChartRecord,
    image_content_sha256,
    make_record_id,
)
from chartqa_dt.data.refchartqa import xywh_to_norm1000

ROOT = "ChartQA Dataset"
SPLITS = ("train", "val", "test")

#: Which file a question came from, and what `question_kind` that makes it.
QA_FILES = {"human": "{split}_human.json", "machine": "{split}_augmented.json"}


class ChartQAError(ValueError):
    """A ChartQA row or annotation could not be read without guessing."""


def split_dir(split: str) -> str:
    if split not in SPLITS:
        raise ChartQAError(f"unknown split {split!r}; expected one of {SPLITS}")
    return f"{ROOT}/{split}"


def qa_path(split: str, kind: str) -> str:
    try:
        return f"{split_dir(split)}/{QA_FILES[kind].format(split=split)}"
    except KeyError:
        raise ChartQAError(f"unknown question kind {kind!r}; "
                           f"expected one of {sorted(QA_FILES)}") from None


def image_path(split: str, imgname: str) -> str:
    return f"{split_dir(split)}/png/{imgname}"


def table_path(split: str, imgname: str) -> str:
    return f"{split_dir(split)}/tables/{Path(imgname).stem}.csv"


def annotation_path(split: str, imgname: str) -> str:
    return f"{split_dir(split)}/annotations/{Path(imgname).stem}.json"


def parse_table(text: str) -> dict[str, Any]:
    """A gold CSV as ``{"columns": [...], "rows": [[...], ...]}``.

    Values stay as written. Coercing them here would quietly change what the gold answer
    is compared against, and the executor already handles numeric parsing where it
    matters.
    """
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        raise ChartQAError("empty table")
    return {"columns": rows[0], "rows": rows[1:]}


#: How a chart type stores its element boxes. Measured over 4,000 random train charts:
#: v_bar and h_bar pair one box per datapoint (97.5% / 98.6% of series), line stores
#: SEGMENTS between consecutive points (85.6% of series have len(bboxes) == len(y) - 1),
#: and pie uses a per-wedge layout with its own keys.
ELEMENT_LAYOUTS = {"v_bar": "series", "h_bar": "series", "line": "segments", "pie": "wedges"}


def _one_colour(raw: Any) -> str | None:
    """A single colour value, normalised, or None when the annotation does not know one.

    Some annotations carry the literal string ``'unk'``; it is not a colour and is dropped
    rather than passed on as if it were.
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None if text and text != "unk" else None


def _element_colours(model: dict[str, Any], count: int) -> list[str | None]:
    """One colour per datapoint, from whichever of the two shapes this annotation uses.

    ChartQA writes the colour two different ways and **21.8% of human-written questions
    mention one** (`DECISIONS.md` 0087), so both have to be read:

    * ``colors`` — a list, one entry per datapoint, on ``v_bar``. Per-datapoint because on
      many charts colour distinguishes categories *within* a series rather than the series
      itself, which is exactly the chart *"the blue bar"* is asked about.
    * ``color`` — singular, on ``line`` / ``h_bar`` / ``pie``, and often already an English
      name such as ``'dark blue'``.

    A shorter ``colors`` list than there are boxes is padded with ``None`` rather than
    zipped short: a wrong colour points at the wrong mark, and a missing one only declines
    to answer.
    """
    listed = model.get("colors")
    if isinstance(listed, str):
        listed = [listed]
    if isinstance(listed, list) and listed:
        values = [_one_colour(c) for c in listed[:count]]
        return values + [None] * (count - len(values))
    single = _one_colour(model.get("color"))
    return [single] * count


def _series_elements(model: dict[str, Any], image_w: int, image_h: int
                     ) -> list[dict[str, Any]]:
    """Bars: `bboxes`, `x` and `y` are parallel arrays, one entry per datapoint.

    A model whose lengths disagree is skipped rather than zipped short — a silent
    misalignment would attach boxes to the wrong values, and nothing downstream could
    detect it.
    """
    boxes = model.get("bboxes") or []
    xs, ys = model.get("x") or [], model.get("y") or []
    if len(boxes) != len(ys) or (xs and len(xs) != len(boxes)):
        return []
    colours = _element_colours(model, len(boxes))
    out = []
    for i, box in enumerate(boxes):
        norm = _norm_or_none(box, image_w, image_h)
        if norm is None:
            continue
        out.append({"series": model.get("name"), "label": str(xs[i]) if xs else None,
                    "value": ys[i], "bbox": norm, "kind": "datapoint",
                    "colour": colours[i]})
    return out


def _wedge_element(model: dict[str, Any], image_w: int, image_h: int
                   ) -> dict[str, Any] | None:
    """Pie: one model per wedge, with its own keys and three observed shapes.

    Of 538 wedge models sampled, 268 carry ``bboxes``, 251 carry a possibly-null
    ``bbox``, and 19 carry neither a label nor a value. Only wedges with a label, a
    value and a usable box are returned; the rest are counted as uncovered rather than
    filled in with a guess.
    """
    label, value = model.get("text_label"), model.get("value")
    if label is None or value is None:
        return None
    raw = model.get("bbox")
    if raw is None:
        candidates = model.get("bboxes") or []
        raw = candidates[0] if candidates else None
    norm = _norm_or_none(raw, image_w, image_h) if raw is not None else None
    if norm is None:
        return None
    return {"series": "pie", "label": str(label), "value": value, "bbox": norm,
            "kind": "wedge", "colour": _one_colour(model.get("color"))}


def _norm_or_none(box: Any, image_w: int, image_h: int) -> list[float] | None:
    try:
        norm = xywh_to_norm1000(box, image_w, image_h)
    except ValueError:
        return None
    return norm if norm[2] > norm[0] and norm[3] > norm[1] else None


def annotation_boxes(annotation: dict[str, Any], image_w: int, image_h: int
                     ) -> list[dict[str, Any]]:
    """Every labelled element of a chart, with its box normalised to 0-1000.

    Bars and pie wedges only. **Line charts are deliberately excluded**: their `bboxes`
    are the segments *between* consecutive points, so a point's position is recoverable
    but a point's box size is not — the annotation never states a marker size. Inventing
    one would put a fabricated box into training data, which is precisely the failure the
    RefChartQA audit gate exists to catch. Lines are 12.9% of ChartQA against 83.9% for
    bars, so the coverage lost is small and the alternative is unverifiable.
    """
    layout = ELEMENT_LAYOUTS.get(str(annotation.get("type")))
    if layout == "wedges":
        wedges = (_wedge_element(m, image_w, image_h) for m in annotation.get("models") or ())
        return [w for w in wedges if w is not None]
    if layout != "series":
        return []
    out: list[dict[str, Any]] = []
    for model in annotation.get("models") or ():
        out.extend(_series_elements(model, image_w, image_h))
    return out


def axis_labels(annotation: dict[str, Any], image_w: int, image_h: int
                ) -> dict[str, list[dict[str, Any]]]:
    """Axis tick labels with their boxes — the text a reader actually points at."""
    info = annotation.get("general_figure_info") or {}
    out: dict[str, list[dict[str, Any]]] = {}
    for axis in ("x_axis", "y_axis"):
        major = (info.get(axis) or {}).get("major_labels") or {}
        boxes, values = major.get("bboxes") or [], major.get("values") or []
        if len(boxes) != len(values):
            continue
        items = []
        for box, value in zip(boxes, values):
            try:
                items.append({"text": str(value),
                              "bbox": xywh_to_norm1000(box, image_w, image_h)})
            except ValueError:
                continue
        out[axis] = items
    return out


def row_to_record(row: dict[str, Any], *, split: str, kind: str,
                  image_path_on_disk: str | Path, image_sha256: str,
                  image_size: tuple[int, int], table: dict[str, Any] | None = None,
                  annotation: dict[str, Any] | None = None) -> ChartRecord:
    """One ChartQA QA row as a `ChartRecord`, with whatever gold data is available."""
    question = str(row["query"])
    width, height = image_size
    meta: dict[str, Any] = {"imgname": row.get("imgname"),
                            "image_size": [width, height]}
    boxes = None
    if annotation is not None:
        elements = annotation_boxes(annotation, width, height)
        meta["chart_type"] = annotation.get("type")
        meta[ELEMENTS_KEY] = elements
        meta["axis_labels"] = axis_labels(annotation, width, height)
        boxes = [e["bbox"] for e in elements] or None

    return ChartRecord(
        record_id=make_record_id("chartqa", split, image_sha256, question),
        source="chartqa",
        split=split,
        image_path=str(image_path_on_disk),
        image_sha256=image_sha256,
        question=question,
        answer=None if row.get("label") is None else str(row["label"]),
        question_kind=kind,
        table=table,
        boxes=boxes,
        plan=None,
        elements=meta.get(ELEMENTS_KEY),
        # `None`: ChartQA annotates the chart, not the question, so nothing here knows
        # which marks answer this one. Only a plan can select (`DECISIONS.md` 0124).
        evidence=None,
        meta=meta,
    )


class ArchiveReader:
    """Reads ChartQA members straight out of the zip, without extracting it.

    Extraction would double 875 MB of disk for no benefit — every consumer wants
    individual members, and `zipfile` seeks to them directly. On a machine with a few
    gigabytes free that difference decides whether Phase 3 runs at all.
    """

    def __init__(self, archive: str | Path) -> None:
        import zipfile

        self.path = Path(archive)
        self._zip = zipfile.ZipFile(self.path)
        self._names = set(self._zip.namelist())

    def __enter__(self) -> ArchiveReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._zip.close()

    def exists(self, name: str) -> bool:
        return name in self._names

    def read(self, name: str) -> bytes:
        return self._zip.read(name)

    def read_text(self, name: str) -> str:
        return self.read(name).decode("utf-8", "replace")

    def read_json(self, name: str) -> Any:
        import json

        return json.loads(self.read(name))

    def image_size(self, name: str) -> tuple[int, int]:
        import io as _io

        from PIL import Image

        with Image.open(_io.BytesIO(self.read(name))) as im:
            return im.size

    def qa_rows(self, split: str, kind: str) -> list[dict[str, Any]]:
        return self.read_json(qa_path(split, kind))


def iter_records_from_archive(reader: ArchiveReader, *, split: str, kind: str,
                              limit: int | None = None,
                              with_annotations: bool = True) -> Iterator[ChartRecord]:
    """Convert QA rows by reading members out of the zip.

    The image hash is over the DECODED PIXELS, not the stored bytes — that is what lets
    `dedup_key` recognise the same chart when RefChartQA ships a re-encoded copy of it
    (`DECISIONS.md` 0048).
    """
    rows = reader.qa_rows(split, kind)
    for row in rows[:limit] if limit else rows:
        img_name = image_path(split, row["imgname"])
        if not reader.exists(img_name):
            continue
        raw = reader.read(img_name)
        size = reader.image_size(img_name)
        digest = image_content_sha256(raw)
        table = annotation = None
        tbl = table_path(split, row["imgname"])
        if reader.exists(tbl):
            table = parse_table(reader.read_text(tbl))
        if with_annotations:
            ann = annotation_path(split, row["imgname"])
            if reader.exists(ann):
                annotation = reader.read_json(ann)
        yield row_to_record(row, split=split, kind=kind, image_path_on_disk=img_name,
                            image_sha256=digest, image_size=size, table=table,
                            annotation=annotation)


def iter_records(rows: Iterable[dict[str, Any]], *, split: str, kind: str,
                 root: str | Path, with_annotations: bool = True) -> Iterator[ChartRecord]:
    """Convert QA rows against an extracted archive on disk."""
    import json

    from PIL import Image

    root = Path(root)
    for row in rows:
        img = root / image_path(split, row["imgname"])
        with Image.open(img) as im:
            size = im.size
        digest = image_content_sha256(img)
        table = annotation = None
        tbl = root / table_path(split, row["imgname"])
        if tbl.exists():
            table = parse_table(tbl.read_text(encoding="utf-8", errors="replace"))
        if with_annotations:
            ann = root / annotation_path(split, row["imgname"])
            if ann.exists():
                annotation = json.loads(ann.read_text(encoding="utf-8"))
        yield row_to_record(row, split=split, kind=kind, image_path_on_disk=img,
                            image_sha256=digest, image_size=size, table=table,
                            annotation=annotation)


__all__ = [
    "QA_FILES",
    "ROOT",
    "SPLITS",
    "ArchiveReader",
    "ChartQAError",
    "annotation_boxes",
    "annotation_path",
    "axis_labels",
    "image_path",
    "iter_records",
    "iter_records_from_archive",
    "parse_table",
    "qa_path",
    "row_to_record",
    "split_dir",
    "table_path",
]
