from __future__ import annotations

import csv
import io
import json
import random
import zipfile
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord, image_content_sha256, make_record_id
from chartqa_dt.data.refchartqa import xywh_to_norm1000
from chartqa_dt.plans.executor import parse_numeric

ROOT = "ChartQA Dataset"

QA_FILES = {"human": "{split}_human.json", "machine": "{split}_augmented.json"}

HELDOUT_IMAGES = Path(__file__).resolve().parents[3] / "data" / "heldout_images.json"


def qa_path(split: str, kind: str) -> str:
    return f"{ROOT}/{split}/{QA_FILES[kind].format(split=split)}"


def image_path(split: str, imgname: str) -> str:
    return f"{ROOT}/{split}/png/{imgname}"


def table_path(split: str, imgname: str) -> str:
    return f"{ROOT}/{split}/tables/{Path(imgname).stem}.csv"


def annotation_path(split: str, imgname: str) -> str:
    return f"{ROOT}/{split}/annotations/{Path(imgname).stem}.json"


def parse_table(text: str) -> dict[str, Any]:
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("empty table")
    return {"columns": rows[0], "rows": rows[1:]}


def canonical_chart_type(raw: object) -> str:
    text = str(raw or "").strip().lower()
    return {"v_bar": "vbar", "h_bar": "hbar"}.get(text, text) or "unknown"


def canonicalise_chart_element(element: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(element)
    raw = out.get("value_raw", out.get("value"))
    inferred_unit: str | None = None
    if isinstance(raw, str):
        text = raw.strip()
        if text.endswith("%"):
            inferred_unit = "%"
        elif text.startswith("$"):
            inferred_unit = "$"
    out["value"] = parse_numeric(raw)
    out["unit"] = out.get("unit") or inferred_unit
    out["value_raw"] = raw
    return out


def _one_colour(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None if text and text != "unk" else None


def _element_colours(model: dict[str, Any], count: int) -> list[str | None]:
    listed = model.get("colors")
    if isinstance(listed, str):
        listed = [listed]
    if isinstance(listed, list) and listed:
        values = [_one_colour(c) for c in listed[:count]]
        return values + [None] * (count - len(values))
    single = _one_colour(model.get("color"))
    return [single] * count


def _norm_or_none(box: Any, image_w: int, image_h: int) -> list[float] | None:
    try:
        norm = xywh_to_norm1000(box, image_w, image_h)
    except ValueError:
        return None
    if norm[2] > norm[0] and norm[3] > norm[1]:
        return norm
    return None


def _series_elements(model: dict[str, Any], image_w: int, image_h: int) -> list[dict[str, Any]]:
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


def _wedge_element(model: dict[str, Any], image_w: int, image_h: int) -> dict[str, Any] | None:
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


def annotation_boxes(annotation: dict[str, Any], image_w: int, image_h: int) -> list[dict[str, Any]]:
    chart_type = str(annotation.get("type"))
    if chart_type == "pie":
        wedges = (_wedge_element(m, image_w, image_h) for m in annotation.get("models") or ())
        return [w for w in wedges if w is not None]
    if chart_type not in {"v_bar", "h_bar"}:
        return []
    out: list[dict[str, Any]] = []
    for model in annotation.get("models") or ():
        out.extend(_series_elements(model, image_w, image_h))
    return out


class ArchiveReader:
    def __init__(self, archive: str | Path) -> None:
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

    def names(self) -> set[str]:
        return self._names

    def read(self, name: str) -> bytes:
        return self._zip.read(name)

    def read_text(self, name: str) -> str:
        return self.read(name).decode("utf-8", "replace")

    def read_json(self, name: str) -> Any:
        return json.loads(self.read(name))

    def image_size(self, name: str) -> tuple[int, int]:
        from PIL import Image

        with Image.open(io.BytesIO(self.read(name))) as im:
            return im.size

    def qa_rows(self, split: str, kind: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = self.read_json(qa_path(split, kind))
        return rows


@lru_cache(maxsize=1)
def heldout_image_hashes() -> frozenset[str]:
    data = json.loads(HELDOUT_IMAGES.read_text(encoding="utf-8"))
    return frozenset(data["val"]) | frozenset(data["test"])


def build_heldout_image_hashes(reader: ArchiveReader) -> dict[str, list[str]]:
    hashes: dict[str, list[str]] = {}
    for split in ("val", "test"):
        prefix = image_path(split, "")
        names = [n for n in reader.names() if n.startswith(prefix) and n.endswith(".png")]
        hashes[split] = sorted({image_content_sha256(reader.read(n)) for n in names})
    return hashes


def chartqa_training_records(reader: ArchiveReader, *, limit: int, seed: int) -> list[ChartRecord]:
    rng = random.Random(seed)
    heldout = heldout_image_hashes()
    dropped = 0
    out: list[ChartRecord] = []
    source_offset = 0
    for kind in ("human", "machine"):
        rows = reader.qa_rows("train", kind)
        indexed_rows = list(enumerate(rows, start=source_offset))
        source_offset += len(rows)
        for source_row_index, row in rng.sample(indexed_rows, min(limit, len(rows))):
            img_name = image_path("train", row["imgname"])
            ann_name = annotation_path("train", row["imgname"])
            if not (reader.exists(img_name) and reader.exists(ann_name)):
                continue
            raw = reader.read(img_name)
            digest = image_content_sha256(raw)
            if digest in heldout:
                dropped += 1
                continue
            width, height = reader.image_size(img_name)
            annotation = reader.read_json(ann_name)
            raw_elements = annotation_boxes(annotation, width, height)
            if not raw_elements:
                continue
            elements = [
                canonicalise_chart_element({
                    **element,
                    "grounding_provenance": "chartqa_annotation",
                    "value_provenance": "chartqa_annotation",
                })
                for element in raw_elements
            ]

            table = None
            tbl_name = table_path("train", row["imgname"])
            if reader.exists(tbl_name):
                try:
                    table = parse_table(reader.read_text(tbl_name))
                except ValueError:
                    table = None

            question = str(row["query"])
            out.append(ChartRecord(
                schema_version=2,
                source_row_index=source_row_index,
                record_id=make_record_id("chartqa", "train", digest, question, source_row_index),
                source="chartqa",
                split="train",
                image_path=img_name,
                image_sha256=digest,
                question=question,
                answer=None if row.get("label") is None else str(row["label"]),
                question_kind=kind,
                table=table,
                boxes=[element["bbox"] for element in elements],
                plan=None,
                meta={
                    "chart_type": canonical_chart_type(annotation.get("type")),
                    "imgname": row["imgname"],
                    "image_size": [width, height],
                    "n_elements": len(elements),
                    ELEMENTS_KEY: elements,
                },
                elements=elements,
                evidence=None,
                source_id=None if row.get("id") is None else str(row["id"]),
            ))
    if dropped:
        print(f"  chartqa: {dropped} rows dropped — a training image identical to a held-out chart")
    return out
