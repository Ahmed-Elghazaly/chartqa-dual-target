from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any

from chartqa_dt.data.chartqa import ArchiveReader, image_path
from chartqa_dt.data.records import image_content_sha256, make_record_id
from chartqa_dt.data.refchartqa import boxes_to_norm1000
from chartqa_dt.data.sources import chartqa_archive_path, load_refchartqa


def chartqa_rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    from PIL import Image

    out: list[dict[str, Any]] = []
    source_row_index = 0
    with ArchiveReader(chartqa_archive_path(data_root)) as reader:
        for kind in ("human", "machine"):
            for row in reader.qa_rows(split, kind):
                raw = reader.read(image_path(split, str(row["imgname"])))
                question = str(row["query"])
                digest = image_content_sha256(raw)
                out.append({
                    "record_id": make_record_id("chartqa", split, digest, question, source_row_index),
                    "question": question,
                    "answer": str(row["label"]),
                    "question_kind": kind,
                    "image": Image.open(io.BytesIO(raw)).convert("RGB"),
                })
                source_row_index += 1
    return out


def select_slice(rows: list[dict[str, Any]], slice_file: Path) -> list[dict[str, Any]]:
    ids = json.loads(slice_file.read_text(encoding="utf-8"))["record_ids"]
    by_id = {row["record_id"]: row for row in rows}
    return [by_id[record_id] for record_id in ids]


def refchartqa_sample(cache_root: Path, split: str, n: int, seed: int = 0) -> list[dict[str, Any]]:
    kinds = ("human", "machine", "pot")
    quotient, remainder = divmod(n, len(kinds))
    capacities = {kind: quotient + int(index < remainder) for index, kind in enumerate(kinds)}
    seen = dict.fromkeys(kinds, 0)
    buckets: dict[str, list[dict[str, Any]]] = {kind: [] for kind in kinds}
    rng = random.Random(seed)
    canonical_split = "val" if split in {"val", "validation"} else split
    for source_row_index, row in enumerate(load_refchartqa(split, cache_root)):
        kind = str(row.get("type", "")).lower()
        if kind not in buckets or capacities[kind] == 0:
            continue
        seen[kind] += 1
        bucket = buckets[kind]
        if len(bucket) < capacities[kind]:
            position = len(bucket)
        else:
            position = rng.randrange(seen[kind])
            if position >= capacities[kind]:
                continue
        image = row["image"].convert("RGB")
        w, h = image.size
        question = str(row["query"])
        item = {
            "record_id": make_record_id("refchartqa", canonical_split, image_content_sha256(image),
                                        question, source_row_index),
            "question": question,
            "answer": str(row["label"]),
            "question_kind": kind,
            "image": image,
            "image_size": (w, h),
            "raw_boxes": list(row.get("grounding_bboxes") or []),
            "gt_boxes": boxes_to_norm1000(row.get("grounding_bboxes"), w, h),
        }
        if position == len(bucket):
            bucket.append(item)
        else:
            bucket[position] = item
    out = [item for kind in kinds for item in buckets[kind]]
    random.Random(seed).shuffle(out)
    return out
