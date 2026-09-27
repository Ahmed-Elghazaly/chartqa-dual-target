from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from chartqa_dt.data.chartqa import ArchiveReader, chartqa_training_records
from chartqa_dt.data.dedup import DedupReport, deduplicate
from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord, qualified_labels
from chartqa_dt.data.refchartqa import CONTEXT_ELEMENTS_KEY, read_jsonl
from chartqa_dt.plans.facts import FactError, resolved_element_value, table_values_for_record


def refchartqa_records(cache: Path, aligned_cache: Path) -> list[ChartRecord]:
    records = [ChartRecord.from_dict(raw) for raw in read_jsonl(cache)]
    by_id = {row["record_id"]: row for row in read_jsonl(aligned_cache)}
    out: list[ChartRecord] = []
    for r in records:
        a = by_id.get(r.record_id)
        if a is None:
            continue
        aligned = [
            {**e, "grounding_provenance": "refchartqa_aligned",
             "value_provenance": "chartqa_annotation"}
            for e in a[ELEMENTS_KEY]
        ]
        meta = {
            **r.meta,
            ELEMENTS_KEY: aligned,
            CONTEXT_ELEMENTS_KEY: a[CONTEXT_ELEMENTS_KEY],
            "aligned_to_chartqa": True,
        }
        out.append(replace(r, meta=meta, table=r.table or a.get("table"), elements=aligned,
                           evidence=list(range(len(aligned)))))
    print(f"  refchartqa: {len(out):,} of {len(records):,} records aligned to ChartQA elements")
    return out


def admissible_context(
    record: ChartRecord,
    context: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    marked = [dict(element) for element in (record.elements or [])]
    marked_labels = qualified_labels(marked)
    table_values = table_values_for_record(record)
    kept: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    for element in context:
        trial = [*marked, *kept, dict(element)]
        labels = qualified_labels(trial)
        if labels[: len(marked)] != marked_labels:
            dropped["renames_marked_fact"] += 1
            continue
        if len(set(labels)) != len(labels):
            dropped["ambiguous_label"] += 1
            continue
        try:
            for label, candidate in zip(labels, trial, strict=True):
                resolved_element_value(record, candidate, label, table_values=table_values)
        except FactError:
            dropped["value_conflict"] += 1
            continue
        kept.append(dict(element))
    return kept, dropped


def attach_context(records: Iterable[ChartRecord]) -> list[ChartRecord]:
    out: list[ChartRecord] = []
    added = 0
    for record in records:
        context = record.meta.get(CONTEXT_ELEMENTS_KEY)
        if context is None:
            out.append(record)
            continue
        meta = {key: value for key, value in record.meta.items() if key != CONTEXT_ELEMENTS_KEY}
        present = [element.get("bbox") for element in (record.elements or [])]
        fresh = [element for element in context if element.get("bbox") not in present]
        kept, _ = admissible_context(record, fresh)
        added += len(kept)
        elements = [*(record.elements or []), *kept]
        if kept:
            meta[ELEMENTS_KEY] = elements
        out.append(replace(record, elements=elements if kept else record.elements, meta=meta))
    print(f"  canonical pool: {added:,} unmarked context facts appended")
    return out


def canonical_real_pool(chartqa: list[ChartRecord], refchartqa: list[ChartRecord]
                        ) -> tuple[list[ChartRecord], DedupReport]:
    merged, report = deduplicate([*chartqa, *refchartqa])
    merged.sort(key=lambda record: (record.key, record.record_id))
    return attach_context(merged), report


def load_real_pool(archive: Path, *, chartqa_limit: int, seed: int, refchartqa_cache: Path,
                   aligned_cache: Path) -> list[ChartRecord]:
    with ArchiveReader(archive) as reader:
        chartqa = chartqa_training_records(reader, limit=chartqa_limit, seed=seed)
    refchartqa = refchartqa_records(refchartqa_cache, aligned_cache)
    records, report = canonical_real_pool(chartqa, refchartqa)
    print(f"  canonical real pool: {report.summary()}")
    return records


def synthetic_records(manifest: Path) -> list[ChartRecord]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    out: list[ChartRecord] = []
    for source_row_index, example in enumerate(data["examples"]):
        if example["holdout"]:
            continue
        out.append(ChartRecord(
            schema_version=2,
            source_row_index=source_row_index,
            record_id=example["example_id"],
            source="synthetic",
            split="train",
            image_path=str(manifest.parent / example["image_path"]),
            image_sha256=example["image_sha256"],
            question=example["question"],
            answer=example["answer"],
            question_kind="synthetic",
            table=example["table"],
            boxes=[ev["bbox"] for ev in example["evidence"]],
            plan=example["plan"],
            elements=[
                {**el, "grounding_provenance": "synthetic_exact",
                 "value_provenance": "synthetic_generated"}
                for el in example["elements"]
            ],
            evidence=example["evidence_index"],
            source_id=example["example_id"],
            meta={
                "level": example["level"],
                "chart_type": example["chart_type"],
                "style_seed": example["style_seed"],
                "data_seed": example["data_seed"],
            },
        ))
    return out
