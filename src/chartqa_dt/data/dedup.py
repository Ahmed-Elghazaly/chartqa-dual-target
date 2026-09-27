from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from chartqa_dt.data.records import ChartRecord

SOURCE_PRIORITY = {"chartqa": 0, "chartqapro": 1, "refchartqa": 2, "synthetic": 3}
BOX_EPSILON = 1.0

_GROUNDING_PRIORITY = {
    "synthetic_exact": 0,
    "refchartqa_aligned": 1,
    "refchartqa_gold": 2,
    "chartqa_annotation": 3,
}
_VALUE_PRIORITY = {
    "synthetic_generated": 0,
    "chartqa_table": 1,
    "chartqa_annotation": 2,
    "unknown": 3,
}
_CORE_ELEMENT_FIELDS = ("label", "value", "unit", "series", "colour")


class DedupConflictError(ValueError):
    pass


@dataclass
class DedupReport:
    input_records: int = 0
    output_records: int = 0
    merges: int = 0
    duplicate_groups: int = 0
    duplicate_rows: int = 0
    merged_pairs: Counter[str] = field(default_factory=Counter)
    answer_conflicts: int = 0
    boxes_gained: int = 0
    plans_gained: int = 0
    quarantined_groups: int = 0
    quarantined_records: int = 0
    conflict_groups: list[dict[str, Any]] = field(default_factory=list)
    cross_split_collisions: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def duplicates_removed(self) -> int:
        return self.input_records - self.output_records - self.quarantined_records

    def summary(self) -> str:
        pairs = ", ".join(f"{k}={v}" for k, v in sorted(self.merged_pairs.items())) or "none"
        return (
            f"{self.input_records:,} in -> {self.output_records:,} out; "
            f"{self.duplicate_groups:,} duplicate groups / {self.duplicate_rows:,} extra rows, "
            f"{self.merges:,} compatible merges ({pairs}); "
            f"{self.quarantined_groups:,} conflicting groups / "
            f"{self.quarantined_records:,} rows quarantined; "
            f"{len(self.cross_split_collisions):,} cross-split collisions"
        )


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _same_box(a: list[float], b: list[float]) -> bool:
    return len(a) == len(b) == 4 and all(abs(float(x) - float(y)) <= BOX_EPSILON
                                         for x, y in zip(a, b))


def _answer_key(answer: str | None) -> str | None:
    return None if answer is None else " ".join(answer.strip().casefold().split())


def union_boxes(*groups: list[list[float]] | None) -> list[list[float]] | None:
    candidates = [list(box) for group in groups for box in (group or ())]
    candidates.sort(key=lambda box: tuple(float(v) for v in box))
    out: list[list[float]] = []
    for box in candidates:
        if not any(_same_box(box, kept) for kept in out):
            out.append(box)
    return out or None


def _preferred(current: Any, proposed: Any, priorities: Mapping[str, int]) -> Any:
    values = [value for value in (current, proposed) if value is not None]
    if not values:
        return None
    return min(values, key=lambda value: (priorities.get(str(value), 99), str(value)))


def _merge_element_dict(existing: dict[str, Any], proposed: Mapping[str, Any]) -> None:
    for key in _CORE_ELEMENT_FIELDS:
        left, right = existing.get(key), proposed.get(key)
        if left is not None and right is not None and left != right:
            raise DedupConflictError(
                f"same element box has conflicting {key}: {left!r} vs {right!r}"
            )
        if left is None and right is not None:
            existing[key] = right
    existing["grounding_provenance"] = _preferred(
        existing.get("grounding_provenance"), proposed.get("grounding_provenance"),
        _GROUNDING_PRIORITY,
    )
    existing["value_provenance"] = _preferred(
        existing.get("value_provenance"), proposed.get("value_provenance"),
        _VALUE_PRIORITY,
    )
    for key in sorted(set(proposed) - set(existing)):
        existing[key] = proposed[key]


def _merge_elements(a: ChartRecord, b: ChartRecord
                    ) -> tuple[list[dict[str, Any]] | None, list[int] | None]:
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for side, elements in enumerate((a.elements or [], b.elements or [])):
        for index, element in enumerate(elements):
            if not isinstance(element, dict):
                raise DedupConflictError("element is not an object")
            candidates.append((side, index, dict(element)))
    candidates.sort(key=lambda item: _canonical(item[2]))

    merged: list[dict[str, Any]] = []
    old_to_intermediate: dict[tuple[int, int], int] = {}
    for side, index, element in candidates:
        box = element.get("bbox")
        match = None
        for kept_index, kept in enumerate(merged):
            kept_box = kept.get("bbox")
            if (isinstance(box, list) and isinstance(kept_box, list)
                    and _same_box(box, kept_box)):
                match = kept_index
                break
            if box is None and kept_box is None and _canonical(element) == _canonical(kept):
                match = kept_index
                break
        if match is None:
            match = len(merged)
            merged.append(element)
        else:
            _merge_element_dict(merged[match], element)
        old_to_intermediate[(side, index)] = match

    ordering = sorted(range(len(merged)), key=lambda i: (
        tuple(float(v) for v in merged[i].get("bbox", [float("inf")] * 4)),
        str(merged[i].get("series") or ""), str(merged[i].get("label") or ""),
        _canonical(merged[i]),
    ))
    old_final = {old: new for new, old in enumerate(ordering)}
    final_elements = [merged[i] for i in ordering]

    remapped: list[set[int] | None] = []
    for side, record in enumerate((a, b)):
        if record.evidence is None:
            remapped.append(None)
        else:
            remapped.append({old_final[old_to_intermediate[(side, i)]]
                             for i in record.evidence})
    known = [indices for indices in remapped if indices is not None]
    if len(known) == 2 and known[0] != known[1]:
        raise DedupConflictError(
            f"conflicting evidence selections after remap: {sorted(known[0])} vs "
            f"{sorted(known[1])}"
        )
    evidence = sorted(known[0]) if known else None
    return final_elements or None, evidence


def _compatible_optional(name: str, a: Any, b: Any) -> Any:
    if a is not None and b is not None and _canonical(a) != _canonical(b):
        raise DedupConflictError(f"conflicting {name}")
    return a if a is not None else b


def _sources(record: ChartRecord) -> list[str]:
    return list(record.meta.get("merged_from") or [record.source])


def merge_pair(a: ChartRecord, b: ChartRecord, report: DedupReport | None = None
               ) -> ChartRecord:
    if a.split != b.split:
        raise ValueError(
            f"refusing to merge across splits: {a.record_id} is {a.split!r} and "
            f"{b.record_id} is {b.split!r}. A shared key across splits is a leak."
        )
    if a.split != "train":
        raise ValueError("evaluation rows must remain independent and are never merged")
    if a.key != b.key:
        raise ValueError("merge_pair requires the same content dedup_key")

    answers = {_answer_key(answer) for answer in (a.answer, b.answer) if answer is not None}
    if len(answers) > 1:
        if report is not None:
            report.answer_conflicts += 1
        raise DedupConflictError(f"conflicting answers: {a.answer!r} vs {b.answer!r}")

    ordered = sorted((a, b), key=lambda record: (
        SOURCE_PRIORITY.get(record.source, 99), record.record_id, record.source_row_index))
    primary, other = ordered
    answer = primary.answer if primary.answer is not None else other.answer
    table = _compatible_optional("tables", primary.table, other.table)
    plan = _compatible_optional("plans", primary.plan, other.plan)
    elements, evidence = _merge_elements(primary, other)
    boxes = union_boxes(primary.boxes, other.boxes)

    if report is not None:
        if boxes and not primary.boxes:
            report.boxes_gained += 1
        if plan and not primary.plan:
            report.plans_gained += 1

    meta: dict[str, Any] = {}
    for record in reversed(ordered):
        meta.update(record.meta)
    meta["merged_from"] = sorted({*_sources(a), *_sources(b)})
    meta["merged_record_ids"] = sorted({
        a.record_id, b.record_id,
        *a.meta.get("merged_record_ids", []), *b.meta.get("merged_record_ids", []),
    })
    return replace(primary, answer=answer, boxes=boxes, plan=plan, table=table,
                   elements=elements, evidence=evidence, meta=meta)


def deduplicate(records: Iterable[ChartRecord]) -> tuple[list[ChartRecord], DedupReport]:
    rows = list(records)
    report = DedupReport(input_records=len(rows))
    groups: dict[tuple[str, str], list[tuple[int, ChartRecord]]] = {}
    splits_by_key: dict[str, set[str]] = {}
    for position, record in enumerate(rows):
        groups.setdefault((record.split, record.key), []).append((position, record))
        splits_by_key.setdefault(record.key, set()).add(record.split)
    for key, splits in splits_by_key.items():
        ordered_splits = sorted(splits)
        for prior, current in itertools.pairwise(ordered_splits):
            report.cross_split_collisions.append((key, prior, current))

    emitted: list[tuple[int, ChartRecord]] = []
    for (split, key), members in groups.items():
        if len(members) == 1:
            emitted.append(members[0])
            continue
        report.duplicate_groups += 1
        report.duplicate_rows += len(members) - 1
        if split != "train":
            emitted.extend(members)
            continue
        sorted_records = sorted((record for _, record in members), key=lambda record: (
            SOURCE_PRIORITY.get(record.source, 99), record.record_id,
            record.source_row_index))
        merged = sorted_records[0]
        group_report = DedupReport()
        try:
            for record in sorted_records[1:]:
                pair = " + ".join(sorted({merged.source, record.source}))
                merged = merge_pair(merged, record, group_report)
                group_report.merges += 1
                group_report.merged_pairs[pair] += 1
        except DedupConflictError as exc:
            report.answer_conflicts += group_report.answer_conflicts
            report.quarantined_groups += 1
            report.quarantined_records += len(members)
            report.conflict_groups.append({
                "dedup_key": key,
                "record_ids": sorted(record.record_id for _, record in members),
                "reason": str(exc),
            })
            continue
        report.merges += group_report.merges
        report.merged_pairs.update(group_report.merged_pairs)
        report.answer_conflicts += group_report.answer_conflicts
        report.boxes_gained += group_report.boxes_gained
        report.plans_gained += group_report.plans_gained
        emitted.append((min(position for position, _ in members), merged))

    out = [record for _, record in sorted(emitted, key=lambda item: item[0])]
    report.output_records = len(out)
    return out, report
