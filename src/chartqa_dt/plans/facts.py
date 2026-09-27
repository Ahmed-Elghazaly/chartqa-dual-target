from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from chartqa_dt.data.records import ChartRecord, fold_for_matching, qualified_labels
from chartqa_dt.plans.executor import parse_numeric


class FactError(ValueError):
    pass


VALUE_AGREEMENT_TOLERANCE = 0.02


def values_agree(table_value: Any, element_value: Any) -> bool:
    table_number = parse_numeric(table_value)
    element_number = parse_numeric(element_value)
    if table_number is None or element_number is None:
        return True
    scale = VALUE_AGREEMENT_TOLERANCE * max(abs(table_number), abs(element_number), 1e-9)
    if abs(table_number - element_number) <= scale:
        return True
    percent_scale = VALUE_AGREEMENT_TOLERANCE * max(abs(element_number), 1e-9)
    return abs(table_number * 100.0 - element_number) <= percent_scale


def table_values_for_record(record: ChartRecord) -> dict[str | tuple[str, str], float]:
    table = record.table
    if not isinstance(table, dict):
        return {}
    columns = [fold_for_matching(column) for column in (table.get("columns") or [])]
    values: dict[str | tuple[str, str], float] = {}
    for row in table.get("rows") or []:
        if not isinstance(row, list) or not row:
            continue
        label = str(row[0]).strip()
        first_numeric = True
        for column_index, cell in enumerate(row[1:], start=1):
            number = parse_numeric(cell)
            if number is None:
                continue
            if column_index < len(columns):
                values.setdefault((columns[column_index], label), number)
            if first_numeric:
                values.setdefault(label, number)
                first_numeric = False
    return values


def resolved_element_value(
    record: ChartRecord,
    element: Mapping[str, Any],
    qualified_label: str,
    *,
    table_values: Mapping[str | tuple[str, str], float] | None = None,
) -> Any:
    indexed = table_values if table_values is not None else table_values_for_record(record)
    bare_label = str(element.get("label"))
    series = fold_for_matching(element.get("series"))
    value = indexed.get((series, bare_label))
    if value is None:
        value = (
            indexed.get(bare_label, element.get("value"))
            if qualified_label == bare_label
            else element.get("value")
        )
    if not values_agree(value, element.get("value")):
        raise FactError(
            f"record {record.record_id!r} has conflicting table and annotation values "
            f"for fact {qualified_label!r}: {value!r} versus {element.get('value')!r}"
        )
    return value


def _valid_box(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(
            not isinstance(coordinate, bool)
            and isinstance(coordinate, (int, float))
            and math.isfinite(float(coordinate))
            for coordinate in value
        )
        and 0 <= value[0] < value[2] <= 1000
        and 0 <= value[1] < value[3] <= 1000
    )


def _finite_json(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(_finite_json(item) for item in value.values())
    return True


def mining_refusal_reason(record: ChartRecord) -> str | None:
    if record.split != "train":
        return "not_training_split"
    if record.answer is None or not str(record.answer).strip():
        return "missing_answer"
    if not isinstance(record.question, str) or not record.question.strip():
        return "missing_question"
    if record.evidence is not None and not record.evidence:
        return "empty_evidence"
    sources = set(record.meta.get("merged_from") or [record.source])
    if sources == {"refchartqa"} and record.meta.get("aligned_to_chartqa") is not True:
        return "unaligned_refchartqa"
    if not record.elements:
        return "missing_facts"
    for element in record.elements:
        label = element.get("label")
        if not isinstance(label, str) or not label.strip():
            return "missing_fact_label"
        if not _valid_box(element.get("bbox")):
            return "invalid_fact_box"
    if not _finite_json(record.table) or not _finite_json(record.elements):
        return "non_finite_fact_data"
    labels = qualified_labels(record.elements)
    table_values = table_values_for_record(record)
    try:
        for label, element in zip(labels, record.elements, strict=True):
            resolved_element_value(record, element, label, table_values=table_values)
    except FactError:
        return "conflicting_fact_value"
    return None


def facts_for_record(record: ChartRecord) -> list[dict[str, Any]]:
    reason = mining_refusal_reason(record)
    if reason is not None:
        raise FactError(f"record {record.record_id!r} is not eligible for plan mining: {reason}")
    elements = record.elements or []
    labels = qualified_labels(elements)
    table_values = table_values_for_record(record)
    return [
        {
            "id": f"f{index + 1}",
            "element_index": index,
            "label": label,
            "value": resolved_element_value(record, element, label, table_values=table_values),
            "unit": element.get("unit"),
            "colour": element.get("colour"),
            "bbox": element.get("bbox"),
        }
        for index, (label, element) in enumerate(zip(labels, elements, strict=True))
    ]


def marked_fact_ids(record: ChartRecord) -> set[str] | None:
    if record.evidence is None:
        return None
    return {f"f{index + 1}" for index in record.evidence}
