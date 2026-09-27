from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord, qualified_labels
from chartqa_dt.model.coords import clamp_for_official_evaluator
from chartqa_dt.model.parsing import parse_record
from chartqa_dt.plans.executor import (
    FOLD_OPS,
    MAX_FACTS,
    EvidenceItem,
    ExecutorError,
    is_fact_ref,
    plan_fact_ids,
    resolve_label_plan,
)
from chartqa_dt.plans.facts import FactError, resolved_element_value, table_values_for_record
from chartqa_dt.plans.roundtrip import check_record
from chartqa_dt.plans.schema import validate_record

COMPACT = (",", ":")


class TargetError(ValueError):
    pass


class NoPlanAvailable(TargetError):
    pass


MIN_TRUNCATED_PREFIX = 8


def rename_truncated_operands(plan: Any, names: Sequence[str]) -> Any:
    if not isinstance(plan, dict):
        if isinstance(plan, str) and plan not in names:
            candidates = [n for n in names
                          if len(n) >= MIN_TRUNCATED_PREFIX and plan.startswith(n)]
            if len(candidates) == 1:
                return candidates[0]
        return plan
    if is_fact_ref(plan) or "op" not in plan:
        return plan
    return {**plan, "args": [rename_truncated_operands(a, names)
                             for a in (plan.get("args") or [])]}


def _label_operands(plan: Any) -> list[str]:
    if not isinstance(plan, dict):
        return []
    labels: list[str] = []
    for arg in plan.get("args") or []:
        if isinstance(arg, str):
            labels.append(arg)
        elif isinstance(arg, dict) and not is_fact_ref(arg):
            labels.extend(_label_operands(arg))
    return labels


def _has_implicit_fold(plan: Any) -> bool:
    if not isinstance(plan, dict) or is_fact_ref(plan):
        return False
    args = plan.get("args")
    return ((plan.get("op") in FOLD_OPS and args == [])
            or any(_has_implicit_fold(arg) for arg in (args or []) if isinstance(arg, dict)))


def _facts_and_plan(record: ChartRecord, *, require_plan: bool, include_plan: bool = True
                    ) -> tuple[list[dict[str, Any]], list[str], dict[str, Any] | None]:
    raw = record.elements if record.elements is not None else record.meta.get(ELEMENTS_KEY)
    elements = [element for element in (raw or []) if isinstance(element, dict)]
    if elements and not any(element.get("label") is not None for element in elements):
        elements = []

    if not elements:
        raise TargetError(
            f"{record.record_id}: labelled chart facts are required; unaligned boxes "
            "cannot be given invented labels or values"
        )

    names = qualified_labels(elements)
    if len(set(names)) != len(names):
        repeated = next(name for name in names if names.count(name) > 1)
        raise TargetError(
            f"{record.record_id}: {repeated!r} still names multiple marks after series "
            "qualification"
        )
    by_name = {name: index for index, name in enumerate(names)}
    table_values = table_values_for_record(record)

    plan = (rename_truncated_operands(record.plan, names)
            if include_plan and isinstance(record.plan, dict) and record.plan.get("op")
            else None)
    if require_plan and plan is None:
        raise NoPlanAvailable(
            f"{record.record_id}: no verified plan; use the Stage-1 target, which has plan=null"
        )

    selected: set[int] = set(record.evidence or [])
    if plan is not None:
        if _has_implicit_fold(plan):
            selected.update(range(len(elements)))
        elif plan_fact_ids(plan):
            for fact_id in plan_fact_ids(plan):
                try:
                    selected.add(int(fact_id[1:]) - 1)
                except (TypeError, ValueError):
                    raise TargetError(
                        f"{record.record_id}: invalid fact reference {fact_id!r}"
                    ) from None
        else:
            for label in _label_operands(plan):
                if label not in by_name:
                    raise TargetError(
                        f"{record.record_id}: plan references {label!r}, which has no chart fact"
                    )
                selected.add(by_name[label])
    if not selected:
        raise TargetError(f"{record.record_id}: target has neither operands nor marked evidence")
    if len(selected) > MAX_FACTS:
        raise TargetError(
            f"{record.record_id}: target requires {len(selected)} facts, over the "
            f"{MAX_FACTS}-fact limit"
        )

    facts: list[dict[str, Any]] = []
    for index in sorted(selected):
        if index < 0 or index >= len(elements):
            raise TargetError(f"{record.record_id}: fact index {index} is out of range")
        element, name = elements[index], names[index]
        try:
            value = resolved_element_value(record, element, name, table_values=table_values)
        except FactError as exc:
            raise TargetError(str(exc)) from exc
        facts.append(_fact(record, index, name, value, element.get("unit"),
                           element.get("bbox")))

    evidence_refs = []
    selected_ids = {int(fact["id"][1:]) - 1: fact["id"] for fact in facts}
    for index in record.evidence or []:
        if index not in selected_ids:
            raise TargetError(f"{record.record_id}: marked evidence {index} was not emitted")
        evidence_refs.append(selected_ids[index])

    typed_plan = plan
    if plan is not None and not plan_fact_ids(plan):
        items = [EvidenceItem(fact["label"], fact["value"], fact["unit"], fact["id"])
                 for fact in facts]
        try:
            typed_plan = resolve_label_plan(plan, items)
        except ExecutorError as exc:
            raise TargetError(f"{record.record_id}: cannot resolve label plan: {exc}") from exc
    return facts, evidence_refs, typed_plan


def _fact(record: ChartRecord, source_index: int, label: Any, value: Any,
          unit: Any, box: Any) -> dict[str, Any]:
    if not (isinstance(box, (list, tuple)) and len(box) == 4
            and all(isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool)
                    and math.isfinite(float(coordinate)) for coordinate in box)):
        raise TargetError(
            f"{record.record_id}: fact {str(label)!r} has no usable box ({box!r})"
        )
    return {
        "id": f"f{source_index + 1}",
        "label": str(label),
        "value": value,
        "unit": unit,
        "bbox": clamp_for_official_evaluator(tuple(box)),
    }


def build_target(record: ChartRecord) -> str:
    if record.answer is None:
        raise TargetError(f"{record.record_id} has no answer; nothing to supervise")
    facts, evidence_refs, plan = _facts_and_plan(record, require_plan=True)
    text = json.dumps({
        "answerable": True,
        "facts": facts,
        "evidence_refs": evidence_refs,
        "plan": plan,
        "model_answer": str(record.answer),
    }, separators=COMPACT, ensure_ascii=False)

    parsed = parse_record(text)
    if not parsed.ok or parsed.record is None:
        raise TargetError(f"{record.record_id}: own target does not parse — {parsed.reason}")
    result = validate_record(parsed.record)
    if not result.ok:
        raise TargetError(f"{record.record_id}: own target fails the schema — "
                          f"{'; '.join(result.errors[:2])}")
    trip = check_record(parsed.record, strict=True)
    if not trip.ok:
        raise TargetError(f"{record.record_id}: own plan does not reproduce its own "
                          f"answer ({trip.outcome}: executed {trip.executed!r} vs stated "
                          f"{trip.stated!r}{'; ' + trip.error if trip.error else ''})")
    return text


def build_grounding_target(record: ChartRecord) -> str:
    if record.answer is None:
        raise TargetError(f"{record.record_id} has no answer; nothing to supervise")
    if record.evidence is None:
        raise TargetError(
            f"{record.record_id}: its boxes describe the whole chart rather than the evidence "
            "for this question; only a plan can select from them"
        )
    facts, evidence_refs, _ = _facts_and_plan(record, require_plan=False, include_plan=False)
    if not evidence_refs:
        raise TargetError(f"{record.record_id} has no question-specific evidence boxes")

    text = json.dumps({"answerable": True, "facts": facts,
                       "evidence_refs": evidence_refs, "plan": None,
                       "model_answer": str(record.answer)},
                      separators=COMPACT, ensure_ascii=False)
    parsed = parse_record(text)
    result = validate_record(parsed.record) if parsed.record is not None else None
    if not parsed.ok or parsed.record is None or result is None or not result.ok:
        why = parsed.reason if not parsed.ok or result is None else "; ".join(result.errors[:2])
        raise TargetError(f"{record.record_id}: invalid grounding target — {why}")
    return text


def build_stage1_target(record: ChartRecord) -> str:
    if record.plan is not None:
        build_target(record)
    return build_grounding_target(record)
