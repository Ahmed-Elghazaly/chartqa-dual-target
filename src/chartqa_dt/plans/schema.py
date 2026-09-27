from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from chartqa_dt.model.coords import OFFICIAL_MAX_COORD
from chartqa_dt.plans.executor import (
    FILTER_COMPARISONS,
    MAX_DEPTH,
    MAX_FACTS,
    OPS,
    plan_depth,
    plan_fact_ids,
    validate_plan_shape,
)

EVALUATION_MAX_FACTS = 64

_BOX = {
    "type": "array",
    "minItems": 4,
    "maxItems": 4,
    "items": {"type": "number", "minimum": 0, "maximum": OFFICIAL_MAX_COORD},
}

OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ChartQADualTarget",
    "type": "object",
    "required": [
        "answerable", "facts", "evidence_refs", "plan", "model_answer",
    ],
    "additionalProperties": False,
    "properties": {
        "answerable": {"type": "boolean"},
        "facts": {
            "type": "array",
            "minItems": 0,
            "maxItems": MAX_FACTS,
            "items": {"$ref": "#/$defs/fact"},
        },
        "evidence_refs": {
            "type": "array",
            "minItems": 0,
            "maxItems": MAX_FACTS,
            "items": {"type": "string", "pattern": "^f[1-9][0-9]*$"},
        },
        "focus_bbox": {"oneOf": [_BOX, {"type": "null"}]},
        "plan": {"oneOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]},
        "model_answer": {"type": "string", "maxLength": 256},
    },
    "$defs": {
        "fact": {
            "type": "object",
            "required": ["id", "label", "value", "unit", "bbox"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "string", "pattern": "^f[1-9][0-9]*$"},
                "label": {"type": "string", "maxLength": 128},
                "value": {"type": ["number", "string", "null"]},
                "unit": {"type": ["string", "null"], "maxLength": 32},
                "bbox": _BOX,
            },
        },
        "ref": {
            "type": "object",
            "required": ["ref"],
            "additionalProperties": False,
            "properties": {"ref": {"type": "string", "pattern": "^f[1-9][0-9]*$"}},
        },
        "expression": {
            "oneOf": [{"$ref": "#/$defs/ref"}, {"$ref": "#/$defs/node"}],
        },
        "node": {
            "type": "object",
            "required": ["op", "args"],
            "additionalProperties": False,
            "properties": {
                "op": {"enum": sorted(OPS)},
                "args": {
                    "type": "array",
                    "maxItems": MAX_FACTS,
                    "items": {"$ref": "#/$defs/expression"},
                },
                "position": {"type": "integer", "minimum": 1, "maximum": MAX_FACTS},
                "order": {"enum": ["ascending", "descending"]},
                "compare": {"enum": list(FILTER_COMPARISONS)},
                "threshold": {"type": "number"},
            },
        },
    },
}


@lru_cache(maxsize=4)
def output_schema(max_facts: int = MAX_FACTS) -> dict[str, Any]:
    def lift(node: Any, key: str = "") -> Any:
        if isinstance(node, dict):
            out = {name: lift(value, name) for name, value in node.items()}
            if out.get("maxItems") == MAX_FACTS:
                out["maxItems"] = max_facts
            if key == "position" and out.get("maximum") == MAX_FACTS:
                out["maximum"] = max_facts
            return out
        if isinstance(node, list):
            return [lift(item, key) for item in node]
        return node

    lifted: dict[str, Any] = lift(copy.deepcopy(OUTPUT_SCHEMA))
    return lifted


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def validate_schema(record: Any, *, max_facts: int = MAX_FACTS) -> list[str]:
    import jsonschema

    validator = jsonschema.Draft202012Validator(output_schema(max_facts))
    return [
        f"{list(error.path) or 'root'}: {error.message}"
        for error in sorted(validator.iter_errors(record), key=lambda error: list(error.path))
    ]


def _box_errors(box: Any, where: str) -> list[str]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return []
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               for value in box):
        return []
    if not all(math.isfinite(float(value)) for value in box):
        return [f"{where}: every coordinate must be finite"]
    x1, y1, x2, y2 = box
    errors = []
    if x1 >= x2:
        errors.append(f"{where}: x1 ({x1}) must be less than x2 ({x2})")
    if y1 >= y2:
        errors.append(f"{where}: y1 ({y1}) must be less than y2 ({y2})")
    return errors


def validate_beyond_schema(record: Any, *, max_facts: int = MAX_FACTS) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(record, dict):
        return ["record is not an object"], warnings

    facts = record.get("facts")
    if not isinstance(facts, list):
        return errors, warnings
    ids: list[str] = []
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            continue
        fact_id = fact.get("id")
        if isinstance(fact_id, str):
            ids.append(fact_id)
        value = fact.get("value")
        if isinstance(value, float) and not math.isfinite(value):
            errors.append(f"facts[{index}].value must be finite")
        errors.extend(_box_errors(fact.get("bbox"), f"facts[{index}].bbox"))
    duplicate_ids = sorted({fact_id for fact_id in ids if ids.count(fact_id) > 1})
    if duplicate_ids:
        errors.append(f"fact ids must be unique; duplicates={duplicate_ids}")
    known = set(ids)

    refs = record.get("evidence_refs")
    if isinstance(refs, list):
        duplicated = sorted({ref for ref in refs if refs.count(ref) > 1})
        if duplicated:
            errors.append(f"evidence_refs must be unique; duplicates={duplicated}")
        missing = sorted(ref for ref in refs if isinstance(ref, str) and ref not in known)
        if missing:
            errors.append(f"evidence_refs contain unknown fact ids: {missing}")
        if len(refs) > 3:
            warnings.append(
                f"{len(refs)} official evidence boxes; false-positive boxes reduce dataset AP"
            )

    focus = record.get("focus_bbox")
    if focus is not None:
        errors.extend(_box_errors(focus, "focus_bbox"))

    answerable = record.get("answerable")
    plan = record.get("plan")
    if answerable is False:
        if plan is not None:
            errors.append("an unanswerable record must have plan=null")
        if facts:
            errors.append("an unanswerable record must have no facts")
        if refs:
            errors.append("an unanswerable record must have no evidence_refs")
        if record.get("model_answer") != "":
            errors.append("an unanswerable record must have an empty model_answer")
    if plan is not None:
        errors.extend(validate_plan_shape(plan, max_facts=max_facts))
        if plan_depth(plan) > MAX_DEPTH:
            errors.append(f"plan depth {plan_depth(plan)} exceeds {MAX_DEPTH}")
        missing = sorted(set(plan_fact_ids(plan)) - known)
        if missing:
            errors.append(f"plan references unknown fact ids: {missing}")
    return errors, warnings


def validate_record(record: Any, *, max_facts: int = MAX_FACTS) -> ValidationResult:
    errors = validate_schema(record, max_facts=max_facts)
    if errors:
        return ValidationResult(False, errors)
    semantic, warnings = validate_beyond_schema(record, max_facts=max_facts)
    return ValidationResult(not semantic, semantic, warnings)
