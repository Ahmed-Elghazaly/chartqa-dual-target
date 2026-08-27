"""The strict output schema, and the validation the schema itself cannot express.

`PLAN.md` Appendix A supplies the JSON Schema. Auditing it in Phase 0 confirmed it
is sound and that it deliberately delegates: it rejects a coordinate above 1000, a
ninth evidence item, an unknown operation, an extra key and a missing field, but
it accepts inverted boxes, zero-area boxes, a coordinate of exactly 1000, plans
deeper than four, and a ``lookup`` of a label that is not in ``evidence``.

Those five are the "validation rules beyond the schema" the plan names, and they
live here. Two carry measured hazards behind them:

* a coordinate of exactly **1000** is silently discarded by the official evaluator
  (`DECISIONS.md` 0004), so it is a validation finding rather than a clamp we make
  quietly;
* the schema's ``maxItems: 8`` on ``evidence`` is a **hazard, not an allowance** —
  dataset-level AP collapses from 1.0000 to 0.3243 when three extra boxes are
  appended per image (`DECISIONS.md` 0014).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chartqa_dt.plans.executor import MAX_DEPTH, OPS, plan_depth
from chartqa_dt.vision.coords import OFFICIAL_MAX_COORD

MAX_EVIDENCE = 8

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["answerable", "evidence", "plan", "model_answer"],
    "additionalProperties": False,
    "properties": {
        "answerable": {"type": "boolean"},
        "evidence": {
            "type": "array", "minItems": 0, "maxItems": MAX_EVIDENCE,
            "items": {
                "type": "object",
                "required": ["label", "bbox"],
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string", "maxLength": 128},
                    "value": {"type": ["number", "string", "null"]},
                    "unit": {"type": ["string", "null"], "maxLength": 32},
                    "bbox": {"type": "array", "minItems": 4, "maxItems": 4,
                             "items": {"type": "number", "minimum": 0, "maximum": 1000}},
                },
            },
        },
        "focus_bbox": {"type": ["array", "null"], "minItems": 4, "maxItems": 4,
                       "items": {"type": "number", "minimum": 0, "maximum": 1000}},
        "plan": {"$ref": "#/$defs/node"},
        "model_answer": {"type": "string", "maxLength": 256},
    },
    "$defs": {
        "node": {
            "type": "object",
            "required": ["op"],
            "additionalProperties": False,
            "properties": {
                "op": {"enum": sorted(OPS)},
                "args": {"type": "array", "maxItems": 4,
                         "items": {"oneOf": [{"$ref": "#/$defs/node"},
                                             {"type": ["string", "number", "boolean", "null"]}]}},
            },
        },
    },
}


@dataclass
class ValidationResult:
    """Why a record failed, in enough detail to be counted by category."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def validate_schema(record: Any) -> list[str]:
    """Errors from the JSON Schema alone."""
    try:
        import jsonschema
    except ImportError:  # pragma: no cover
        return ["jsonschema is not installed"]
    validator = jsonschema.Draft202012Validator(OUTPUT_SCHEMA)
    return [f"{list(e.path) or 'root'}: {e.message}" for e in
            sorted(validator.iter_errors(record), key=lambda e: list(e.path))]


def validate_beyond_schema(record: Any) -> tuple[list[str], list[str]]:
    """The five rules the schema delegates. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(record, dict):
        return ["record is not an object"], []

    evidence = record.get("evidence") or []
    labels: set[str] = set()

    for i, item in enumerate(evidence):
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        if isinstance(label, str):
            if label in labels:
                errors.append(f"evidence[{i}]: duplicate label {label!r}; lookups become ambiguous")
            labels.add(label)
        errors.extend(_box_errors(item.get("bbox"), f"evidence[{i}].bbox"))
        warnings.extend(_box_warnings(item.get("bbox"), f"evidence[{i}].bbox"))

    if record.get("focus_bbox") is not None:
        errors.extend(_box_errors(record["focus_bbox"], "focus_bbox"))
        warnings.extend(_box_warnings(record["focus_bbox"], "focus_bbox"))

    plan = record.get("plan")
    if plan is not None:
        depth = plan_depth(plan)
        if depth > MAX_DEPTH:
            errors.append(f"plan depth {depth} exceeds {MAX_DEPTH}")
        for label in _lookup_labels(plan):
            if label not in labels:
                errors.append(f"plan references label {label!r}, which is not in evidence")

    # Not an error: the schema permits it. But dataset AP collapses when a model
    # lists everything plausible (DECISIONS.md 0014), so it is surfaced.
    if len(evidence) > 3:
        warnings.append(
            f"{len(evidence)} evidence items; every box beyond the true ones is a global "
            "false positive at dataset level (DECISIONS.md 0014)"
        )
    return errors, warnings


def _box_errors(box: Any, where: str) -> list[str]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return []                       # the schema already reports the shape
    x1, y1, x2, y2 = box
    out = []
    if not (x1 < x2):
        out.append(f"{where}: x1 ({x1}) must be less than x2 ({x2})")
    if not (y1 < y2):
        out.append(f"{where}: y1 ({y1}) must be less than y2 ({y2})")
    return out


def _box_warnings(box: Any, where: str) -> list[str]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return []
    if any(v > OFFICIAL_MAX_COORD for v in box):
        return [f"{where}: coordinate above {OFFICIAL_MAX_COORD}; the official evaluator "
                "discards the whole box (DECISIONS.md 0004) — clamp before scoring"]
    return []


def _lookup_labels(node: Any) -> list[str]:
    """Every label a `lookup` in this tree depends on."""
    if not isinstance(node, dict):
        return []
    out: list[str] = []
    args = node.get("args") or []
    if node.get("op") == "lookup":
        out += [a for a in args if isinstance(a, str)]
    for a in args:
        out += _lookup_labels(a)
    return out


def validate_record(record: Any) -> ValidationResult:
    """Full validation: the schema, then the rules it delegates."""
    errors = validate_schema(record)
    extra, warnings = validate_beyond_schema(record)
    errors += extra
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)
