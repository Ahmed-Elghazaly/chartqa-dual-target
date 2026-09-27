from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from chartqa_dt.plans.executor import MAX_FACTS
from chartqa_dt.plans.schema import EVALUATION_MAX_FACTS, validate_record

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
STRAY_QUOTE_AFTER_ARRAY_RE = re.compile(r"(\]\s*)\"(\s*[},])")
SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})

REQUIRED_FIELDS = (
    "answerable", "facts", "evidence_refs", "plan", "model_answer",
)


@dataclass
class ParseResult:
    ok: bool
    record: dict[str, Any] | None = None
    reason: str = ""
    repairs: list[str] = field(default_factory=list)
    raw: str = ""


def _candidates(text: str) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = [(text.strip(), [])]

    fenced = FENCE_RE.search(text)
    if fenced:
        out.append((fenced.group(1).strip(), ["stripped code fence"]))

    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        block = text[start:end + 1]
        if block.strip() != text.strip():
            out.append((block, ["extracted the outermost JSON object"]))

    extended: list[tuple[str, list[str]]] = []
    for body, repairs in out:
        if '"' not in body and "“" in body:
            extended.append((body.translate(SMART_QUOTES), [*repairs, "normalised quotes"]))
        if TRAILING_COMMA_RE.search(body):
            extended.append((TRAILING_COMMA_RE.sub(r"\1", body),
                             [*repairs, "removed trailing comma"]))
        if STRAY_QUOTE_AFTER_ARRAY_RE.search(body):
            cleaned = STRAY_QUOTE_AFTER_ARRAY_RE.sub(r"\1\2", body)
            extended.append((cleaned, [*repairs, "removed stray quote after an array"]))
            if TRAILING_COMMA_RE.search(cleaned):
                extended.append((TRAILING_COMMA_RE.sub(r"\1", cleaned),
                                 [*repairs, "removed stray quote after an array",
                                  "removed trailing comma"]))
    return out + extended


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value}")


def parse_record(text: str) -> ParseResult:
    if not text or not text.strip():
        return ParseResult(False, reason="empty generation", raw=text or "")

    last_error = "no JSON object found"
    for body, repairs in _candidates(text):
        try:
            obj = json.loads(body, parse_constant=_reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"invalid JSON: {exc.args[0] if exc.args else exc}"
            continue
        if not isinstance(obj, dict):
            last_error = f"top level is {type(obj).__name__}, expected an object"
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in obj]
        if missing:
            return ParseResult(False, reason=f"missing required field(s): {missing}",
                               repairs=repairs, raw=text)
        return ParseResult(True, record=obj, repairs=repairs, raw=text)

    return ParseResult(False, reason=last_error, raw=text)


def coerce_boxes(record: dict[str, Any]) -> list[list[float]]:
    out: list[list[float]] = []
    facts = {fact.get("id"): fact for fact in record.get("facts") or []
             if isinstance(fact, dict)}
    for fact_id in record.get("evidence_refs") or []:
        item = facts.get(fact_id)
        if not isinstance(item, dict):
            continue
        box = item.get("bbox")
        if not isinstance(box, Sequence) or isinstance(box, str) or len(box) != 4:
            continue
        try:
            values = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in values):
            continue
        out.append(values)
    return out


def answer_of(record: dict[str, Any]) -> str:
    value = record.get("model_answer")
    return "" if value is None else str(value)


def schema_ok(record: dict[str, Any], *, max_facts: int = MAX_FACTS) -> tuple[bool, str]:
    result = validate_record(record, max_facts=max_facts)
    return result.ok, "; ".join(result.errors[:2])


def evaluation_schema_ok(record: dict[str, Any]) -> tuple[bool, str]:
    return schema_ok(record, max_facts=EVALUATION_MAX_FACTS)
