"""Turning raw generation into a record — and refusing to invent one when it cannot.

**Non-negotiable rule 3: invalid outputs count as failures.** That is the whole design
constraint here. It would be easy to write a parser that always returns *something*: pull
the last number out of the text, guess a box, fall back to an empty plan. Every one of
those choices converts a model failure into a silently plausible number, and the resulting
score would measure the parser rather than the model.

So this module separates two things that are often conflated:

* **Recovery from formatting noise** — a code fence, a stray sentence before the JSON,
  a trailing comma, smart quotes. The model produced the record; the transport was untidy.
  Recovering here is legitimate, and every recovery is *counted* so the rate is visible.
* **Invention** — supplying a field the model did not produce. Never done. A record
  missing `model_answer` is a failure, and `ParseResult.ok` is False.

The difference is whether the information came from the model or from us.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: Repairs that only touch transport, never content. Ordered cheapest first.
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
#: `"bbox":[10,20,30,40]"` — a quotation mark immediately after a closing bracket, where
#: no valid JSON can have one. Measured at 11-21 occurrences per affected record in the
#: zero-shot probe. Unambiguously transport noise: there is exactly one reading, and
#: removing it invents nothing.
STRAY_QUOTE_AFTER_ARRAY_RE = re.compile(r"(\]\s*)\"(\s*[},])")
SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})

REQUIRED_FIELDS = ("answerable", "evidence", "plan", "model_answer")


@dataclass
class ParseResult:
    """What came back, and exactly how much of it was ours rather than the model's."""

    ok: bool
    record: dict[str, Any] | None = None
    reason: str = ""
    repairs: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def repaired(self) -> bool:
        return bool(self.repairs)

    def __bool__(self) -> bool:
        return self.ok


def _candidates(text: str) -> list[tuple[str, list[str]]]:
    """Progressively more forgiving readings of `text`, each with its repair list."""
    out: list[tuple[str, list[str]]] = [(text.strip(), [])]

    fenced = FENCE_RE.search(text)
    if fenced:
        out.append((fenced.group(1).strip(), ["stripped code fence"]))

    # The outermost {...}: models often prefix a sentence before the object.
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


def parse_record(text: str) -> ParseResult:
    """Read one record from raw generation, or explain why there is not one.

    Never fills in a missing field. A record that does not carry all of
    `REQUIRED_FIELDS` is a failure, because supplying one would mean scoring our default
    instead of the model's output.
    """
    if not text or not text.strip():
        return ParseResult(False, reason="empty generation", raw=text or "")

    last_error = "no JSON object found"
    for body, repairs in _candidates(text):
        try:
            obj = json.loads(body)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"invalid JSON: {exc.args[0] if exc.args else exc}"
            continue
        if not isinstance(obj, dict):
            last_error = f"top level is {type(obj).__name__}, expected an object"
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in obj]
        if missing:
            # Not repaired: a missing field is the model's failure, not the transport's.
            return ParseResult(False, reason=f"missing required field(s): {missing}",
                               repairs=repairs, raw=text)
        return ParseResult(True, record=obj, repairs=repairs, raw=text)

    return ParseResult(False, reason=last_error, raw=text)


def coerce_boxes(record: dict[str, Any]) -> list[list[float]]:
    """Evidence boxes as a plain list, dropping only entries that are not boxes at all.

    Clamping to the official evaluator's 0–999 range happens at emit time
    (`clamp_for_official_evaluator`, `DECISIONS.md` 0004), not here — this reports what
    the model said.
    """
    out: list[list[float]] = []
    for item in record.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        box = item.get("bbox")
        if not isinstance(box, Sequence) or isinstance(box, str) or len(box) != 4:
            continue
        try:
            out.append([float(v) for v in box])
        except (TypeError, ValueError):
            continue
    return out


def answer_of(record: dict[str, Any]) -> str:
    """The model's answer string, without substituting anything for a missing one."""
    value = record.get("model_answer")
    return "" if value is None else str(value)


def schema_ok(record: dict[str, Any]) -> tuple[bool, str]:
    """Does the record satisfy `OUTPUT_SCHEMA`, not merely parse as JSON?

    These come apart, and the gap is not academic. A probe produced this, which parses
    cleanly and is still unusable::

        {"plan": {"op": "lookup", "args": {"label": "Zara", "value": 99}}}

    `args` must be an array; the executor rejects an object. Counting that as a success
    would report a validity rate the pipeline cannot actually act on — and non-negotiable
    rule 3 makes a record the executor rejects a **failure**.
    """
    from chartqa_dt.plans.schema import validate_record

    result = validate_record(record)
    return result.ok, "; ".join(result.errors[:2])


@dataclass
class ParseStats:
    """Aggregate parse health, which `PLAN.md` 5.2 gates the variant choice on.

    Tracks two rates, because they measure different things: `valid` is "this is JSON
    with the right keys", `schema_valid` is "the executor will accept it". The second is
    the one that matters.
    """

    total: int = 0
    valid: int = 0
    schema_valid: int = 0
    repaired: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    schema_reasons: dict[str, int] = field(default_factory=dict)

    def add(self, result: ParseResult) -> None:
        self.total += 1
        if result.ok:
            self.valid += 1
            self.repaired += result.repaired
            ok, why = schema_ok(result.record or {})
            if ok:
                self.schema_valid += 1
            else:
                key = (why or "schema").split(":")[0][:60]
                self.schema_reasons[key] = self.schema_reasons.get(key, 0) + 1
        else:
            key = result.reason.split(":")[0]
            self.reasons[key] = self.reasons.get(key, 0) + 1

    @property
    def schema_valid_fraction(self) -> float:
        return self.schema_valid / self.total if self.total else 0.0

    @property
    def valid_fraction(self) -> float:
        return self.valid / self.total if self.total else 0.0

    @property
    def repaired_fraction(self) -> float:
        return self.repaired / self.valid if self.valid else 0.0

    def describe(self) -> str:
        lines = [f"  valid JSON      : {self.valid}/{self.total} "
                 f"({100 * self.valid_fraction:.1f}%)",
                 f"  schema-valid    : {self.schema_valid}/{self.total} "
                 f"({100 * self.schema_valid_fraction:.1f}%)  <- the gate that matters",
                 f"  needed a repair : {self.repaired} "
                 f"({100 * self.repaired_fraction:.1f}% of valid)"]
        for reason, n in sorted(self.reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"    parse   {n:>5}  {reason}")
        for reason, n in sorted(self.schema_reasons.items(), key=lambda kv: -kv[1]):
            lines.append(f"    schema  {n:>5}  {reason}")
        return "\n".join(lines)


__all__ = ["REQUIRED_FIELDS", "ParseResult", "ParseStats", "answer_of",
           "coerce_boxes", "parse_record", "schema_ok"]
