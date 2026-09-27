from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from chartqa_dt.data.colours import describe_colour
from chartqa_dt.data.records import ChartRecord
from chartqa_dt.plans.executor import MAX_DEPTH, OPS
from chartqa_dt.plans.facts import facts_for_record, marked_fact_ids
from chartqa_dt.plans.verify import Verdict, verify

SIGNATURES: dict[str, str] = {
    "lookup": "lookup(ref) -> the value of that fact",
    "label_of": "label_of(ref) -> the label of that fact",
    "count": "count(ref, ref, ...) -> how many explicit facts were listed",
    "sum": "sum(a, b, ...) -> sum of 2-12 facts or numeric operations",
    "mean": "mean(a, b, ...) -> mean of 2-12 facts or numeric operations",
    "median": "median(ref, ref, ...) -> median of 2-12 explicit facts",
    "min": "min(ref, ref, ...) -> the smallest VALUE",
    "max": "max(ref, ref, ...) -> the largest VALUE",
    "argmin": "argmin(ref, ref, ...) -> LABEL of the unique smallest fact",
    "argmax": "argmax(ref, ref, ...) -> LABEL of the unique largest fact",
    "difference": "difference(a, b) -> a - b",
    "absolute_difference": "absolute_difference(a, b) -> |a - b|",
    "ratio": "ratio(a, b) -> a / b",
    "percentage": "percentage(a, b) -> 100 * a / b",
    "percent_change": "percent_change(a, b) -> 100 * (a - b) / b",
    "compare": "compare(a, b) -> 'greater' | 'less' | 'equal'",
    "greater_than": "greater_than(a, b) -> 'Yes' or 'No'",
    "less_than": "less_than(a, b) -> 'Yes' or 'No'",
    "equal_to": "equal_to(a, b) -> 'Yes' or 'No'",
    "trend": "trend(ref, ref, ...) -> increasing/decreasing/flat/mixed in listed order",
    "rank": "rank(refs; position=N, order=ascending|descending) -> selected fact label",
    "rank_value": "rank_value(refs; position=N, order=ascending|descending) -> VALUE of that fact",
    "count_where": "count_where(refs; compare, threshold) -> how many listed facts pass",
    "sum_where": "sum_where(refs; compare, threshold) -> sum of the listed facts that pass",
    "mean_where": "mean_where(refs; compare, threshold) -> mean of the listed facts that pass",
    "label_where": "label_where(refs; compare, threshold) -> LABEL of the one fact that passes",
    "category_of": "category_of(ref or label operation) -> the category part of a 'series · category' label",
    "meets_threshold": "meets_threshold(a; compare, threshold) -> 'Yes' or 'No' for a number vs the threshold",
    "unanswerable": "unanswerable() -> the chart does not contain the answer",
}

OFFERED: tuple[str, ...] = tuple(sorted(OPS))

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

_THRESHOLD_FIELDS = """ rank_value takes the same "position" and \
"order". The *_where operations and meets_threshold emit "compare" (greater, \
greater_or_equal, less, less_or_equal, or equal) and a numeric "threshold": the threshold is \
the only number a plan may contain, and it must be a number written in the QUESTION itself, \
never one taken from the answer or the data."""


def _render_table(table: dict[str, Any] | None, *, max_rows: int = 20) -> str:
    if not table:
        return "(no table available)"
    cols = " | ".join(str(c) for c in table.get("columns") or [])
    rows = table.get("rows") or []
    body = "\n".join(" | ".join(str(c) for c in r) for r in rows[:max_rows])
    more = f"\n… and {len(rows) - max_rows} further rows" if len(rows) > max_rows else ""
    return f"{cols}\n{body}{more}"


def build_system() -> str:
    ops = "\n".join(f"  - {SIGNATURES[o]}" for o in OFFERED if o in SIGNATURES)
    return f"""You are given a chart question, the chart's underlying data, and the correct \
answer. Write the small program that computes that answer from the data.

OPERATIONS you may use, and nothing else:
{ops}

Reply with a JSON object in a ```json block:
  {{"op": "<operation>", "args": [{{"ref":"f1"}}, ...]}}
Every operand is an explicit fact reference or a nested numeric operation. Raw label
strings, raw numbers, booleans and null are forbidden. Folds/extrema/rank/count/trend name
2-12 fact references explicitly; an empty or one-item fold is invalid. Nesting may go at
most {MAX_DEPTH} levels deep. For rank also emit integer "position" and "order" as
"ascending" or "descending".{_THRESHOLD_FIELDS}

Choose the operation the QUESTION asks for, not merely one that reaches the right number. \
If the question names a specific item, that is a lookup even when the item happens to be the \
largest. If the question asks which item is largest, that is argmax even when you could name \
it directly.

Never reverse-engineer a colour, series, category, or operand mapping from the gold answer or
from arithmetic coincidence. The plan must remain semantically justified if the gold answer
were hidden. A context field is needed only when its absence could change the correct
interpretation; do not request a title or image when the supplied facts already determine the
question. If the chart is understandable but the requested output dimension (for example a
fact's colour or series name) is not expressible, use needs_operator instead of needs_context.

If no combination of these operations expresses what the question asks, do NOT force a fit.
You have three ways to say so, and the difference matters:

  * No safe plan can be supplied — the question is ambiguous, the gold answer contradicts the
    data, the chart does not contain what it asks for, or its required explicit scope exceeds
    the 12-fact safety cap:
    ```json
    {{"refused": "<one short reason>"}}
    ```
  * The question is perfectly answerable, but the operation it needs is missing from the list
    above. Say what that operation would be:
    ```json
    {{"needs_operator": {{"name": "<short_snake_case>", \
"signature": "<name(args) -> result>", "why": "<what this question needs it for>"}}}}
    ```

  * The question may be answerable, but the supplied structured context is missing something
    you need to decide (for example a title, axis, legend, annotation, image, missing value,
    incomplete comparison set, or ambiguous output dimension):
    ```json
    {{"needs_context": {{"kind": "title|axis|legend|annotation|image|missing_value|incomplete_facts|dimension|other", \
"why": "<what is missing and why it matters>"}}}}
    ```

Refusing is a correct answer, and naming a missing operation or context field is a *useful*
one — those suggestions are collected and ranked. A needs_context reply is diagnostic only;
it is never training supervision. A plan that reaches the right number by the wrong route is
worse than any of these outcomes, because it will teach the model the wrong reasoning."""


def build_prompt(
    *,
    question: str,
    answer: Any,
    table: dict[str, Any] | None,
    evidence: list[dict[str, Any]],
    marked_fact_ids: set[str] | None = None,
) -> str:
    normalised = [
        {**entry, "id": str(entry.get("id") or f"f{index}")} for index, entry in enumerate(evidence, start=1)
    ]
    items = (
        "\n".join(
            f"  - {e['id']}: {e.get('label')!r} = {e.get('value')!r}"
            + (f" {e['unit']}" if e.get("unit") else "")
            + (f"   [{describe_colour(e['colour'])}]" if e.get("colour") else "")
            + ("   [MARKED]" if marked_fact_ids and e["id"] in marked_fact_ids else "")
            for e in normalised
        )
        or "  (none)"
    )
    marked_note = (
        "\nSome items are tagged [MARKED]. A human annotator marked those regions as the "
        "official question-grounding target. Your plan must consume every marked fact. It "
        "may also consume explicit unmarked context facts when the requested operation needs "
        "a comparison set (for example argmax, argmin, rank, trend, or an aggregate). Never "
        "omit a marked fact and never use an unrelated distractor.\n"
        if marked_fact_ids
        else ""
    )

    return f"""QUESTION: {question}
CORRECT ANSWER: {answer!r}

CHART DATA:
{_render_table(table)}

FACTS you may reference by id (use these ids exactly):
{items}
{marked_note}"""


@dataclass
class Reply:
    plan: dict[str, Any] | None = None
    refused: bool = False
    needs_operator: dict[str, Any] | None = None
    needs_context: dict[str, str] | None = None
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.plan is not None


def parse_reply(text: str) -> Reply:
    blocks = _JSON_BLOCK.findall(text or "")
    candidate = blocks[-1] if blocks else (text or "")
    try:
        obj = json.loads(candidate.strip())
    except (json.JSONDecodeError, ValueError):
        return Reply(note="reply did not contain parsable JSON")
    if not isinstance(obj, dict):
        return Reply(note="reply was not a JSON object")
    signals = [name for name in ("op", "refused", "needs_operator", "needs_context") if name in obj]
    if len(signals) != 1:
        return Reply(note="reply must contain exactly one outcome")
    if "needs_operator" in obj:
        ask = obj["needs_operator"]
        if not isinstance(ask, dict) or not str(ask.get("name", "")).strip():
            return Reply(note="needs_operator without a name")
        return Reply(
            needs_operator={
                "name": str(ask.get("name"))[:64],
                "signature": str(ask.get("signature", ""))[:200],
                "why": str(ask.get("why", ""))[:300],
            }
        )
    if "needs_context" in obj:
        ask = obj["needs_context"]
        allowed = {
            "title",
            "axis",
            "legend",
            "annotation",
            "image",
            "missing_value",
            "incomplete_facts",
            "dimension",
            "other",
        }
        if not isinstance(ask, dict) or set(ask) != {"kind", "why"}:
            return Reply(note="needs_context requires exactly `kind` and `why`")
        kind = ask["kind"]
        why = ask["why"]
        if kind not in allowed or not isinstance(why, str) or not why.strip():
            return Reply(note="needs_context has an unsupported kind or empty reason")
        return Reply(needs_context={"kind": kind, "why": why.strip()[:300]})
    if "refused" in obj:
        reason = obj["refused"]
        if not isinstance(reason, str) or not reason.strip():
            return Reply(note="refused requires a non-empty reason")
        return Reply(refused=True, note=reason.strip()[:200])
    return Reply(plan=obj)


def request_for(record: ChartRecord) -> dict[str, str]:
    return {
        "custom_id": record.record_id,
        "user": build_prompt(
            question=record.question,
            answer=record.answer,
            table=record.table,
            evidence=facts_for_record(record),
            marked_fact_ids=marked_fact_ids(record),
        ),
    }


def score_reply(record: ChartRecord, raw: str) -> tuple[str, dict[str, Any] | None]:
    if not raw.strip():
        return "missing_response", None
    reply = parse_reply(raw)
    if reply.needs_operator:
        return "needs_operator", None
    if reply.needs_context:
        return "needs_context", None
    if reply.refused:
        return "teacher_refused", None
    if reply.plan is None:
        return "malformed_response", None
    verdict: Verdict = verify(
        reply.plan,
        answer=record.answer,
        question=record.question,
        evidence=facts_for_record(record),
        marked_fact_ids=marked_fact_ids(record),
        marking_required=record.evidence is not None,
    )
    return verdict.status, verdict.plan if verdict.accepted else None


_AGGREGATE = re.compile(r"\b(across all|maximum|minimum|highest|lowest|largest|smallest|biggest|greatest|"
                        r"difference|median|(second|third|fourth) (largest|highest|smallest|lowest|biggest|"
                        r"greatest|most|least))\b", re.I)
_THRESHOLD_WORDED = re.compile(
    r"\b(larger|greater|more|less|smaller|fewer|higher|lower|above|below|over|under|exceed\w*)\b"
    r"(\s+than)?\s+-?[\d.,]+", re.I)
_AGGREGATE_WORDED = re.compile(r"\b(average|mean|sum|total|how many|count|number of|which)\b", re.I)
_NAMES_A_POINT = re.compile(r"\b(1[89]\d\d|20\d\d|january|february|march|april|may|june|july|august|"
                            r"september|october|november|december|q[1-4])\b", re.I)


def question_of(user: str) -> str:
    for line in user.splitlines():
        if line.startswith("QUESTION:"):
            return line[9:].strip()
    return ""


def _has_threshold(plan: object) -> bool:
    return isinstance(plan, dict) and ("threshold" in plan or any(_has_threshold(a) for a in plan.get("args") or []))


def _is_shortcut(plan: dict[str, Any]) -> bool:
    args = plan.get("args") or []
    return plan.get("op") in ("lookup", "label_of") and len(args) == 1 and set(args[0]) == {"ref"}


def quarantine_reason(plan: dict[str, Any], question: str) -> str | None:
    if _is_shortcut(plan) and _AGGREGATE.search(question) and not _NAMES_A_POINT.search(question):
        return "shortcut"
    if not _has_threshold(plan) and _THRESHOLD_WORDED.search(question) and _AGGREGATE_WORDED.search(question):
        return "hidden_filter"
    return None
