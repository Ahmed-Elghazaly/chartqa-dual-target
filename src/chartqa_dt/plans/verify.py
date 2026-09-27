from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from chartqa_dt.plans.executor import (
    FOLD_OPS,
    MAX_DEPTH,
    MAX_FACTS,
    OPS,
    EvidenceItem,
    execute,
    parse_numeric,
    plan_depth,
    plan_fact_ids,
    validate_plan_shape,
)
from chartqa_dt.plans.roundtrip import matches_gold

OK = "accepted"
BAD_SHAPE = "rejected:malformed_plan"
BAD_OP = "rejected:unknown_operation"
TOO_DEEP = "rejected:too_deep"
TOO_MANY_ARGS = "rejected:too_many_arguments"
UNKNOWN_LABEL = "rejected:operand_not_in_evidence"
RAISES = "rejected:executor_refused"
WRONG_ANSWER = "rejected:does_not_reproduce_the_answer"
WRONG_OPERANDS = "rejected:does_not_cover_the_marked_regions"
MISSING_MARKINGS = "rejected:missing_marked_fact_ids"
TOO_MUCH_EVIDENCE = "rejected:needs_more_evidence_than_the_schema_allows"
VACUOUS_FOLD = "rejected:vacuous_single_item_fold"
UNGROUNDED_THRESHOLD = "rejected:threshold_not_stated_in_the_question"

_QUESTION_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def plan_thresholds(plan: Any) -> list[Any]:
    if not isinstance(plan, dict):
        return []
    found = [plan["threshold"]] if "threshold" in plan else []
    for arg in plan.get("args") or []:
        found.extend(plan_thresholds(arg))
    return found


def question_numbers(question: str) -> set[float]:
    out: set[float] = set()
    for token in _QUESTION_NUMBER.findall(question):
        number = parse_numeric(token)
        if number is not None:
            out.add(number)
    return out


@dataclass
class Verdict:
    status: str
    plan: dict[str, Any] | None = None
    executed: Any = None
    detail: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == OK


def _shape_ok(plan: Any) -> tuple[bool, str]:
    if not isinstance(plan, dict) or not isinstance(plan.get("op"), str):
        return False, BAD_SHAPE
    if plan.get("op") not in OPS:
        return False, BAD_OP
    if plan_depth(plan) > MAX_DEPTH:
        return False, TOO_DEEP
    args = plan.get("args")
    if isinstance(args, list) and len(args) > MAX_FACTS:
        return False, TOO_MANY_ARGS
    if validate_plan_shape(plan):
        return False, BAD_SHAPE
    return True, OK


def verify(
    plan: Any,
    *,
    answer: Any,
    question: str | None = None,
    evidence: Sequence[dict[str, Any]],
    marked_fact_ids: set[str] | None = None,
    marking_required: bool = False,
) -> Verdict:
    args = plan.get("args") if isinstance(plan, dict) else None
    if (isinstance(plan, dict) and plan.get("op") in FOLD_OPS and isinstance(args, list)
            and len(args) < 2 and all(isinstance(arg, dict) for arg in args)):
        return Verdict(VACUOUS_FOLD, plan=plan,
                       detail="folds, extrema, rank, count, and trend require at least two explicit facts")
    ok, why = _shape_ok(plan)
    if not ok:
        return Verdict(why, detail=f"plan={plan!r}")
    thresholds = plan_thresholds(plan)
    if thresholds:
        stated = question_numbers(question) if isinstance(question, str) else set()
        unstated = [value for value in thresholds if float(value) not in stated]
        if unstated:
            return Verdict(UNGROUNDED_THRESHOLD, plan=plan,
                           detail=f"threshold {unstated[0]!r} is not a number stated in the question")

    items = [
        EvidenceItem(str(e.get("label")), e.get("value"), e.get("unit"), str(e.get("id") or f"f{index}"))
        for index, e in enumerate(evidence, start=1)
    ]
    if marking_required and marked_fact_ids is None:
        return Verdict(MISSING_MARKINGS, plan=plan,
                       detail="this grounded record has no marked fact ids")
    used = set(plan_fact_ids(plan))
    if len(used) > MAX_FACTS:
        return Verdict(TOO_MUCH_EVIDENCE, plan=plan,
                       detail=f"the plan needs {len(used)} facts, over the limit of {MAX_FACTS}")
    missing = sorted(used - {str(item.fact_id) for item in items})
    if missing:
        return Verdict(UNKNOWN_LABEL, plan=plan, detail=f"{missing[0]!r} is not in evidence")

    try:
        got = execute(plan, items)
    except Exception as exc:
        return Verdict(RAISES, plan=plan, detail=f"{type(exc).__name__}: {exc}")

    if isinstance(got, bool):
        agrees = str(answer).strip().lower() == ("yes" if got else "no")
    elif isinstance(got, str):
        agrees = got.strip().lower() == str(answer).strip().lower()
    else:
        agrees = got is not None and matches_gold(got, answer)
    if not agrees:
        return Verdict(WRONG_ANSWER, plan=plan, executed=got,
                       detail=f"executed {got!r} against answer {answer!r}")

    if marked_fact_ids is not None and not marked_fact_ids and used:
        return Verdict(WRONG_OPERANDS, plan=plan, executed=got,
                       detail="the record explicitly marks no facts for a non-empty plan")
    if marked_fact_ids is not None and not marked_fact_ids <= used:
        omitted = sorted(marked_fact_ids - used)
        return Verdict(WRONG_OPERANDS, plan=plan, executed=got,
                       detail=f"plan does not consume marked region {omitted[0]!r}")
    return Verdict(OK, plan=plan, executed=got)
