from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chartqa_dt.eval.metrics import relaxed_correctness, to_float
from chartqa_dt.plans.executor import MAX_FACTS, EvidenceItem, execute, parse_numeric

RELAXED_TOLERANCE = 0.05


def gold_tolerance(target: Any) -> float:
    text = str(target).strip().replace(",", "").replace("$", "").rstrip("%").strip()
    frac = text.split(".")[1] if "." in text else ""
    digits = len(frac.rstrip())
    return 0.5 * (10.0 ** -digits)


def matches_gold(value: Any, target: Any) -> bool:
    v, t = parse_numeric(value), parse_numeric(target)
    if v is None or t is None:
        return False
    tolerance = gold_tolerance(target)
    epsilon = 1e-12 * max(abs(v), abs(t), 1.0)
    return abs(v - t) <= tolerance + epsilon


@dataclass
class RoundTrip:
    outcome: str
    executed: Any = None
    stated: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.outcome == "agrees"


def check_record(record: dict[str, Any], *, strict: bool = False,
                 max_facts: int = MAX_FACTS) -> RoundTrip:
    plan = record.get("plan")
    stated = str(record.get("model_answer", ""))
    if not isinstance(plan, dict) or not plan.get("op"):
        return RoundTrip("no_plan", stated=stated)

    evidence = [
        EvidenceItem(
            str(fact.get("label")), fact.get("value"), fact.get("unit"),
            str(fact.get("id")),
        )
        for fact in (record.get("facts") or []) if isinstance(fact, dict)
    ]
    try:
        got = execute(plan, evidence, max_facts=max_facts)
    except Exception as exc:
        return RoundTrip("raises", stated=stated, error=f"{type(exc).__name__}: {exc}")

    agrees = answers_agree_at_gold_precision(stated, got) if strict else answers_agree(stated, got)
    return RoundTrip("agrees" if agrees else "disagrees", executed=got, stated=stated)


def answers_agree_at_gold_precision(stated: str, got: Any) -> bool:
    if got is None:
        return False
    if isinstance(got, bool):
        return stated.strip().lower() == ("yes" if got else "no")
    if isinstance(got, str):
        return got.strip().lower() == stated.strip().lower()
    return matches_gold(got, stated)


def answers_agree(stated: str, got: Any) -> bool:
    if got is None:
        return relaxed_correctness(stated, "")
    a, b = to_float(stated), to_float(str(got))
    if a is None or b is None:
        return relaxed_correctness(stated, str(got))
    scale = max(abs(a), abs(b))
    return scale == 0.0 or abs(a - b) <= RELAXED_TOLERANCE * scale
