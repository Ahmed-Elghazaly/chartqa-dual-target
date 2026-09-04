"""LLM-proposed plans, and the verification every one of them must survive.

`Prompt.md` Idea 7B. The deterministic miner is **94% precise and 19% recall**
(`DECISIONS.md` 0078): it accepts little, and its dominant failure is ambiguity — several
operations reproduce the gold answer and it cannot tell which the question meant, *because
it never reads the question*. Even handed gold operands it settles only 17.7% (0079), and
the remainder are questions whose wording names the operation outright.

So a language model is asked to choose, and is **never trusted**. This module is the
verifier, not the proposer: it defines what a proposal must survive before it becomes
supervision, and nothing here calls a model.

Every proposal passes five gates, in order, and each rejection is counted separately so the
teacher's failure profile is measurable rather than anecdotal:

1. **Shape** — parses, `op` is in `OPS`, depth ≤ `MAX_DEPTH`, arity within the schema.
2. **Grounded operands** — every bare string argument names an evidence item that exists.
3. **Executes** — the deterministic executor runs it without raising.
4. **Reproduces the gold answer** — at the precision the answer was written to
   (`mining.matches_gold`), not the 5% scoring tolerance, for the reason `DECISIONS.md` 0045
   records: 5% of the year 2014 is a century.
5. **Uses the marked regions** — where RefChartQA grounding exists, the operands must be the
   regions the annotation marks. This is the gate the deterministic miner cannot apply to
   itself, and it is what makes a teacher's proposal checkable against something other than
   arithmetic.

A proposal that fails any gate is **discarded**, never repaired. Repairing it would make the
pipeline the author of the supervision.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from chartqa_dt.plans.distinguish import coincidences, indistinguishable_from
from chartqa_dt.plans.executor import (
    MAX_DEPTH,
    OPS,
    EvidenceItem,
    execute,
    folds_over_evidence,
    plan_depth,
    plan_labels,
)
from chartqa_dt.plans.mining import matches_gold
from chartqa_dt.plans.schema import MAX_EVIDENCE

#: The schema's cap on a single node's argument list.
MAX_ARGS = 4

#: Verdicts, in the order the gates run. Kept as constants so counters cannot drift.
OK = "accepted"
BAD_SHAPE = "rejected:malformed_plan"
BAD_OP = "rejected:unknown_operation"
TOO_DEEP = "rejected:too_deep"
TOO_MANY_ARGS = "rejected:too_many_arguments"
UNKNOWN_LABEL = "rejected:operand_not_in_evidence"
RAISES = "rejected:executor_refused"
WRONG_ANSWER = "rejected:does_not_reproduce_the_answer"
WRONG_OPERANDS = "rejected:operands_outside_the_marked_regions"
TOO_MUCH_EVIDENCE = "rejected:needs_more_evidence_than_the_schema_allows"


@dataclass
class Verdict:
    status: str
    plan: dict | None = None
    executed: Any = None
    #: Rival readings this record's evidence cannot tell the plan apart from
    #: (`plans.distinguish`). Non-empty means the plan passed every arithmetic gate but the
    #: chart does not contain the information needed to know it was the reading intended —
    #: 0080's blind spot, now visible. Recorded on the verdict rather than rejected, so the
    #: cost of refusing them can be measured before anyone decides to (`DECISIONS.md` 0097).
    underdetermined: list[dict] = field(default_factory=list)
    #: Other operand choices of the SAME operation that also reach the gold answer. A
    #: different failure from `underdetermined`: the operation is certain and *which marks it
    #: is about* is not. Measured at 22.6% of the deterministic miner's "unique" verdicts
    #: (`DECISIONS.md` 0106), and invisible to every arithmetic gate because each coincidence
    #: reproduces the answer by definition.
    coincident_operands: list[dict] = field(default_factory=list)
    detail: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == OK


@dataclass
class VerificationStats:
    counts: dict[str, int] = field(default_factory=dict)
    #: Accepted plans whose record could not distinguish them from another reading.
    underdetermined: int = 0
    #: Accepted plans where another operand pair reaches the same answer.
    coincident: int = 0

    def note(self, status: str) -> None:
        self.counts[status] = self.counts.get(status, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def accepted(self) -> int:
        return self.counts.get(OK, 0)

    @property
    def precision(self) -> float:
        """Share of proposals that survive every gate. Not the model's *accuracy* — a
        proposal can be rejected for proposing nothing usable as well as for being wrong."""
        return self.accepted / self.total if self.total else 0.0

    def describe(self) -> str:
        lines = [f"  proposals verified : {self.total}",
                 f"  accepted           : {self.accepted} ({100 * self.precision:.1f}%)"]
        for status, n in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            if status != OK:
                lines.append(f"    {status:<46}{n:>5}")
        if self.underdetermined:
            lines.append(f"    {'of the accepted, underdetermined':<46}"
                         f"{self.underdetermined:>5}   <-- passed every gate, but the "
                         f"evidence cannot tell them from another reading")
        if self.coincident:
            lines.append(f"    {'of the accepted, operands coincide':<46}"
                         f"{self.coincident:>5}   <-- another operand pair reaches the "
                         f"same answer")
        return "\n".join(lines)


def _labels_in(plan: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(plan, dict):
        return out
    for arg in plan.get("args") or []:
        if isinstance(arg, str):
            out.append(arg)
        elif isinstance(arg, dict):
            out.extend(_labels_in(arg))
    return out


def _shape_ok(plan: Any) -> tuple[bool, str]:
    if not isinstance(plan, dict) or not isinstance(plan.get("op"), str):
        return False, BAD_SHAPE
    stack = [plan]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        if node.get("op") not in OPS:
            return False, BAD_OP
        args = node.get("args")
        if args is None or not isinstance(args, list):
            return False, BAD_SHAPE
        if len(args) > MAX_ARGS:
            return False, TOO_MANY_ARGS
        stack.extend(a for a in args if isinstance(a, dict))
    if plan_depth(plan) > MAX_DEPTH:
        return False, TOO_DEEP
    return True, OK


def verify(plan: Any, *, answer: Any, evidence: Sequence[dict[str, Any]],
           marked_labels: set[str] | None = None) -> Verdict:
    """Run every gate over one proposal. Never repairs, never partially accepts."""
    ok, why = _shape_ok(plan)
    if not ok:
        return Verdict(why, detail=f"plan={plan!r}")

    items = [EvidenceItem(str(e.get("label")), e.get("value"), e.get("unit"))
             for e in evidence]
    # The cap applies to the evidence the TARGET will carry, which `train.targets`
    # selects as the items the plan names -- not to the pool of candidates it chose from.
    # Applying it to the pool rejected `lookup('2019')` on any chart with more than eight
    # elements, which is 64.4% of ChartQA, and reported it as a malformed plan. Only a
    # plan that folds over everything needs the whole chart, and for those the cap is real.
    needed = len(items) if folds_over_evidence(plan) else len(set(plan_labels(plan)))
    if needed > MAX_EVIDENCE:
        return Verdict(TOO_MUCH_EVIDENCE, plan=plan,
                       detail=f"the plan needs {needed} evidence items, over the "
                              f"schema's {MAX_EVIDENCE}")
    known = {i.label for i in items}
    used = set(_labels_in(plan))
    missing = sorted(used - known)
    if missing:
        return Verdict(UNKNOWN_LABEL, plan=plan, detail=f"{missing[0]!r} is not in evidence")

    try:
        got = execute(plan, items)
    except Exception as exc:                       # noqa: BLE001 — any refusal is one gate
        return Verdict(RAISES, plan=plan, detail=f"{type(exc).__name__}: {exc}")

    if isinstance(got, str):
        agrees = got.strip().lower() == str(answer).strip().lower()
    else:
        agrees = got is not None and matches_gold(got, answer)
    if not agrees:
        return Verdict(WRONG_ANSWER, plan=plan, executed=got,
                       detail=f"executed {got!r} against answer {answer!r}")

    # An EMPTY set means "this source has no grounding", not "nothing may be used". Passing
    # `set()` for an ungrounded ChartQA record put every operand outside the marked regions
    # and rejected all 25 correct proposals in the first end-to-end run.
    if marked_labels and used and not used <= marked_labels:
        outside = sorted(used - marked_labels)
        return Verdict(WRONG_OPERANDS, plan=plan, executed=got,
                       detail=f"{outside[0]!r} is not a marked region")

    return Verdict(OK, plan=plan, executed=got,
                   underdetermined=indistinguishable_from(plan, items),
                   coincident_operands=coincidences(plan, items, answer))


def plan_key(plan: Any) -> str:
    """A stable identity for a plan, so two samples of the same plan compare equal.

    Argument *order* is kept: `difference(a, b)` and `difference(b, a)` are different plans
    and only one of them reproduces the answer.
    """
    return json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class Consensus:
    """What K samples of the same record agreed on."""

    plan: dict | None = None
    votes: int = 0
    samples: int = 0
    distinct: int = 0

    @property
    def agreement(self) -> float:
        return self.votes / self.samples if self.samples else 0.0


def consensus(proposals: Sequence[Any], *, answer: Any,
              evidence: Sequence[dict[str, Any]],
              marked_labels: set[str] | None = None,
              threshold: float = 0.5) -> Consensus:
    """Pick the plan that K independent samples agree on, among those that verify.

    **Why sample at all, when there is already a verifier.** The gates are arithmetic: they
    settle whether a plan *computes* the answer, and cannot settle which reading was meant
    when several compute it (`DECISIONS.md` 0080, 0097). Self-consistency is the standard
    answer to exactly that — correct reasoning paths converge while wrong ones spray — and it
    supplies the one thing the evidence cannot: what the reader *repeatedly* thought the
    question was asking.

    Verification runs **first**, so a plan that does not reproduce the answer cannot win a
    vote by being popular. Among the survivors, the most frequent plan wins if it clears
    `threshold`; a tie or a scattered vote returns no plan, because a reader that answers
    differently every time has told us it does not know.

    **The threshold's denominator is every sample, not every surviving sample**, and the
    difference matters. Three samples that fail arithmetic and one that passes gives 1/4, not
    1/1, so the survivor does not win. A reader that miscomputes this record three times out
    of four has said something about its grasp of it, and the one time it happened to be
    right is not evidence to the contrary. The looser denominator would let a single lucky
    sample carry a record that K-fold sampling was bought to protect.

    This is worth its K-fold cost only where a single sample is not enough — the records
    `Verdict.underdetermined` flags — not on every record.
    """
    verdicts, _ = verify_many([
        {"plan": p, "answer": answer, "evidence": evidence, "marked_labels": marked_labels}
        for p in proposals])
    kept = [v.plan for v in verdicts if v.accepted]
    if not kept:
        return Consensus(samples=len(proposals))
    counts: dict[str, int] = {}
    first: dict[str, dict] = {}
    for plan in kept:
        key = plan_key(plan)
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, plan)
    top = max(counts.values())
    winners = [k for k, n in counts.items() if n == top]
    result = Consensus(samples=len(proposals), votes=top, distinct=len(counts))
    if len(winners) == 1 and top / len(proposals) >= threshold:
        result.plan = first[winners[0]]
    return result


def verify_many(proposals: Sequence[dict[str, Any]]) -> tuple[list[Verdict], VerificationStats]:
    """Verify a batch. Each proposal supplies `plan`, `answer`, `evidence`, `marked_labels`."""
    stats = VerificationStats()
    out: list[Verdict] = []
    for p in proposals:
        verdict = verify(p.get("plan"), answer=p.get("answer"),
                         evidence=p.get("evidence") or [],
                         marked_labels=p.get("marked_labels"))
        stats.note(verdict.status)
        if verdict.underdetermined:
            stats.underdetermined += 1
        if verdict.coincident_operands:
            stats.coincident += 1
        out.append(verdict)
    return out, stats


__all__ = [
    "BAD_OP",
    "BAD_SHAPE",
    "MAX_ARGS",
    "OK",
    "RAISES",
    "TOO_DEEP",
    "TOO_MANY_ARGS",
    "TOO_MUCH_EVIDENCE",
    "UNKNOWN_LABEL",
    "WRONG_ANSWER",
    "WRONG_OPERANDS",
    "Consensus",
    "Verdict",
    "VerificationStats",
    "consensus",
    "plan_key",
    "verify",
    "verify_many",
]
