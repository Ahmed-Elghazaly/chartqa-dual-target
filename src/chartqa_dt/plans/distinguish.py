"""Telling apart plans that agree on this chart but mean different things.

**The spurious-program problem.** Weakly supervised semantic parsing calls a program
*spurious* when it has the wrong semantics but the right denotation — it reaches the gold
answer by the wrong route. `DECISIONS.md` 0080 hit exactly this and could not solve it:

> *"Where the marked evidence has one element, `argmax`, `argmin` and `lookup` all trivially
> return it, so arithmetic verification cannot distinguish them. A careless teacher scores
> 100% here while being semantically wrong three times."*

Every gate in `plans.llm_mining` runs the plan on **one** input — the record's own evidence —
so a plan that happens to coincide with the truth on that input passes.

**The fix comes from the literature.** Lee, Kim and Jung (EMNLP 2023), *Weakly Supervised
Semantic Parsing with Execution-based Spurious Program Filtering*, build a program's semantic
representation by executing it **under various inputs** and compare programs by those results.
Two programs with different semantics diverge somewhere, even when they agree on the gold
input.

Applied here, the question is not *"which of two plans is right"* but the sharper one:
**does this chart's evidence contain enough information to tell the proposed plan apart from a
different reading of the same question?** If it does not, accepting the proposal teaches an
arbitrary choice, and refusing costs a record we could never have got right.

**Perturbation by shuffling, not by scaling.** Values are permuted among the labels. That
keeps every number the chart actually contains — so units, magnitudes and any executor guard
still behave — while breaking the label-to-value association, which is precisely what
separates `lookup("Nigeria")` from `max()`. Multiplying values instead would leave the
ordering intact, and ordering is what the extrema read.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence
from typing import Any

from chartqa_dt.plans.executor import EvidenceItem, execute

#: How many permutations to fingerprint over. Eight separates the operations that actually
#: collide on ChartQA — `lookup` against the extrema and the folds — while staying cheap
#: enough to run on every proposal. A plan is only ever compared with plans fingerprinted in
#: the same call, so the number matters less than that it is fixed and seeded.
TRIALS = 8

#: The readings a question is most often torn between, and the ones 0080 found colliding.
#: Each is built from the operands the proposal already uses, so an alternative is always
#: executable wherever the proposal is.
RIVAL_OPS: tuple[str, ...] = ("lookup", "argmax", "argmin", "max", "min", "mean", "sum",
                              "median", "count")


def fingerprint(plan: Any, evidence: Sequence[EvidenceItem], *,
                trials: int = TRIALS, seed: int = 0) -> tuple:
    """What this plan computes across permutations of the evidence — its semantics, sampled.

    A plan that raises on a permutation records that as part of its behaviour rather than
    being discarded: refusing to run on some inputs *is* a semantic difference, and two plans
    that refuse in different places are not the same plan.
    """
    rng = random.Random(seed)
    values = [e.value for e in evidence]
    out: list[Any] = []
    for _ in range(trials):
        shuffled = list(values)
        rng.shuffle(shuffled)
        perturbed = [EvidenceItem(e.label, v, e.unit)
                     for e, v in zip(evidence, shuffled)]
        try:
            out.append(("ok", execute(plan, perturbed)))
        except Exception as exc:                 # noqa: BLE001 — refusing is behaviour too
            out.append(("raises", type(exc).__name__))
    return tuple(out)


#: Operations whose meaning depends on WHICH operands were chosen, and in what order.
PAIRWISE: frozenset[str] = frozenset({"difference", "ratio", "percent_change"})

#: How many alternative operand pairs to test. The evidence a plan is verified against is a
#: whole chart, so enumerating every pair is quadratic and mostly wasted; this samples the
#: nearest alternatives, which is where a coincidence actually lives.
MAX_OPERAND_RIVALS = 12


def rivals_for(plan: Any, evidence: Sequence[EvidenceItem]) -> list[dict]:
    """Other readings of the same question, as different operations AND as different operands.

    **Both kinds matter, and only the first was obvious.** `Prompt.md` Idea 8 asks for special
    attention to the case where *one operation type is unique but multiple concrete programs
    of that type produce the same answer*, and measurement found it in **22.6%** of the
    records where the deterministic miner called the operation unique — one of them with 176
    operand pairs reaching the same number, and one computing a difference between a label and
    **itself** on a chart where that label repeats.

    `difference("Alpha","Beta")` and `difference("Gamma","Delta")` are different claims about
    which marks a question is about — but they are also genuinely different *functions*, so a
    fingerprint separates them and they never appear here. Operand ambiguity is a different
    question and `coincidences` answers it.
    """
    from chartqa_dt.plans.executor import plan_labels

    named = list(dict.fromkeys(plan_labels(plan))) if isinstance(plan, dict) else []
    out: list[dict] = []
    for op in RIVAL_OPS:
        if op == "lookup":
            out.extend({"op": "lookup", "args": [label]} for label in named[:4])
        else:
            out.append({"op": op, "args": []})

    return [r for r in out if r != plan]


def indistinguishable_from(plan: Any, evidence: Sequence[EvidenceItem], *,
                           trials: int = TRIALS, seed: int = 0) -> list[dict]:
    """Rival readings this evidence cannot tell apart from `plan`.

    A non-empty result means the record is **underdetermined**: the chart does not contain
    the information needed to know which reading the question meant, so a plan accepted here
    was accepted by luck.
    """
    if not isinstance(plan, dict) or not evidence:
        return []
    mine = fingerprint(plan, evidence, trials=trials, seed=seed)
    # A plan that raises on every permutation has no discriminating behaviour to compare;
    # calling everything indistinguishable from it would be noise, not a finding.
    if all(kind == "raises" for kind, _ in mine):
        return []
    return [rival for rival in rivals_for(plan, evidence)
            if fingerprint(rival, evidence, trials=trials, seed=seed) == mine]


def coincidences(plan: Any, evidence: Sequence[EvidenceItem], answer: Any) -> list[dict]:
    """Other operand choices of the SAME operation that also reach the gold answer.

    **A different question from `indistinguishable_from`, and both are needed.** That one asks
    *is this plan a different function from its rivals?* — and `difference("A","B")` genuinely
    is a different function from `difference("C","D")`, so a fingerprint separates them. This
    asks the question that actually threatens supervision: *on this chart, does another choice
    of operands land on the same answer?* If it does, the record cannot say which pair the
    question meant, and a plan accepted here was accepted by coincidence.

    `Prompt.md` Idea 8 asks for special attention to exactly this, and measurement found it in
    **22.6%** of the records where the deterministic miner called the operation unique — one
    with 176 pairs reaching the same number, and one taking a difference between a label and
    *itself* on a chart where that label repeats (`DECISIONS.md` 0106).

    Only the pairwise operations, because they are the ones where the choice is a claim. An
    empty list means the evidence determines the operands.
    """
    from chartqa_dt.plans.mining import matches_gold

    op = plan.get("op") if isinstance(plan, dict) else None
    if op not in PAIRWISE:
        return []
    named = [a for a in (plan.get("args") or []) if isinstance(a, str)]
    if len(named) != 2:
        return []

    out: list[dict] = []
    for a, b in itertools.permutations(evidence, 2):
        if len(out) >= MAX_OPERAND_RIVALS:
            break
        candidate = {"op": op, "args": [a.label, b.label]}
        if candidate == plan:
            continue
        try:
            got = execute(candidate, list(evidence))
        except Exception:                        # noqa: BLE001 — a refusal is not a rival
            continue
        if got is not None and matches_gold(got, answer):
            out.append(candidate)
    return out


__all__ = ["MAX_OPERAND_RIVALS", "PAIRWISE", "RIVAL_OPS", "TRIALS", "coincidences",
           "fingerprint", "indistinguishable_from", "rivals_for"]
