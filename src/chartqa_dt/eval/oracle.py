"""Oracle decomposition — `PLAN.md` 9.1. Where does the error actually come from?

Four configurations over the *same* records:

| evidence | plan | tells you |
|---|---|---|
| predicted | predicted | the real system |
| **gold** | predicted | how much error came from *seeing* |
| predicted | **gold** | how much error came from *reasoning* |
| gold | gold | the executor's own ceiling |

The design constraint that matters is the words *the same records*. A cell computed on a
different set is not comparable with the others, and the difference between two such cells
measures the sets rather than the substitution. Since the gold-plan rows only exist for
records that *have* a gold plan — most ChartQA rows do not (`DECISIONS.md` 0045 refuses to
guess one) — the eligible set is the intersection, and `decompose` reports its size beside
every cell so a reader can see what the table is about.

The second constraint is that a substitution can *break* a record: a predicted plan may
reference a label that the gold evidence does not contain, and then the executor refuses.
That is a real result — the plan does not fit the truth — and it is counted as a failure,
not skipped. Skipping it would make the gold-evidence column look better precisely where
the model was most wrong.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from chartqa_dt.eval.metrics import relaxed_correctness
from chartqa_dt.plans.executor import EvidenceItem, execute

#: The four cells, in the order `PLAN.md` 9.1 tabulates them.
CONFIGURATIONS = (
    ("pred_pred", False, False, "the real system"),
    ("gold_pred", True, False, "error from seeing"),
    ("pred_gold", False, True, "error from reasoning"),
    ("gold_gold", True, True, "the executor's own ceiling"),
)


@dataclass
class OracleItem:
    """One record with both its predicted and its gold halves."""

    record_id: str
    gold_answer: str
    pred_evidence: list[dict[str, Any]] = field(default_factory=list)
    gold_evidence: list[dict[str, Any]] = field(default_factory=list)
    pred_plan: dict | None = None
    gold_plan: dict | None = None

    @property
    def eligible(self) -> bool:
        """Usable in all four cells. Anything less and the table stops being a 2x2."""
        return bool(self.gold_plan and self.gold_evidence)


@dataclass
class CellResult:
    name: str
    tells_you: str
    n: int = 0
    correct: int = 0
    executor_refused: int = 0
    no_plan: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "tells_you": self.tells_you, "n": self.n,
                "correct": self.correct, "accuracy": self.accuracy,
                "executor_refused": self.executor_refused, "no_plan": self.no_plan}


def _items(evidence: Sequence[dict[str, Any]]) -> list[EvidenceItem]:
    return [EvidenceItem(str(e.get("label")), e.get("value"), e.get("unit"))
            for e in evidence if isinstance(e, dict)]


def run_one(item: OracleItem, *, gold_evidence: bool, gold_plan: bool) -> tuple[bool, str]:
    """Execute one configuration. Returns whether it was correct and why not.

    A missing plan and a refusing executor are distinct outcomes and both are failures.
    """
    plan = item.gold_plan if gold_plan else item.pred_plan
    if not isinstance(plan, dict) or not plan.get("op"):
        return False, "no_plan"
    evidence = _items(item.gold_evidence if gold_evidence else item.pred_evidence)
    try:
        got = execute(plan, evidence)
    except Exception:                       # noqa: BLE001 — every refusal is one outcome
        return False, "executor_refused"
    return relaxed_correctness(item.gold_answer, "" if got is None else str(got)), ""


def decompose(items: Sequence[OracleItem]) -> dict[str, Any]:
    """The whole table, computed on the records eligible for every cell."""
    eligible = [i for i in items if i.eligible]
    cells: dict[str, CellResult] = {}
    for name, gold_e, gold_p, tells in CONFIGURATIONS:
        cell = CellResult(name=name, tells_you=tells, n=len(eligible))
        for item in eligible:
            ok, why = run_one(item, gold_evidence=gold_e, gold_plan=gold_p)
            cell.correct += ok
            if why == "executor_refused":
                cell.executor_refused += 1
            elif why == "no_plan":
                cell.no_plan += 1
        cells[name] = cell

    real, seeing = cells["pred_pred"], cells["gold_pred"]
    reasoning, ceiling = cells["pred_gold"], cells["gold_gold"]
    return {
        "n_eligible": len(eligible),
        "n_total": len(items),
        "n_excluded_no_gold_plan": sum(1 for i in items if not i.gold_plan),
        "cells": {name: cell.to_dict() for name, cell in cells.items()},
        # Each attribution is a difference between two cells over the same records, so it
        # is a like-for-like comparison rather than two separately-scored populations.
        "attribution": {
            "visual_error_points": 100 * (seeing.accuracy - real.accuracy),
            "reasoning_error_points": 100 * (reasoning.accuracy - real.accuracy),
            "executor_ceiling_pct": 100 * ceiling.accuracy,
        },
    }


def describe(result: dict[str, Any]) -> str:
    lines = [f"oracle decomposition over {result['n_eligible']:,} records "
             f"(of {result['n_total']:,}; "
             f"{result['n_excluded_no_gold_plan']:,} have no gold plan)",
             f"  {'evidence':<10}{'plan':<10}{'accuracy':>10}{'refused':>10}  tells you"]
    for name, gold_e, gold_p, _ in CONFIGURATIONS:
        cell = result["cells"][name]
        lines.append(f"  {'gold' if gold_e else 'pred':<10}"
                     f"{'gold' if gold_p else 'pred':<10}"
                     f"{100 * cell['accuracy']:>9.2f}%{cell['executor_refused']:>10}"
                     f"  {cell['tells_you']}")
    a = result["attribution"]
    lines.append(f"  visual error {a['visual_error_points']:+.2f} pts   "
                 f"reasoning error {a['reasoning_error_points']:+.2f} pts   "
                 f"executor ceiling {a['executor_ceiling_pct']:.2f}%")
    return "\n".join(lines)


__all__ = ["CONFIGURATIONS", "CellResult", "OracleItem", "decompose", "describe", "run_one"]
