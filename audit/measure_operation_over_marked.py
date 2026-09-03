#!/usr/bin/env python3
"""AUDIT · if the operands are gold, how hard is it to find the OPERATION?

`mine_plan` searches the whole table, so it must find the operation *and* the operands, and
45.7% of the time several combinations reproduce the answer — ambiguity, its dominant
failure.

But RefChartQA already tells us the operands: the regions it marks. Searching only over
those collapses the problem. With two marked bars there are a handful of operations, and
usually at most one gives the answer.

This measures how much of the 1,443 plan-less-but-grounded backlog that recovers, and how
often the result is still ambiguous.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.data.records import ELEMENTS_KEY  # noqa: E402
from chartqa_dt.env import get_env  # noqa: E402
from chartqa_dt.plans.executor import EvidenceItem, execute  # noqa: E402
from chartqa_dt.plans.mining import matches_gold  # noqa: E402
from chartqa_dt.train.targets import TargetError, build_target  # noqa: E402

#: Operations worth trying over an already-correct operand set. `lookup` is covered by the
#: single-box derivation; the rest are the ways several marks combine into one answer.
CANDIDATE_OPS = ["sum", "mean", "difference", "ratio", "percent_change", "min", "max",
                 "count", "median"]
#: Operations whose result is a LABEL, for questions like "in which year was it highest?"
LABEL_OPS = ["argmax", "argmin"]


def operations_giving(answer, evidence: list[EvidenceItem]) -> list[dict]:
    """Every candidate plan over exactly this evidence that reproduces the answer."""
    hits: list[dict] = []
    labels = [e.label for e in evidence]
    for op in CANDIDATE_OPS:
        for args in ([], labels) if len(evidence) > 1 else ([labels[0]] if labels else [],):
            plan = {"op": op, "args": list(args) if isinstance(args, list) else [args]}
            try:
                got = execute(plan, evidence)
            except Exception:                          # noqa: BLE001
                continue
            if got is not None and matches_gold(got, answer):
                hits.append(plan)
                break
    for op in LABEL_OPS:
        plan = {"op": op, "args": []}
        try:
            got = execute(plan, evidence)
        except Exception:                              # noqa: BLE001
            continue
        if isinstance(got, str) and got.strip().lower() == str(answer).strip().lower():
            hits.append(plan)
    return hits


def main() -> int:
    from scripts.build_mixtures import refchartqa_records

    root = Path(get_env().data_root)
    records = refchartqa_records(cap=100000, cache=root / "refchartqa_train.jsonl")

    stats = collections.Counter()
    by_op = collections.Counter()
    examples = []
    for r in records:
        try:
            build_target(r)
            continue                                   # already supervised
        except TargetError as exc:
            if "no mined plan" not in str(exc):
                continue
        els = [e for e in (r.meta.get(ELEMENTS_KEY) or []) if isinstance(e, dict)]
        if not els or r.answer is None:
            stats["not_named"] += 1
            continue
        stats["candidates"] += 1
        evidence = [EvidenceItem(str(e.get("label")), e.get("value"), e.get("unit"))
                    for e in els]
        hits = operations_giving(r.answer, evidence)
        if not hits:
            stats["no_operation_fits"] += 1
        elif len(hits) == 1:
            stats["UNIQUE operation found"] += 1
            by_op[hits[0]["op"]] += 1
            if len(examples) < 6:
                examples.append({"q": r.question[:78], "answer": r.answer,
                                 "plan": hits[0],
                                 "marks": [(str(e.get("label")), e.get("value"))
                                           for e in els][:4]})
        else:
            stats["still ambiguous"] += 1

    n = stats["candidates"]
    print(f"\nplan-less but semantically grounded records: {n:,}\n")
    for k in ("UNIQUE operation found", "still ambiguous", "no_operation_fits"):
        print(f"  {k:<26}{stats[k]:>6,}  ({100 * stats[k] / max(n, 1):5.1f}%)")
    print(f"\n  operations recovered: {dict(by_op.most_common())}")
    print("\nexamples:")
    for e in examples:
        print(f"\n  Q: {e['q']}\n     answer {e['answer']!r}  marks {e['marks']}"
              f"\n     -> {json.dumps(e['plan'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
