#!/usr/bin/env python3
"""EXPERIMENT · an LLM teacher's proposals on the calibration sample, verified.

The teacher here is Claude (this session), reading each record's question, marked regions
and gold answer, and proposing a plan in the project's DSL — or refusing. Proposals were
written before running the verifier, and the refusals are as much a result as the
acceptances: a teacher that proposes something for every record would score well on
arithmetic and badly on meaning.

Each entry records the decision and the reason, so a reader can disagree with a specific
judgement rather than with an aggregate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from chartqa_dt.plans.llm_mining import verify_many  # noqa: E402

#: index -> (plan or None, category, reason)
PROPOSALS: dict[int, tuple[dict | None, str, str]] = {
    # --- the question asks for a maximum or minimum, and the marked region is the answer
    **{i: ({"op": "argmin", "args": []}, "argmin", "question asks for the minimum")
       for i in (2, 4, 12, 29, 37)},
    **{i: ({"op": "argmax", "args": []}, "argmax", "question asks for the maximum")
       for i in (5, 6, 10, 13, 16, 18, 19, 25, 26, 32, 34, 35, 38)},
    3: ({"op": "argmin", "args": []}, "argmin", "'lowest share'"),
    # --- genuine aggregates over several marked regions
    22: ({"op": "mean", "args": []}, "aggregate", "'average ... per Characteristic'"),
    36: ({"op": "median", "args": []}, "aggregate", "'median Number of days'"),
    # --- REFUSED: the DSL cannot express a yes/no comparison.
    #     `compare` returns 'greater'/'less'/'equal'; `boolean` takes one argument.
    **dict.fromkeys((0, 1, 7, 11, 14, 17, 24, 30, 33), (None, "refused:no_yes_no_operator", "'is X less than Y' -> Yes/No is not expressible")),
    # --- REFUSED: the question asks for RANK 2, not the extremum. `argmax` would verify
    #     against a single marked region and be semantically wrong.
    **dict.fromkeys((15, 20, 21), (None, "refused:rank_not_extremum", "question asks for the SECOND highest")),
    # --- REFUSED: reverse lookup — given a value, name the label. No such operator.
    **dict.fromkeys((8, 23, 27, 31, 39), (None, "refused:reverse_lookup", "value is given, the LABEL is asked for")),
    # --- REFUSED: argmax over a COMPUTED quantity, and the marks share one label
    9: (None, "refused:argmax_over_computed", "max of a per-row difference; duplicate labels"),
    # --- REFUSED: the marked region carries no value
    28: (None, "refused:no_value", "marked region has value null"),
}


def main() -> int:
    rows = [json.loads(line) for line in
            Path("audit/llm_mining_sample.jsonl").read_text(encoding="utf-8").splitlines()
            if line]
    proposed, refused = [], []
    for i, row in enumerate(rows):
        plan, category, reason = PROPOSALS.get(i, (None, "refused:unclassified", ""))
        if plan is None:
            refused.append((i, category, reason))
            continue
        proposed.append({"index": i, "plan": plan, "answer": row["gold_answer"],
                         "evidence": row["marked_regions"],
                         "marked_labels": {m["label"] for m in row["marked_regions"]},
                         "category": category})

    verdicts, stats = verify_many(proposed)

    print(f"calibration sample: {len(rows)} records the deterministic miner could not settle\n")
    print(f"  teacher proposed a plan for : {len(proposed)}  ({100 * len(proposed) / len(rows):.0f}%)")
    print(f"  teacher refused             : {len(refused)}  ({100 * len(refused) / len(rows):.0f}%)\n")
    print(stats.describe())

    bad = [(p, v) for p, v in zip(proposed, verdicts) if not v.accepted]
    if bad:
        print("\n  proposals the verifier rejected:")
        for p, v in bad:
            print(f"    [{p['index']}] {v.status}  {v.detail}")

    import collections
    print("\n  why the teacher refused:")
    for cat, n in collections.Counter(c for _, c, _ in refused).most_common():
        print(f"    {cat:<34}{n:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
