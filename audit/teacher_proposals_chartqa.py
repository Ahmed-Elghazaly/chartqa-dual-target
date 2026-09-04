#!/usr/bin/env python3
"""EXPERIMENT · a teacher's proposals on 40 UNBIASED ChartQA records, written to JSONL.

Seed 1, so these are records Claude had not already reasoned about while judging the seed-0
expressiveness sample -- scoring yourself twice on the same records measures memory, not
judgement.

The proposals below were written by reading each prompt exactly as the pipeline builds it
(evidence = the chart's annotated elements, not the table's rows) and before running the
verifier. Refusals are recorded with a reason, because a teacher that answers everything is
not a good teacher, it is an unverified one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT))

from mine_with_llm import load_requests  # noqa: E402


def L(label: str) -> dict:
    return {"op": "lookup", "args": [label]}


#: index -> plan, for the records the teacher was willing to answer
PLANS: dict[int, dict] = {
    0:  {"op": "argmax", "args": []},          # "when was the highest"
    1:  L("2018"), 3: L("2019"), 6: L("2017"), 9: L("2020"), 10: L("Obese"),
    5:  {"op": "median", "args": []},          # "find the median of all bar values"
    12: L("Mexico"), 13: L("2017"), 14: L("2013"),
    15: L("League of Legends 2019 World"),
    17: L("2017"), 18: L("2015/2016"), 20: L("2019"), 24: L("2017*"), 25: L("2019"),
    26: {"op": "max", "args": []},             # "the peak of ... between 2000 and 2016"
    27: {"op": "argmax", "args": []},          # "which country registered the largest"
    28: L("2013 to 2015"), 29: L("2020"), 30: L("2019"), 31: L("2018"),
    32: L("Libtayo"), 37: L("2018"),
    # --- answerable only once labels carry their series (AUDIT.md H3, decision 0083).
    #     Before qualification these three read as "'2020' appears twice, nothing says
    #     which", and were refused.
    4:  L("Very important · Black"),
    11: L("Menswear · 2020"),
    21: L("Induced contribution to GDP · 2028**"),
}

#: index -> why the teacher would not answer
REFUSALS: dict[int, str] = {
    2:  "answer is a LIST of two numbers; one plan yields one value",
    7:  "'first reported' is a position, not an extremum; argmin would be right by accident",
    8:  "answer 'New England Patriots' (27) is not the maximum (34); not in the data",
    16: "the question names no year; the series is now nameable but the YEAR is not",
    19: "cannot reconstruct 0.5527… from these values under any reading",
    22: "argmax over all items gives 'Coronavirus SARS-' (45 213), not the gold answer",
    23: "gold says 2019/20 (12.1) but 2010/11 (12.3) is higher; gold contradicts the data",
    33: "every year increases; the question does not pick one out",
    34: "'is the sum of the smallest two greater than the largest' -> Yes; no such operator",
    35: "asks for 6 age categories; count() counts the 12 marks, not the 6 categories",
    36: "a release date is not computable from the values",
    39: "'lowest pass grade' is about the grade scale, not the smallest percentage",
    38: "the annotation carries no 2020 element -- see the note below",
}


# Record 38 was originally PROPOSED as `lookup("2020")` and the verifier rejected it with
# `operand_not_in_evidence`: the chart's annotation stops before 2020. The guess came from
# reading a truncated dump rather than the prompt, which lists every item — a real teacher
# reading the prompt would have seen the gap and refused. It is recorded as a refusal, and
# the original error is left documented here rather than quietly deleted.


def main() -> int:
    reqs = load_requests("chartqa", limit=40, seed=1)
    out, refused = [], []
    for i, r in enumerate(reqs):
        if i in PLANS:
            out.append({"record_id": r.record_id, "plan": PLANS[i], "index": i})
        else:
            out.append({"record_id": r.record_id, "refused": REFUSALS.get(i, "unclassified"),
                        "index": i})
            refused.append(i)
    path = Path("audit/teacher_proposals_chartqa.jsonl")
    path.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out),
                    encoding="utf-8")
    print(f"{len(PLANS)} proposals, {len(refused)} refusals -> {path}")
    assert set(PLANS) | set(REFUSALS) == set(range(len(reqs))), "every record must be decided"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
