"""Asking a reader which label a question is talking about — and nothing else.

`plans.forward` builds the plan a question asks for and checks it against the answer, which
covers 44.9% of ChartQA. Of the 55% it abstains on, **31.8% are lookups it could not
identify**: the gold answer *is* one element's value, but the question refers to that element
in words the label does not contain. Measured on real misses:

| the question says | the label says | what is needed |
|---|---|---|
| "the 2019/20 season" | `2019/2020` | format normalisation |
| "the second quarter of 2018" | `Mobile · Q2 '18` | notation |
| "the previous year's revenue" | `2019` | a relative reference |
| "second division of German professional soccer" | `2. Bundesliga` | **world knowledge** |

No pattern gets `2. Bundesliga` from *"second division of German professional soccer"*. This
is the part of mining a language model is genuinely better at, and it is worth being precise
about **why**: not because it reasons about arithmetic — `plans.forward` already does that,
deterministically and testably — but because it resolves an entity across a paraphrase.

**So the model is asked a yes/no question, never asked to write a plan.** The candidate label
is chosen by arithmetic (it is the one element whose value equals the gold answer) and the
reader only judges *fidelity*: is this the thing the question is asking about? That split
matters. The arithmetic half cannot be got wrong, and the half that can be is a single
judgement a reader can check, not a program it might get subtly right for the wrong reason.

It also makes the task cheap. A binary judgement is a few tokens, so hundreds fit in one
request — which is what makes this runnable on a subscription rather than an API budget.

**The circularity that is avoided.** Picking the candidate by value and *stopping there*
would be the backwards search all over again (`DECISIONS.md` 0085): on a sorted chart the
answer's holder is often just the largest bar. Requiring a reader to agree the question names
that element is what keeps it a reading of the question rather than a restatement of the
label — and a `no` is a useful answer, not a failure.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from chartqa_dt.eval.metrics import to_float
from chartqa_dt.plans.executor import parse_numeric

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


@dataclass
class Question:
    """One record where the arithmetic is settled and only the reading is in doubt."""

    record_id: str
    question: str
    answer: str
    candidate: str
    #: Every label on the chart, so a reader can see what else it might have meant.
    labels: list[str] = field(default_factory=list)

    def as_prompt_line(self, index: int) -> str:
        others = [x for x in self.labels if x != self.candidate][:10]
        return (f'{index}. Q: {self.question}\n'
                f'   answer {self.answer!r} is the value of: {self.candidate!r}\n'
                f'   other labels: {others}')


def candidates(*, question: str, answer: Any,
               evidence: Sequence[Mapping[str, Any]]) -> str | None:
    """The single element whose value is the gold answer, or `None` if it is not unique.

    Not unique means the reading cannot be settled this way — two elements holding the same
    number are indistinguishable by arithmetic, so no yes/no answer would determine a plan.
    """
    # The ANSWER is parsed with the official parser and the chart VALUES with ours. They are
    # different kinds of text and the difference is load-bearing: `to_float` divides a
    # trailing `%` by 100 because the official evaluator does, which is right for an answer
    # and wrong for a bar (`DECISIONS.md` 0082).
    target = to_float(answer)
    if target is None:
        return None
    holders = [str(e.get("label")) for e in evidence
               if parse_numeric(e.get("value")) is not None
               and abs(parse_numeric(e.get("value")) - target) < 1e-9]
    return holders[0] if len(holders) == 1 else None


def build_batch(questions: Sequence[Question]) -> str:
    """One request covering many records. A binary judgement each, so they pack tightly."""
    body = "\n\n".join(q.as_prompt_line(i) for i, q in enumerate(questions))
    return f"""For each numbered item below, decide ONE thing: is the named label the thing \
the question is asking about?

The arithmetic is already settled — that label's value IS the given answer. You are only \
judging whether the question refers to that label, which often needs a paraphrase resolved:
  "the 2019/20 season"                     -> a label written '2019/2020'      yes
  "the second quarter of 2018"             -> a label written "Q2 '18"          yes
  "second division of German soccer"       -> a label written '2. Bundesliga'   yes
  "Tibet's RURAL population"               -> a label 'Urban population · Tibet' NO

Answer "no" whenever the question is about something else, is about the whole chart rather \
than one item, or you cannot tell. A wrong "yes" teaches a model to point at the wrong mark, \
which is worse than teaching it nothing.

{body}

Reply with one JSON object in a ```json block, mapping each item number to true or false:
```json
{{"0": true, "1": false, ...}}
```"""


def parse_batch(reply: str, *, expected: int) -> dict[int, bool]:
    """Read the verdicts back. Anything missing or malformed is simply absent, never `False`.

    The distinction matters: an absent verdict means "not judged" and the record is left for
    a later pass, while `False` means "judged, and the question is about something else".
    Collapsing them would silently discard records a reader never saw.
    """
    blocks = _JSON_BLOCK.findall(reply or "")
    try:
        obj = json.loads((blocks[-1] if blocks else reply or "").strip())
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[int, bool] = {}
    for key, value in obj.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if 0 <= index < expected and isinstance(value, bool):
            out[index] = value
    return out


def plan_for(label: str) -> dict:
    """The plan a `yes` implies. Trivial by construction, which is the point."""
    return {"op": "lookup", "args": [label]}


__all__ = ["Question", "build_batch", "candidates", "parse_batch", "plan_for"]
