"""RefChartQA ships gold derivations. This turns them into our DSL.

**We were about to buy something we already had.** 35,304 of the 55,486 cached RefChartQA
records — 63.6% — carry a `response` field holding a *program of thought*: a commented,
step-by-step derivation of the answer. `data/refchartqa.py` read it into `meta` and nothing
ever used it, while `BLOCKED` listed an LLM mining run at roughly USD 213 whose entire job
is to recover derivations (`DECISIONS.md` 0133).

The format is rigidly structured, which is what makes this a parsing problem rather than an
inference problem:

    <comment># Get the value of 'Number of foreign students' in 'China', set to Y_1</comment>
    <step>Y_1=7562</step>
    <comment># Divide Y_1 by Y_2, set to Answer</comment>
    <step>Answer=np.divide(Y_1, Y_2)</step>

**Measured over all 35,304:** every one parses into comment/step pairs, and they take only
**29 distinct step-shapes**, of which the twelve commonest cover **93.4%**. A closed set that
small can be handled deterministically, and what cannot be handled is refused rather than
guessed.

**Nothing here is trusted.** A converted plan is a *candidate*: it goes through the same
five gates as any other, and the one that matters is that it must execute against the
record's own evidence and reproduce the gold answer. The derivation supplies a hypothesis;
the executor decides.
"""

from __future__ import annotations

import re
from typing import Any

#: One `<comment>#  …</comment><step>…</step>` pair.
STEP = re.compile(r"<comment>#\s*(.*?)</comment>\s*<step>(.*?)</step>", re.S)
_QUOTED = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def quoted(text: str) -> list[str]:
    """Every single- or double-quoted run, in order. Labels arrive quoted."""
    return [a or b for a, b in _QUOTED.findall(text)]


def steps_of(response: str) -> list[tuple[str, str]]:
    return [(c.strip(), s.strip()) for c, s in STEP.findall(response or "")]


def classify(comment: str) -> str:
    """What one step does, from its comment.

    Read the comment rather than the code in `<step>`: the comment states the *intent*
    (*"Get the index that minimize Y"*) while the step states an implementation
    (`np.argmin`), and intent is what a plan records.
    """
    c = comment.lower()
    if "index that minimize" in c:
        return "argmin"
    if "index that maximize" in c:
        return "argmax"
    if c.startswith("get the value of"):
        return "lookup"
    if c.startswith("get all the values"):
        return "fold_values"
    if c.startswith("get the names of all"):
        return "fold_labels"
    if "divide" in c:
        return "ratio"
    if "subtract" in c or "difference" in c:
        return "difference"
    if "sum of" in c or c.startswith("add"):
        return "sum"
    if "average" in c or "mean" in c:
        return "mean"
    if "maximum value" in c or c.startswith("get the maximum"):
        return "max"
    if "minimum value" in c or c.startswith("get the minimum"):
        return "min"
    return "?"


#: Step-shape → the plan it means. Only shapes seen in the data, and only ones whose
#: meaning is unambiguous. A shape absent from here yields `None`, which is a refusal.
SHAPES: dict[tuple[str, ...], dict[str, Any]] = {
    ("fold_values", "max"): {"op": "max", "args": []},
    ("fold_values", "min"): {"op": "min", "args": []},
    ("fold_values", "sum"): {"op": "sum", "args": []},
    ("fold_values", "mean"): {"op": "mean", "args": []},
    ("fold_labels", "fold_values", "argmax", "lookup"): {"op": "argmax", "args": []},
    ("fold_labels", "fold_values", "argmin", "lookup"): {"op": "argmin", "args": []},
}

#: Shapes whose final operation takes two named operands, read from the two lookups.
BINARY_SHAPES: dict[tuple[str, ...], str] = {
    ("lookup", "lookup", "ratio"): "ratio",
    ("lookup", "lookup", "difference"): "difference",
}


def to_plan(response: str) -> dict[str, Any] | None:
    """A DSL plan for this derivation, or `None` when its shape is not handled.

    `None` is the honest answer for 36.7% of derivations and is left that way: a shape
    guessed at would produce a plan that might still execute to the right number, which is
    exactly the spurious-program failure the gates exist to catch and the one thing that
    cannot be detected downstream.
    """
    steps = steps_of(response)
    if not steps:
        return None
    shape = tuple(classify(c) for c, _ in steps)
    if shape in SHAPES:
        return dict(SHAPES[shape])
    if shape in BINARY_SHAPES:
        first, second = (quoted(steps[0][0]), quoted(steps[1][0]))
        if first and second:
            return {"op": BINARY_SHAPES[shape], "args": [first[-1], second[-1]]}
        return None
    if shape == ("lookup",):
        labels = quoted(steps[0][0])
        if labels:
            return {"op": "lookup", "args": [labels[-1]]}
    return None


__all__ = ["BINARY_SHAPES", "SHAPES", "STEP", "classify", "quoted", "steps_of", "to_plan"]
