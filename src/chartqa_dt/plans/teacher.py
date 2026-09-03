"""Asking a language model for a plan, and pinning exactly what was asked.

`Prompt.md` Idea 7B. This is the proposing half; `llm_mining.py` is the verifying half, and
they are separate files on purpose — nothing here may decide what becomes supervision.

**Why a model at all.** The deterministic miner refuses 53.9% of ChartQA training rows as
`ambiguous`, which does not mean two cells hold the answer: it means two *operations*
reproduce it. Half of those collisions are `lookup` against an extremum — 26.6% of all rows,
`lookup+max` alone 775 times in a 4,000-row sample. ChartQA charts are usually sorted, so the
answer cell is simultaneously `lookup(<its label>)` and `max` of its column. *"How many
internet users did Nigeria have"* wants the lookup and *"which country had the most"* wants
the extremum; the two are identical in the table and one word apart in the question. The
miner never reads the question (`DECISIONS.md` 0081, `AUDIT.md` H4).

**The teacher sees the gold answer, and that is a deliberate trade.** Mining looks for the
*reasoning* behind an answer already known, so withholding it would make the task guesswork.
The cost is that a teacher can reverse-engineer a plan that lands on the number without
meaning it — and where RefChartQA marks a single region, `argmax`, `argmin` and `lookup` all
trivially return it, so no arithmetic gate can tell them apart (0080). Two things bound that:
the teacher is told to refuse rather than guess, and precision is reported separately for
multi-element evidence, where the gates have something to bite on.

**Reproducibility.** The prompt is built deterministically from the record and hashed. A
cached proposal is keyed by record, model and prompt hash together, so changing the wording
of the prompt invalidates the cache rather than silently mixing two experiments. Every
accepted plan carries the model name and prompt hash into `meta`, so any target in the
training set can be traced back to the exact request that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from chartqa_dt.plans.executor import MAX_DEPTH, NEEDS_TABLE, OPS

#: Bumped whenever the prompt text changes in a way that could change proposals. Part of
#: the cache key, so an old cache cannot masquerade as a new experiment.
PROMPT_VERSION = "1"

#: The `meta` key holding where a plan came from. Written for every LLM-mined plan.
PROVENANCE_KEY = "plan_provenance"

#: Signatures the executor actually implements, written out for the teacher. Derived from
#: `OPS` below so an operation cannot be offered here and be missing there.
SIGNATURES: dict[str, str] = {
    "lookup": "lookup(label) -> the value of that one item",
    "count": "count() -> how many items; count(a, b, ...) -> how many were listed",
    "sum": "sum() over all items, or sum(a, b, ...) over the named ones",
    "mean": "mean() over all items, or mean(a, b, ...)",
    "median": "median() over all items, or median(a, b, ...)",
    "min": "min() -> the smallest VALUE (a number)",
    "max": "max() -> the largest VALUE (a number)",
    "argmin": "argmin() -> the LABEL of the smallest item (a name)",
    "argmax": "argmax() -> the LABEL of the largest item (a name)",
    "difference": "difference(a, b) -> a - b",
    "ratio": "ratio(a, b) -> a / b",
    "percent_change": "percent_change(a, b) -> 100 * (a - b) / b",
    "compare": "compare(a, b) -> 'greater' | 'less' | 'equal'",
    "trend": "trend() -> 'increasing' | 'decreasing' | 'flat', first item to last",
    "boolean": "boolean(a) -> true if a is non-zero",
    "unanswerable": "unanswerable() -> the chart does not contain the answer",
}

#: Offered to the teacher: implemented operations only. `filter`, `rank` and
#: `multiple_choice` are in `OPS` but raise in the executor, so proposing them would produce
#: guaranteed rejections and a misleading failure profile.
OFFERED: tuple[str, ...] = tuple(sorted(OPS - NEEDS_TABLE))

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class TeacherError(RuntimeError):
    """The teacher could not be reached or did not return usable text."""


@dataclass
class Proposal:
    """What the teacher said, before any of it is believed."""

    record_id: str
    plan: dict | None
    refused: bool
    raw: str
    model: str
    prompt_sha256: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"record_id": self.record_id, "plan": self.plan, "refused": self.refused,
                "model": self.model, "prompt_sha256": self.prompt_sha256,
                "note": self.note, "raw": self.raw}


@dataclass
class TeacherRequest:
    """A prompt and everything needed to reproduce it."""

    record_id: str
    prompt: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    answer: Any = None
    marked_labels: set[str] = field(default_factory=set)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()


def _render_table(table: dict | None, *, max_rows: int = 20) -> str:
    if not table:
        return "(no table available)"
    cols = " | ".join(str(c) for c in table.get("columns") or [])
    rows = table.get("rows") or []
    body = "\n".join(" | ".join(str(c) for c in r) for r in rows[:max_rows])
    more = f"\n… and {len(rows) - max_rows} further rows" if len(rows) > max_rows else ""
    return f"{cols}\n{body}{more}"


def build_prompt(*, question: str, answer: Any, table: dict | None,
                 evidence: list[dict[str, Any]],
                 marked_labels: set[str] | None = None) -> str:
    """The exact text sent to the teacher. Deterministic — same record, same string.

    Evidence is listed in the form the executor resolves labels against, because a label the
    teacher invents is rejected by the `operand_not_in_evidence` gate and produces a wasted
    call rather than a plan.
    """
    ops = "\n".join(f"  - {SIGNATURES[o]}" for o in OFFERED if o in SIGNATURES)
    items = "\n".join(
        f"  - {e.get('label')!r} = {e.get('value')!r}"
        + (f" {e['unit']}" if e.get("unit") else "")
        + ("   [MARKED]" if marked_labels and e.get("label") in marked_labels else "")
        for e in evidence) or "  (none)"
    marked_note = (
        "\nSome items are tagged [MARKED]. A human annotator marked exactly those regions of "
        "the chart as the ones needed to answer. Your plan's operands must be those items.\n"
        if marked_labels else "")

    return f"""You are given a chart question, the chart's underlying data, and the correct \
answer. Write the small program that computes that answer from the data.

QUESTION: {question}
CORRECT ANSWER: {answer!r}

CHART DATA:
{_render_table(table)}

ITEMS you may reference by label (use these labels exactly):
{items}
{marked_note}
OPERATIONS you may use, and nothing else:
{ops}

Reply with a JSON object in a ```json block:
  {{"op": "<operation>", "args": [...]}}
An argument is either a label string from the list above, or a nested {{"op":…,"args":[…]}}.
Nesting may go at most {MAX_DEPTH} levels deep and a node takes at most 4 arguments.

Choose the operation the QUESTION asks for, not merely one that reaches the right number. \
If the question names a specific item, that is a lookup even when the item happens to be the \
largest. If the question asks which item is largest, that is argmax even when you could name \
it directly.

If no combination of these operations expresses what the question asks, reply exactly:
```json
{{"refused": "<one short reason>"}}
```
Refusing is a correct answer. A plan that reaches the right number by the wrong route is \
worse than no plan, because it will be used to teach a model the wrong reasoning."""


def parse_proposal(text: str) -> tuple[dict | None, bool, str]:
    """Pull the plan out of the teacher's reply. Returns (plan, refused, note)."""
    blocks = _JSON_BLOCK.findall(text or "")
    candidate = blocks[-1] if blocks else (text or "")
    try:
        obj = json.loads(candidate.strip())
    except (json.JSONDecodeError, ValueError):
        return None, False, "reply did not contain parsable JSON"
    if not isinstance(obj, dict):
        return None, False, "reply was not a JSON object"
    if "refused" in obj:
        return None, True, str(obj["refused"])[:200]
    if "op" not in obj:
        return None, False, "JSON object had no `op`"
    return obj, False, ""


def call_anthropic(prompt: str, *, model: str, max_tokens: int = 700,
                   api_key: str | None = None) -> str:
    """The one place that touches the network. Raises if it cannot run — never invents.

    A Claude or ChatGPT *subscription* does not authorise this; it needs a console API key
    in `ANTHROPIC_API_KEY`. Without one this raises, so a run that could not reach a model
    fails loudly instead of producing an empty result that looks like a measurement.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise TeacherError(
            "ANTHROPIC_API_KEY is not set. Mining at scale needs a console API key; a "
            "Claude.ai or ChatGPT subscription cannot drive a pipeline. Use "
            "--proposals <file> to score proposals produced elsewhere.")
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise TeacherError("the `anthropic` package is not installed") from exc

    client = anthropic.Anthropic(api_key=key)
    reply = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])
    return "".join(getattr(b, "text", "") for b in reply.content)


def provenance(*, model: str, prompt_sha256: str, verifier_gates: list[str]) -> dict:
    """What gets written into `meta` so a target can be traced back to its request."""
    return {"method": "llm_teacher", "model": model,
            "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_sha256,
            "gates_passed": verifier_gates}
