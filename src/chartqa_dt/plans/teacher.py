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


Contamination, licensing and reproducibility
--------------------------------------------

`Prompt.md` Idea 7 requires that a closed teacher's **contamination, licensing and
dependency risk** be addressed explicitly rather than assumed away. They were not, until
`DECISIONS.md` 0134.

**Contamination — low, and for a structural reason.** ChartQA is a public 2022 benchmark, so
any frontier model has almost certainly seen it. That would matter if we asked the teacher
for the *answer*; we do not. We supply the question, the data table **and the gold answer**,
and ask only for the program that connects them. Recall of the answer cannot inflate our
yield, because the answer is an input. Recall of a *plan* is not possible either for the
ChartQA half: ChartQA publishes no programs.

It is **not** zero for RefChartQA. Its `pot` subset publishes derivations (0133), so a
teacher may have seen those. The mitigation is that we do not need a teacher there — the
derivations are converted deterministically, with no model in the loop.

**Licensing — a live constraint, and Ahmed's call.** Anthropic's published policy prohibits
using outputs to train or develop AI models *without written permission*, while permitting
non-competing specialised tools and prohibiting general-purpose chatbots. A chart-QA model
is a specialised tool rather than a chatbot, which is the permitted category, but the
requirement is stated broadly enough that this should be settled before any mining spend
rather than after.

**Reproducibility — the argument for open weights.** A closed model can be deprecated, and
then a headline number cannot be reproduced at all. `PROVENANCE_KEY` pins the model id and
the prompt hash, which records *what* was used but cannot resurrect it. An open-weights
teacher pinned by weight hash removes the licensing question and satisfies reproducibility
in a way no closed model can. This is recorded as the recommended alternative.

"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from chartqa_dt.data.colours import names_for
from chartqa_dt.plans.executor import MAX_DEPTH, NEEDS_TABLE, OPS

#: Bumped whenever the prompt text changes in a way that could change proposals. Part of
#: the cache key, so an old cache cannot masquerade as a new experiment.
PROMPT_VERSION = "2"

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
    "within": ('within(series, op) -> apply an operation to ONE series only, e.g. '
               '{"op":"within","args":["Hyperscale",{"op":"argmax","args":[]}]} for '
               '"which year was highest in hyperscale". Inside it, labels lose the '
               '"series · " prefix, so argmax returns "2021" not "Hyperscale · 2021"'),
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


def describe_colour(colour: Any) -> str:
    """The most specific word for this colour, for a reader to match against a question.

    **21.8% of human-written ChartQA questions mention a colour** and 0.5% of machine ones
    (`DECISIONS.md` 0087), so a reader that cannot see colour cannot answer a fifth of the
    half of the benchmark that decides the score. The most specific name is shown — *dark
    blue* rather than *blue* — because a question saying "blue" still matches it, while one
    saying "dark blue" would not match the other way round.
    """
    words = names_for(colour)
    return max(words, key=lambda w: (len(w.split()), len(w))) if words else ""


def _render_table(table: dict | None, *, max_rows: int = 20) -> str:
    if not table:
        return "(no table available)"
    cols = " | ".join(str(c) for c in table.get("columns") or [])
    rows = table.get("rows") or []
    body = "\n".join(" | ".join(str(c) for c in r) for r in rows[:max_rows])
    more = f"\n… and {len(rows) - max_rows} further rows" if len(rows) > max_rows else ""
    return f"{cols}\n{body}{more}"


def build_system() -> str:
    """The half of the request that is identical for every record, so it can be cached.

    Prompt caching is a **prefix** match: stable content must come first or nothing caches.
    The first version of this prompt opened with the question and its table — the most
    volatile text in the request — and closed with the operation catalogue, which is byte
    identical every time. That ordering makes caching impossible, and across tens of
    thousands of records the catalogue and the instructions would be paid for in full on
    every single call. They live here instead, sent as a cached `system` block.
    """
    ops = "\n".join(f"  - {SIGNATURES[o]}" for o in OFFERED if o in SIGNATURES)
    return f"""You are given a chart question, the chart's underlying data, and the correct \
answer. Write the small program that computes that answer from the data.

OPERATIONS you may use, and nothing else:
{ops}

Reply with a JSON object in a ```json block:
  {{"op": "<operation>", "args": [...]}}
An argument is either a label string from the ITEMS list, or a nested {{"op":…,"args":[…]}}.
Nesting may go at most {MAX_DEPTH} levels deep and a node takes at most 4 arguments.

Choose the operation the QUESTION asks for, not merely one that reaches the right number. \
If the question names a specific item, that is a lookup even when the item happens to be the \
largest. If the question asks which item is largest, that is argmax even when you could name \
it directly.

If no combination of these operations expresses what the question asks, do NOT force a fit.
You have two ways to say so, and the difference matters:

  * The question cannot be answered from this chart at all — it is ambiguous, the gold answer
    contradicts the data, or it asks for something the chart does not contain:
    ```json
    {{"refused": "<one short reason>"}}
    ```
  * The question is perfectly answerable, but the operation it needs is missing from the list
    above. Say what that operation would be:
    ```json
    {{"needs_operator": {{"name": "<short_snake_case>", \
"signature": "<name(args) -> result>", "why": "<what this question needs it for>"}}}}
    ```

Refusing is a correct answer, and naming a missing operation is a *useful* one — those
suggestions are collected and ranked, and the operations asked for most often get built. A
plan that reaches the right number by the wrong route is worse than either, because it will
be used to teach a model the wrong reasoning."""


def build_prompt(*, question: str, answer: Any, table: dict | None,
                 evidence: list[dict[str, Any]],
                 marked_labels: set[str] | None = None) -> str:
    """The per-record half of the request. Deterministic — same record, same string.

    Evidence is listed in the form the executor resolves labels against, because a label the
    teacher invents is rejected by the `operand_not_in_evidence` gate and produces a wasted
    call rather than a plan.
    """
    items = "\n".join(
        f"  - {e.get('label')!r} = {e.get('value')!r}"
        + (f" {e['unit']}" if e.get("unit") else "")
        + (f"   [{describe_colour(e['colour'])}]" if e.get("colour") else "")
        + ("   [MARKED]" if marked_labels and e.get("label") in marked_labels else "")
        for e in evidence) or "  (none)"
    marked_note = (
        "\nSome items are tagged [MARKED]. A human annotator marked exactly those regions of "
        "the chart as the ones needed to answer. Your plan's operands must be those items.\n"
        if marked_labels else "")

    return f"""QUESTION: {question}
CORRECT ANSWER: {answer!r}

CHART DATA:
{_render_table(table)}

ITEMS you may reference by label (use these labels exactly):
{items}
{marked_note}"""


@dataclass
class Reply:
    """One teacher reply, read but not yet believed.

    Exactly one of `plan`, `refused` and `needs_operator` is meaningful; `note` carries the
    reason a reply was none of the three.
    """

    plan: dict | None = None
    refused: bool = False
    needs_operator: dict | None = None
    note: str = ""

    @property
    def usable(self) -> bool:
        return self.plan is not None


def parse_proposal(text: str) -> Reply:
    """Read the teacher's reply into one of its three shapes.

    A missing operation is deliberately NOT folded into `refused`. A refusal says *this
    question cannot be answered*; a suggestion says *this question is fine and our DSL is
    not*. Counting them together would hide the second, which is the only one that tells us
    what to build next.
    """
    blocks = _JSON_BLOCK.findall(text or "")
    candidate = blocks[-1] if blocks else (text or "")
    try:
        obj = json.loads(candidate.strip())
    except (json.JSONDecodeError, ValueError):
        return Reply(note="reply did not contain parsable JSON")
    if not isinstance(obj, dict):
        return Reply(note="reply was not a JSON object")
    if "needs_operator" in obj:
        ask = obj["needs_operator"]
        if not isinstance(ask, dict) or not str(ask.get("name", "")).strip():
            return Reply(note="needs_operator without a name")
        return Reply(needs_operator={"name": str(ask.get("name"))[:64],
                                     "signature": str(ask.get("signature", ""))[:200],
                                     "why": str(ask.get("why", ""))[:300]})
    if "refused" in obj:
        return Reply(refused=True, note=str(obj["refused"])[:200])
    if "op" not in obj:
        return Reply(note="JSON object had no `op`")
    return Reply(plan=obj)


#: USD per million tokens, from the Claude API pricing table (cached 2026-06-24). Batch
#: requests are half these; a cache read is about a tenth of the input rate.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

#: How hard the model should think per record. `medium` by default rather than `high`:
#: reading a question and choosing one operator is not a long-horizon task, and effort is
#: the first quality-for-cost lever. **Sweep it on a calibration sample before a full run** —
#: which effort a workload repays is a property of the workload, not something to assume.
DEFAULT_EFFORT = "medium"


def call_anthropic(prompt: str, *, system: str | None = None, model: str,
                   max_tokens: int = 4096, effort: str = DEFAULT_EFFORT,
                   api_key: str | None = None) -> str:
    """The one place that touches the network. Raises if it cannot run — never invents.

    A Claude or ChatGPT *subscription* does not authorise this; it needs a console API key
    in `ANTHROPIC_API_KEY`. Without one this raises, so a run that could not reach a model
    fails loudly instead of producing an empty result that looks like a measurement.

    The system half is sent with `cache_control`, so the operation catalogue and the
    instructions — identical on every one of tens of thousands of calls — are billed at the
    cache-read rate after the first. Thinking is left at its default (adaptive on Opus 5)
    and depth is controlled with `effort`.
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
        output_config={"effort": effort},
        system=[{"type": "text", "text": system or build_system(),
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}])
    return "".join(getattr(b, "text", "") for b in reply.content)


#: Requests per batch submission. The API's own ceiling is far higher; this is small enough
#: that a failure costs one chunk rather than a whole run, and that progress is visible.
BATCH_CHUNK = 5_000


def run_batch(items: Sequence[tuple[str, str]], *, model: str,
              effort: str = DEFAULT_EFFORT, max_tokens: int = 4096,
              api_key: str | None = None, poll_seconds: int = 30,
              on_progress: Callable[[str], None] | None = None) -> dict[str, str]:
    """Submit `(record_id, prompt)` pairs through the Message Batches API. Half price.

    Mining is not latency-sensitive — nothing waits on any single answer — which is exactly
    the workload batching is for, and it halves the bill. Results come back in **any order**,
    so they are keyed by `custom_id` and never by position.

    Returns `{record_id: raw reply text}` for every request that succeeded. A request that
    errored, was cancelled or expired is simply absent: it is not a refusal and must not be
    counted as one, so the caller sees a missing key rather than an empty string.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise TeacherError(
            "ANTHROPIC_API_KEY is not set. Mining at scale needs a console API key; a "
            "Claude.ai or ChatGPT subscription cannot drive a pipeline.")
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise TeacherError("the `anthropic` package is not installed") from exc

    client = anthropic.Anthropic(api_key=key)
    system = build_system()
    out: dict[str, str] = {}
    say = on_progress or (lambda _m: None)

    for start in range(0, len(items), BATCH_CHUNK):
        chunk = items[start:start + BATCH_CHUNK]
        say(f"submitting {len(chunk):,} requests "
            f"({start + len(chunk):,}/{len(items):,}) …")
        batch = client.messages.batches.create(requests=[
            {"custom_id": rid,
             "params": {"model": model, "max_tokens": max_tokens,
                        "output_config": {"effort": effort},
                        "system": [{"type": "text", "text": system,
                                    "cache_control": {"type": "ephemeral"}}],
                        "messages": [{"role": "user", "content": prompt}]}}
            for rid, prompt in chunk])
        while True:
            status = client.messages.batches.retrieve(batch.id).processing_status
            if status == "ended":
                break
            say(f"  batch {batch.id}: {status}")
            time.sleep(poll_seconds)
        kept = 0
        for result in client.messages.batches.results(batch.id):
            if result.result.type != "succeeded":
                continue      # errored / cancelled / expired -- absent, not refused
            out[result.custom_id] = "".join(
                getattr(b, "text", "") for b in result.result.message.content)
            kept += 1
        say(f"  batch {batch.id}: {kept:,} of {len(chunk):,} succeeded")
    return out


def estimate_cost(*, n_records: int, system_chars: int, record_chars: int,
                  model: str, batch: bool) -> dict[str, float]:
    """A budget for a run, before any of it is spent.

    Tokens are estimated from characters at ~3.6 chars/token, which is a **rough** figure
    for English prose with numbers — it is a planning aid, not a bill. The exact count is
    available from `client.messages.count_tokens` once a key exists; this exists so the
    decision to buy one can be made first.

    The system half is counted once at the full input rate and thereafter at the cache-read
    rate, because it is byte-identical across records.
    """
    per_token = 3.6
    in_rate, out_rate = PRICING.get(model, PRICING["claude-opus-5"])
    if batch:
        in_rate, out_rate = in_rate / 2, out_rate / 2
    sys_tok = system_chars / per_token
    rec_tok = record_chars / per_token
    # A reply is a short JSON object plus adaptive thinking; 700 output tokens is a
    # deliberately generous per-record figure so the estimate errs high.
    out_tok = 700.0
    first = sys_tok * in_rate / 1e6
    cached = sys_tok * (in_rate * 0.1) / 1e6 * max(n_records - 1, 0)
    records = rec_tok * in_rate / 1e6 * n_records
    output = out_tok * out_rate / 1e6 * n_records
    return {"input_system_first": first, "input_system_cached": cached,
            "input_records": records, "output": output,
            "total": first + cached + records + output}


def provenance(*, model: str, prompt_sha256: str, verifier_gates: list[str]) -> dict:
    """What gets written into `meta` so a target can be traced back to its request."""
    return {"method": "llm_teacher", "model": model,
            "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_sha256,
            "gates_passed": verifier_gates}
