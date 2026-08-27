"""The prompts, recorded verbatim — `PLAN.md` 5.1 and 5.5.

Two prompts, and the distinction between them is the whole experiment.

* `STRUCTURED_PROMPT` elicits the strict JSON record of Appendix A: answerability,
  evidence boxes, a typed expression tree, and an answer.
* `PLAIN_PROMPT` is the elicitation the Qwen3-VL report used to produce the published
  ChartQA figure of 79.1 (`verification/phase0.md` F9), reproduced **exactly**, down to
  the sentence "Answer the question using a single word or phrase."

`PLAIN_PROMPT` exists so that "structured output costs N points" is measured against the
same elicitation the published number came from. Paraphrasing it — even improving it —
would make the comparison meaningless, because the difference would then include a prompt
change as well as the format change.

`PLAN.md` 5.5 requires the final prompt text to be committed in `PREREGISTRATION.md`
before any test split is opened, so these strings are the source of truth for that file
rather than being retyped into it.
"""

from __future__ import annotations

import hashlib
import json

#: Verbatim from the Qwen3-VL technical report's evaluation-prompt appendix, for
#: `DocVQA | InfoVQA | ChartQA_TEST`. Do not edit: it anchors the 79.1 comparison.
PLAIN_PROMPT = "{question}\nAnswer the question using a single word or phrase."

#: The operations the executor accepts. Listed in the prompt because a model that invents
#: an operation produces a record the executor must reject, and a rejected record counts
#: as a failure (non-negotiable rule 3) rather than being silently repaired.
ALLOWED_OPS = (
    "lookup", "count", "sum", "mean", "median", "min", "max",
    "difference", "ratio", "percent_change", "argmax", "argmin",
    "compare", "rank", "trend", "filter", "boolean", "multiple_choice",
    "unanswerable",
)

STRUCTURED_PROMPT = """\
Read the chart and answer the question.

Reply with ONE JSON object and nothing else. No markdown, no code fence, no explanation.

{{
  "answerable": true or false,
  "evidence": [{{"label": "<axis or series label>", "value": <number or string or null>,
                "unit": "<unit or null>", "bbox": [x1, y1, x2, y2]}}],
  "plan": {{"op": "<operation>", "args": [...]}},
  "model_answer": "<the answer>"
}}

Rules:
- bbox coordinates are integers from 0 to 999, measured on the image as you see it:
  x1,y1 is the top-left corner and x2,y2 the bottom-right.
- Put a box on every chart element the answer depends on, and on nothing else.
  Order them most important first.
- "op" must be one of: {ops}.
- An argument is either a label string naming one of your evidence items, or a nested
  {{"op": ..., "args": [...]}} object.
- Aggregations over every evidence item take an empty args list, e.g.
  {{"op": "sum", "args": []}}.
- If the chart does not contain the answer, set "answerable" to false, use
  {{"op": "unanswerable", "args": []}}, and leave "model_answer" empty.
- "model_answer" is the final answer only: a single word, phrase or number.

Question: {{question}}\
""".format(ops=", ".join(ALLOWED_OPS))


def build_structured_prompt(question: str) -> str:
    return STRUCTURED_PROMPT.replace("{question}", question)


def build_plain_prompt(question: str) -> str:
    return PLAIN_PROMPT.replace("{question}", question)


def prompt_fingerprint() -> dict[str, str]:
    """SHA-256 of each prompt, so a silent edit is detectable.

    `PLAN.md` 5.5 seals the prompt at pre-registration. A hash makes "the prompt did not
    change between the baseline and the trained run" checkable instead of remembered.
    """
    return {
        "structured": hashlib.sha256(STRUCTURED_PROMPT.encode()).hexdigest(),
        "plain": hashlib.sha256(PLAIN_PROMPT.encode()).hexdigest(),
    }


def example_record() -> dict:
    """A syntactically valid record, used in tests and in the pre-registration."""
    return {
        "answerable": True,
        "evidence": [
            {"label": "2019", "value": 245, "unit": "millions", "bbox": [412, 180, 468, 640]},
            {"label": "2018", "value": 210, "unit": "millions", "bbox": [330, 240, 386, 640]},
        ],
        "plan": {"op": "difference", "args": ["2019", "2018"]},
        "model_answer": "35",
    }


def example_record_json() -> str:
    return json.dumps(example_record(), separators=(",", ":"))


__all__ = ["ALLOWED_OPS", "PLAIN_PROMPT", "STRUCTURED_PROMPT", "build_plain_prompt",
           "build_structured_prompt", "example_record", "example_record_json",
           "prompt_fingerprint"]
