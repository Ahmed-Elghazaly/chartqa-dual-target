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

#: Read from `OUTPUT_SCHEMA` rather than restated, so the prompt cannot drift from the
#: schema it is trying to satisfy. The first compact-prompt probe failed on all three of
#: these limits — args of 5 and 8 elements, a 35-character unit, and duplicate labels —
#: because the prompt never mentioned them.
MAX_EVIDENCE = 8
MAX_ARGS = 4
MAX_UNIT_CHARS = 32

#: The operations the executor accepts. Listed in the prompt because a model that invents
#: an operation produces a record the executor must reject, and a rejected record counts
#: as a failure (non-negotiable rule 3) rather than being silently repaired.
ALLOWED_OPS = (
    "lookup", "count", "sum", "mean", "median", "min", "max",
    "difference", "ratio", "percent_change", "argmax", "argmin",
    "compare", "rank", "trend", "filter", "boolean", "multiple_choice",
    "unanswerable",
)

# Iterated on validation data only (`PLAN.md` 5.1), from what the first probe measured:
#
# * The model **imitates the example's formatting**. A pretty-printed example produced
#   pretty-printed records with a median of 308 tokens, 33% of which hit the 512-token cap
#   mid-record. Four of five parse failures were pure truncation — well-formed JSON that
#   simply ran out of budget. The example is now compact, and `PLAN.md` 5.2's own wording
#   asks for "valid **compact** JSON".
# * `plan.args` came back as an object — `{"label": "Zara", "value": 99}` — where the
#   schema requires an array. That record parses as JSON and the executor still rejects
#   it, so the prompt now shows args as a list in every example and says so.
# * One failure emitted `{"answerable": false, "evidence": []}` with no `plan` and no
#   `model_answer`, so the unanswerable case gets a complete worked example rather than a
#   description.
STRUCTURED_PROMPT = """\
Read the chart and answer the question.

Reply with ONE compact JSON object on a single line. No markdown, no code fence, no
newlines, no indentation, no explanation.

Format:
{{"answerable":true,"evidence":[{{"label":"<label>","value":<number|string|null>,\
"unit":"<unit|null>","bbox":[x1,y1,x2,y2]}}],"plan":{{"op":"<operation>","args":[...]}},\
"model_answer":"<answer>"}}

Example — "How many stores does Zara have?":
{{"answerable":true,"evidence":[{{"label":"Zara","value":99,"unit":"stores",\
"bbox":[340,180,650,200]}}],"plan":{{"op":"lookup","args":["Zara"]}},"model_answer":"99"}}

Example — "What is the difference between 2019 and 2018?":
{{"answerable":true,"evidence":[{{"label":"2019","value":245,"unit":null,\
"bbox":[412,180,468,640]}},{{"label":"2018","value":210,"unit":null,\
"bbox":[330,240,386,640]}}],"plan":{{"op":"difference","args":["2019","2018"]}},\
"model_answer":"35"}}

Example — the chart does not contain the answer:
{{"answerable":false,"evidence":[],"plan":{{"op":"unanswerable","args":[]}},\
"model_answer":""}}

Rules:
- All four keys are required every time, including "plan" and "model_answer".
- "evidence": at most {max_evidence} items, and FEWER IS BETTER. Include only the chart
  elements the answer actually depends on, most important first. Do not list every bar or
  every year.
- Each "label" appears at most ONCE. Never repeat a label.
- "unit": at most {max_unit} characters, or null. Use "USD" or "%" rather than a phrase.
- "args" is always a LIST with at most {max_args} elements. Each element is either a label
  string naming one of your evidence items, or a nested {{"op":...,"args":[...]}} object.
  Never an object with "label" or "value" keys.
- To aggregate over EVERY evidence item, use an empty list — never list the labels:
  {{"op":"sum","args":[]}} means "sum all the evidence".
- bbox coordinates are integers 0-999 on the image as you see it; x1,y1 is top-left and
  x2,y2 is bottom-right.
- "op" must be EXACTLY one of these strings: {ops}.
  Use "mean" (not "average"), "difference" (not "subtract").
- Choose the op by WHAT THE ANSWER IS:
  * the answer is a category name ("which year", "which country") -> "argmax" or
    "argmin" over the evidence, NOT "lookup". {{"op":"argmax","args":[]}} returns the
    label of the largest evidence item.
  * the answer is a number read straight off the chart -> "lookup" with ONE label.
  * the answer is a computed number -> "difference", "ratio", "sum", "mean", ...
  * the answer is "yes"/"no" -> "boolean"; "greater"/"less" -> "compare".
- Argument counts are fixed: "lookup" takes exactly 1 label; "compare", "difference",
  "ratio" and "percent_change" take exactly 2; "sum", "mean", "median", "min", "max",
  "count", "argmax" and "argmin" take either an explicit list or [] for all evidence.
- The plan must PRODUCE "model_answer" when run against your evidence. If running your
  own plan would give a different value, the plan is wrong — fix it before answering.
- "model_answer" is the final answer only: a single word, phrase or number.

Question: {{question}}\
""".format(ops=", ".join(ALLOWED_OPS), max_evidence=MAX_EVIDENCE,
           max_unit=MAX_UNIT_CHARS, max_args=MAX_ARGS)


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
