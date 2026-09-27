from __future__ import annotations

from chartqa_dt.plans.executor import OPS
from chartqa_dt.plans.schema import OUTPUT_SCHEMA

PLAIN_PROMPT = "{question}\nAnswer the question using a single word or phrase."

MAX_EVIDENCE = OUTPUT_SCHEMA["properties"]["evidence_refs"]["maxItems"]
MAX_ARGS = OUTPUT_SCHEMA["$defs"]["node"]["properties"]["args"]["maxItems"]
MAX_UNIT_CHARS = OUTPUT_SCHEMA["$defs"]["fact"]["properties"]["unit"]["maxLength"]
ALLOWED_OPS: tuple[str, ...] = tuple(sorted(OPS))

STRUCTURED_PROMPT = """\
Read the chart and answer the question.

Reply with ONE compact JSON object on a single line. No markdown, no code fence, no
newlines, no indentation, no explanation.

Format:
{{"answerable":true,"facts":[{{"id":"f1","label":"<label>",\
"value":<number|string|null>,"unit":"<unit|null>","bbox":[x1,y1,x2,y2]}}],\
"evidence_refs":["f1"],"plan":{{"op":"<operation>","args":[{{"ref":"f1"}}]}},\
"model_answer":"<answer>"}}

Example — "How many stores does Zara have?":
{{"answerable":true,"facts":[{{"id":"f1","label":"Zara",\
"value":99,"unit":"stores","bbox":[340,180,650,200]}}],"evidence_refs":["f1"],\
"plan":{{"op":"lookup","args":[{{"ref":"f1"}}]}},"model_answer":"99"}}

Example — "What is the difference between 2019 and 2018?":
{{"answerable":true,"facts":[{{"id":"f1","label":"2019",\
"value":245,"unit":null,"bbox":[412,180,468,640]}},{{"id":"f2","label":"2018",\
"value":210,"unit":null,"bbox":[330,240,386,640]}}],"evidence_refs":["f1","f2"],\
"plan":{{"op":"difference","args":[{{"ref":"f1"}},{{"ref":"f2"}}]}},\
"model_answer":"35"}}

Example — the chart does not contain the answer:
{{"answerable":false,"facts":[],"evidence_refs":[],"plan":null,\
"model_answer":""}}

Rules:
- All five keys are required. plan may be null only when no
  verified computation is available; never invent a plan from the answer.
- "facts" are the self-contained chart operands. Give each a unique id f1..f{max_evidence}.
  At most {max_evidence} facts are allowed. If the computation needs more, the record is
  invalid: do not truncate an aggregate or silently change its meaning.
- "evidence_refs" lists ONLY the fact ids whose boxes directly ground this question for
  official scoring. A computation may use additional context facts not listed there.
- "unit": at most {max_unit} characters, or null. Use "USD" or "%" rather than a phrase.
- "args" is always a LIST with at most {max_args} operands. Every operand is an explicit
  {{"ref":"f1"}} or a nested numeric operation. Raw numbers, strings, null, and booleans
  are forbidden as operands. There are no implicit empty-list folds.
- bbox is four integers 0-999: x1,y1 is top-left and x2,y2 is bottom-right.
- "op" must be EXACTLY one of these strings: {ops}.
  Use "mean" (not "average"), "difference" (not "subtract").
- Choose the op by WHAT THE ANSWER IS:
  * the answer is a category name -> "argmax", "argmin", "rank", or "label_of".
  * the answer is a value read straight off the chart -> "lookup" with ONE fact ref.
  * the answer is a computed number -> "difference", "ratio", "sum", "mean", ...
  * the answer is Yes/No -> "greater_than", "less_than", or "equal_to".
  * the answer is "greater"/"less"/"equal" -> "compare".
- Binary operations take exactly 2 operands. Aggregates, extrema, count, rank, and trend
  name 2-{max_evidence} explicit fact refs; one-item or empty folds are invalid.
- The plan must PRODUCE "model_answer" when run against your facts. If running your
  own plan would give a different value, the plan is wrong — fix it before answering.
- "model_answer" is the final answer only: a single word, phrase or number.

Question: {{question}}\
""".format(ops=", ".join(ALLOWED_OPS), max_evidence=MAX_EVIDENCE,
           max_unit=MAX_UNIT_CHARS, max_args=MAX_ARGS)


TRAINING_PROMPT = """\
Answer the question about the chart. Reply with one compact JSON object:
{"answerable":<bool>,"facts":[{"id":"f1","label":<str>,\
"value":<num|str|null>,"unit":<str|null>,"bbox":[x1,y1,x2,y2]}],\
"evidence_refs":["f1"],"plan":{"op":<str>,"args":[{"ref":"f1"}]},\
"model_answer":<str>}
bbox is four integers 0-999. Plan operands must be fact refs, never raw values.

Question: {question}\
"""


GROUNDING_PROMPT = """\
Ground the answer to the question in the chart. Reply with one compact JSON object:
{"answerable":<bool>,"facts":[{"id":"f1","label":<str>,\
"value":<num|str|null>,"unit":<str|null>,"bbox":[x1,y1,x2,y2]}],\
"evidence_refs":["f1"],"plan":null,"model_answer":<str>}
Use only question-specific marked facts. bbox is four integers 0-999. plan must be null.

Question: {question}\
"""


def build_training_prompt(question: str) -> str:
    return TRAINING_PROMPT.replace("{question}", question)


def build_grounding_prompt(question: str) -> str:
    return GROUNDING_PROMPT.replace("{question}", question)


def build_structured_prompt(question: str) -> str:
    return STRUCTURED_PROMPT.replace("{question}", question)


def build_plain_prompt(question: str) -> str:
    return PLAIN_PROMPT.replace("{question}", question)
