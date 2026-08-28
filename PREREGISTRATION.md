# Pre-registration

**Frozen before any test split is opened.** `PLAN.md` 5.5 requires this file to be
committed and clean before `chartqa_dt.splits` will allow a sealed split, and
`assert_split_allowed` enforces it mechanically rather than on trust
(`DECISIONS.md` 0031).

Everything below is generated from its authoritative source by
`scripts/write_prereg.py`. Retyping numbers into prose is how a pre-registration ends up
describing a run that never happened.

---

## 1. Backbone and variant

| | |
|---|---|
| backbone | `Qwen/Qwen3-VL-2B-Instruct` (`DECISIONS.md` 0035, 0036) |
| variant selected | **instruct** |
| reason | only one variant was measured |
| visual token factor | 32 (derived from the processor, `DECISIONS.md` 0008) |
| quantisation | 4-bit NF4, vision tower excluded (`DECISIONS.md` 0012) |
| LoRA | r=16, alpha=32, dropout=0.05, on **both** vision and language |
| image budget | 512 px long side |

### 5.2 comparison, on the frozen 200-question slice

| variant | relaxed accuracy | valid JSON | repaired | median latency |
|---|---:|---:|---:|---:|
| instruct | 50.00% | 66.5% | 12.0% | 11.44 s |

`PLAN.md` 5.2's gate — Thinking only if **all three** hold: ≥ 2 accuracy points better,
≥ 90% valid JSON, ≤ 2× Instruct's median latency. The thresholds were written into
`scripts/run_zeroshot.py` before any number existed.

## 2. Prompts, verbatim

Structured prompt — SHA-256 `8eca79176557b679323770fd2914387584ee206c71153af61d370fa561ed7bd6`:

```
Read the chart and answer the question.

Reply with ONE compact JSON object on a single line. No markdown, no code fence, no
newlines, no indentation, no explanation.

Format:
{"answerable":true,"evidence":[{"label":"<label>","value":<number|string|null>,"unit":"<unit|null>","bbox":[x1,y1,x2,y2]}],"plan":{"op":"<operation>","args":[...]},"model_answer":"<answer>"}

Example — "How many stores does Zara have?":
{"answerable":true,"evidence":[{"label":"Zara","value":99,"unit":"stores","bbox":[340,180,650,200]}],"plan":{"op":"lookup","args":["Zara"]},"model_answer":"99"}

Example — "What is the difference between 2019 and 2018?":
{"answerable":true,"evidence":[{"label":"2019","value":245,"unit":null,"bbox":[412,180,468,640]},{"label":"2018","value":210,"unit":null,"bbox":[330,240,386,640]}],"plan":{"op":"difference","args":["2019","2018"]},"model_answer":"35"}

Example — the chart does not contain the answer:
{"answerable":false,"evidence":[],"plan":{"op":"unanswerable","args":[]},"model_answer":""}

Rules:
- All four keys are required every time, including "plan" and "model_answer".
- "evidence": NEVER more than 8 items. Fewer is better. Include only the
  chart elements the answer actually depends on, most important first.
- If a question covers more elements than that (a whole-chart total or average over a
  long chart), still stop at 8: give the correct "model_answer" from the
  whole chart, and ground the 8 most relevant elements. Do NOT keep listing
  elements — an unfinished record scores zero.
- Each "label" appears at most ONCE. Never repeat a label.
- "unit": at most 32 characters, or null. Use "USD" or "%" rather than a phrase.
- "args" is always a LIST with at most 4 elements. Each element is either a label
  string naming one of your evidence items, or a nested {"op":...,"args":[...]} object.
  Never an object with "label" or "value" keys.
- To aggregate over EVERY evidence item, use an empty list — never list the labels:
  {"op":"sum","args":[]} means "sum all the evidence".
- bbox is four integers 0-999: x1,y1 is top-left and x2,y2 is bottom-right.
- "op" must be EXACTLY one of these strings: lookup, count, sum, mean, median, min, max, difference, ratio, percent_change, argmax, argmin, compare, rank, trend, filter, boolean, multiple_choice, unanswerable.
  Use "mean" (not "average"), "difference" (not "subtract").
- Choose the op by WHAT THE ANSWER IS:
  * the answer is a category name ("which year", "which country") -> "argmax" or
    "argmin" over the evidence, NOT "lookup". {"op":"argmax","args":[]} returns the
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

Question: {question}
```

Plain prompt — SHA-256 `40f71b4847119aeba27cd390d54c05945142e380cec7681c651d731b419b5049`. This is the Qwen3-VL report's own ChartQA
elicitation, reproduced exactly so that "structured output costs N points" is measured
against the elicitation that produced the published 79.1 (`verification/phase0.md` F9):

```
{question}
Answer the question using a single word or phrase.
```

## 3. Decoding

Greedy, fixed: `{"do_sample": false, "temperature": null, "top_p": null, "top_k": null, "num_beams": 1}`. Max new tokens: 900
(structured), 32 (plain). Sampling is not used — the "before" number
must be exactly reproducible from this file, or the before/after comparison inherits noise
it cannot separate from a real effect.

## 4. Answer normaliser

`chartqa_dt.eval.metrics.normalise_prediction` — `str(text).strip()`, applied to the
model's output **before** scoring. The metric itself is left byte-identical to the
official one, which does *not* strip (`DECISIONS.md` 0053). Normalising in the pipeline
rather than in the metric keeps our numbers comparable with published ones.

## 5. Evaluators

| evaluator | file | SHA-256 |
|---|---|---|
| RefChartQA (official) | `verification/refchartqa_eval/evaluate.py` | `d0c9f87d68d999da7963ea655935a7140fc35f245ad2c26c53e28e4f651c0dd8` |
| ChartQA (pix2struct, official) | `verification/chartqa_eval/metrics.py` | `375d5970dd3c05f71b27934eabfa3e0a400c0eb459ba540a387e5c7ec6e8cecd` |

Both are vendored byte-identical and hash-checked by `tests/test_vendored_integrity.py`.
`DECISIONS.md` 0003 makes the official evaluator the scorer of record; our implementation
agrees with it on 11,690 real predictions to within 0.07 percentage points of AP.

## 6. Datasets, pinned

| dataset | revision | integrity |
|---|---|---|
| ChartQA | `af8b6f5c08c95085271561c2a3f9d15f2b5a9031` | archive SHA-256 `1bf310e5a51101681495c4a24f4f29d2…` |
| RefChartQA | `c6b6504adb96cf72f0852a5f73ba4c62b718f843` | 9 parquet SHA-256s recorded before download |

## 7. Frozen validation slices

| slice | n | SHA-256 |
|---|---:|---|
| `chartqa_variant_200` | 200 | `f2fbf013994d5759817dcec60df2b9d64a66334d121f7648e970c5a9806c73b1` |
| `chartqa_val` | 1920 | `34e889ed16b6899be17a415f59fb7b2d4c3b64fb95f8cb4d320be599f864754d` |

Sampled once, before any prompt existed. Test splits are untouched.

## 8. Training mixtures

| | stage 1 | stage 2 |
|---|---:|---:|
| total | 10,304 | 6,304 |
| synthetic | 6,000 | 2,000 |
| ChartQA | 2,408 | 2,408 |
| RefChartQA | 1,896 | 1,896 |
| with boxes | 10,304 | 6,304 |
| with a plan | 8,406 | 4,406 |
| of those, compositional | 4,824 | 1,820 |

Deduplicated: 179 merges, of which
162 across ChartQA and RefChartQA.
Zero validation or test records, asserted in code, at the **image** level as well as the
split label (`DECISIONS.md` 0048, 0049).

## 9. Training hyperparameters

Measured in the Phase 2 smoke run (`verification/measured_facts.json` → `phase2`):

| | |
|---|---|
| peak reserved | 5.572 GiB |
| seconds per step | 11.903 |
| projected full run | 9.92 h |
| LoRA params (vision / language) | 7,208,960 / 17,432,576 |
| batch | 2 × 4 gradient accumulation, single device (`DECISIONS.md` 0025) |

## 10. Early stopping

Validation relaxed accuracy and AP@0.5, evaluated at fixed intervals. Stop when neither
improves for two consecutive evaluations. The checkpoint reported is the last one that
improved, not the last one trained — and the rule is fixed here so it cannot be relaxed
after seeing a curve.

## 11. What counts as success

| target | claim level | success |
|---|---|---|
| ChartQA relaxed accuracy | **B** — published 79.1, exact checkpoint, exact prompt, verified evaluator | fine-tuned beats our own zero-shot baseline, CIs disjoint |
| RefChartQA AP@0.5 | **C** — published 32.83 is not independently reproducible (`DECISIONS.md` 0052) | fine-tuned beats our own zero-shot baseline, CIs disjoint |
| executable plans | — | the executor reproduces the emitted answer on a majority of records; invalid records count as failures |

**The primary claim is the internal before/after**, both arms measured by us with the
byte-identical official evaluator on the same sealed split. 32.83 is cited as context and
labelled as unverified, because nobody — including its authors' released artefacts — can
reproduce it.

### How much of each test split is evaluated, decided now

Declared here because it is a choice, and a choice made after seeing test numbers is not
a choice a reader can trust.

| split | rows | evaluated | why |
|---|---:|---:|---|
| ChartQA test | 2,500 | **all 2,500** | the published 79.1 is measured on the whole split, and the plain-prompt arm costs about 0.8 GPU hours at a 32-token cap, so there is no reason to sample |
| RefChartQA test | 11,690 | **1,800, stratified** | evaluating all of them for three systems is roughly 37 GPU hours against a 30 h weekly quota; 1,800 is the same size as the 5.4 validation slice, stratified the same way |

At 1,800 rows a proportion near 40% carries a 95% Wilson interval of **±2.26 points**,
against **±0.89** for the whole 11,690-row split. So sampling costs about 1.4 points of
resolution and buys back roughly 31 GPU hours; the sample resolves differences of about five
points and not smaller ones. Every RefChartQA test figure is
reported with its interval and with `n = 1,800` beside it, never as though it came from the
whole split. If a difference this project cares about turns out to be smaller than the
interval, that is reported as *not resolved at this sample size* rather than as a result.

The sample is drawn once, by the same stratified procedure as the validation slice, and its
SHA-256 is recorded before any test row is read.

## 12. The zero-shot baselines this project must beat

Section 11's success condition is *"beats our own zero-shot baseline, CIs disjoint"*. The
baselines are recorded here, before any test split is opened, so the bar cannot move
afterwards. Both are the selected checkpoint on the frozen validation slices, scored with
the vendored official evaluators.

| protocol | subset | zero-shot, 95% CI |
|---|---|---|
| ChartQA relaxed accuracy | human | **TBD** (no result; needs n≥1,920) |
| ChartQA relaxed accuracy | machine | **TBD** (no result; needs n≥1,920) |
| ChartQA relaxed accuracy | all | **TBD** (no result; needs n≥1,920) |
| ChartQA, plain published prompt | all | **TBD** (no result; needs n≥1,920) |
| RefChartQA AP@0.5 | human | **TBD** (no result; needs n≥1,800) |
| RefChartQA AP@0.5 | machine | **TBD** (no result; needs n≥1,800) |
| RefChartQA AP@0.5 | PoT | **TBD** (no result; needs n≥1,800) |
| RefChartQA P@F1 | all | **TBD** (no result; needs n≥1,800) |

The plain-prompt row is the published-prompt condition, kept beside the structured one so
the cost of asking for a record rather than a bare answer is visible in the same table.

## 13. Extensions and their entry gates

Planned only if the core result lands and quota remains: ChartQAPro transfer (entry gate:
Phase 7 complete and the extension approved), and the RefChartQA scaling ladder at
4,000 / 10,000 / 25,000 training rows (entry gate: Phase 6 stage 2 complete).

---

_Generated by `scripts/write_prereg.py`. Regenerating after the test split is opened would
defeat the purpose; the committed version is the record._
