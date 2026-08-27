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
| variant selected | **TBD — 5.2 has not run** |
| reason | — |
| visual token factor | 32 (derived from the processor, `DECISIONS.md` 0008) |
| quantisation | 4-bit NF4, vision tower excluded (`DECISIONS.md` 0012) |
| LoRA | r=16, alpha=32, dropout=0.05, on **both** vision and language |
| image budget | 512 px long side |

### 5.2 comparison, on the frozen 200-question slice

_5.2 has not run yet; this section is filled by `scripts/write_prereg.py`._

`PLAN.md` 5.2's gate — Thinking only if **all three** hold: ≥ 2 accuracy points better,
≥ 90% valid JSON, ≤ 2× Instruct's median latency. The thresholds were written into
`scripts/run_zeroshot.py` before any number existed.

## 2. Prompts, verbatim

Structured prompt — SHA-256 `fb3ae905d9949c7a6c3c2f474c9793006540a7a903d19b5b3f32c1d428b2bd86`:

```
Read the chart and answer the question.

Reply with ONE JSON object and nothing else. No markdown, no code fence, no explanation.

{
  "answerable": true or false,
  "evidence": [{"label": "<axis or series label>", "value": <number or string or null>,
                "unit": "<unit or null>", "bbox": [x1, y1, x2, y2]}],
  "plan": {"op": "<operation>", "args": [...]},
  "model_answer": "<the answer>"
}

Rules:
- bbox coordinates are integers from 0 to 999, measured on the image as you see it:
  x1,y1 is the top-left corner and x2,y2 the bottom-right.
- Put a box on every chart element the answer depends on, and on nothing else.
  Order them most important first.
- "op" must be one of: lookup, count, sum, mean, median, min, max, difference, ratio, percent_change, argmax, argmin, compare, rank, trend, filter, boolean, multiple_choice, unanswerable.
- An argument is either a label string naming one of your evidence items, or a nested
  {"op": ..., "args": [...]} object.
- Aggregations over every evidence item take an empty args list, e.g.
  {"op": "sum", "args": []}.
- If the chart does not contain the answer, set "answerable" to false, use
  {"op": "unanswerable", "args": []}, and leave "model_answer" empty.
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

Greedy, fixed: `{"do_sample": false, "temperature": null, "top_p": null, "top_k": null, "num_beams": 1}`. Max new tokens: 512
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
| total | 12,000 | 12,000 |
| synthetic | 6,000 | 1,825 |
| ChartQA | 6,000 | 7,087 |
| RefChartQA | 0 | 3,088 |
| with boxes | 12,000 | 12,000 |
| with a plan | 6,952 | 2,930 |
| of those, compositional | 5,080 | 1,883 |

Deduplicated: 628 merges, of which
609 across ChartQA and RefChartQA.
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

## 12. Extensions and their entry gates

Planned only if the core result lands and quota remains: ChartQAPro transfer (entry gate:
Phase 7 complete and the extension approved), and the RefChartQA scaling ladder at
4,000 / 10,000 / 25,000 training rows (entry gate: Phase 6 stage 2 complete).

---

_Generated by `scripts/write_prereg.py`. Regenerating after the test split is opened would
defeat the purpose; the committed version is the record._
