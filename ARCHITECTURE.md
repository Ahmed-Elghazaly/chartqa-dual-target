# How the system works, as it stands

`Prompt.md` Phase 1.8. The current state after the audit, in one pass from a chart image to a
scored number. Every claim here is either code you can read or a decision record you can check;
nothing is aspirational. Where the audit changed something, the old behaviour is named, because
the reason is usually more useful than the rule.

---

## 1. The shape of the thing

One small vision-language model (Qwen3-VL-2B-Instruct, QLoRA: 24.6M trainable of 1.45B) is
fine-tuned to answer a chart question by emitting **one JSON record**:

```json
{"answerable":true,
 "evidence":[{"label":"2019","value":245,"unit":"m","bbox":[412,180,486,742]}],
 "plan":{"op":"difference","args":["2019","2018"]},
 "model_answer":"47"}
```

Four things at once: whether the question is answerable, **where** it looked, **how** the answer
is computed, and the answer. Two of those are scored against published benchmarks — the boxes as
grounding (AP@0.5, P@F1 on RefChartQA) and the answer as relaxed accuracy (ChartQA). The plan is
the project's own contribution and has no published counterpart on these benchmarks (0094).

**What the plan is for, precisely.** A deterministic CPU executor re-runs it against the model's
own evidence. Today that makes the arithmetic **checkable**, not automatic: every scoring path
takes `model_answer`, and the executor's verdict is a reported diagnostic. Whether scoring the
*executed* value instead would do better is an open, cheap experiment — `answer_under` scores
three policies from one set of generations (0096, 0102). The README claimed the stronger version
until the audit found it untrue.

---

## 2. From a chart to a training target

```
ChartQA archive ──► chartqa_records()  ─┐
RefChartQA cache ─► refchartqa_records()├─► deduplicate ─► build_stage1 / build_stage2 ─► mixture (record IDs only)
synthetic manifest► synthetic_records() ─┘                                                      │
                                                                                                ▼
                                                        rehydrate ─► build_target() ─► one JSON string
```

**Records are built complete, then plans are attached** (0088). `chartqa_records` assembles the
image, boxes, labels, values, **series** and **colour**, and mines nothing. A reader mines plans
from those finished records (`scripts/mine_plans.py`), each plan passes five gates, and
`attach_mined_plans` joins the survivors back by record id. Plans used to be mined inside the
record builder, which fused two unrelated jobs and meant the miner only ever saw what the builder
happened to hand it — never the colours, which were being dropped one function away (0087).

**The join happens in the reader, never downstream.** A mixture stores record ids and training
rehydrates from these readers, so anything added after this point is discarded before training
sees it. That is how the dedup merge was silently lost (`AUDIT.md` H2).

**Element identity carries its series.** On a grouped chart `"2019"` names one bar per series,
and the target builder and the executor used to resolve that differently — first match versus
last. Colliding labels are now qualified as `"Democratic · 2019"`; the 77.4% of charts with no
collision keep their labels exactly (0083).

### What a target must survive

`build_target` refuses rather than repairs, and every refusal names its cause:

* the plan's operands exist as elements with boxes
* the gold table and the annotation agree about the value of each mark (0075)
* a label that still collides after qualification is refused, not resolved by position (0083)
* the record **round-trips**: its own plan, run against its own evidence, reproduces its own
  answer at the answer's own precision — not the 5% scoring tolerance, because 5% of the year
  2014 is a century (0045)
* it parses, and satisfies `OUTPUT_SCHEMA`

---

## 3. Where plans come from

**A language model, and only a language model** (0085, 0088). The deterministic miner is off the
supervision path.

That miner searched **backwards** — *which operations reproduce this gold answer?* — and had to
refuse whenever several did, which is 53.9% of rows. On a sorted bar chart the top value is
simultaneously `lookup(<its label>)` and `max` of its column, so an answer-first search must find
both and cannot choose. **That is what working backwards means, not a defect to patch.** It
survives as an independent cross-check in measurement, where its forced verdicts are useful.

Every proposal passes five gates, and one that fails any is **discarded, never repaired** —
repairing would make the pipeline the author of its own supervision:

1. shape — parses, known operation, within depth and arity
2. operands exist in the evidence
3. executes without raising
4. reproduces the gold answer at the answer's precision
5. stays inside the marked regions, where RefChartQA grounding exists

Two things the gates cannot do, both now handled:

* **They run on one input**, so a plan that coincides with the truth there passes. This is the
  *spurious program* problem from weakly supervised semantic parsing, and
  `plans/distinguish.py` detects it by executing under permuted evidence — recorded on the
  verdict, not yet a rejection (0097).
* **They are arithmetic**, so they cannot say which reading was meant when several compute the
  answer. `llm_mining.consensus` samples the reader K times and votes among plans that already
  verified, spent only where the evidence cannot decide (0100).

---

## 4. Training

Two stages, a curriculum, and a control.

| | data | what it is for |
|---|---|---|
| **stage 1** | synthetic (L1→L4) + audited real boxes | grounding, and the output format |
| **stage 2** | real ChartQA + RefChartQA + synthetic replay | the task |
| **control** | same as stage 2, answer only | isolates what the structure buys |

Stage 1 is a **curriculum stage**, so its distribution matters (0101): the literature's remedy
for a synthetic-to-real gap is to bridge it progressively, and a stage that teaches format on
charts maximally unlike the target is the worst case. L1–L2 stay uniform for format; L3–L4
should match ChartQA's operation mix and chart density, and **do not yet** — the corpus is
13.8× over-weighted on `difference` and no synthetic chart has more than 7 marks against a
real median of 10 (0091, 0098).

**Numbers that constrain everything.** `max_seq_len` 1,024; effective batch 8 (2 × 4);
12,000 records per stage — which is not a round number but the compute budget backwards:
12,000 ÷ 8 × 2 stages × 11.903 s/step = 9.92 h against a 10-hour Kaggle session (0092). That
gate has since been lifted — three accounts, and a resume verified against an uninterrupted
run — which is why resolution is now **native** rather than 512px, buying 11.9 points of
targets that were too small for a single visual token to resolve (0095).

Checkpoints every 100 steps carry adapter weights, optimizer state, scheduler state, RNG states
and the dataloader position. Early stopping maximises **negative** validation loss, because AP
cannot resolve a stopping signal at any affordable slice size (0069).

---

## 5. Evaluation

The same path evaluates the trained model and the zero-shot baseline, `--adapter` being the only
difference. Reported per subset — RefChartQA-H, -M and -PoT — never aggregated, because the
subsets differ by 30 points and an average of them means nothing (0002).

* **relaxed accuracy**, byte-faithful to the official ChartQA implementation, quirks included
* **AP@0.5** and **P@F1**, matching RefChartQA's own metrics
* **round-trip agreement**, a headline number: does the plan reproduce the stated answer (0059)

Our metrics agree with RefChartQA's official evaluator to **0.068 points** across 11,690 real
predictions, and the vendored prediction file is TinyChart's output — its M and PoT numbers
reproduce the paper exactly (0093). The 32.83 the project was originally anchored on **does not
appear in the RefChartQA paper**; results are reported against its Table 2 instead, where
ChartGemma (2B, 448px) is the size-matched baseline.

---

## 6. What is not built, and what is uncertain

Honest list, because a current-state document that reads as finished is a lie.

* **The plan cache holds two records.** The mining pipeline is built, tested and verified end to
  end; the mining itself has not been run at volume.
* **No training run has happened** under the audited configuration.
* **Six DSL operations are wanted and missing**, each with a number: a Yes/No comparison (8.0%
  of human questions), a threshold filter, a count of series, a constancy check, `product`, and
  an argmax over a computed quantity (0090).
* **L3–L4 need regenerating** against ChartQA's operation mix and density (0101).
* **The answer policy is undecided** — `stated`, `executed` or `executed_or_stated` — and Phase 5
  settles it from data it will produce anyway (0096).
* **`meta[elements]` still means two things**: the operands on a synthetic record, the whole
  chart on a ChartQA one (0098).
