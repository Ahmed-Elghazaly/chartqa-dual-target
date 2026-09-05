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

---

## 7. Subsystem by subsystem

`Prompt.md`'s SOURCE OF TRUTH section asks each major subsystem to be stated as *original
intention → current implementation → observed behaviour → limitations → new evidence →
revised conclusion*. Ten subsystems, in the order data moves through them.

### Data adapters — ChartQA, RefChartQA, synthetic

* **Intention** — three sources behind one `ChartRecord`, so everything downstream is
  source-agnostic.
* **Implementation** — `chartqa_records`, `refchartqa_records`, `synthetic_records` in
  `scripts/build_mixtures.py`.
* **Observed** — the abstraction leaked. `record.boxes` meant three different things by
  source (C2), and `meta[elements]` meant the *operands* on a synthetic record and the
  *whole chart* on a ChartQA one (M3). `record.table` still has two shapes.
* **Limitations** — a consumer reading those fields generically got different semantics per
  source, silently, and four defects came from it (0067, 0071, 0098, 0116).
* **New evidence** — targets agreed only because `_evidence_from` pruned ChartQA's elements
  to the plan's labels, which hid the divergence for months; then 0116 built a target that
  did not prune and produced *"point at everything"*.
* **Revised** — **unified (0124).** `ChartRecord.elements` is every mark; `ChartRecord.evidence`
  is the indices that answer *this* question, or `None` for unknown, which is what ChartQA
  is. Each source also tags per-element grounding and value provenance (0126). `record.table`
  having two shapes is the one part of this that remains open.

### Deduplication and cross-source fusion

* **Intention** — one function to find and merge duplicates across sources.
* **Implementation** — `data/dedup.py`, called before the stage cap.
* **Observed** — the merge produced fused records that **nothing downstream ever read**,
  because a mixture stores ids and training rehydrates from the readers (H2).
* **Limitations** — dedup and fusion were the same function, so a silent loss in one looked
  like normal behaviour in the other.
* **New evidence** — the two datasets share 86.9% of images but only 42.1% of questions, so a
  question-keyed merge recovers less than half of what matching on boxes does.
* **Revised** — separated. Dedup stops double-counting; fusion happens geometrically in
  `align_refchartqa.py` at 98.9% IoU ≥ 0.9, attached **in the reader** (0077, 0105).

### Element and evidence representation

* **Intention** — a chart element is a label, a value, a unit and a box; and the question's
  *evidence* is the subset of those elements that answers it.
* **Implementation** — `ChartRecord.elements` and `ChartRecord.evidence`, first-class fields
  since 0124. `annotation_boxes` still builds the elements; `meta[ELEMENTS_KEY]` is written
  only so records cached before the fields existed stay readable.
* **Observed** — labels are not unique on 22.6% of charts, and the target builder and the
  executor resolved collisions differently — first match against last (H3). Colour was
  produced by the annotation and dropped (H7).
* **Limitations** — identity was a label alone, which is not an identity.
* **New evidence** — every colliding chart carries a series name, and (series, label) is
  unique on 94.4% of them.
* **Revised** — colliding labels qualified as `"Democratic · 2019"`; colour carried on 96.9%
  of elements; a label that still collides is refused, not resolved by position (0083, 0087).
  ELEMENTS and EVIDENCE are now separate fields rather than one overloaded key, which is what
  `Prompt.md` Ideas 1, 2 and 5 asked for and what four defects argued for (0124). An element
  whose label is a *truncated render* of the plan's operand rejoins by unique prefix (0129).

### Plan mining

* **Intention** — recover the reasoning behind a gold answer, exactly, or not at all.
* **Implementation** — was `plans/mining.py`, searching backwards from the answer; now a
  language model reading finished records through five gates.
* **Observed** — the backwards miner is 94% precise and 15–25% recall, and its dominant
  failure is not a bug: several operations reproduce any given answer and it cannot choose
  (H4, H5).
* **Limitations** — the gates are arithmetic, so they cannot catch a plan that is right by
  luck, and they run on one input.
* **New evidence** — weakly supervised semantic parsing names this the *spurious program*
  problem and filters it by executing under perturbed inputs.
* **Revised** — LLM-only on the supply path (0085, 0088); `plans/distinguish.py` detects the
  undecidable cases (0097); `consensus` samples K times only where the evidence is silent
  (0100). The backwards miner survives as an independent cross-check.

### DSL and executor

* **Intention** — a small typed language whose programs a CPU can re-run exactly.
* **Implementation** — `plans/executor.py`, 20 operations, depth ≤ 4, arity ≤ 4.
* **Observed** — 93.3% of a random corpus sample is expressible, but that sample is dominated
  by templated questions; 40 human questions produced seven requests for missing operations.
* **Limitations** — `filter`, `rank` and `multiple_choice` are declared in `OPS` and raise.
* **New evidence** — a series-restricted fold is 8.6% of human questions and 0.1% of machine
  ones — the single most-requested gap.
* **Revised** — `within` added on measured demand (0090). Six requested operations remain,
  each with a number attached rather than an impression.

### Target construction

* **Intention** — a target the model can be taught, that its own executor accepts.
* **Implementation** — `train/targets.py`; refuses rather than repairs, and every refusal
  names its cause.
* **Observed** — four separate join defects reached it (0067, 0071, 0075, 0082), and a bare
  aggregate was silently truncated to eight items while the round-trip blamed the plan (C4).
* **Limitations** — it required a plan, discarding 31.2% of RefChartQA records that carry gold
  boxes, for a stage that is grounding-only by design.
* **New evidence** — those records supply +23,357 real grounding examples, nearly double the
  stage-1 cap.
* **Revised** — `build_grounding_only_target` emits boxes and answer and omits the plan
  (0104). Built and tested; **not yet wired into a mixture**.

### Synthetic generation

* **Intention** — exact boxes, exact answers and exact plans, by construction.
* **Implementation** — `synth/generator.py`, 24,000 examples, 8 chart types × 4 levels.
* **Observed** — it does what it claims: the geometry is proven against rendered pixels for
  every chart type. But 25% was chart families ChartQA does not contain, `difference` is 13.8×
  over-weighted, and no chart exceeds 7 marks against a real median of 10.
* **Limitations** — it was built to prove the *format* can be learned, which it did, and it
  does not resemble the target domain.
* **New evidence** — the curriculum literature: bridge a synthetic-to-real gap progressively;
  a stage that teaches format on charts maximally unlike the target is the worst case.
* **Revised** — absent chart types dropped by selection (0091). L1–L2 stay uniform for format;
  L3–L4 should match ChartQA's density and operation mix and **need regenerating** (0101).

### Image preprocessing

* **Intention** — let the processor own resizing; port coordinates to match it exactly.
* **Implementation** — `vision/coords.py`, factor 32 derived from the loaded processor.
* **Observed** — correct. No double resize; the coordinate port matches `smart_resize`.
* **Limitations** — the pixel budget is applied inside a silent `except: pass` (M1, open).
* **New evidence** — resolution is a reported column in RefChartQA's Table 2, and a 3B model
  at 768px matches a 9.6B model at 448px on grounding.
* **Revised** — verified unchanged (G1); resolution moved 512px → native once the session gate
  lifted (0095).

### Training

* **Intention** — two stages, a curriculum, and a control that isolates what structure buys.
* **Implementation** — `train/loop.py`, `collate.py`, QLoRA, effective batch 8, 12,000 per
  stage.
* **Observed** — the collate contract is careful and its subtleties are handled — the prompt
  boundary is measured with the image, the end-of-turn token is supervised, truncation is
  refused rather than accepted.
* **Limitations** — the loss spends 35.6% on boxes and **3.7% on the answer**, which carry
  half the metric each; 17.1% goes to labels and values no metric scores.
* **New evidence** — that imbalance only matters once the answer policy is known, because
  which answer is *scored* decides what the objective should weight (0096).
* **Revised** — nothing changed. The imbalance is recorded and the policy experiment is the
  prerequisite for touching it.

### Evaluation

* **Intention** — the official metrics, per subset, with the trained model and the baseline
  down the same path.
* **Implementation** — `eval/runner.py`, `eval/metrics.py` byte-faithful to upstream.
* **Observed** — correct, and better than believed: our metrics match RefChartQA's official
  evaluator to **0.068 points** across 11,690 predictions, which was being printed and never
  read as the cross-check it is.
* **Limitations** — it scores the model's stated answer, so the executor's verdict does not
  affect any number (H8).
* **New evidence** — 32.83 is not in the RefChartQA paper; the vendored file is TinyChart's
  output, and its M and PoT numbers reproduce exactly.
* **Revised** — report against Table 2's six models (0093); make all three answer policies
  scoreable from one generation set and decide on data (0096).

---

## The scripts, and what each one produces

**Nothing here is dead, and that is the point of the table.** Two scripts have zero
references from anywhere in the repo — `build_val_slices.py` and `refchartqa_audit.py` —
and both are essential: the first froze the validation slice every Phase 5 measurement uses,
the second produced the 200-row box audit that decision 0047 rests on. A reference count is
not a liveness check for a script whose job is to *produce an artifact once*, and deleting
one would make its artifact unreproducible (`DECISIONS.md` 0132 nearly made that mistake in
the other direction, on code that genuinely was dead).

| script | what it does | writes |
|---|---|---|
| `align_refchartqa.py` | give RefChartQA's boxes semantic identity from ChartQA elements | `<data>/refchartqa_aligned.jsonl` |
| `build_mixtures.py` | build the stage-1 and stage-2 mixtures | `data/mixture_stage{1,2}.json` |
| `build_sealed_images.py` | record the pixel hash of every val/test image | `data/sealed_images.json` |
| `build_semantic_audit.py` | the stratified manual audit set (0129) | `<data>/semantic_audit.jsonl` |
| `build_val_slices.py` | **freeze the validation slices Phase 5 iterates on** | `<data>/slices/*.jsonl` |
| `cache_refchartqa.py` | stream RefChartQA into a local record cache | `<data>/refchartqa_train.jsonl` |
| `characterise_official_evaluator.py` | run the official evaluator to characterise it | `verification/` |
| `check_ci.py` | CI status for *this* commit, not the last run | — |
| `check_credentials.py` | verify every credential and say what is wrong | — |
| `crosscheck_evaluators.py` | our metrics against the official ones | `verification/evaluator_crosscheck.json` |
| `e2e.py` | build real targets, **print some**, fail on composition drift | `data/composition_snapshot.json` |
| `gpu_budget.py` | live Kaggle quota against what is still to spend | — |
| `kaggle_run.py` | drive a Kaggle GPU kernel non-interactively | — |
| `make_presentation_figures.py` | figures, from our own generated charts | `presentation/figures/` |
| `measure_resolution_ladder.py` | visual-token cost vs input resolution (0095) | — |
| `measure_subtoken.py` | what fraction of grounding targets are sub-token | — |
| `measure_target_yield.py` | how many records become training examples, on CPU | — |
| `mine_plans.py` | mine and verify a plan per record with a language model | `audit/plans/` |
| `record_parquet_hashes.py` | expected SHA-256 per parquet, without downloading | `data/` manifest |
| `refchartqa_audit.py` | **the 200-row box audit behind decision 0047** | `data/refchartqa_audit.jsonl` |
| `reproduce_level_b.py` | the Level-B reproduction | `verification/level_b_reproduction.json` |
| `run_zeroshot.py` | zero-shot baselines on GPU | `outputs/phase5/` |
| `write_prereg.py` | generate `PREREGISTRATION.md` from recorded facts | `PREREGISTRATION.md` |

`audit/` holds one-off measurement scripts — 18 of them, tracked deliberately. Each is the
*evidence* for a number in `DECISIONS.md`: how it was measured, not just what it came to.
They are kept for reproducibility and are not expected to be re-run.
