# Findings, in the record `Prompt.md` asks for

The PRIORITIZATION section requires fifteen fields per meaningful finding, and TASK 2 requires
fourteen per major proposed change. The two overlap; where a finding carries a change, the
union is given — seventeen fields, with *why it fits this project* and *what would make us
reject it* added.

`AUDIT.md` carries the narrative for each of these and `DECISIONS.md` the full reasoning; this
is the structured index the brief specifies.

Confidence is stated as **high** only where a number was measured on real data with a stated
sample size.

---

## C1 — An evidence entry's value and box can describe different marks

| field | |
|---|---|
| **1. Current behavior** | `_evidence_from` took the value from the gold table and the box from the annotation, without checking they describe the same mark |
| **2. Original rationale** | 0067 — the table is authoritative for values because the annotation rounds differently |
| **3. Evidence from code** | `entry(label, table_values.get(label), element["bbox"])` joined two sources on a label alone |
| **4. Problem** | 9.2% of ChartQA evidence entries pointed at one mark and stated another's number |
| **5. External research** | — (a correctness bug, not a design question) |
| **6. Alternatives** | drop the table and use annotation values (rejected: 35 of 105 records then disagreed with their own answer); drop the check (rejected: silent) |
| **7. Recommended action** | refuse the record when the two sources disagree beyond 2% |
| **8. Expected benefit** | removes 9.2% of silently wrong supervision |
| **9. Risks** | loses records where the disagreement is benign rounding; the 2% tolerance is the guard |
| **10. Files** | `train/targets.py` |
| **11. Migration** | none — refusal only |
| **12. Tests** | `test_targets.py`, value-agreement cases |
| **13. Experiment** | none needed |
| **14. Priority** | **CRITICAL** |
| **15. Confidence** | high — measured, n=9.2% of a real sample |
| **Status** | ✅ fixed, 0075 |

## C3 — Two numeric parsers disagreed by 100× on every percentage

| field | |
|---|---|
| **1. Current behavior** | `mining.to_number('5.3%')` → 5.3; `executor.to_number('5.3%')` → 0.053. `'3 071'` raised |
| **2. Original rationale** | each was written for its own module; `eval.metrics.to_float` divides percents because the official evaluator does (0045) |
| **3. Evidence from code** | two independent implementations, neither referencing the other |
| **4. Problem** | a plan mined against 5.3 was executed against 0.053, so every percentage chart failed its own round-trip. 20.7% of charts also carry a spaced thousand |
| **5. External research** | — (the official evaluator is the authority and was not to be changed) |
| **6. Alternatives** | change `to_float` (**rejected**: it must stay byte-faithful or our numbers become incomparable); tolerate 100× in the comparison (rejected: hides real errors) |
| **7. Recommended action** | one shared `parse_numeric` for chart values; `to_float` untouched for answers |
| **8. Expected benefit** | measured: LLM mining acceptance 44% → 88% |
| **9. Risks** | none identified; 0 of 32,719 ChartQA and 0 of 3,996 RefChartQA answers carry a `%`, so the undivided scale cannot mismatch an answer |
| **10. Files** | `plans/executor.py`, `plans/mining.py`, `train/targets.py`, `plans/resolve.py` |
| **11. Migration** | none stored; changes values inside newly built targets only |
| **12. Tests** | `test_value_parsers.py`, including an AST walk that fails on any new misuse |
| **13. Experiment** | before/after on the same 25 proposals |
| **14. Priority** | **CRITICAL** |
| **15. Confidence** | high — measured, n=25 proposals and 3,000 charts |
| **Status** | ✅ fixed, 0082 and 0089 |

## H4 — The miner's dominant refusal is a collision the question resolves

| field | |
|---|---|
| **1. Current behavior** | `mine_plan` searches backwards from the gold answer and refuses when several operations reproduce it |
| **2. Original rationale** | the uniqueness rule was the guarantee of semantic correctness — one operation, no ambiguity |
| **3. Evidence from code** | `mining.py:282` returns `ambiguous` when `len(hits) > 1` |
| **4. Problem** | 53.9% of rows refused; 26.6% of *all* rows are `lookup` against an extremum, which one word of the question separates |
| **5. External research** | weakly supervised semantic parsing; spurious programs (Lee/Kim/Jung, EMNLP 2023) |
| **6. Alternatives** | a question-intent tie-breaker over the miner (**rejected**: 86–89% precision against the miner's 94%); more operations (rejected: worth ~7%, not 45%) |
| **7. Recommended action** | mine forwards from the question with a language model; retire the backwards miner to cross-check duty |
| **8. Expected benefit** | ~55% of unbiased ChartQA verified against 15–25% |
| **9. Risks** | a reader can be wrong in ways arithmetic cannot catch — mitigated by 0097's detector and 0100's consensus |
| **10. Files** | `plans/llm_mining.py`, `plans/teacher.py`, `scripts/mine_plans.py`, `scripts/build_mixtures.py` |
| **11. Migration** | plans move to a cache joined in the reader; records carry `plan=None` until mined |
| **12. Tests** | `test_teacher.py`, `test_distinguish.py`, plus the five gates' own tests |
| **13. Experiment** | acceptance rate and refusal profile on the first volume run |
| **14. Priority** | **HIGH** |
| **15. Confidence** | high — measured, n=4,000 rows and two 40-record hand audits |
| **Status** | ✅ resolved, 0085 and 0088 |

## H7 — 21.8% of human questions mention a colour, and nothing read the colour

| field | |
|---|---|
| **1. Current behavior** | `annotation_boxes` returned label, value, series and box, and dropped the colour field |
| **2. Original rationale** | **none recorded** — it was never a decision |
| **3. Evidence from code** | `_series_elements` never referenced `colors` or `color` |
| **4. Problem** | 21.8% of human questions (0.5% of machine) mention a colour and were unanswerable from our representation. Human questions are half the test split |
| **5. External research** | ChartQA's own literature confirms annotations *"omit essential visual encoding information such as bar or line colors"* on some charts |
| **6. Alternatives** | extract colour from the image (rejected: the annotation already has it); ignore colour questions (rejected: a fifth of half the metric) |
| **7. Recommended action** | read both annotation shapes; name a colour as a *set* of acceptable words |
| **8. Expected benefit** | 61.9% of colour-mentioning questions gain a matching series; 96.9% of elements carry a colour |
| **9. Risks** | a colour word inside a label is not a colour reference — guarded by requiring the whole label in the question |
| **10. Files** | `data/colours.py` (new), `data/chartqa.py`, `plans/teacher.py` |
| **11. Migration** | additive: a new key on elements, which live in `meta`, not in the output schema |
| **12. Tests** | `test_colours.py`, 19 cases |
| **13. Experiment** | human-split accuracy, before and after |
| **14. Priority** | **HIGH** |
| **15. Confidence** | high — measured, n=1,200 colour questions and 1,500 charts |
| **Status** | ✅ fixed, 0087 |

## H8 — The interpreter checks the answer; it does not replace it

| field | |
|---|---|
| **1. Current behavior** | every scoring path takes `model_answer`; the executor's verdict is a reported diagnostic |
| **2. Original rationale** | 0059 states it correctly — the executor makes arithmetic *"checkable rather than asserted"*. The README overstated it |
| **3. Evidence from code** | `eval/runner.py::score_item(gen.answer)`; no substitution anywhere |
| **4. Problem** | the project's headline claim was not implemented, and 3.7% of the loss carries half the metric |
| **5. External research** | execution-guided decoding arrives at the same intervention from the other direction (0102) |
| **6. Alternatives** | substitute unconditionally (**rejected**: a plan that will not run would score nothing); leave it and correct the claim (partially adopted — the claim is corrected) |
| **7. Recommended action** | make all three policies scoreable from one generation set; decide on data |
| **8. Expected benefit** | unknown by construction — that is the point |
| **9. Risks** | none: reporting only until a policy is chosen |
| **10. Files** | `plans/roundtrip.py`; `eval/runner.py` and `cli/evaluate.py` to report three |
| **11. Migration** | none |
| **12. Tests** | `test_roundtrip.py`, seven policy cases |
| **13. Experiment** | **required** — three accuracies on the Phase 5 generations |
| **14. Priority** | **HIGH** |
| **15. Confidence** | high that the gap is real; **none** on which policy wins |
| **Status** | ✅ measurable, 0096 |

## H9 — Synthetic charts are half the density of real ones

| field | |
|---|---|
| **1. Current behavior** | the generator emits 3–7 marks per chart, uniformly over 8 chart types and 4 levels |
| **2. Original rationale** | *"the primary source of plan supervision, given that the uniqueness rule admits only ~5.7% of real questions"* — true when written |
| **3. Evidence from code** | `synth/generator.py`'s own docstring; the manifest's table sizes |
| **4. Problem** | no synthetic chart exceeds 7 marks; 63.9% of real charts exceed 8, median 10, max 77. The model practises among 4 distractors and is tested among 10–77 |
| **5. External research** | synthetic-to-real curriculum literature: bridge the gap progressively, do not ignore it (0101) |
| **6. Alternatives** | reweight by selection (**impossible**: no selection produces a density never generated); ignore (rejected: grounding is half the metric and density *is* the difficulty) |
| **7. Recommended action** | regenerate L3–L4 against ChartQA's density and operation mix; keep L1–L2 uniform for format |
| **8. Expected benefit** | closes a 2.8× density gap and a 13.8× operation skew |
| **9. Risks** | hours of compute; sealed holdout seeds must not move |
| **10. Files** | `synth/generator.py`, then full regeneration |
| **11. Migration** | the 24,000-example manifest is replaced; the old one is outside the repo and not overwritten in place |
| **12. Tests** | existing geometry tests; rerun `audit/measure_synthetic_fit.py` |
| **13. Experiment** | **required** — human-split accuracy specifically |
| **14. Priority** | **HIGH** |
| **15. Confidence** | high on the gap (n=24,000 vs 1,500); **medium** that closing it improves the score |
| **Status** | specified, not implemented — 0091, 0098, 0101 |

## Grounding-only targets — supervision refused for want of a plan

| field | |
|---|---|
| **1. Current behavior** | `build_record` requires a plan and refuses without one |
| **2. Original rationale** | a target whose plan cannot execute teaches non-executable plans (0067) |
| **3. Evidence from code** | `build_record` raises *"no mined plan, and one cannot be derived"* |
| **4. Problem** | 31.2% of RefChartQA records have gold boxes and no plan, and **stage 1 is grounding-only by design** |
| **5. External research** | — (an internal contradiction, not a design question) |
| **6. Alternatives** | fill the plan with `unanswerable` (**rejected**: false); derive one (rejected: forbidden by `PLAN.md` 3.6) |
| **7. Recommended action** | emit boxes and answer, omit the plan, for stage 1 only |
| **8. Expected benefit** | RefChartQA 56.6% → **98.5%** supervisable; +23,357 real grounding records |
| **9. Risks** | stage 1 sees a different shape from stage 2 — mitigated by making it a strict *subset*, so stage 1 teaches a prefix stage 2 completes |
| **10. Files** | `train/targets.py`; `data/mixture.py` and `cli/train.py` to wire it |
| **11. Migration** | deliberately not `OUTPUT_SCHEMA`-valid; anything validating stage-1 targets must select by stage, as `build_answer_only_target` already does |
| **12. Tests** | `test_targets.py`, seven cases including the schema-invalidity assertion |
| **13. Experiment** | AP after stage 1, with and without |
| **14. Priority** | **HIGH** |
| **15. Confidence** | high on the supply (n=3,996); **medium** that it improves AP |
| **Status** | built and tested; **not yet wired** — 0104 |

## Native resolution

| field | |
|---|---|
| **1. Current behavior** | `image_max_pixels = 512 × 512` |
| **2. Original rationale** | 0060: native costs 17.72 h against a 10 h Kaggle session |
| **3. Evidence from code** | the constant, and a second copy at `cli/train.py`'s call site |
| **4. Problem** | the gate is gone (three accounts, verified resume), and native moves 11.9 points of targets out of the sub-visual-token bucket |
| **5. External research** | RefChartQA Table 2: TinyChart 3B at 768px matches Qwen-VL-Chat 9.6B at 448px on grounding |
| **6. Alternatives** | 448px (rejected: worse on both counts); raise `max_seq_len` too (**unnecessary** — measured p99 864 of 1,024) |
| **7. Recommended action** | `image_max_pixels = None` |
| **8. Expected benefit** | 53.2% → 41.3% of targets below one visual token |
| **9. Risks** | 17.72 h vs 9.92 h per 3,000 steps; one config value, trivially reverted |
| **10. Files** | `config.py`, `cli/train.py` |
| **11. Migration** | none stored; changes what training sees |
| **12. Tests** | existing coordinate tests are budget-parameterised |
| **13. Experiment** | AP@0.5 before and after — this is the change most likely to move grounding |
| **14. Priority** | **HIGH** |
| **15. Confidence** | high on the sub-token measurement; **medium** that it converts to AP |
| **Status** | ✅ adopted, 0095 |

---

## The remaining findings, in brief

Each carries the same fields in its `AUDIT.md` section and its decision record; they are
summarised here rather than repeated.

| # | finding | priority | files | migration | experiment | confidence | status |
|---|---|---|---|---|---|---|---|
| C2 | `record.boxes` used as AP ground truth across sources | CRITICAL | `cli/train.py` | none | none | high | ✅ 0076 |
| C4 | a bare aggregate silently truncated to 8 items | CRITICAL | `train/targets.py` | none | none | high | ✅ 0082 |
| C5 | the answer parser used on chart values in 3 more places | CRITICAL | `targets.py`, `resolve.py` | none | none | high | ✅ 0089 |
| H1 | RefChartQA grounding aligns to ChartQA elements | HIGH | `scripts/align_refchartqa.py` | a new cache file | matching threshold sweep | high | ✅ 0077 |
| H2 | the dedup merge is discarded before training | HIGH | `scripts/build_mixtures.py` | join must be in the reader | none | high | ✅ 0077 |
| H3 | labels non-unique on 22.6% of charts | HIGH | `data/records.py`, `train/targets.py` | labels change on colliding charts only | none | high | ✅ 0083 |
| H5 | the mining *direction* was the constraint | HIGH | see H4 | see H4 | see H4 | high | ✅ 0085 |
| H6 | pattern matching recognises templates, not language | HIGH | — | — | human-split accuracy | high | ✅ measured |
| H10 | folds indistinguishable on one element | HIGH | `plans/distinguish.py` | none | cost of refusing | high | ✅ 0097 |
| M1 | the pixel budget sits in a silent `except: pass` | MEDIUM | `modeling/backends/` | none | none | high | open |
| M2 | synthetic operation mix 13.8× skewed | MEDIUM | `synth/generator.py` | manifest regeneration | human-split accuracy | high | partly ✅ 0091 |
| M3 | `meta[elements]` means two different things | MEDIUM | `data/`, `train/targets.py` | would change synthetic records | none | high | recorded, 0098 |
| M4 | four constants copied rather than derived | MEDIUM | 4 modules | none | none | high | ✅ |
| G1 | no double resize | *no change* | — | — | — | high | ✅ verified |
| G2 | 32.83 is not in the RefChartQA paper | *reframes* | reporting only | none | none | high | ✅ 0093 |
| G3 | early stopping is correct | *no change* | — | — | — | high | ✅ verified |
| G4 | the output format is right | *no change* | — | — | — | high | ✅ 0094 |
| G5 | constrained decoding is disqualified | *no change* | — | — | — | high | ✅ 0099 |
