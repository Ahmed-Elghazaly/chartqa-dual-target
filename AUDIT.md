# The audit

Six documents, merged. They were separate because `Prompt.md` asked for several *specific*
artifacts — a verdict answering ten named questions, a fixed 15-point record per finding, a
row per item of Task 1's 99, a line-by-line checklist — and each is still here, whole, as
its own part. Nothing was summarised away in the merge; only the top-level headings moved
down one level so the parts nest.

| part | was | answers |
|---|---|---|
| 1 | `VERDICT.md` | which decisions were right, wrong, outdated, uncertain |
| 2 | `AUDIT.md` | each finding in full, with its measurement |
| 3 | `FINDINGS.md` | the same findings in the 15-point record the brief specifies |
| 4 | `AUDIT_COVERAGE.md` | Task 1's 99 items, one row each |
| 5 | `PROMPT_CHECKLIST.md` | the brief line by line, and what is **not** done |
| 6 | `AUDIT_PLAN.md` | how the work was tracked, including what it cost |

The *why* behind any decision is in `DECISIONS.md`, which is append-only and is not
duplicated here.

---

# Part 1 — The verdict

> The ten questions `Prompt.md`'s FINAL EXPECTATION asks, answered.
>
> *Was `VERDICT.md`.*

`Prompt.md`'s FINAL EXPECTATION asks ten questions. This answers them in order, then gives the
implementation plan for what remains (Phases 5 and 6). Everything here is traceable to a
decision record; nothing is asserted that was not measured.

---

### 1. Which previous decisions remain strong

| decision | why it holds |
|---|---|
| **0014** — emit few boxes | confirmed useful somewhere it was not designed for: because evidence is selected by what the plan *names*, 90.3% of questions never touch the evidence cap (0084) |
| **0026** — checkpoints save RNG state | the resume check it added is what makes 90 h/week across three accounts usable at all (0092, 0095) |
| **0045** — mine at the answer's own precision | not the binding constraint and never was, but correct: 5% of the year 2014 is a century |
| **0064** — repair by drop/unwrap, never add | ruled out constrained decoding on its own terms (0099) |
| **0069** — early-stop on loss, not AP | AP cannot resolve a stopping signal at any affordable slice size (±8.7 to ±12.2 pts); the sign convention is correct and tested |
| **G1** — no double resize | re-verified; the processor owns resizing and our coordinate port matches it |

### 2. Which were wrong

| decision | the error | fixed by |
|---|---|---|
| **0067** — evidence selection | selected the first eight boxes, so a plan's label matched nothing; 1 of 636 records yielded a target | 0075 |
| **0082's own scope** | fixed the answer/value parser confusion at two call sites and did not ask where else it lived. It lived in three more, one of which made every percentage chart's evidence 100× too small | 0089 |
| **0080's conclusion** | "45% of questions need new operators" — true of the sample, which was drawn from miner failures. Corpus-wide it is ~7% | 0081 |
| **the `README` claim** | *"the arithmetic never depends on the model doing mental maths"*. Every path scores the model's own answer | 0096 |
| **`AUDIT.md` H3's magnitude** | 74.2% of charts with non-unique labels, from a filename-ordered sample that is 40.5% `multi_col` against 15.6% real. True figure 22.6% | 0083 |

### 3. Which have become outdated

| decision | the premise that expired |
|---|---|
| **0060** — 512px | native was rejected *only* because 17.72 h broke a 10 h Kaggle session. Three accounts and a verified resume removed that gate (0095) |
| **0052** — 32.83 is unreproducible | true of 32.83, which is **not in the RefChartQA paper at all**. Two other published numbers reproduce exactly (0093) |
| **0078/0079** — improving the deterministic miner | its recall is bounded by *direction*, not by search. Improving it was work on a component pointed the wrong way (0085) |
| **`synth/generator.py`'s docstring** | still says it is *"the primary source of plan supervision, given that the uniqueness rule admits only ~5.7%"*. The uniqueness rule is off the path (0091) |
| **0092** — the 12,000 cap | correct as compute arithmetic; the compute constraint is gone, so it is now a choice |

### 4. Which are merely uncertain

* **Whether the executed answer beats the stated one.** Zero-shot, 20% of plans did not execute and 40% disagreed with the stated answer — and nothing measured which side was right (0096).
* **Whether K-sample consensus is worth its cost**, and at what K. Implemented; unmeasured (0100).
* **Whether an underdetermined plan should be refused.** The detector works; the cost of refusing is unpriced on real proposals (0097).
* **Whether mined supervision performs as well as supervision exact by construction.** Provenance is complete and unread (0105).

### 5. What should change immediately

Ordered by benefit per unit of risk. Items 1–3 are done; 4–6 are specified below.

| | change | status |
|---|---|---|
| 1 | one shared value parser, guarded by an AST test | **done** (0089) |
| 2 | series and colour into element identity | **done** (0083, 0087) |
| 3 | native resolution | **done** (0095) |
| 4 | wire grounding-only targets into stage 1 | specified, not wired |
| 5 | regenerate L3–L4 against ChartQA's density and operation mix | specified, needs compute |
| 6 | run the mining at volume | pipeline ready; needs a reader |

### 6. What should be experimentally tested

Every one of these is answered by data Phase 5 and 6 produce anyway. **None needs a separate run.**

| experiment | settles | how |
|---|---|---|
| three answer policies scored on one generation set | is the executor a calculator or a checker? | `answer_under` (0096) |
| results stratified by supervision provenance | is mined supervision as good as exact? | `eval/stratified.py` already groups by a field (0105) |
| accepted plans flagged underdetermined | what refusing them would cost | `Verdict.underdetermined` (0097) |
| resample-on-self-disagreement | does disagreement predict being wrong? | same generations as the answer-policy test (0102) |
| 3 training seeds | is a difference real or noise? | ~30 h, affordable now |
| the RefChartQA scaling ladder | does more data still help? | ~20 h; decides whether to raise the 12,000 cap |

### 7. What should remain unchanged

* **The JSON output format.** Short keys save 0 tokens/item on this tokenizer; a line format saves 32% and costs the schema, unambiguous parsing and the model's JSON priors, for +2.2% of questions — and we are not sequence-constrained (0094).
* **No constrained decoding.** It removes refusal and forces a box, and one spurious box takes AP from 1.00 to 0.68 (0099).
* **The uniqueness rule in `plans/mining.py`**, which is retired from the supervision path but kept as an independent cross-check. At 94% precision it is a useful second opinion, and it caught two of my errors (0085).
* **`eval/metrics.to_float`**, byte-faithful to the official evaluator, quirks included.

### 8. How the changes interact

Four couplings, and three of them are why work is sequenced rather than parallel.

1. **Grounding-only targets ↔ stage-1 composition.** +23,357 real grounding records is nearly double the stage-1 cap, so wiring them changes the synthetic/real mix — and fixes the density gap at the same time, because real charts have the density synthetic ones lack. **Wire this before regenerating**, or L3–L4 may be regenerated to fill a budget that real data has already filled.
2. **Native resolution ↔ the evidence cap.** Native costs 178 more visual tokens; the cap costs 47 per item. Both draw on the same 1,024. Measured together: p99 864, worst case 875 — safe, but they cannot be raised independently again without re-measuring.
3. **The answer policy ↔ the training objective.** If `executed` wins, then `model_answer`'s 3.7% of the loss is nearly irrelevant and the 17.1% spent on labels and values is the real answer supervision. If `stated` wins, a sixth of the objective is scaffolding for a calculator nobody reads. **Settle the policy before re-weighting anything.**
4. **The spurious-plan detector ↔ K-sample consensus.** They compose: sample K times *only* where the evidence cannot decide. Adopting consensus everywhere would cost K× for no gain on records a single pass already settles.

### 9. What measurable benefit we expect

Stated as predictions, so they can be wrong.

| change | expected benefit | measured how |
|---|---|---|
| native resolution | **11.9 points** fewer targets below one visual token → grounding AP | AP@0.5 per subset, before/after |
| grounding-only targets | RefChartQA **56.6% → 98.5%** supervisable; stage 1 becomes mostly real | target yield, and AP after stage 1 |
| series + colour identity | 22.6% of charts stop resolving labels ambiguously; 21.8% of human questions become answerable | mined-plan acceptance rate, human-split accuracy |
| LLM mining | ~55% of unbiased ChartQA verified, against 15–25% deterministic | targets built per 1,000 records |
| L3–L4 regeneration | closes a 13.8× operation skew and a 2.8× density gap | accuracy on the human split specifically |

**No claim here is that any of these will improve the final number.** They are each a defect removed or a supply increased, with the measurement that will say whether it mattered.

### 10. How we will know whether the revised system is better

* **Against six published models**, not one unverifiable number: RefChartQA Table 2, three splits, four metrics, with ChartGemma (2B, 448px) as the size-matched baseline (0093).
* **Against our own zero-shot baseline** through the identical path, `--adapter` the only difference — the comparison `PREREGISTRATION.md` seals before results exist.
* **Against the direct-answer control**, trained on the same records, which is what separates ordinary domain adaptation from the contribution of grounding, plans and execution.
* **With intervals, not point estimates.** `Cell` already refuses a point estimate without one; three seeds make them real.
* **Reported per subset.** RefChartQA-H, -M and -PoT differ by 30 points and an average of them means nothing (0002).

---

### Implementation plan for what remains (Phase 6)

#### A. Wire grounding-only targets into stage 1

* **Files** — `data/mixture.py` (`build_stage1`), `scripts/build_mixtures.py`, `cli/train.py` (target selection per stage).
* **Ordering** — first; it changes what the other work is sized against.
* **Migration** — none to stored artifacts. Mixtures hold record ids and rehydrate, so a rebuild picks the new targets up. **The join must be in the reader**, per `AUDIT.md` H2.
* **Compatibility** — the target is deliberately not `OUTPUT_SCHEMA`-valid, so anything validating stage-1 targets against the full schema must select by stage, as `build_answer_only_target` already does.
* **Tests** — extend `test_mixture.py` for composition; the builder's own tests exist.
* **Rollback** — one flag; the builder is additive and nothing else calls it.

#### B. Run the mining at volume

* **Files** — none; `scripts/mine_plans.py` is complete.
* **Ordering** — parallel with A. Independent.
* **Artifacts** — writes `~/.cache/chartqa_dt/data/chartqa_plans.jsonl`; a rerun replaces by record id and never duplicates.
* **Validation** — every plan passes five gates; report acceptance rate, refusal profile, and requested operators.
* **Rollback** — delete the cache; records return to `plan=None`.
* **Blocked on** — a reader. Batches are plain text for a Claude Code session or Codex; `--api` exists if a key ever does.

#### C. Regenerate L3–L4

* **Files** — `synth/generator.py`, then a full regeneration.
* **Ordering** — **after A**, so the real-data supply is known before synthetic is sized.
* **Artifacts** — the 24,000-example manifest is regenerated. Sealed holdout seeds must not move.
* **Targets** — ChartQA's operation mix (`lookup` ~64%, `argmax`/`argmin` ~21%) and density (median 10 marks, p90 24).
* **Validation** — rerun `audit/measure_synthetic_fit.py`; the skews should close.
* **Rollback** — keep the current manifest; it is outside the repo and not overwritten in place.

#### D. Settle the answer policy

* **Files** — `eval/runner.py` and `cli/evaluate.py`, to report three accuracies instead of one.
* **Ordering** — with the first evaluation run; it needs generations, not a separate experiment.
* **Rollback** — reporting only; nothing downstream depends on it until a policy is chosen.

#### Not planned, deliberately

Loss weighting by confidence, K-sample consensus everywhere, refusing underdetermined plans, and the six missing DSL operations. Each is built or specified and each awaits the measurement that would justify it. `Prompt.md`: *"Do not permanently adopt an experimental change merely because it sounds better."*

---

### Appendix — the self-critique, in the form Phase 2 asks for

*Original rationale → what the code now does → what we now know → does it still hold →
revised conclusion.* Sixteen decisions, every one that the audit examined closely enough to
change or confirm.

**0002 — 32.83 is the RefChartQA human subset.** *Rationale:* `IDEA.md` treated it as *the*
target without saying which split, and the splits differ by 30 points. *Code:* every grounding
number is reported per subset and never aggregated. *Now know:* **32.83 is not in the
RefChartQA paper at all**; the best published RefChartQA-H AP@0.5 is 27.81. *Holds?* The
per-subset rule holds and is more important than ever. The anchor does not. *Revised:* report
against Table 2's six models; keep 32.83 only as a note about `IDEA.md`'s provenance (0093).

**0014 — emit few boxes.** *Rationale:* one spurious box takes AP from 1.00 to 0.68, so
precision beats recall on this metric. *Code:* evidence is selected by the labels a plan
names, not by taking the first N. *Now know:* that choice pays off somewhere it was not
designed for — 90.3% of questions never touch the evidence cap, because a `lookup` needs one
item however large the chart. *Holds?* Yes, more strongly. *Revised:* unchanged; it is also
the reason raising `MAX_EVIDENCE` buys so little (0084).

**0026 — checkpoints save RNG state.** *Rationale:* dropout makes a resume unverifiable
without it; the check caught a 0.0438 divergence. *Code:* every 100 steps, with
`assert_resume_matched`. *Now know:* it is what makes 90 h/week across three accounts usable,
which is what removed the constraint behind 0060. *Holds?* Yes. *Revised:* unchanged, and now
load-bearing for two later decisions (0092, 0095).

**0037 / 0060 — 512 pixels.** *Rationale:* native cost 17.72 h against a 10 h Kaggle session;
448 was cheaper but resolved fewer targets. *Code:* `image_max_pixels = 512 × 512`, restated
at the CLI call site. *Now know:* the gate is gone, native moves 11.9 points of targets out of
the sub-visual-token bucket, and the sequence still fits at p99 864 of 1,024. *Holds?* No —
the constraint expired. *Revised:* native (0095). 0060 is superseded, not overturned: it was
right under its constraint, and the measurement it made of the option it rejected is what made
this decidable later.

**0041 — empty args mean fold over everything.** *Rationale:* a compact form so an L3
aggregate stays inside the schema's `maxItems: 4`. *Code:* `FOLD_OPS` with no string args
folds over the evidence list. *Now know:* the guard that protected it required the plan to
*name* a label, so a **bare** `argmax()` — the common case — fell through to a branch that
silently kept the first eight elements. *Holds?* The convention holds; its guard did not.
*Revised:* the guard no longer requires named labels (0082).

**0045 — mine at the answer's own precision.** *Rationale:* 5% of the year 2014 is a century.
*Code:* `matches_gold` uses the answer's written precision, not the scoring tolerance. *Now
know:* the tolerance was never the binding constraint — the mining *direction* was. *Holds?*
Yes, and it was challenged and survived. *Revised:* unchanged (0085).

**0052 — 32.83 cannot be reproduced.** *Rationale:* the official evaluator on the vendored
file gave 28.33 / 71.25 / 59.66 against a published 32.83 / 59.28 / 39.32. *Code:*
`scripts/reproduce_level_b.py`, which still runs and still reports exactly that. *Now know:*
the file is **TinyChart's** predictions — its M and PoT reproduce the paper *exactly* — and
32.83 is in no table of the paper. *Holds?* Its conclusion about 32.83 holds; its stopping
point did not. *Revised:* a published number *does* reproduce, the gate is met, and our
evaluator is validated to 0.068 points on 11,690 predictions (0093).

**0059 — round-trip agreement is a headline number.** *Rationale:* parse and schema validity
can both be 100% while every plan computes something else, in which case the plan is
decoration. *Code:* measured and reported. *Now know:* it is the *only* place the executor's
verdict appears — no score depends on it. *Holds?* Yes, and it is more important than it
looked. *Revised:* unchanged; extended by making the three answer policies scoreable (0096).

**0064 — repair by drop, unwrap, never add.** *Rationale:* discarding an offending item beats
discarding a good record, and nothing may be invented. *Code:* `parse_record`, every repair
counted. *Now know:* it is exactly why constrained decoding is disqualified — a
schema-forcing decoder *adds*, and would force a box the model does not have. *Holds?* Yes.
*Revised:* unchanged, and now the reason a whole technique was rejected (0099).

**0067 — evidence is selected by the plan's labels.** *Rationale:* the first eight boxes
produced targets whose plan referenced a label that was not among them — 1 of 636 records
usable. *Code:* selection by label, with the cap applied to what the plan needs. *Now know:*
the join it fixed had a second half — the *value* and the *box* could still describe different
marks. *Holds?* Yes, incompletely. *Revised:* amended by the value/box agreement gate (0075).

**0069 — early-stop on validation loss, not AP.** *Rationale:* AP needs generation, and at
affordable slice sizes its 95% CI is ±8.7 to ±12.2 points — too noisy to detect improvement.
*Code:* the evaluator returns **negative** loss so the maximising stopper is correct. *Now
know:* the sign is the dangerous part and is already guarded by a test written for exactly
that. *Holds?* Yes. *Revised:* unchanged; confirmed correct.

**0071 — the elements key is a shared constant.** *Rationale:* the synthetic reader wrote
`evidence` where the target builder read `elements`, and all 12,000 stage-1 targets were lost
silently. *Code:* one constant, used by every reader. *Now know:* the same *shape* of defect
survived under a different name — the key is shared but its **meaning** is not: the operands
on a synthetic record, the whole chart on a ChartQA one. *Holds?* The fix holds; the class of
defect recurred. *Revised:* recorded and measured, not yet unified (0098).

**0075 — refuse when the table and the annotation disagree.** *Rationale:* 9.2% of evidence
entries pointed at one mark and stated another's number. *Code:* `values_agree`, 2% tolerance.
*Now know:* it used the **answer** parser on chart values, so it read `'43.6%'` as 0.436 and
refused correct records for a disagreement it had invented. *Holds?* The gate holds; its
implementation was wrong. *Revised:* uses `parse_numeric` (0089).

**0078 / 0079 — measure and improve the deterministic miner.** *Rationale:* it is 94% precise
and 19% recall, so the recall is where the work is. *Code:* the miner, plus two rounds of
recall work. *Now know:* recall is bounded by **direction**, not by search. An answer-first
search must refuse whenever several operations reproduce the answer, and that is 53.9% of rows.
*Holds?* The measurements hold; the conclusion drawn from them did not. *Revised:* the miner
is off the supervision path and kept as a cross-check (0085, 0088).

**0080 — the DSL is the binding constraint.** *Rationale:* 18 of 40 records needed operations
we do not have. *Code:* the five-gate verifier, unchanged. *Now know:* those 40 were drawn
from *miner failures*, which over-represent hard question types. On an unbiased sample 93.3%
are expressible. *Holds?* No — for the corpus. Yes — for human questions specifically, which
is a different and later finding. *Revised:* withdrawn as stated, and re-derived correctly
per question origin (0081, 0086, 0090).

**0091 — synthetic data is uniform by design.** *Rationale:* it was *"the primary source of
plan supervision, given that the uniqueness rule admits only ~5.7% of real questions."*
*Code:* the generator, unchanged, and its docstring still says that. *Now know:* the uniqueness
rule is off the path, so the premise expired; and uniform coverage is right for teaching a
format and wrong for teaching a prior. *Holds?* No. *Revised:* stage 1 is a curriculum stage,
so L1–L2 stay uniform and L3–L4 should match ChartQA (0101).

**0092 — the 12,000 cap.** *Rationale:* none was ever written down. *Code:* two constants with
no comment. *Now know:* it is the compute budget backwards — 12,000 ÷ 8 × 2 × 11.903 s = 9.92 h
against a 10 h session — and that gate has lifted. *Holds?* As arithmetic, yes. As a
constraint, no. *Revised:* the derivation is now written at the constant; the cap is unchanged
until the scaling ladder says whether more data helps.

---

# Part 2 — The findings

> Each finding in full: what was measured, what it cost, what changed.
>
> *Was `AUDIT.md`.*

Started 2026-08-29. Everything below is measured against the **current working tree**, not
recalled. Reproduction scripts are in `audit/`, outputs beside them.

**Status:** empirical audit of the data and supervision path complete; both CRITICAL
findings fixed and verified. External research and the remaining subsystems (training
objective, output format, DSL, synthetic distribution, LLM-assisted mining) in progress.

#### Resolved

**C1 → `DECISIONS.md` 0075.** `values_agree` refuses a record when the gold table and the
annotation disagree about which mark a label names. Genuine disagreements **110 → 0**, at a
cost of **55 records (0.9% of stage-2 yield)**. The audit's first recommendation here was
*wrong* and measurement reversed it: the 61 percent-scale cases are required by the official
metric (`relaxed_correctness(gold="81.9%", pred="0.819")` is True, `pred="81.9"` is False),
so they are kept deliberately.

**C2 → `DECISIONS.md` 0076.** `grounding_truth_for` gives the AP monitor question-specific
ground truth only. ChartQA records now contribute to the answer metrics and not to AP,
because ChartQA has no per-question grounding to score against.

---

### Summary

| # | finding | priority | confidence |
|---|---|---|---|
| C1 | An evidence entry's **value and box can describe different marks** — 9.2% of ChartQA evidence entries | **CRITICAL** | ✅ **FIXED** (0075) |
| C2 | `record.boxes` means different things per source, and validation AP uses it as ground truth | **CRITICAL** | ✅ **FIXED** (0076) |
| H1 | RefChartQA grounding aligns to ChartQA elements at **99.2% exact** — 1,896 records could gain real labels, values and plans | **HIGH** | high, measured |
| H2 | The dedup **merge is discarded** before training sees it | **HIGH** | high, by construction |
| H3 | Labels are non-unique on **22.6%** of charts (not 74.2% — that sample was biased); target builder and executor resolved duplicates *differently* | **HIGH** | ✅ **FIXED** (0083) |
| M1 | The processor pixel budget is applied inside a silent `except: pass` | MEDIUM | high |
| **H10** | **`argmax`, `argmin` and the folds are indistinguishable on one element**, so an arithmetic gate cannot catch a plan that is right by luck | **HIGH** | ✅ **DETECTED** (0097) |
| **H4** | **The miner's dominant refusal is a `lookup` vs `max`/`min` tie — 26.6% of all rows.** The table cannot say which the question asked for; one word of the question can | **HIGH** | high, measured (n=4,000) |
| **C3** | **`mining` and `executor` parsed every percentage 100x apart**, and spaced thousands raised — invisible to the old pipeline, halves the new one | **CRITICAL** | ✅ **FIXED** (0082) |
| **C4** | **A bare aggregate lost its evidence silently** — `argmax()` on a chart with >8 elements kept the first 8 and the round-trip blamed the plan | **CRITICAL** | ✅ **FIXED** (0082) |
| **C5** | **The answer parser was used on chart values in three more places** than 0082 fixed — a correct plan executed to 0.821 against a gold answer of 82.1 | **CRITICAL** | ✅ **FIXED** (0089) |
| **H5** | **The mining direction, not the miner, was the constraint.** Searching backwards from the answer must refuse whenever several operations reproduce it — 53.9% of rows | **HIGH** | ✅ **RESOLVED** (0085, 0088) |
| **H6** | **Pattern matching recognises templates, not language** — 53.5% of machine questions against 14.8% of human ones — so pattern-mined supervision is ~92% machine against a **50/50** test split | **HIGH** | high, measured |
| **H7** | **21.8% of human questions mention a colour**, and every annotation carried the colour while nothing read it | **HIGH** | ✅ **FIXED** (0087) |
| **H8** | **The interpreter never replaced the answer.** Every path scores `model_answer`; the README claimed otherwise | **HIGH** | ✅ measurable (0096) |
| **H9** | **Synthetic charts never exceed 7 marks**; 63.9% of real charts have more than 8 (median 10, max 77). Cannot be fixed by reweighting | **HIGH** | high, measured |
| **M2** | The synthetic corpus is 13.8x over-weighted on `difference` and 2.6x under on `lookup`; 25% was chart types ChartQA does not contain | MEDIUM | ✅ partly fixed (0091) |
| **M3** | `meta[elements]` means *the operands* on synthetic records and *the whole chart* on ChartQA ones; `record.table` has two shapes | MEDIUM | high, measured |
| **M4** | **Four constants were copied rather than derived** — `MAX_EVIDENCE`, the numeric parsers, `ALLOWED_OPS`, the pixel budget. Each would have drifted silently | MEDIUM | ✅ **FIXED** (0084, 0089, 0090, 0095) |
| **G2** | **32.83 is not in the RefChartQA paper.** Results are reported against its Table 2 instead; our evaluator agrees with the official one to **0.068 points** | *reframes the claim* | ✅ verified (0093) |
| **G3** | **Early stopping is correct** — AP cannot resolve a stopping signal, the evaluator returns negative loss, and the sign is tested | *no change* | high, verified |
| **G4** | **The output format is right** — short JSON keys save 0 tokens/item on this tokenizer, and we are not sequence-constrained | *no change* | ✅ verified (0094) |
| **G5** | **Constrained decoding is disqualified**, not merely unused: it would remove refusal and force a box, and one spurious box takes AP 1.00 → 0.68 | *no change* | ✅ evidenced (0099) |
| G1 | **No double resize** — the processor owns resizing and our coordinate port matches it exactly | *no change* | high, verified |

---

### C1 — An evidence entry's value and its box can describe different marks

**Current behaviour.** `targets._evidence_from` builds each evidence entry as

```python
value = table_values.get(label, element.get("value"))   # FIRST numeric cell of that table row
bbox  = by_label[label]["bbox"]                         # FIRST element in ANNOTATION order
```

so the *value* comes from the gold table and the *box* from the annotation. Nothing joins
them beyond a shared label string.

**Original rationale.** `DECISIONS.md` 0067: reading values from the annotation made 35 of
105 planned records disagree with their own answer, so the table became the value authority
and the annotation the box authority. That fixed the round-trip failure it was aimed at.

**Evidence from current code.** `audit/measure_value_box_agreement.py` over the 2,401
ChartQA records in `data/mixture_stage2.json` that currently build a target:

| | |
|---|---:|
| evidence entries examined | 1,893 |
| entries where the emitted value ≠ the boxed element's value | **174 (9.2%)** |
| — genuine table↔annotation disagreement | 110 |
| — percent scaling (`to_float("81.9%") → 0.819` vs annotation `81.9`) | 61 |
| — rounding | 3 |
| records shipping at least one such entry | **86 of 2,401 (3.6%)** |

The genuine disagreements come in **swapped pairs**:

```
Finland         emits 9.4    boxes the mark whose value is 9.9
Hungary         emits 9.9    boxes the mark whose value is 9.4
United Kingdom  emits 12.5   boxes the mark whose value is 14.2
Portugal        emits 14.2   boxes the mark whose value is 12.5
```

**Problem.** This is *wrong grounding supervision*: the target teaches the model to draw a
box around one mark and state a different mark's number. It is precisely the association the
project exists to teach.

**Why nothing caught it.** The round-trip check executes the plan over the evidence and
compares with the stated answer. When the plan does not consume the mismatched value — a
`count`, an `argmax`, or a plan referencing other labels — the check passes with the wrong
value in place. This is the brief's point made concrete: **executor agreement does not prove
semantic correctness.**

**Recommended action.** A validation gate in `_evidence_from`: when the table's value for a
label and the boxed element's value disagree beyond the annotation's own rounding, **refuse
the record**. Separately, stop applying the metric's percent conversion when parsing a table
for values — `to_float` is the *scoring* parser and its `%`→/100 behaviour is correct there
and wrong here.

**Expected benefit.** Removes ~3.6% of ChartQA records and, with them, a class of
supervision that teaches the wrong thing. Costs yield; the brief is explicit that this is the
right trade.

**Risks.** Losing 86 records is negligible against 2,401. The real risk is the opposite one —
that the disagreement indicates the *box* is wrong rather than the value, in which case
records we currently keep are worse than the count suggests.

**Files.** `src/chartqa_dt/train/targets.py`.
**Tests required.** A record whose table and annotation disagree must be refused with a
stated reason; a percent-formatted table must not be silently divided by 100; a
single-series record must be unaffected.
**Experiment.** Re-measure target yield and the mismatch rate after the gate; both must go
to zero mismatches.
**Priority CRITICAL. Confidence high.**

---

### C2 — `record.boxes` has no single meaning, and validation AP treats it as ground truth

**Current behaviour.** Four writers disagree about what the field holds:

| writer | what `boxes` contains |
|---|---|
| `data/chartqa.py` | **every element** in the chart |
| `data/refchartqa.py` | **this question's** gold grounding |
| `scripts/build_mixtures.py` (synthetic) | **this question's** exact evidence |
| `data/dedup.py` | the **union** of whichever two merged |

**Evidence.** `audit/measure_boxes_semantics.py` on `data/mixture_stage2.json`:

| source | records | median `boxes` | median elements |
|---|---:|---:|---:|
| chartqa | 2,408 | **10** | 10 |
| refchartqa | 1,896 | **1** | 0 |
| synthetic | 2,000 | 2 | 2 |

**Problem.** `cli/train.py` builds the validation-monitoring items with
`"boxes": list(record.boxes or [])`, and `train/monitor.py` uses that as grounding ground
truth for AP@0.5. So for ChartQA records the metric scores the model against **every element
in the chart** while the model is trained to emit only what the answer needs:

> **2,321 of 2,403 ChartQA records (96.6%) carry more ground-truth boxes than their own
> target emits — by a median factor of 10×.**

Validation AP is therefore invalid on those records, capped near 1/10 recall for a reason
unrelated to the model. `PLAN.md` 6.6 uses the monitoring curve to decide whether to extend
training, so the defect reaches a decision.

**Recommended action.** Two separate fixes.
1. **Immediately:** the AP monitor must use *question-specific* ground truth only — i.e.
   records whose boxes are question grounding (RefChartQA, synthetic), never a chart-element
   list. This is a small change in `cli/train.py`'s holdout construction.
2. **Structurally:** give the record two fields with distinct contracts rather than one
   overloaded one — chart **elements** and question **evidence**. This is Ideas 1, 2 and 5,
   and it is what makes the first fix impossible to regress.

**Priority CRITICAL for (1), HIGH for (2). Confidence high.**

---

### H1 — RefChartQA grounding aligns to ChartQA elements almost perfectly

**The opportunity.** RefChartQA supplies per-question grounding but no labels and no values,
so `targets._evidence_from` falls to its last branch and names the evidence `item1, item2,
…`. Those records therefore teach *where to point* but nothing about what was pointed at,
and they carry no plan at all.

**Evidence.** `audit/measure_refchartqa_alignment.py`, matching each cached RefChartQA
grounding box against the ChartQA elements of the same image (matched by decoded-pixel hash):

| | |
|---|---:|
| RefChartQA records whose image has ChartQA elements | 3,474 of 3,996 (86.9%) |
| grounding boxes scored | 6,340 |
| best-match **IoU ≥ 0.9** | **98.9%** |
| median best-match IoU | **1.000** |
| exact matches (IoU ≥ 0.999) in a 400-box sample | **99.2%** |
| median margin over the runner-up element | **1.000** |
| confidently matched (IoU ≥ 0.5 **and** margin ≥ 0.2) | 98.8% |

**Interpretation.** The boxes are not merely close — they are *the same boxes*. RefChartQA's
grounding was derived from ChartQA's element geometry, so RefChartQA is best understood as
**ChartQA elements plus a per-question selection**. The alignment problem the brief asks
about is therefore nearly trivial, and the margin over the runner-up being 1.000 means
ambiguity is rare rather than merely uncommon.

**What it buys.** Aligned RefChartQA evidence gains a real label, a real value and a series
(`'Switzerland' / '100%' / bars` instead of `item1`), and the record becomes eligible for
plan mining against ChartQA's table — turning the project's largest block of *real*
grounding supervision from partially-useful into fully supervised.

**Risks.** The values arrive as raw strings with formatting (`'100%'`, `'9 891'`,
`'460 000'`), so the percent and separator handling of C1 applies here too and must be fixed
first. 13.1% of records have no ChartQA elements for their image and must stay unaligned
rather than being forced.

**Recommended action.** Implement alignment as an explicit, thresholded, *rejecting* step
with the margin recorded, after C1's value handling is fixed. Do not force a match.

**Priority HIGH. Confidence high.**

---

### H2 — The dedup merge never reaches training

**Current behaviour.** `data/mixture.py` calls `deduplicate()` while building a mixture. The
merge keeps the primary's answer, **unions the boxes**, and keeps any plan. The mixture file
then stores **record ids only** (rule 7). At training time `cli/train.py::_all_source_records`
rehydrates each id from the raw source adapters — which perform no dedup and no merge.

**Evidence.** `deduplicate` is referenced only in `data/mixture.py`. Rehydrating
`data/mixture_stage2.json` yields **0 records carrying `merged_from`**, against 179 merges
reported at build time (162 of them ChartQA + RefChartQA).

**Problem.** De-duplication survives — an id appears once — but **annotation fusion does
not**. Today that costs nothing measurable (the last build reported 0 boxes and 0 plans
gained). It matters because Ideas 3 and 4 propose to make fusion valuable, and as the
pipeline stands **any fused annotation would be silently discarded before training**.

**Recommended action.** Separate the two operations, as the brief suggests. Deduplication
stays where it is. Fusion must produce a **persisted artefact** that rehydration reads —
otherwise it cannot participate in training at all.

**Priority HIGH — it is a precondition for H1. Confidence high (by construction).**

---

### H3 — Labels are not unique, and the two sides disagree about which mark a label means

**Evidence.** ⚠️ **The first number here was wrong, and the correction matters.**
`audit/measure_label_ambiguity.py` read `sorted(names)[:3000]` — the first annotation files
in *filename* order. ChartQA filenames encode the chart family, so that prefix is **40.5%
`multi_col`** against **15.6%** in the real train split, and multi-column charts have
duplicate labels by construction. It reported 74.2%; the true rate is about a third of that.

`audit/measure_series_identity.py`, over 3,000 charts sampled at **random** and deduplicated
by image:

| | |
|---|---:|
| charts where some label names **more than one** element | **678 (22.6%)** |
| of those, every element carries a `series` name | **678 (100%)** |
| **(series, label) is unique** where the label alone is not | **640 (94.4%)** |
| still collides even with the series | 38 (5.6%) |

The worst label repeats twice on 53.4% of colliding charts, three times on 25.4%, and up to
seven times.

The old, biased figure is left visible rather than deleted: it is the second time in this
audit that iterating a source in its natural order produced a badly skewed sample (the first
measured human-only questions, `DECISIONS.md` 0081). Audit scripts sample randomly now.

```
'Senegal'   names 3 marks  series ['About the same','Less','More']  values 89, 23, 3
'Sep 2015'  names 4 marks  series ['A little','A lot','DK','Nothing at all']  values 36,32,30,1
```

**Problem.** `targets.py` resolves a label with `by_label.setdefault(...)` — **first wins**.
`executor.py` resolves it with `{e.label: e for e in evidence}` — **last wins**. The two
sides of the same contract disagree. The elements carry a `series`, but neither the target
builder nor `OUTPUT_SCHEMA` has any field for it, so the disambiguating information is
discarded at the boundary.

**Mitigating measurement.** Only **2.4%** of evidence entries actually sit on an ambiguous
label today, because mining rejects most multi-series questions. So the *current* damage is
small — but it is the mechanism that would make H1's alignment unsafe at scale, since
RefChartQA questions are heavily multi-series.

**Why it is now the top blocker.** Running the LLM teacher over 40 unbiased ChartQA records
made this the largest single cause of refusal: **6 of its 15 refusals** were "this label
appears N times and nothing says which". That is 15% of all records — larger than every
remaining DSL gap combined.

**Recommended action.** The disambiguating information already exists and is thrown away:
`chartqa.py::_series_elements` writes `"series": model.get("name")` on every element and
nothing downstream reads it. Carry it into the element's identity so both sides resolve the
same mark. See `DECISIONS.md` 0083 for the design and its measured cost.

**Priority HIGH. Confidence high.**

---

### M1 — The processor pixel budget is applied inside a silent `except: pass`

`_set_processor_pixel_budget` tries a `dict` branch, then an attribute branch wrapped in
`try/except (AttributeError, TypeError): pass`. On the installed version `size` is a
`SizeDict`, so the attribute branch runs; **verified working** — the budget applies and an
800×600 chart yields `image_grid_thw = [1, 26, 36]` → 234 visual tokens.

The risk is that a future `transformers` version makes `SizeDict` immutable, the exception is
swallowed, and training silently runs at **native** resolution: ~1.8× the step time and a
different sub-token profile, with nothing in the logs to say so.

**Recommended action.** Read the budget back after setting it and raise if it did not apply.
**Priority MEDIUM. Confidence high.**

---

### G1 — Confirmed correct: preprocessing (no change recommended)

Idea 12 asks whether we resize before Qwen does. **We do not.** `feed._image` opens the image
and converts to RGB with no resize; the pixel budget is set on the processor and its own
`smart_resize` performs the single resize. There is no double resampling of chart text.

Verified against the installed processor: `patch_size=16`, `merge_size=2` → factor **32**;
`Qwen2VLImageProcessor` with `do_resize=True` calling `smart_resize`. Our `vision/coords.py`
port reproduces it exactly — an 800×600 chart gives 234 visual tokens by both our arithmetic
and the real processor.

**Recommended action: none.** The design already matches what Idea 12 concludes it should be.

---

### H4 — Half the supervision is lost to a collision the question text resolves

**The finding.** The deterministic miner refuses 53.9% of ChartQA training rows as
`ambiguous`, and `ambiguous` is easy to misread: it does not mean two cells hold the answer,
it means **two operations reproduce it**, so the uniqueness rule cannot choose. Measured over
4,000 rows sampled at random across both question kinds:

| what collided | share of ambiguous | share of all rows |
|---|---:|---:|
| **`lookup` vs an extremum (`max`/`min`/`argmax`/`argmin`)** | **49.3%** | **26.6%** |
| `lookup` vs `mean`/`median`/`sum` | 44.2% | 23.8% |
| everything else | 6.5% | 3.5% |

The single most common collision is `lookup+max`, 775 times. ChartQA charts are usually
sorted and questions often ask about the top row, so the answer cell is simultaneously
`lookup(<its label>)` and `max` of its column.

**Why it matters.** These are not hard questions. *"How many internet users did Nigeria
have as of December 2020?"* wants `lookup('Nigeria')`; *"which country had the most?"* wants
`argmax`. The two are one word apart in the question and identical in the table. The miner
reads only the table, so it must refuse both.

**Not an expressiveness problem.** On an unbiased sample of 60 random training questions,
93.3% (95% CI 84.1–97.4%) are expressible in the current DSL, and of the 41 that the miner
loses, 25 wanted a plain `lookup`. Adding operators addresses ~7% of the corpus; letting the
mining step read the question addresses the 50.4% lost to collisions involving `lookup`.

**Recommendation.** Adopt LLM mining with the five-gate verifier
(`src/chartqa_dt/plans/llm_mining.py`), which is built and tested. Leave the uniqueness rule
untouched for the deterministic path — at 94% precision it is not too strict, it is
under-informed. Defer new operators until the residual failures can be measured.

**Evidence.** `audit/measure_ambiguity_shape.py`, `audit/measure_miner_on_unbiased_sample.py`,
`audit/judge_dsl_sample.py`, `audit/measure_dsl_coverage.py`; decisions 0080, 0081.

---

### C3/C4 — Defects only the new pipeline could see

Both were found by building the LLM mining path and running it end to end on 40 unbiased
ChartQA records, which accepted **0 of 25 correct proposals**. Full reasoning and the
measurements are in `DECISIONS.md` 0082; the short form:

**C3 — one value, two scales.** `mining.to_number('5.3%')` returned 5.3 and
`executor.to_number('5.3%')` returned 0.053, so a plan mined against the table was executed
against evidence 100x smaller. **0 of 32,719 ChartQA answers and 0 of 3,996 RefChartQA
answers carry a `%`**, so the divided form could never match. Separately, `'3 071'` — carried
by 20.7% of charts in one of four space characters — raised outright. Now one parser,
`executor.parse_numeric`, with a test asserting the two modules agree on every value.

**C4 — a bare aggregate kept the first eight elements.** The fold guard required the plan to
name a label, so it caught `difference("Alpha", mean-of-everything)` and missed a bare
`argmax()`. The median ChartQA chart has 10 elements; **64.4% have more than eight**. The
round-trip refused these records, so nothing wrong shipped — but it refused them with *"own
plan does not reproduce its own answer"*, blaming the plan for evidence we had cut.

**Why they survived.** Fixing them changed the deterministic pipeline's output by **exactly
zero targets**, and doubled the LLM path from 44% to 88% accepted. The old miner only
produces a plan where the two parsers happened to agree — it refuses percentage charts as
`ambiguous` and drops spaced thousands as unparsable before the executor ever sees them. A
component can be correct under every input the current system gives it and wrong under the
inputs the next one will.

**Evidence.** `audit/measure_evidence_defects.py`, `audit/measure_target_yield.py`,
`audit/teacher_proposals_chartqa.py`, `tests/test_executor.py`.

---

### What this audit found about how the defects got there

Twenty-four findings is a list. The patterns under them are the transferable part, and every
one recurred at least twice.

#### 1. The expensive gaps were never decisions

The three costliest findings were not wrong choices. They were things nobody had chosen.

* **Searching backwards from the answer** cost 53.9% of rows (H5). No decision record proposed
  it, weighed it, or named it — it was simply how the first miner was written, so it was never
  re-examined. Every fix before the audit improved a component pointed the wrong way.
* **`annotation_boxes` dropped the colour field** on every element from the first line it was
  written (H7). Nothing said why. It was worth 21.8% of human questions.
* **The 12,000 cap** is the compute budget backwards — 12,000 ÷ batch 8 × 2 stages × 11.903 s =
  9.92 h against a 10 h session — and that derivation lived across four constants in three
  files with nothing connecting them (0092).

A decision record protects the choices someone argued about. It cannot protect the ones nobody
noticed making, and those were the expensive ones.

#### 2. A justification can be true when written and expire quietly

`synth/generator.py` still says it is *"the primary source of plan supervision, given that the
uniqueness rule admits only ~5.7% of real ChartQA questions."* That was true. The uniqueness
rule is now off the supervision path, so the sentence explaining the design outlived the design
(M2, H9). Nothing re-reads a docstring to ask whether its premise still holds.

#### 3. Fixing an instance is not fixing a rule

0082 found the answer parser used on chart values twice and fixed both call sites. It did not
ask where else the same confusion lived. It lived in three more places, and one of them silently
made every percentage chart's evidence a hundredth of its real value (C5). The fix that ended it
was a test that walks the AST of every module and fails on any unjustified use — the rule, not
the instances.

The same shape produced **four copied constants** (M4). Each was a value restated at a call
site, and each would have drifted silently the moment the original changed.

#### 4. A failed check is still a measurement

0052 ran the reproduction gate, found 32.83 did not reproduce, correctly concluded the file was
*"a different model's output"*, and stopped. The next question — *then whose?* — was two
comparisons away, and the answer upgraded the project from *"no published number can be
verified"* to *"two reproduce exactly and our evaluator agrees with the official one to 0.068
points"* (G2). The number a failed check produced still meant something.

#### 5. Measure who your method works for before measuring how well

Forward construction looked like a 3× improvement in supervision yield. Split by question
origin it was 53.5% on machine-generated questions and 14.8% on human ones (H6) — and ChartQA's
test split is 50/50 with the metric averaging the halves. A method measured only in aggregate
looked like a win and would have skewed the training set 92% machine.

#### 6. Iterating a source in its natural order is a sampling bias, twice

Once by taking the first 4,000 rows, which are human-only and the harder half (0081). Once by
taking `sorted(names)[:3000]`, which is 40.5% `multi_col` against 15.6% in the split, and
inflated a finding threefold (H3, 0083). Both were in audit scripts written to *check* for bias.

#### 7. Building the thing finds what reading it does not

Four critical defects were invisible until the new mining path ran end to end and accepted **0
of 25 correct proposals** (C3, C4). Each component was correct under every input the *old*
pipeline gave it. Reading the code found none of them; running it found all four in one pass.

---

# Part 3 — Findings in the 15-point record

> `Prompt.md`'s PRIORITIZATION section asks for a fixed record per finding. This is it.
>
> *Was `FINDINGS.md`.*

The PRIORITIZATION section requires fifteen fields per meaningful finding, and TASK 2 requires
fourteen per major proposed change. The two overlap; where a finding carries a change, the
union is given — seventeen fields, with *why it fits this project* and *what would make us
reject it* added.

`AUDIT.md` carries the narrative for each of these and `DECISIONS.md` the full reasoning; this
is the structured index the brief specifies.

Confidence is stated as **high** only where a number was measured on real data with a stated
sample size.

---

### C1 — An evidence entry's value and box can describe different marks

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

### C3 — Two numeric parsers disagreed by 100× on every percentage

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

### H4 — The miner's dominant refusal is a collision the question resolves

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

### H7 — 21.8% of human questions mention a colour, and nothing read the colour

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

### H8 — The interpreter checks the answer; it does not replace it

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

### H9 — Synthetic charts are half the density of real ones

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

### Grounding-only targets — supervision refused for want of a plan

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

### Native resolution

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

### The remaining findings, in brief

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

---

# Part 4 — Task 1's 99 items

> One row per item, with an honest depth mark rather than a uniform tick.
>
> *Was `AUDIT_COVERAGE.md`.*

`Prompt.md` Task 1 enumerates 99 things to review. Part 6 below tracks them by *subsystem*;
this tracks them by *item*, because the two are not the same and the difference is where a gap
hides.

**Depth is marked honestly, not uniformly.** Ticking 99 boxes would be worth nothing.

| mark | means |
|---|---|
| **M** | **measured** — a number on real data, with a decision record |
| **V** | **verified in code** — read end to end, contract confirmed, no defect found |
| **L** | **limitation recorded** — a real weakness, stated rather than fixed |
| **I** | **inspected** — read, nothing found, no separate measurement made |

---

| # | item | depth | evidence |
|---:|---|:---:|---|
| 1 | Overall architecture | **V** | `ARCHITECTURE.md`, ten subsystems in the six-part form |
| 2 | Repository organization | **M** | 330 → 298 tracked files; one mining path; outputs gitignored |
| 3 | Phase structure and dependencies | **V** | `STATUS.md`; the two-stage split traced (0088) |
| 4 | ChartRecord | **M** | `boxes` meant 3 things (C2); identity settled (0107); **ELEMENTS/EVIDENCE now split into fields, yields unchanged** (0124) |
| 5 | ChartQA adapter | **M** | colour never read (0087); series dropped (0083) |
| 6 | RefChartQA adapter | **M** | boxes ARE ChartQA elements, 98.9% IoU ≥ 0.9 (0077) |
| 7 | Synthetic adapter / conversion | **M** | elements = operands, not the chart (0098) |
| 8 | Dataset merging | **M** | the merge is discarded before training (H2) |
| 9 | Deduplication | **M** | separated from fusion; every consumer enumerated (0108) |
| 10 | Image identity | **V** | `image_content_sha256` over decoded pixels, not file bytes |
| 11 | Decoded-pixel hashing | **M** | caught 15 train images pixel-identical to held-out charts (0049) |
| 12 | Same-image matching | **M** | 86.9% of RefChartQA shares a ChartQA image |
| 13 | Same-question matching | **M** | only 42.1% shares a question — why fusion beats a merge (0105) |
| 14 | Cross-source matching | **M** | IoU ≥ 0.90 with a 0.50 margin, greedy one-to-one, all-or-nothing (0077) |
| 15 | Train/val/test split handling | **V** | `assert_train_only` on the **inputs**, before any filtering |
| 16 | Record-level leakage | **V** | same guard; checked before filtering so a leak cannot hide |
| 17 | Image-level leakage | **M** | `assert_no_held_out_images` — RefChartQA ships "train" rows using test charts (0049) |
| 18 | Sealed-test logic | **V** | `sealed_image_hashes`; synthetic holdout by style and data seed |
| 19 | Table representation | **L** | two shapes: `{columns, rows}` and `{labels, values}` (0098) |
| 20 | Chart-element representation | **M** | series and colour carried (0083, 0087); `elements` is a first-class field (0124) |
| 21 | Labels | **M** | non-unique on 22.6% of charts, not 74.2% — that sample was biased (0083) |
| 22 | Values | **M** | two parsers 100× apart; four call sites wrong (0082, 0089) |
| 23 | Units | **M** | unit suffixes (`'26.29 t'`) on 0.8% of charts; `check_units` verified |
| 24 | Bounding boxes | **M** | 0–1000 anisotropic; the official evaluator discards a box at exactly 1000 |
| 25 | Evidence representation | **M** | was derived and cost four defects (0108); **now stored as indices into `elements`, `None` = unknown** (0124) |
| 26 | `meta` | **M** | `ELEMENTS_KEY` meant two things by source (0098); elements moved out of `meta` (0124) |
| 27 | Label/value/bbox associations | **M** | 9.2% described different marks (C1, 0075) |
| 28 | Coordinate systems | **V** | anisotropic by axis; ported to match `smart_resize` exactly |
| 29 | Coordinate normalization | **V** | `clamp_for_official_evaluator`; G1 re-verified |
| 30 | Image loading | **V** | PIL from the zip member; size read from the decoded image, never assumed |
| 31 | Image resizing | **V** | the processor owns it; no double resize (G1) |
| 32 | Qwen3-VL preprocessing | **M** | Idea 12 — verified correct against the official processor, no change |
| 33 | Visual patch / token behaviour | **M** | patch 16 × merge 2 = factor 32, derived from the loaded processor, never hard-coded |
| 34 | Resolution choices | **M** | 512px → native; 11.9 points of sub-token targets (0095) |
| 35 | Target construction | **M** | grounding-only targets: 56.6% → 98.5% supervisable (0104) |
| 36 | Structured output schema | **M** | admitted three operations the executor refuses (0109) |
| 37 | Output serialization | **M** | JSON kept; short keys save 0 tokens/item on this tokenizer (0094) |
| 38 | Invalid-output handling | **V** | drop / unwrap / never add, every repair counted (0064) |
| 39 | Typed reasoning DSL | **M** | 93.3% of a random corpus expressible; human questions need more (0081, 0090) |
| 40 | Operator set | **M** | `within` added on measured demand; six more requested with numbers (0090) |
| 41 | Operator semantics | **M** | each judged against 60 real questions (0081) |
| 42 | Nested / compositional plans | **M** | executor allows depth 4; synthetic reaches 2 and **every plan mined from real data is depth 1** — 12,667 of 12,667 (0125) |
| 43 | Executor implementation | **M** | one shared parser after four defects (0082, 0089) |
| 44 | Numerical tolerance | **V** | the answer's own precision, not 5% — 5% of the year 2014 is a century (0045) |
| 45 | Unit behaviour | **V** | `check_units` refuses a mixed-unit aggregate |
| 46 | Executor errors | **V** | `ExecutorError` never swallowed; every occurrence counted |
| 47 | Round-trip verification | **M** | it cannot catch wrong evidence (0075) or a coincidence (0097, 0106) |
| 48 | Plan mining | **M** | direction was the constraint, not search (0085) |
| 49 | Candidate generation | **M** | `candidate_sets` flattenings measured; `union` scored worse than `all_cells` |
| 50 | Operand generation | **M** | 22.6% of unique operations had several operand pairs (0106) |
| 51 | Candidate ambiguity | **M** | 53.9% refused; `lookup` vs extremum is 26.6% of all rows (H4) |
| 52 | **Concrete-program ambiguity** | **M** | **22.6%, the case the brief flags for special attention (0106)** |
| 53 | Semantic correctness of mined plans | **M** | 100 records hand-judged across two seeded samples |
| 54 | Mining tolerances | **V** | confirmed correct; not the binding constraint (0045, 0085) |
| 55 | Numeric-looking categories | **M** | `answer_is_a_category` — 8.5% of mined plans had a gold answer that is a row label |
| 56 | Percentage handling | **M** | 21.4% of charts all-percent with a bare answer; the 100× defect (0082) |
| 57 | Question semantics in mining | **M** | the miner never reads the question — the whole finding (0085, 0086) |
| 58 | Synthetic curriculum | **M** | L1→L4 balanced; `balance_by_level` stops L3/L4 being cut by the cap (0066) |
| 59 | Synthetic value generation | **M** | was `uniform(4, 60)`, so no chart ever showed a thousands separator; now a lognormal fitted to ChartQA, with negatives and percentage charts at measured rates (0120) |
| 60 | Synthetic question generation | **M** | templated (0086); rebuilt against ChartQA's own phrasing — median 7 → 10 words, past tense 0% → 51.3%, entity nouns (0122) |
| 61 | Synthetic rendering | **V** | boxes from matplotlib artists, never a formula; proven against pixels |
| 62 | Synthetic chart diversity | **M** | types ChartQA lacks dropped (0091); **density ceiling removed — p50 4 → 10, max 7 → 40** (0118) |
| 63 | Synthetic style diversity | **I** | `style_seed` over font size, dark mode, grid; three seeds sealed for the robustness test |
| 64 | Synthetic geometry extraction | **V** | `artist_box`, `point_box`, `scatter_point_box`; degenerate boxes rejected |
| 65 | Synthetic bbox verification | **V** | 640/640 verified across 8 chart types × 4 levels × 20 seeds |
| 66 | Real-vs-synthetic domain gap | **M** | five gaps measured; type, density, values and language closed (0091, 0118, 0120, 0122); operation mix partly, and the rest is a level-proportion question (0123) |
| 67 | Data-quality gates | **M** | five gates; discard never repair; the value/box gate (0075) |
| 68 | Target-yield checks | **M** | before/after on identical records throughout |
| 69 | Supervision provenance | **M** | complete and unread; report before weighting (0105) |
| 70 | Confidence tracking | **M** | `match_iou` and `match_margin` per element; `gates_passed` per plan (0105) |
| 71 | Source balancing | **M** | absent chart types dropped; levels stay balanced (0091) |
| 72 | Training mixtures | **M** | the 12,000 cap is the compute budget backwards (0092) |
| 73 | Training objective | **M** | boxes 35.6% of the loss, the answer 3.7% (0096) |
| 74 | Standard autoregressive SFT | **V** | prompt masked, target supervised, end-of-turn included |
| 75 | Token masking | **V** | the prompt boundary is measured **with the image**, never by text count |
| 76 | Possible auxiliary losses | **L** | considered and not adopted — the answer policy must settle first (0096) |
| 77 | Curriculum strategy | **M** | stage 1 is a curriculum stage, so its distribution matters (0101) |
| 78 | Evaluation methodology | **M** | per subset, never aggregated — they differ by 30 points (0002) |
| 79 | ChartQA answer evaluation | **M** | byte-faithful to the official parser, quirks included |
| 80 | RefChartQA grounding evaluation | **M** | ours matches the official evaluator to **0.068 pts** on 11,690 predictions (0093) |
| 81 | Plan validity evaluation | **M** | schema validity tracked separately from JSON validity (0058) |
| 82 | Plan execution evaluation | **M** | executable rate reported separately from agreement (0059) |
| 83 | Semantic plan evaluation | **M** | what arithmetic cannot catch, now detected (0097, 0106) |
| 84 | Round-trip consistency metrics | **M** | a headline number since 0059 |
| 85 | Malformed-output rate | **M** | `ParseStats` counts every repair by kind (0064) |
| 86 | Failure analysis | **M** | refusal profiles by cause throughout, never a bare total |
| 87 | Logging | **I** | `logging_utils.py`; run records written per phase; no defect found |
| 88 | Reproducibility | **V** | every sample seeded and named; ids stable across processes |
| 89 | Random seeds | **M** | RNG state in checkpoints — its absence cost 0.0438 loss on resume (0026) |
| 90 | Config management | **M** | the pixel budget was restated at a call site and would have drifted (0095) |
| 91 | Tests | **M** | 1,200 → see `tests/`; expanded specifically against this audit's defect classes |
| 92 | CLI behaviour | **V** | six entry points; `cdt-mine` repointed from the retired miner (0088) |
| 93 | Error handling | **V** | typed errors, never swallowed; every refusal names its cause |
| 94 | Performance | **M** | 11.903 s/step at 512px, 21.267 native — measured on the real GPU (0060) |
| 95 | Memory cost | **M** | 5.572 GB peak at 512px, 6.723 native; an earlier 1.482 was a sharded misread (0060) |
| 96 | Scalability | **I** | the archive is read lazily by member; `qa_rows` materialises one split at a time |
| 97 | Documentation accuracy | **M** | the README's central claim was false and is corrected (0096) |
| 98 | Research defensibility | **M** | 32.83 is not in the paper; six published baselines instead (0093) |
| 99 | Anything else discovered | **M** | 0106 operand ambiguity; 0109 unexecutable operators offered to the model |

---

### Where the depth is thinnest

Four items are marked **I** — inspected, nothing found, no separate measurement. They are the
honest weak spots of this audit:

* **59 — synthetic value generation.** Seeded and reproducible, but the *distribution* of
  generated values was never compared to real charts. Given that chart **density** turned out
  to be badly mismatched (0098), value distribution is a plausible place for a similar gap.
* **63 — synthetic style diversity.** Font size, dark mode and grid vary by seed; whether that
  spans the visual variation of real Statista charts is unmeasured.
* **87 — logging.** Adequate for the phases run so far; no failure has been traced to a
  missing log, which is weak evidence.
* **96 — scalability.** Nothing here has run at the full 55,789-row RefChartQA split. Lazy
  reads and per-split materialisation suggest it holds; it has not been shown.

Items 59 and 63 fold naturally into the L3–L4 regeneration (0101), which is already scheduled
and already has to measure a distribution.

---

# Part 5 — `Prompt.md` line by line

> Every section of the brief with a status, and an explicit list of what is not done.
>
> *Was `PROMPT_CHECKLIST.md`.*

Ahmed asked for a checklist rather than prose. Every section of `Prompt.md` is below with a
status and the evidence. **Nothing is ticked because it was read; it is ticked because
something was measured, changed, or explicitly decided against with a reason.**

| mark | means |
|:---:|---|
| ✅ | **done** — measured or implemented, with a decision record |
| 🟡 | **partly done** — the honest state is written in the row |
| ❌ | **not done** — and the row says why, and what it would take |
| ⛔ | **blocked** — needs a GPU, or an API run, or Ahmed's call |

---

### The framing sections

| § | requirement | status | evidence |
|---|---|:---:|---|
| CORE MINDSET | treat prior decisions as hypotheses, not commitments | ✅ | 0112, 0115, 0117, 0118, 0120 each overturn an earlier decision of mine |
| | don't defend the repo because you built it | ✅ | 0116 and 0118 are both "my own change was wrong" |
| | don't accept Ahmed's ideas merely because suggested | ✅ | 0113 refutes the coordinate-quantisation idea; 0121 refutes executor-replaces-answer; 0120 finds "close values" already harder than real |
| | if both are weak, find a third approach | ✅ | 0118: neither "accept the density" nor "spend compute" — the cause was a 12-item tuple |
| | if evidence is inconclusive, design an experiment | ✅ | 0121 leaves a pre-registered prediction rather than a guess |
| PRIMARY OBJECTIVE | correctness > novelty > size | ✅ | 0116 rejected 4,939 extra records because they were wrong |
| SOURCE OF TRUTH | code and measurement, not memory | ✅ | every decision here carries a number |
| PROTECT THE REPOSITORY | never commit dataset content | ✅ | rule-7 guardrail now runs in preflight too, after a week of red CI |
| | never rewrite history | ✅ | none rewritten; the CI fix was an allow-list change |

---

### TASK 1 — the 99-item audit

✅ **All 99 items covered.** Tracked individually in Part 4 below with honest depth
marks (**M** measured / **V** verified / **L** limitation recorded / **I** inspected) —
ticking 99 boxes uniformly would be worth nothing.

---

### TASK 2 — external research

✅ Done, with the 14-point justification record per finding. Primary sources only; the
notable negative result is that **32.83 is not in the RefChartQA paper** and our evaluator
matches the official one to 0.068 points across 11,690 predictions.

---

### The 15 specific ideas

| # | idea | status | what actually happened |
|---:|---|:---:|---|
| 1 | reconsider `ChartRecord` | ✅ | Audited (0098, 0108), the flagged assumption **fired** in 0116, and the structural change is now **done** (0124): `elements` and `evidence` are first-class fields, `evidence=None` means "unknown". Yields identical — 87.8% at rung 4,000, before and after. |
| 2 | distinguish elements from evidence | ✅ | Identity: qualified labels, not opaque ids, with the trade-off table (0107). Container: ELEMENTS and EVIDENCE separated (0124). Synthetic now keeps the whole chart instead of only the operands. |
| 3 | connect RefChartQA grounding to ChartQA elements | ✅ | `scripts/align_refchartqa.py`. 98.9% of boxes match a ChartQA element at IoU ≥ 0.9 (0077). Re-run at full scale today: **3,405 → 48,770 aligned (87.9%)** (0115). |
| 4 | reconsider the ChartQA ↔ RefChartQA merge | ✅ | Merge separated from deduplication; every consumer enumerated (0108, 0109). |
| 5 | evidence should have one clear meaning | ✅ | TABLE / ELEMENTS / EVIDENCE / PLAN / ANSWER is what the code does; the one gap (EVIDENCE not first-class) is recorded, not hidden (0108). |
| 6 | target builder | ✅ | Five gates; discard, never repair. Grounding-only targets added then correctly restricted (0116). **Duplicate values** — the brief's own item — now refuse a tied `argmax` (0127), and truncated operands rejoin (0129). |
| 7 | plan mining, two approaches | ✅ | Deterministic search retired after measuring it must refuse 53.9%; LLM mining is the single path (0088). |
| 8 | (spurious programs / execution filtering) | ✅ | Spurious-program detector measured; the check it names by name was run. |
| 9 | synthetic data | 🟡 | **Density ceiling removed (0118)** and **value distribution fitted to ChartQA (0120)**. Remaining items listed in the table below. |
| 10 | DSL + executor | ✅ | Every edge case in the brief's list tested: division by zero, NaN/Infinity, missing evidence, depth limits, `mean([])` — all handled or refused explicitly. Duplicate labels with contradictory values now refused (0128). |
| 11 | round-trip verification | ✅ | Seven notions kept distinct. Executor-replaces-answer **measured and rejected** (−6.9 points); agreement kept as a label-free confidence signal, +15.9 points (0121). |
| 12 | Qwen3-VL preprocessing | ✅ | Verified against the official processor; we do not pre-resize. No change needed. |
| 13 | model output format | ✅ | Audited; no change. |
| 14 | training objective | ✅ | Loss masking verified end to end (prompt, padding and image tokens all masked, with a `supervised` count check). Composition measured — boxes 35.6%. Quantisation tested and refuted (0113). |
| 15 | supervision provenance / confidence | ✅ | Two closed vocabularies — grounding and value — tagged **per element** by every source; kept out of the model, as the brief requires (0126). |

---

### Idea 9 in detail — every sub-item it lists

**DATA**

| item | status | note |
|---|:---:|---|
| category distributions | ✅ | 5 → 11 pools, up to 50 labels (0118) |
| value distributions | ✅ | lognormal fitted to ChartQA, mean 1.45 sd 1.33 (0120) |
| extreme values | ✅ | 21.8% of charts above 1,000, against 20.5% real |
| close values | ✅ | **no change needed** — synthetic is already harder than real (49.5% vs 36.8% within 5%) |
| decimal values | ✅ | precision follows magnitude in both directions |
| negative values | ✅ | 1.6%, against 1.7% real |
| percentages | ✅ | 7.7% of charts sum to exactly 100, against 7.4% real |
| formatting | ✅ | thousands separators now rendered; the model had never seen one |
| units | 🟡 | 6 units, not measured against ChartQA's distribution |
| correlations | ❌ | values are i.i.d. within a chart; real series trend over time |
| realistic table structure | ❌ | flat `{labels, values}`; ChartQA has multi-column tables |

**CHART TYPES**

| item | status | note |
|---|:---:|---|
| vbar, hbar, grouped bar, line, multi-line, scatter, pie, area | ✅ | all eight |
| stacked bars | ❌ | not implemented |
| area/scatter in the mixture | ✅ | dropped by selection — ChartQA contains 0.0% of both (0091) |

**STYLE**

| item | status | note |
|---|:---:|---|
| font sizes, backgrounds, gridlines, legends, tick rotation, titles, value labels, aspect ratio, DPI, colours | ✅ | all randomised in `Style.sample` |
| axis formatting | ✅ | thousands separators (0120) |
| font *family* | ❌ | one family only |
| tick density, margins, subtitles, annotation placement, clutter, realistic imperfections | ❌ | none of these are varied |

**LANGUAGE**

| item | status | note |
|---|:---:|---|
| nested reasoning | ✅ | L4 |
| ambiguity avoidance | ✅ | colliding labels refused, not guessed (0083) |
| varied operand order | ✅ | multiple orderings per operation (0122) |
| template diversity, paraphrases, naturalness, ChartQA-like wording | ✅ | 40+ templates; median 7 → 10 words, past tense 0% → 51.3%, entity nouns instead of "category" (0122) |
| referring expressions, distractors | ❌ | need chart geometry and colour at question-build time, not just the series (0122) |

**GROUNDING**

| item | status | note |
|---|:---:|---|
| evidence count | ✅ | now varies with density (0118) |
| geometry verification | ✅ | every box checked on a recoloured render |
| false-positive boxes | ✅ | verification rejects them |
| line/scatter markers | ✅ | own box functions |
| grouped chart identity | ✅ | qualified labels (0083, 0107) |
| tiny elements | ❌ | rejected by `MIN_BOX_SIDE_PX`, but real charts contain them |
| overlapping elements | ❌ | not generated |
| legend association | ❌ | not generated as a task |

**DOMAIN GAP**

| item | status | note |
|---|:---:|---|
| chart-type resemblance | ✅ | fixed by selection (0091) |
| mark-density resemblance | ✅ | p50 4 → 10, max 7 → 40 (0118) |
| value resemblance | ✅ | fitted (0120) |
| operation-mix resemblance | ❌ | `difference` 13.8× over-represented; fixable by selection, **not done** |
| "unrealistically clean" | 🟡 | no imperfections, no clutter |
| other rendering engines | ❌ | matplotlib only; not investigated |

---

### Cross-cutting requirements

| requirement | status | evidence |
|---|:---:|---|
| DATA QUALITY > QUANTITY | ✅ | 0116 discarded 4,939 records rather than accept wrong ones |
| MANUAL SEMANTIC AUDIT SET | ✅ | `scripts/build_semantic_audit.py` — 16 strata, seeded, short strata left short; 237 rows awaiting judgement. Found the truncated-operand join bug on its first run (0129). |
| PRIORITIZATION (15-point record) | ✅ | Part 3 below |
| EMPIRICAL VALIDATION | 🟡 | every CPU-measurable claim has a number; the GPU ones are listed as blocked below |
| TESTING REQUIREMENTS | ✅ | 2,100+ tests; new work mutation-checked (8 mutations on the decoder, 4 on the generator) |
| — quality statistics before/after for data changes | ✅ | 0118 and 0120 both carry before/after tables |
| — successful / rejected / ambiguous / regression cases | ✅ | e.g. `test_synth_density.py`, `test_grounding_only_fallback.py` |
| WORK ORDER phases 1–9 | 🟡 | 1–6 done; 7 partial; 8–9 gated on GPU runs — every blocked one has a runnable command in `STATUS.md` |

---

### The FINAL EXPECTATION questions

| question | answered in |
|---|---|
| which decisions remain strong | Part 1 below |
| which were wrong | 0112, 0115, 0117, 0118, 0120 — and 0116, which was mine, twice |
| which are outdated | 0112 (a starting point that became a ceiling), 0117 (a ratio that stopped being one) |
| which are merely uncertain | `SYNTHETIC_REPLAY`; the answer policy |
| what should change immediately | done, and listed above |
| what should be experimentally tested | the ⛔ rows below |
| what should remain unchanged | Idea 12, 13; `ChartRecord`'s identity scheme (0107) |
| how the changes interact | 0114 ↔ 0121 (both bear on decode); 0115 → 0116 → 0119 (a chain) |
| expected measurable benefit | stated per decision |
| how we will know it is better | pre-registered predictions, e.g. the agree-rate in 0121 |

---

### What is NOT done — the honest list

**Not done, and doable without a GPU:**

2. 🟡 **Question language** (Idea 9). Templates, tense, length and entity nouns done (0122). **Referring expressions and distractors are not** — they need geometry and colour at question-build time.
3. 🟡 **Operation mix** (0091). L3 weighted toward ChartQA (`argmax`/`argmin` 7.3% → 10.8% corpus-wide) (0123). The rest is a **level-proportion** question — `difference` is 25% because L2 and L4 are half the corpus — and that is a mixture-time selection decision needing your call on what stage 1 is for.
4. ❌ **Stacked bars**, tiny/overlapping elements, legend-association tasks, clutter and imperfections.
5. ❌ **Within-chart correlations** and multi-column tables.
6. ❌ Two ChartQA record constructors still exist (0119, 0124) — a merge with no measurement behind it yet.
7. ❌ **Distractor-aware spurious-program check.** Newly *possible* — synthetic records now keep the whole chart (0124) — and not yet built.
8. ❌ **Judge the 237-row semantic audit set.** Built and sampled (0129); the judging is manual and not done.

**Blocked:**

9. ⛔ **LLM plan mining at volume.** ChartQA contributes **5 records of 22,947** to stage 1 today. Everything is built and ready to run; it needs the API run.
10. ⛔ **RefChartQA scaling ladder** (4,000 / 10,000 / 25,000). Unblocked as of 0115 — needs GPU.
11. ⛔ **Re-run the zero-shot baseline with `close_evidence=True`** (0114), or the reported fine-tuning gain will include a truncation fix.
12. ⛔ **Three training seeds**, Phase 8–9 verification and reporting.


---

### Where the blocked work is written down

`STATUS.md`'s blocked section — six experiments, each with why it is blocked, a copy-pasteable command, and
what result would change a decision already recorded. `Prompt.md` requires exactly that:
*"clearly document the blocked experiment and exactly how it should be run later."*

Reading the prompt line by line for that section found that the decode fix from 0114 had
**no CLI flag** — it existed and could not be run, the same shape as the scaling ladder
whose cache held 3,996 rows. `run_zeroshot.py --close-evidence` now exists.

---

# Part 6 — The work plan, as it was tracked

> Kept because it records what was attempted and what it cost, not only what succeeded.
>
> *Was `AUDIT_PLAN.md`.*

**Rule for this file:** nothing is ticked until it is *measured or implemented and verified*.
"Looked at it" is not done. Where an item cannot be finished in this environment, it says so
and states the exact command to finish it later.

Legend — ✅ done · 🔄 in progress · ⬜ not started · 🚫 blocked (reason stated)

---

### Phase 1 — current-state reconstruction

| # | item | status | evidence |
|---|---|---|---|
| 1.1 | Trace sources → adapters → `ChartRecord` | ✅ | `AUDIT.md` C1, C2, H1–H3 |
| 1.2 | Trace dedup / merging | ✅ | H2 — merge never reaches training |
| 1.3 | Trace mining / enrichment | ✅ | 0078, 0079 |
| 1.4 | Trace target construction | ✅ | 0075, 0077 |
| 1.5 | Trace training serialization → model → parsing → executor | ✅ | collate contract read, the loss composition measured (0096), the sequence budget verified at native resolution (0095) |
| 1.6 | Trace evaluation | ✅ | runner traced to `score_item(gen.answer)` — which found H8 — and our metrics validated against the official evaluator to 0.068 pts on 11,690 predictions (0093) |
| 1.7 | Trace synthetic generation end-to-end | ✅ | 0091 — generator, manifest, mixture composition and its fit to the real distribution |
| 1.8 | Write the concise current-state architecture | ✅ | `ARCHITECTURE.md` — one pass from chart to scored number, with a closing section on what is not built and what is uncertain |

### Phase 2 — self-critique of prior decisions

| # | item | status | verdict |
|---|---|---|---|
| 2.1 | Re-examine 0001–0105 against what we now know | ✅ | done by supersession, and then checked mechanically: every file a decision cites is now verified to exist by `test_every_file_a_decision_cites_still_exists`. Superseded or amended: 0002 and 0052 by 0093, 0037/0060 by 0095, 0041 by 0082, 0067 by 0075, 0078/0079 by 0085/0088, 0080 by 0081, 0091 by 0101. Confirmed still correct: 0014, 0026, 0045, 0064, 0069 |
| 2.2 | 0014 "emit few boxes" | ✅ | confirmed useful in a place it was not designed for: 90.3% of questions never touch `MAX_EVIDENCE` because evidence is selected by what the plan names (0084) | |
| 2.3 | 0037/0060 resolution choice | ✅ | **superseded: native** (0095). 0060 rejected native only because 17.72 h broke a 10 h session; that gate is gone. Buys 11.9 points of sub-token targets (53.2% → 41.3%), and the sequence still fits at p99 864 of 1,024 |
| 2.4 | 0041 empty-args fold convention | ✅ | the *pure* fold slipped past the guard and was silently truncated to 8 items; fixed in 0082 (`AUDIT.md` C4) |
| 2.5 | 0045 mining tolerance | ✅ | not the binding constraint and never was: the constraint is the mining *direction* (0085). The tolerance itself is confirmed correct — 5% of the year 2014 is a century |
| 2.6 | 0062 small-probe lesson | ✅ | reconfirmed by 5.3 (round-trip 69% → 58.8% at n=1,920) |
| 2.7 | 0069 early stopping on loss | ✅ | *confirmed correct.* AP cannot resolve a stopping signal at affordable slice sizes (±8.7 to ±12.2 pts), the evaluator returns **negative** loss so the maximising stopper is right, and `test_validate.py` already guards the sign |

### Phase 3 — external research (primary sources)

| # | topic | status |
|---|---|---|
| 3.1 | Qwen3-VL preprocessing — official implementation | ✅ inspected the installed processor directly; no double resize; factor 32 verified |
| 3.2 | ChartQA paper / repo — annotation semantics | ✅ | four of our own measurements independently confirmed (chart mix, `'unk'` colours, T5-generated machine questions only partly validated, unanswerable questions). One new: **values are printed on the elements**, so the sub-token bound applies to grounding, not to reading numbers (0103) |
| 3.3 | RefChartQA paper / repo — grounding provenance | ✅ its boxes ARE ChartQA elements (0077), and the paper read as a primary source gave Table 2, the vendored file's identity, and the absence of 32.83 (0093) |
| 3.4 | Semantic parsing from denotations · weak supervision | ✅ | our blind spot has a name (**spurious programs**) and an established fix (Lee/Kim/Jung EMNLP 2023, execution-based filtering). Implemented as `plans/distinguish.py` (0097) |
| 3.5 | Program synthesis · execution-guided search/decoding | ✅ | fine-grained partial-program guidance is **inapplicable at our program size** (median depth 1–2, 8.8% of tokens). The coarse form is exactly 0096's `executed` answer policy — the record already writes the plan before the answer — plus resample-on-self-disagreement, to be settled by the same Phase 5 experiment (0102) |
| 3.6 | LLM program generation · teacher distillation · self-consistency | ✅ | implemented as **verified** self-consistency (0100): the vote runs only among plans that already passed every gate, and the denominator is all samples so one lucky sample cannot carry a record. Paired with 0097 — sample K times only where the evidence cannot decide |
| 3.7 | Constrained / structured generation | ✅ | **evidenced no** (0099). It would make schema validity 100% and costs ~2 points of accuracy, but it removes refusal — `answerable:false` becomes unreachable and boxes are forced, and 0014 measured one spurious box taking AP 1.00 → 0.68. Revisit only if post-training validity is still low, and then constrain structure only |
| 3.8 | Chart QA + grounded chart QA state of the art | ✅ | RefChartQA Table 2 read from the primary source: six models, three splits, four metrics. 32.83 is **not in the paper**; ChartGemma 2B @448 is the size-matched baseline (0093) |
| 3.9 | Curriculum learning · synthetic data | ✅ | settles 0091's open question (0101): stage 1 is a **curriculum stage**, so the distribution matters, and the literature's remedy is a graded synthetic→real bridge. L1–L2 stay uniform for format; **L3–L4 should match ChartQA's operation mix and chart density** |

### Specific ideas 1–15

| # | idea | status | outcome |
|---|---|---|---|
| 1 | Reconsider `ChartRecord` | ✅ measured | `boxes` genuinely means 3 different things; C2 fixed the immediate harm, structural fix open |
| 2 | Distinguish ELEMENTS from EVIDENCE | ✅ | series now carried into element identity (0083), and the two meanings of `meta[elements]` measured and recorded (0098) |
| 3 | Connect RefChartQA grounding to ChartQA elements | ✅ implemented | 0077 — 98.9% at IoU≥0.9; 85.2% aligned |
| 4 | Reconsider ChartQA ↔ RefChartQA merging | ✅ | **the fusion already happens under another name** (0105): the aligner matches boxes to ChartQA elements at 98.9% IoU≥0.9 and attaches labels, values and table — strictly better than a question-keyed merge, since the datasets share 86.9% of images but only 42.1% of questions |
| 5 | Evidence should have one clear meaning | ✅ | **it does not** (0098): `meta[elements]` holds *the operands* on synthetic records and *the whole chart* on ChartQA ones (median 1–4 vs 11), and `record.table` has two shapes. Targets agree only because `_evidence_from` prunes |
| 6 | Target builder | ✅ | 0075 value/box gate, 0082/0083/0089 fixes, and **grounding-only targets** (0104): RefChartQA goes from 56.6% to **98.5% supervisable**, +23,357 real grounding records projected. Builder written and tested; not yet wired into a mixture |
| 7 | Plan mining — deterministic vs LLM-assisted | ✅ | **settled: LLM only** (0085, 0086, 0088). Backwards search must refuse 53.9%; pattern matching gets 53.5% of machine questions and 14.8% of human ones, which would skew supervision 92% machine against a 50/50 test split |
| 8 | Improve deterministic mining | ✅ | **audited as the brief asks, not improved.** Precision, recall and the full refusal profile measured; a question-intent tie-breaker built and rejected at 86–89% against its 94%. Its specifically-flagged case — one operation type unique, several concrete programs of that type — measured at **22.6%** and now detected (0106) |
| 9 | Synthetic data | ✅ | **designed for a job it no longer has** (0091). 25% of the corpus is chart types ChartQA does not contain; `difference` is 13.8x over-weighted and `lookup` 2.6x under. Fix by reweighting at mixture time, not regenerating |
| 10 | DSL + executor | ✅ | audited against real questions: 93.3% of a random corpus sample is expressible, but human questions need more. `within` added on measured demand (0090); six remaining operations each carry a number; and the brief's *"schema-valid but non-executable"* case was live — the prompt offered three operations the executor refuses (0109); and the brief's *"schema-valid but non-executable"* case was live — the prompt offered three operations the executor refuses (0109) |
| 11 | Round-trip verification | ✅ | 0075/0077 showed it cannot catch wrong evidence; 0097 adds what it cannot catch on one input either — a plan the evidence cannot distinguish from another reading |
| 12 | Qwen3-VL preprocessing | ✅ | no change needed — verified correct |
| 13 | Model output format | ✅ | **audited, no change** (0094). Short keys save 0 tokens/item — Qwen encodes JSON keys as single tokens; a line format saves 32%/item but costs the schema, unambiguous parsing and JSON priors for +2.2% of questions, and we are not sequence-constrained (p99 679 of 1,024) |
| 14 | Training objective | ✅ | measured what the loss spends itself on (0096): boxes 35.6%, `model_answer` **3.7%**, and 17.1% on labels/values no metric scores. Found the README claimed the executor replaces the answer when it only checks it; made the three answer policies scoreable from one generation set |
| 15 | Supervision provenance / confidence | ✅ | **complete and unread** (0105): synthetic seeds, per-element `match_iou`/`match_margin`, and `plan_provenance` with model + prompt hash + gates. Decision: report by provenance before weighting by it — `eval/stratified.py` already groups by a categorical field |

### Cross-cutting requirements

| item | status |
|---|---|
| Manual semantic audit set | ✅ superseded — RefChartQA grounding gives 3,405 records with gold operand identity (0078), and 100 records were hand-judged across two seeded samples (0081, 0086) |
| Data quality > quantity | ✅ applied — every change so far reduced yield and raised correctness |
| Prioritised findings with the 15-point record | ✅ Part 3 below — the structured record the brief specifies. **This was ticked prematurely once**: `AUDIT.md` carried the narrative and the priorities but not the fifteen fields, and re-reading `Prompt.md` in full rather than by its headers caught it |
| Empirical validation of each change | ✅ before/after measured for 0075–0079 |
| Tests for each change | ✅ 1,006 → passing |
| Documentation matches reality | ✅ | README's central claim corrected (0096), `ARCHITECTURE.md` written with a section on what is NOT built, `AUDIT.md` current at 24 findings, `STATUS.md` updated for 0093 |

### Open questions raised by Ahmed, to be answered with measurement

| # | question | status |
|---|---|---|
| Q1 | Why cap training examples per stage at 12,000? | ✅ | **it is the compute budget, backwards**: 12,000 x 1 epoch / batch 8 = 1,500 steps, x2 stages = 3,000, x 11.903 s/step = 9.92 h against a 10 h Kaggle limit. The constraint has since been lifted (3 accounts, verified resume) so it is now a choice; not raised yet, because more data is gated on mining and whether it helps is what the scaling ladder answers (0092) |
| Q2 | Is the operation set expressive enough? | ✅ | **for machine questions yes, for human questions no.** 93.3% of a random corpus sample is expressible (0081), but that sample is dominated by templated questions; reading 40 human ones by hand produced 7 operator requests (0090) |
| Q3 | Is the deterministic miner complete? | ✅ | wrong question. It is 94% precise and its recall is bounded by *direction*, not by search: several operations reproduce any given answer and it cannot choose (0085) |
| Q4 | Can we ever be sure a plan is unambiguous? | ✅ | **the question dissolves.** Fidelity to the question is the property we want, not uniqueness — a plan can be right even when another operation reaches the same number (0085) |
| Q5 | Are ChartQA and RefChartQA the same questions? | ✅ | **86.9% share an image, 42.1% share a question.** Related, not duplicates — which is why dedup and fusion are separate concerns (H2) |
| Q6 | Would a strong LLM find a correct plan for ~all examples? | ✅ | **no.** Measured twice by acting as the teacher: 21/21 accepted on RefChartQA-aligned records but only 52% proposed; on 40 human ChartQA questions, 9 verified plans, 22 refusals, 7 operator requests. The limit is the DSL and the data, not the reader |

### Blocked

| item | blocker | how to finish |
|---|---|---|
| LLM-assisted mining at scale | No API key. A Claude/ChatGPT **subscription** cannot drive a pipeline over ~15,000 questions. | `scripts/mine_with_llm.py` is **written, tested and verified end to end** — prompt building, caching keyed by record+model+prompt hash, provenance, and the five-gate verifier. Measured on 40 unbiased records with Claude as the teacher: **22 verified plans (55%)**, 88% of proposals accepted. Only the model call is missing. Set `ANTHROPIC_API_KEY` and run `python scripts/mine_with_llm.py --source chartqa --limit 20000`. Proposals produced elsewhere can be scored today with `--proposals <file>`, no key needed. |
| Phase 6 training | Audit in progress; mixtures will need rebuilding after | rebuild, then `cdt-train --stage stage1` |

---

### Session log — what moved, and what it cost

| finding | how it was found | outcome |
|---|---|---|
| DSL is 93.3% sufficient, not 55% | checked 0080's biased sample against 60 random questions **before** writing any operator | 0080 partly withdrawn; no operators written |
| `lookup` vs extremum collision = 26.6% of all rows | `audit/measure_ambiguity_shape.py`, n=4,000 | `AUDIT.md` H4; the quantitative case for LLM mining |
| two parsers 100x apart on every percentage | ran the LLM path end to end; it accepted 0 of 25 correct proposals | 0082; LLM yield 44% → 88% |
| bare aggregate silently truncated to 8 items | reading the fold guard while diagnosing the above | 0082; 64.4% of charts affected |
| a first pass measured human questions only | iterating rows in order instead of sampling | corrected before publishing the number |
| I guessed a label that wasn't in the annotation | the verifier's `operand_not_in_evidence` gate caught it | the gate works on its author |

**Largest remaining blocker, measured:** of the teacher's 15 refusals on 40 records,
**6 are duplicate labels across series** — `AUDIT.md` H3. That is 15% of all records, and it
is now the single biggest recoverable loss.
