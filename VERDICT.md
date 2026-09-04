# The audit's verdict

`Prompt.md`'s FINAL EXPECTATION asks ten questions. This answers them in order, then gives the
implementation plan for what remains (Phases 5 and 6). Everything here is traceable to a
decision record; nothing is asserted that was not measured.

---

## 1. Which previous decisions remain strong

| decision | why it holds |
|---|---|
| **0014** — emit few boxes | confirmed useful somewhere it was not designed for: because evidence is selected by what the plan *names*, 90.3% of questions never touch the evidence cap (0084) |
| **0026** — checkpoints save RNG state | the resume check it added is what makes 90 h/week across three accounts usable at all (0092, 0095) |
| **0045** — mine at the answer's own precision | not the binding constraint and never was, but correct: 5% of the year 2014 is a century |
| **0064** — repair by drop/unwrap, never add | ruled out constrained decoding on its own terms (0099) |
| **0069** — early-stop on loss, not AP | AP cannot resolve a stopping signal at any affordable slice size (±8.7 to ±12.2 pts); the sign convention is correct and tested |
| **G1** — no double resize | re-verified; the processor owns resizing and our coordinate port matches it |

## 2. Which were wrong

| decision | the error | fixed by |
|---|---|---|
| **0067** — evidence selection | selected the first eight boxes, so a plan's label matched nothing; 1 of 636 records yielded a target | 0075 |
| **0082's own scope** | fixed the answer/value parser confusion at two call sites and did not ask where else it lived. It lived in three more, one of which made every percentage chart's evidence 100× too small | 0089 |
| **0080's conclusion** | "45% of questions need new operators" — true of the sample, which was drawn from miner failures. Corpus-wide it is ~7% | 0081 |
| **the `README` claim** | *"the arithmetic never depends on the model doing mental maths"*. Every path scores the model's own answer | 0096 |
| **`AUDIT.md` H3's magnitude** | 74.2% of charts with non-unique labels, from a filename-ordered sample that is 40.5% `multi_col` against 15.6% real. True figure 22.6% | 0083 |

## 3. Which have become outdated

| decision | the premise that expired |
|---|---|
| **0060** — 512px | native was rejected *only* because 17.72 h broke a 10 h Kaggle session. Three accounts and a verified resume removed that gate (0095) |
| **0052** — 32.83 is unreproducible | true of 32.83, which is **not in the RefChartQA paper at all**. Two other published numbers reproduce exactly (0093) |
| **0078/0079** — improving the deterministic miner | its recall is bounded by *direction*, not by search. Improving it was work on a component pointed the wrong way (0085) |
| **`synth/generator.py`'s docstring** | still says it is *"the primary source of plan supervision, given that the uniqueness rule admits only ~5.7%"*. The uniqueness rule is off the path (0091) |
| **0092** — the 12,000 cap | correct as compute arithmetic; the compute constraint is gone, so it is now a choice |

## 4. Which are merely uncertain

* **Whether the executed answer beats the stated one.** Zero-shot, 20% of plans did not execute and 40% disagreed with the stated answer — and nothing measured which side was right (0096).
* **Whether K-sample consensus is worth its cost**, and at what K. Implemented; unmeasured (0100).
* **Whether an underdetermined plan should be refused.** The detector works; the cost of refusing is unpriced on real proposals (0097).
* **Whether mined supervision performs as well as supervision exact by construction.** Provenance is complete and unread (0105).

## 5. What should change immediately

Ordered by benefit per unit of risk. Items 1–3 are done; 4–6 are specified below.

| | change | status |
|---|---|---|
| 1 | one shared value parser, guarded by an AST test | **done** (0089) |
| 2 | series and colour into element identity | **done** (0083, 0087) |
| 3 | native resolution | **done** (0095) |
| 4 | wire grounding-only targets into stage 1 | specified, not wired |
| 5 | regenerate L3–L4 against ChartQA's density and operation mix | specified, needs compute |
| 6 | run the mining at volume | pipeline ready; needs a reader |

## 6. What should be experimentally tested

Every one of these is answered by data Phase 5 and 6 produce anyway. **None needs a separate run.**

| experiment | settles | how |
|---|---|---|
| three answer policies scored on one generation set | is the executor a calculator or a checker? | `answer_under` (0096) |
| results stratified by supervision provenance | is mined supervision as good as exact? | `eval/stratified.py` already groups by a field (0105) |
| accepted plans flagged underdetermined | what refusing them would cost | `Verdict.underdetermined` (0097) |
| resample-on-self-disagreement | does disagreement predict being wrong? | same generations as the answer-policy test (0102) |
| 3 training seeds | is a difference real or noise? | ~30 h, affordable now |
| the RefChartQA scaling ladder | does more data still help? | ~20 h; decides whether to raise the 12,000 cap |

## 7. What should remain unchanged

* **The JSON output format.** Short keys save 0 tokens/item on this tokenizer; a line format saves 32% and costs the schema, unambiguous parsing and the model's JSON priors, for +2.2% of questions — and we are not sequence-constrained (0094).
* **No constrained decoding.** It removes refusal and forces a box, and one spurious box takes AP from 1.00 to 0.68 (0099).
* **The uniqueness rule in `plans/mining.py`**, which is retired from the supervision path but kept as an independent cross-check. At 94% precision it is a useful second opinion, and it caught two of my errors (0085).
* **`eval/metrics.to_float`**, byte-faithful to the official evaluator, quirks included.

## 8. How the changes interact

Four couplings, and three of them are why work is sequenced rather than parallel.

1. **Grounding-only targets ↔ stage-1 composition.** +23,357 real grounding records is nearly double the stage-1 cap, so wiring them changes the synthetic/real mix — and fixes the density gap at the same time, because real charts have the density synthetic ones lack. **Wire this before regenerating**, or L3–L4 may be regenerated to fill a budget that real data has already filled.
2. **Native resolution ↔ the evidence cap.** Native costs 178 more visual tokens; the cap costs 47 per item. Both draw on the same 1,024. Measured together: p99 864, worst case 875 — safe, but they cannot be raised independently again without re-measuring.
3. **The answer policy ↔ the training objective.** If `executed` wins, then `model_answer`'s 3.7% of the loss is nearly irrelevant and the 17.1% spent on labels and values is the real answer supervision. If `stated` wins, a sixth of the objective is scaffolding for a calculator nobody reads. **Settle the policy before re-weighting anything.**
4. **The spurious-plan detector ↔ K-sample consensus.** They compose: sample K times *only* where the evidence cannot decide. Adopting consensus everywhere would cost K× for no gain on records a single pass already settles.

## 9. What measurable benefit we expect

Stated as predictions, so they can be wrong.

| change | expected benefit | measured how |
|---|---|---|
| native resolution | **11.9 points** fewer targets below one visual token → grounding AP | AP@0.5 per subset, before/after |
| grounding-only targets | RefChartQA **56.6% → 98.5%** supervisable; stage 1 becomes mostly real | target yield, and AP after stage 1 |
| series + colour identity | 22.6% of charts stop resolving labels ambiguously; 21.8% of human questions become answerable | mined-plan acceptance rate, human-split accuracy |
| LLM mining | ~55% of unbiased ChartQA verified, against 15–25% deterministic | targets built per 1,000 records |
| L3–L4 regeneration | closes a 13.8× operation skew and a 2.8× density gap | accuracy on the human split specifically |

**No claim here is that any of these will improve the final number.** They are each a defect removed or a supply increased, with the measurement that will say whether it mattered.

## 10. How we will know whether the revised system is better

* **Against six published models**, not one unverifiable number: RefChartQA Table 2, three splits, four metrics, with ChartGemma (2B, 448px) as the size-matched baseline (0093).
* **Against our own zero-shot baseline** through the identical path, `--adapter` the only difference — the comparison `PREREGISTRATION.md` seals before results exist.
* **Against the direct-answer control**, trained on the same records, which is what separates ordinary domain adaptation from the contribution of grounding, plans and execution.
* **With intervals, not point estimates.** `Cell` already refuses a point estimate without one; three seeds make them real.
* **Reported per subset.** RefChartQA-H, -M and -PoT differ by 30 points and an average of them means nothing (0002).

---

## Implementation plan for what remains (Phase 6)

### A. Wire grounding-only targets into stage 1

* **Files** — `data/mixture.py` (`build_stage1`), `scripts/build_mixtures.py`, `cli/train.py` (target selection per stage).
* **Ordering** — first; it changes what the other work is sized against.
* **Migration** — none to stored artifacts. Mixtures hold record ids and rehydrate, so a rebuild picks the new targets up. **The join must be in the reader**, per `AUDIT.md` H2.
* **Compatibility** — the target is deliberately not `OUTPUT_SCHEMA`-valid, so anything validating stage-1 targets against the full schema must select by stage, as `build_answer_only_target` already does.
* **Tests** — extend `test_mixture.py` for composition; the builder's own tests exist.
* **Rollback** — one flag; the builder is additive and nothing else calls it.

### B. Run the mining at volume

* **Files** — none; `scripts/mine_plans.py` is complete.
* **Ordering** — parallel with A. Independent.
* **Artifacts** — writes `~/.cache/chartqa_dt/data/chartqa_plans.jsonl`; a rerun replaces by record id and never duplicates.
* **Validation** — every plan passes five gates; report acceptance rate, refusal profile, and requested operators.
* **Rollback** — delete the cache; records return to `plan=None`.
* **Blocked on** — a reader. Batches are plain text for a Claude Code session or Codex; `--api` exists if a key ever does.

### C. Regenerate L3–L4

* **Files** — `synth/generator.py`, then a full regeneration.
* **Ordering** — **after A**, so the real-data supply is known before synthetic is sized.
* **Artifacts** — the 24,000-example manifest is regenerated. Sealed holdout seeds must not move.
* **Targets** — ChartQA's operation mix (`lookup` ~64%, `argmax`/`argmin` ~21%) and density (median 10 marks, p90 24).
* **Validation** — rerun `audit/measure_synthetic_fit.py`; the skews should close.
* **Rollback** — keep the current manifest; it is outside the repo and not overwritten in place.

### D. Settle the answer policy

* **Files** — `eval/runner.py` and `cli/evaluate.py`, to report three accuracies instead of one.
* **Ordering** — with the first evaluation run; it needs generations, not a separate experiment.
* **Rollback** — reporting only; nothing downstream depends on it until a policy is chosen.

### Not planned, deliberately

Loss weighting by confidence, K-sample consensus everywhere, refusing underdetermined plans, and the six missing DSL operations. Each is built or specified and each awaits the measurement that would justify it. `Prompt.md`: *"Do not permanently adopt an experimental change merely because it sounds better."*

---

## Appendix — the self-critique, in the form Phase 2 asks for

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
