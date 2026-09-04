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
