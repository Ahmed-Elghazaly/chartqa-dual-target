# Audit work plan — every item in `Prompt.md`, tracked

**Rule for this file:** nothing is ticked until it is *measured or implemented and verified*.
"Looked at it" is not done. Where an item cannot be finished in this environment, it says so
and states the exact command to finish it later.

Legend — ✅ done · 🔄 in progress · ⬜ not started · 🚫 blocked (reason stated)

---

## Phase 1 — current-state reconstruction

| # | item | status | evidence |
|---|---|---|---|
| 1.1 | Trace sources → adapters → `ChartRecord` | ✅ | `AUDIT.md` C1, C2, H1–H3 |
| 1.2 | Trace dedup / merging | ✅ | H2 — merge never reaches training |
| 1.3 | Trace mining / enrichment | ✅ | 0078, 0079 |
| 1.4 | Trace target construction | ✅ | 0075, 0077 |
| 1.5 | Trace training serialization → model → parsing → executor | 🔄 | collate/loss path not yet audited |
| 1.6 | Trace evaluation | 🔄 | metrics audited in walkthrough ch7; runner path not re-verified |
| 1.7 | Trace synthetic generation end-to-end | ✅ | 0091 — generator, manifest, mixture composition and its fit to the real distribution |
| 1.8 | Write the concise current-state architecture | ⬜ | |

## Phase 2 — self-critique of prior decisions

| # | item | status | verdict |
|---|---|---|---|
| 2.1 | Re-examine 0001–0079 against what we now know | 🔄 | 0067 amended by 0075; 0045 challenged by 0079 |
| 2.2 | 0014 "emit few boxes" | ✅ | confirmed useful in a place it was not designed for: 90.3% of questions never touch `MAX_EVIDENCE` because evidence is selected by what the plan names (0084) | |
| 2.3 | 0037/0060 resolution choice | ✅ | **superseded: native** (0095). 0060 rejected native only because 17.72 h broke a 10 h session; that gate is gone. Buys 11.9 points of sub-token targets (53.2% → 41.3%), and the sequence still fits at p99 864 of 1,024 |
| 2.4 | 0041 empty-args fold convention | 🔄 | interacted badly with 0067 → 0071 |
| 2.5 | 0045 mining tolerance | 🔄 | see 0079 — tolerance is not the binding constraint, ambiguity is |
| 2.6 | 0062 small-probe lesson | ✅ | reconfirmed by 5.3 (round-trip 69% → 58.8% at n=1,920) |
| 2.7 | 0069 early stopping on loss | ✅ | *confirmed correct.* AP cannot resolve a stopping signal at affordable slice sizes (±8.7 to ±12.2 pts), the evaluator returns **negative** loss so the maximising stopper is right, and `test_validate.py` already guards the sign |

## Phase 3 — external research (primary sources)

| # | topic | status |
|---|---|---|
| 3.1 | Qwen3-VL preprocessing — official implementation | ✅ inspected the installed processor directly; no double resize; factor 32 verified |
| 3.2 | ChartQA paper / repo — annotation semantics | ✅ | four of our own measurements independently confirmed (chart mix, `'unk'` colours, T5-generated machine questions only partly validated, unanswerable questions). One new: **values are printed on the elements**, so the sub-token bound applies to grounding, not to reading numbers (0103) |
| 3.3 | RefChartQA paper / repo — grounding provenance | 🔄 measured: its boxes ARE ChartQA elements (0077) |
| 3.4 | Semantic parsing from denotations · weak supervision | ✅ | our blind spot has a name (**spurious programs**) and an established fix (Lee/Kim/Jung EMNLP 2023, execution-based filtering). Implemented as `plans/distinguish.py` (0097) |
| 3.5 | Program synthesis · execution-guided search/decoding | ✅ | fine-grained partial-program guidance is **inapplicable at our program size** (median depth 1–2, 8.8% of tokens). The coarse form is exactly 0096's `executed` answer policy — the record already writes the plan before the answer — plus resample-on-self-disagreement, to be settled by the same Phase 5 experiment (0102) |
| 3.6 | LLM program generation · teacher distillation · self-consistency | ✅ | implemented as **verified** self-consistency (0100): the vote runs only among plans that already passed every gate, and the denominator is all samples so one lucky sample cannot carry a record. Paired with 0097 — sample K times only where the evidence cannot decide |
| 3.7 | Constrained / structured generation | ✅ | **evidenced no** (0099). It would make schema validity 100% and costs ~2 points of accuracy, but it removes refusal — `answerable:false` becomes unreachable and boxes are forced, and 0014 measured one spurious box taking AP 1.00 → 0.68. Revisit only if post-training validity is still low, and then constrain structure only |
| 3.8 | Chart QA + grounded chart QA state of the art | ✅ | RefChartQA Table 2 read from the primary source: six models, three splits, four metrics. 32.83 is **not in the paper**; ChartGemma 2B @448 is the size-matched baseline (0093) |
| 3.9 | Curriculum learning · synthetic data | ✅ | settles 0091's open question (0101): stage 1 is a **curriculum stage**, so the distribution matters, and the literature's remedy is a graded synthetic→real bridge. L1–L2 stay uniform for format; **L3–L4 should match ChartQA's operation mix and chart density** |

## Specific ideas 1–15

| # | idea | status | outcome |
|---|---|---|---|
| 1 | Reconsider `ChartRecord` | ✅ measured | `boxes` genuinely means 3 different things; C2 fixed the immediate harm, structural fix open |
| 2 | Distinguish ELEMENTS from EVIDENCE | 🔄 | 74.2% of charts have non-unique labels; series discarded at the boundary (H3) |
| 3 | Connect RefChartQA grounding to ChartQA elements | ✅ implemented | 0077 — 98.9% at IoU≥0.9; 85.2% aligned |
| 4 | Reconsider ChartQA ↔ RefChartQA merging | 🔄 | H2 found fusion is discarded; dedup vs fusion now separated in practice |
| 5 | Evidence should have one clear meaning | ✅ | **it does not** (0098): `meta[elements]` holds *the operands* on synthetic records and *the whole chart* on ChartQA ones (median 1–4 vs 11), and `record.table` has two shapes. Targets agree only because `_evidence_from` prunes |
| 6 | Target builder | 🔄 | 0075 added the value/box gate; grounding-only targets still open |
| 7 | Plan mining — deterministic vs LLM-assisted | ✅ | **settled: LLM only** (0085, 0086, 0088). Backwards search must refuse 53.9%; pattern matching gets 53.5% of machine questions and 14.8% of human ones, which would skew supervision 92% machine against a 50/50 test split |
| 8 | Improve deterministic mining | ✅ | *not pursued, deliberately* — deterministic mining is off the supervision path (0088). `plans/mining.py` stays as an independent cross-check only |
| 9 | Synthetic data | ✅ | **designed for a job it no longer has** (0091). 25% of the corpus is chart types ChartQA does not contain; `difference` is 13.8x over-weighted and `lookup` 2.6x under. Fix by reweighting at mixture time, not regenerating |
| 10 | DSL + executor | 🔄 | `within` added on measured demand (0090). Six requested operations remain, each now a number: Yes/No comparison 8.0% of human questions, threshold filter, count-of-series, constancy, `product`, argmax-over-computed |
| 11 | Round-trip verification | ✅ | 0075/0077 showed it cannot catch wrong evidence; 0097 adds what it cannot catch on one input either — a plan the evidence cannot distinguish from another reading |
| 12 | Qwen3-VL preprocessing | ✅ | no change needed — verified correct |
| 13 | Model output format | ✅ | **audited, no change** (0094). Short keys save 0 tokens/item — Qwen encodes JSON keys as single tokens; a line format saves 32%/item but costs the schema, unambiguous parsing and JSON priors for +2.2% of questions, and we are not sequence-constrained (p99 679 of 1,024) |
| 14 | Training objective | ✅ | measured what the loss spends itself on (0096): boxes 35.6%, `model_answer` **3.7%**, and 17.1% on labels/values no metric scores. Found the README claimed the executor replaces the answer when it only checks it; made the three answer policies scoreable from one generation set |
| 15 | Supervision provenance / confidence | 🔄 | match IoU + margin recorded by 0077; not yet used for weighting |

## Cross-cutting requirements

| item | status |
|---|---|
| Manual semantic audit set | 🔄 partly superseded — RefChartQA grounding gives 3,405 records with gold operand identity (0078) |
| Data quality > quantity | ✅ applied — every change so far reduced yield and raised correctness |
| Prioritised findings with the 15-point record | 🔄 `AUDIT.md` |
| Empirical validation of each change | ✅ before/after measured for 0075–0079 |
| Tests for each change | ✅ 1,006 → passing |
| Documentation matches reality | 🔄 |

## Open questions raised by Ahmed, to be answered with measurement

| # | question | status |
|---|---|---|
| Q1 | Why cap training examples per stage at 12,000? | ✅ | **it is the compute budget, backwards**: 12,000 x 1 epoch / batch 8 = 1,500 steps, x2 stages = 3,000, x 11.903 s/step = 9.92 h against a 10 h Kaggle limit. The constraint has since been lifted (3 accounts, verified resume) so it is now a choice; not raised yet, because more data is gated on mining and whether it helps is what the scaling ladder answers (0092) |
| Q2 | Is the operation set expressive enough? | ✅ | **for machine questions yes, for human questions no.** 93.3% of a random corpus sample is expressible (0081), but that sample is dominated by templated questions; reading 40 human ones by hand produced 7 operator requests (0090) |
| Q3 | Is the deterministic miner complete? | ✅ | wrong question. It is 94% precise and its recall is bounded by *direction*, not by search: several operations reproduce any given answer and it cannot choose (0085) |
| Q4 | Can we ever be sure a plan is unambiguous? | ✅ | **the question dissolves.** Fidelity to the question is the property we want, not uniqueness — a plan can be right even when another operation reaches the same number (0085) |
| Q5 | Are ChartQA and RefChartQA the same questions? | 🔄 measuring |
| Q6 | Would a strong LLM find a correct plan for ~all examples? | ✅ | **no.** Measured twice by acting as the teacher: 21/21 accepted on RefChartQA-aligned records but only 52% proposed; on 40 human ChartQA questions, 9 verified plans, 22 refusals, 7 operator requests. The limit is the DSL and the data, not the reader |

## Blocked

| item | blocker | how to finish |
|---|---|---|
| LLM-assisted mining at scale | No API key. A Claude/ChatGPT **subscription** cannot drive a pipeline over ~15,000 questions. | `scripts/mine_with_llm.py` is **written, tested and verified end to end** — prompt building, caching keyed by record+model+prompt hash, provenance, and the five-gate verifier. Measured on 40 unbiased records with Claude as the teacher: **22 verified plans (55%)**, 88% of proposals accepted. Only the model call is missing. Set `ANTHROPIC_API_KEY` and run `python scripts/mine_with_llm.py --source chartqa --limit 20000`. Proposals produced elsewhere can be scored today with `--proposals <file>`, no key needed. |
| Phase 6 training | Audit in progress; mixtures will need rebuilding after | rebuild, then `cdt-train --stage stage1` |

---

## Session log — what moved, and what it cost

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
