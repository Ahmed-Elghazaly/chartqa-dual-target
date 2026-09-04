# Deep audit — Phase 1–4 findings

Started 2026-08-29. Everything below is measured against the **current working tree**, not
recalled. Reproduction scripts are in `audit/`, outputs beside them.

**Status:** empirical audit of the data and supervision path complete; both CRITICAL
findings fixed and verified. External research and the remaining subsystems (training
objective, output format, DSL, synthetic distribution, LLM-assisted mining) in progress.

### Resolved

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

## Summary

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

## C1 — An evidence entry's value and its box can describe different marks

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

## C2 — `record.boxes` has no single meaning, and validation AP treats it as ground truth

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

## H1 — RefChartQA grounding aligns to ChartQA elements almost perfectly

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

## H2 — The dedup merge never reaches training

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

## H3 — Labels are not unique, and the two sides disagree about which mark a label means

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

## M1 — The processor pixel budget is applied inside a silent `except: pass`

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

## G1 — Confirmed correct: preprocessing (no change recommended)

Idea 12 asks whether we resize before Qwen does. **We do not.** `feed._image` opens the image
and converts to RGB with no resize; the pixel budget is set on the processor and its own
`smart_resize` performs the single resize. There is no double resampling of chart text.

Verified against the installed processor: `patch_size=16`, `merge_size=2` → factor **32**;
`Qwen2VLImageProcessor` with `do_resize=True` calling `smart_resize`. Our `vision/coords.py`
port reproduces it exactly — an 800×600 chart gives 234 visual tokens by both our arithmetic
and the real processor.

**Recommended action: none.** The design already matches what Idea 12 concludes it should be.

---

## H4 — Half the supervision is lost to a collision the question text resolves

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

## C3/C4 — Defects only the new pipeline could see

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

## What this audit found about how the defects got there

Twenty-four findings is a list. The patterns under them are the transferable part, and every
one recurred at least twice.

### 1. The expensive gaps were never decisions

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

### 2. A justification can be true when written and expire quietly

`synth/generator.py` still says it is *"the primary source of plan supervision, given that the
uniqueness rule admits only ~5.7% of real ChartQA questions."* That was true. The uniqueness
rule is now off the supervision path, so the sentence explaining the design outlived the design
(M2, H9). Nothing re-reads a docstring to ask whether its premise still holds.

### 3. Fixing an instance is not fixing a rule

0082 found the answer parser used on chart values twice and fixed both call sites. It did not
ask where else the same confusion lived. It lived in three more places, and one of them silently
made every percentage chart's evidence a hundredth of its real value (C5). The fix that ended it
was a test that walks the AST of every module and fails on any unjustified use — the rule, not
the instances.

The same shape produced **four copied constants** (M4). Each was a value restated at a call
site, and each would have drifted silently the moment the original changed.

### 4. A failed check is still a measurement

0052 ran the reproduction gate, found 32.83 did not reproduce, correctly concluded the file was
*"a different model's output"*, and stopped. The next question — *then whose?* — was two
comparisons away, and the answer upgraded the project from *"no published number can be
verified"* to *"two reproduce exactly and our evaluator agrees with the official one to 0.068
points"* (G2). The number a failed check produced still meant something.

### 5. Measure who your method works for before measuring how well

Forward construction looked like a 3× improvement in supervision yield. Split by question
origin it was 53.5% on machine-generated questions and 14.8% on human ones (H6) — and ChartQA's
test split is 50/50 with the metric averaging the halves. A method measured only in aggregate
looked like a win and would have skewed the training set 92% machine.

### 6. Iterating a source in its natural order is a sampling bias, twice

Once by taking the first 4,000 rows, which are human-only and the harder half (0081). Once by
taking `sorted(names)[:3000]`, which is 40.5% `multi_col` against 15.6% in the split, and
inflated a finding threefold (H3, 0083). Both were in audit scripts written to *check* for bias.

### 7. Building the thing finds what reading it does not

Four critical defects were invisible until the new mining path ran end to end and accepted **0
of 25 correct proposals** (C3, C4). Each component was correct under every input the *old*
pipeline gave it. Reading the code found none of them; running it found all four in one pass.
