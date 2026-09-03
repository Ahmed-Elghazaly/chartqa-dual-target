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
| H3 | Labels are non-unique on **74.2%** of charts; target builder and executor resolve duplicates *differently* | **HIGH** | high, measured |
| M1 | The processor pixel budget is applied inside a silent `except: pass` | MEDIUM | high |
| **H4** | **The miner's dominant refusal is a `lookup` vs `max`/`min` tie — 26.6% of all rows.** The table cannot say which the question asked for; one word of the question can | **HIGH** | high, measured (n=4,000) |
| **C3** | **`mining` and `executor` parsed every percentage 100x apart**, and spaced thousands raised — invisible to the old pipeline, halves the new one | **CRITICAL** | ✅ **FIXED** (0082) |
| **C4** | **A bare aggregate lost its evidence silently** — `argmax()` on a chart with >8 elements kept the first 8 and the round-trip blamed the plan | **CRITICAL** | ✅ **FIXED** (0082) |
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

**Evidence.** `audit/measure_label_ambiguity.py` over 2,070 ChartQA train annotations:

| | |
|---|---:|
| charts where some label names **more than one** element | **1,536 (74.2%)** |
| v_bar | 87.6% |
| h_bar | 64.1% |
| charts with a single series | 537 (25.9%) |

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

**Recommended action.** Decide the contract explicitly and enforce it on both sides. The
element identity should carry `series` (or an equivalent discriminator) internally; whether
the *model-facing* label needs it is a separate question the brief rightly separates.

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
