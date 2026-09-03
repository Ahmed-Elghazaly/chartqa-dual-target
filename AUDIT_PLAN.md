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
| 1.7 | Trace synthetic generation end-to-end | ⬜ | |
| 1.8 | Write the concise current-state architecture | ⬜ | |

## Phase 2 — self-critique of prior decisions

| # | item | status | verdict |
|---|---|---|---|
| 2.1 | Re-examine 0001–0079 against what we now know | 🔄 | 0067 amended by 0075; 0045 challenged by 0079 |
| 2.2 | 0014 "emit few boxes" | ⬜ | |
| 2.3 | 0037 resolution choice (512 vs 448) | ⬜ | |
| 2.4 | 0041 empty-args fold convention | 🔄 | interacted badly with 0067 → 0071 |
| 2.5 | 0045 mining tolerance | 🔄 | see 0079 — tolerance is not the binding constraint, ambiguity is |
| 2.6 | 0062 small-probe lesson | ✅ | reconfirmed by 5.3 (round-trip 69% → 58.8% at n=1,920) |
| 2.7 | 0069 early stopping on loss | ⬜ | |

## Phase 3 — external research (primary sources)

| # | topic | status |
|---|---|---|
| 3.1 | Qwen3-VL preprocessing — official implementation | ✅ inspected the installed processor directly; no double resize; factor 32 verified |
| 3.2 | ChartQA paper / repo — annotation semantics | ⬜ |
| 3.3 | RefChartQA paper / repo — grounding provenance | 🔄 measured: its boxes ARE ChartQA elements (0077) |
| 3.4 | Semantic parsing from denotations · weak supervision | ⬜ |
| 3.5 | Program synthesis · execution-guided search/decoding | ⬜ |
| 3.6 | LLM program generation · teacher distillation · self-consistency | ⬜ |
| 3.7 | Constrained / structured generation | ⬜ |
| 3.8 | Chart QA + grounded chart QA state of the art | ⬜ |
| 3.9 | Curriculum learning · synthetic data | ⬜ |

## Specific ideas 1–15

| # | idea | status | outcome |
|---|---|---|---|
| 1 | Reconsider `ChartRecord` | ✅ measured | `boxes` genuinely means 3 different things; C2 fixed the immediate harm, structural fix open |
| 2 | Distinguish ELEMENTS from EVIDENCE | 🔄 | 74.2% of charts have non-unique labels; series discarded at the boundary (H3) |
| 3 | Connect RefChartQA grounding to ChartQA elements | ✅ implemented | 0077 — 98.9% at IoU≥0.9; 85.2% aligned |
| 4 | Reconsider ChartQA ↔ RefChartQA merging | 🔄 | H2 found fusion is discarded; dedup vs fusion now separated in practice |
| 5 | Evidence should have one clear meaning | ⬜ | depends on idea 2 |
| 6 | Target builder | 🔄 | 0075 added the value/box gate; grounding-only targets still open |
| 7 | Plan mining — deterministic vs LLM-assisted | 🔄 | 0078/0079 measured the deterministic ceiling; LLM design pending |
| 8 | Improve deterministic mining | 🔄 | 0079 done; label-answer gap open |
| 9 | Synthetic data | ⬜ | |
| 10 | DSL + executor | ⬜ | operator completeness not yet audited |
| 11 | Round-trip verification | 🔄 | 0075/0077 showed it cannot catch wrong evidence |
| 12 | Qwen3-VL preprocessing | ✅ | no change needed — verified correct |
| 13 | Model output format | ⬜ | |
| 14 | Training objective | ⬜ | |
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
| Q1 | Why cap training examples per stage at 12,000? | 🔄 measuring |
| Q2 | Is the operation set expressive enough? | ⬜ idea 10 |
| Q3 | Is the deterministic miner complete — does it try all combinations? | 🔄 measuring the label-answer gap |
| Q4 | Can we ever be sure a plan is unambiguous? | 🔄 — no, and this is a real limit |
| Q5 | Are ChartQA and RefChartQA the same questions? | 🔄 measuring |
| Q6 | Would a strong LLM find a correct plan for ~all examples? | 🔄 needs the calibration experiment |

## Blocked

| item | blocker | how to finish |
|---|---|---|
| LLM-assisted mining at scale | No API key. A Claude/ChatGPT **subscription** cannot drive a pipeline over ~15,000 questions. | An Anthropic or OpenAI **console API key** (pay-per-token). Then `scripts/mine_with_llm.py` (to be written) with caching, model+prompt pinning, and deterministic verification. |
| Phase 6 training | Audit in progress; mixtures will need rebuilding after | rebuild, then `cdt-train --stage stage1` |
