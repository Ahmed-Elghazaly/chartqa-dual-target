# Status

> ## Where this stops, 2026-09-05
>
> Everything is committed and pushed, CI is green, the working tree is clean.
>
> **The synthetic corpus is NOT regenerated.** The generator changed substantially
> (0118 density, 0120 values, 0122 language, 0123 operation mix, 0127 tied extrema, 0124
> whole-chart elements), and the corpus in `~/.cache/chartqa_dt/data/synthetic/` is still
> the **old** one — 24,000 examples from the pre-change generator. A regeneration was
> started three times and stopped three times, each time because another generator bug
> surfaced after it had begun. It is deliberately left undone: it takes ~90 minutes and
> should be the **last** step before training, not a step repeated after every change.
>
> Nothing is half-written. The partial output was deleted; the active corpus is untouched
> and still valid for the *old* generator.
>
> **Next session begins with a repo cleanup**, agreed with Ahmed: 15 markdown files where
> six of them (`AUDIT`, `AUDIT_COVERAGE`, `AUDIT_PLAN`, `FINDINGS`, `VERDICT`,
> `PROMPT_CHECKLIST`) all answer *"what did we check and find"*, and three (`STATUS`,
> `RUNS`, `BLOCKED`) all answer *"where are we"*. Target is ~7 files with nothing lost.
>
> **The process rule that came out of this session**, because most of its bugs were one
> mistake repeated: *change one end of a pipeline and not the other, discover it only by
> running the whole thing.* The fallback that was dead code, the provenance that missed the
> legacy path, the decode fix with no CLI flag — all the same shape. **Run the end-to-end
> pipeline before committing, not after, and do not start a long job until the changes
> have settled.**


**Phase 5 in progress** (5.1 and 5.2 done, 5.3 running, 5.4 queued). **Phases 6 to 9 are
built and tested ahead of their gates** — everything that does not need a GPU or a sealed
split. Phases 0–4 complete. Cost so far: **USD 0**.

The binding constraint is GPU quota, not code. So the working rule this week has been:
build every measurement first, run it on the CPU where it can be run, and let the GPU do
only what nothing else can. That has caught five defects that would each have wasted a
ten-hour training run (`DECISIONS.md` 0071–0074).

## Where each phase stands

| phase | state | note |
|---|---|---|
| 0–3 | complete | data, mixtures, audit gate, mining |
| 4 | complete | metrics agree with both official evaluators on 11,690 real predictions |
| 5.1 prompt | complete | three prompts, sealed by hash |
| 5.2 variant | **complete** | **Instruct selected**, n=200 |
| 5.3 ChartQA zero-shot | running | full 1,920 validation split, ~6 h |
| 5.4 RefChartQA zero-shot | queued | 1,800 stratified rows |
| 5.5 pre-registration | drafted | generated from source; the seal guard rejects it while it says "TBD" |
| 6 | **built, untrained** | feed, collator, checkpointing, loop, validation, monitoring, kill-and-resume verified |
| 7 | table and claim guards built | `Cell` refuses a point estimate without an interval; `Claims` cannot say training was reproduced; `--adapter` evaluates through the identical Phase 5 path |
| 8.2 | calibrator and crop policy built | fitted on validation only, self-reported confidence refused, never a third pass |
| 9 | **all seven analyses built** | oracle, stratification, plan diagnostics, transfer, robustness, calibration, figures |
| 10 | skeleton compiles, generator built | `cdt-report` fills tables from recorded JSON; 16 pages, 0 undefined refs |

**1,069 tests pass**; `ruff check src tests scripts` clean; preflight green.

## Phase 5.2 result — the first properly powered measurement

| | value | of what |
|---|---:|---|
| relaxed accuracy | 50.0% | all 200 questions |
| schema-valid (after repair) | 46.5% | all 200 questions |
| **round-trip agreement** | **69.0%** | the 71 schema-valid records |
| plans that execute at all | 94.4% | the 71 schema-valid records |
| median latency | 11.4 s | per question |

The last two are conditional on a usable record, not on a question. Read as a share of all
200, agreement is 24.5%: half the loss is records that never parse.

The n=24 probes had reported round-trip at 40–50%. At n=200 it is 69%, confirming
`DECISIONS.md` 0062: three prompt iterations were run on noise, and the probe could not
have detected any effect it was used to justify.

## The supply of real supervision, measured

`scripts/measure_target_yield.py`, CPU only. A record is *usable* when it becomes a
training target: it parses, satisfies the schema, and its plan reproduces its own answer.

| source | pool | usable | |
|---|---:|---:|---:|
| synthetic | 24,000 | 23,966 | 99.9% |
| ChartQA train | 22,947 | 2,420 | 10.5% |
| RefChartQA train (7% of the split, cached) | 3,996 | 2,063 | 51.6% |
| **all real** | 26,943 | **4,483** | 16.6% |

**The entire supply of real chart supervision this project can build is 4,483 records.**
ChartQA's 10.5% is the mining yield showing through — 19,634 of its rows have no plan that
uniquely explains their answer, and `DECISIONS.md` 0045 refuses to guess one. Synthetic
supplies the rest of both mixtures, which is why `PLAN.md` 9.4's synthetic-to-real transfer
measurement is not a side ablation but the assumption the training set rests on.

Asking this question before training rather than after found three defects that would have
wasted the run (`DECISIONS.md` 0071) and one that would have quartered it (0072).

## What Phase 6's design pass found, before spending 10 GPU hours

Each of these would have produced a plausible-looking failure rather than an error.

1. **Training examples did not fit `max_seq_len`** (0064). The zero-shot prompt is 980
   tokens; with visual tokens and a target the example is 1,363–1,498 against a limit of
   1,024. Every example would have been silently truncated. Fixed with a 117-token training
   prompt — 389 tokens of headroom, no extra compute. Raising the limit was measured and
   rejected: ≥14.9 h against a 10 h gate.
2. **Targets did not reproduce their own answers** (0067). Four separate join defects; at
   worst **1 of 636** ChartQA records produced an executable target, and **100%** of
   RefChartQA targets failed the round-trip. Now 69% of planned ChartQA records, and every
   emitted target round-trips by construction.
3. **No end-of-turn token in the target.** A model trained that way is never taught to
   stop, and every generation runs to the token cap.
4. **Early stopping on AP is unsound** (0069). At an affordable slice the AP interval is
   ±8.7 points, which cannot detect "has not improved". Stopping moved to validation loss —
   free, low variance, and directly sensitive to the boxes because the target contains them.

## What the pre-flight measurement found, before spending any GPU hours

`scripts/measure_target_yield.py` asks, on the CPU, how many of a mixture's 12,000 records
actually become training examples. For stage 1 the answer was **zero**. Not a crash — the
feed catches a refusal, counts it and moves on, so the run would have finished on schedule
and reported 3,000 steps and 24,000 presentations truthfully.

| # | defect | cost, silently |
|---|---|---:|
| 0071 | synthetic element metadata written under `evidence`, read as `elements` | 100% of stage 1 |
| 0071 | a fold-over-evidence plan handed only the labels it names | 100% of level 4 |
| 0071 | round-trip agreement inheriting the evaluator's `"0" != "0.0"` quirk | 512 records |
| 0072 | mixture slots filled with records that yield no target | 46% of stage 2 |
| 0073 | ChartQA images read from disk when they live inside the zip | 38% of stage 2 |
| — | `--steps` defaulting to the smoke value, with an unreachable fallback | 96.7% of the budget |

All five share a shape: an `except` that counts a failure and continues is
indistinguishable, from outside, from there being no failures. `DECISIONS.md` 0074 makes
the refusal rate a **gate** — below a 90% usable floor the run stops, after 200 offered
records, about two minutes in.

Sequence lengths were then measured through the real processor and collator on real chart
images: median 610, p90 683, p99 973, and **1 of 200 over the 1,024 limit**.

## Open items

- **Deferred by Ahmed until the core result is in**: three training seeds (~30 h) and the
  RefChartQA scaling ladder (~30 h). Both measure or document a result rather than improve
  it.
- **The plan-rich mixture arm** is built and waiting: 4,820 compositional plans against the
  pre-registered arm's 1,820. The two arms are **not the same size** (10,304 against 6,304),
  so a difference between them confounds composition with volume; they cannot be separated
  at a real-record supply of 4,483, and the confound is recorded rather than papered over
  (0072). They do get identical step counts, so the comparison is at least matched on
  compute.
- **32.83 is not in the RefChartQA paper** (0093). Reading the primary source found the
  number nowhere in it — the best RefChartQA-H AP@0.5 in Table 2 is 27.81 — and no Qwen2/2.5-VL
  row exists at all. It is no longer a target; results are reported against Table 2's six
  models, of which **ChartGemma (2B, 448×448) is the size-matched comparison at AP@0.5
  19.95 (H) / 60.62 (M) / 43.44 (PoT)**.
- **A published number does reproduce, and our evaluator is validated** (0093). The vendored
  `filtered_results.jsonl` is **TinyChart's** output — its M (71.25) and PoT (59.66) reproduce
  the paper **exactly** with the official evaluator — and our own metrics agree with the
  official ones to **0.068 points** across 11,690 predictions. `PLAN.md` 4.4's gate is met and
  4.2's shared-prediction cross-check has passed.
- The ChartQA reproduction of 79.1 is reachable but only at Phase 7, on the test split (0063).

## The audit (`Prompt.md`) — complete

All 51 items in `AUDIT_PLAN.md` are closed. `AUDIT.md` carries 24 findings and the seven
recurring patterns under them; `DECISIONS.md` runs to 0105; `ARCHITECTURE.md` describes the
system as it now stands, including what is not built.

**Changed by the audit, in rough order of consequence:**

| | |
|---|---|
| mining direction | backwards search retired; a reader mines plans from finished records, five gates, discard never repair (0085, 0088) |
| resolution | 512px → **native**, buying 11.9 points of targets too small for one visual token (0095) |
| grounding supply | RefChartQA 56.6% → **98.5%** supervisable via grounding-only targets (0104) |
| element identity | series carried into labels; colour read for the first time (0083, 0087) |
| parsers | one shared value parser, guarded by an AST test after four separate defects (0082, 0089) |
| comparability | 32.83 is **not in the RefChartQA paper**; results go against its Table 2, and our evaluator matches the official one to 0.068 points (0093) |
| stage 1 | 25% was chart types ChartQA does not contain, now dropped (0091) |

**The three biggest things still open**, each with a measured size:

1. **Mining has not been run at volume.** The pipeline is built, tested and verified end to
   end; the plan cache holds two records.
2. **Stage 1 needs regenerating** against ChartQA's operation mix and chart density — no
   synthetic chart exceeds 7 marks against a real median of 10 (0098, 0101).
3. **The answer policy is undecided** — `stated`, `executed` or `executed_or_stated` — and
   Phase 5 settles it from data it produces anyway (0096).

## Next

5.3 and 5.4 finish → their numbers fill section 12 of `PREREGISTRATION.md`, which records
**the zero-shot baselines the project promises to beat** so the bar cannot move afterwards →
commit it → the seal opens → Phase 6 trains stage 1, both stage-2 arms and the direct-answer
control → Phase 7 evaluates through the identical Phase 5 path with `--adapter`.

Everything after that is already written and tested. The remaining GPU work is training and
evaluation; the remaining CPU work is running the analyses on their outputs.
