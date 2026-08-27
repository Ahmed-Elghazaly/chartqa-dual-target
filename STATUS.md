# Status

**Phase 3 — Data: COMPLETE.** Phases 0, 1 and 2 done. Cost so far: **USD 0** (local, or free-tier Kaggle).

## Phase 3 acceptance criteria

| criterion (`PLAN.md` 3) | status | evidence |
|---|---|---|
| Archives downloaded and hash-verified; `MANIFEST.json` written | **pass** | ChartQA 875,370,872 B, sha256 `1bf310e5a5110168…`. RefChartQA is **streamed, not downloaded** (disk); its 9 parquet SHA-256s are recorded from the pinned revision *before* any fetch, so verification happens on Kaggle against a hash committed beforehand. Recorded total 2,884,503,702 B matches Phase 0's independent measurement exactly. |
| `--dev` mode works end to end without the full download | **pass** | `cdt-data dev`; RefChartQA streamed, never downloaded in full |
| Dedup merge count reported | **pass** | 628 merges, **609 of them ChartQA↔RefChartQA** — zero before the pixel-hash fix (0048) |
| Audit complete, decision recorded, `refchartqa_audit.jsonl` written | **pass** | **200/200 acceptable, gate PASSED** (0047) |
| Generator produces all chart types; box-verification test passes | **pass** | 8 types × 4 levels × 20 seeds = **640/640**; 6,000-example pool generated with 0 rejections |
| Plan yield reported by chart source, with the two rates | **pass** | human **15.41%**, machine **13.60%**, all **14.07%**, full 28,299-question training split |
| Both mixture files written with composition breakdowns | **pass** | `data/mixture_stage1.json`, `data/mixture_stage2.json`, 12,000 each |
| **Zero val/test records in either mixture**, asserted in code | **pass** | plus an image-level guard (0049) and an end-to-end check on the written files |

**567 tests pass**; `ruff check src tests scripts` clean; **CI green**.

### Mixtures

| | stage 1 (curriculum, L1→L4, not shuffled) | stage 2 (joint, shuffled) |
|---|---|---|
| total | 12,000 | 12,000 |
| synthetic | 6,000 | 1,825 (replay) |
| ChartQA | 6,000 | 7,087 |
| RefChartQA | — | 3,088 |
| with boxes | 12,000 | 12,000 |
| with plan | 6,952 | 2,930 |
| of those, compositional | 5,080 | 1,883 |

## The four findings that change what comes next

1. **ChartQA carries its own element boxes** (0042). 80.8% of training charts, 12.7 boxes each, bar extent
   linear in the gold value at r² = 0.9999. The plan treats RefChartQA as the only source of real grounding
   supervision; it is not. Recorded *before* the audit ran.
2. **Mining yields 14.07%, not the estimated 5.7%** — but 73.6% are bare `lookup` and every machine-generated
   one is. Compositional plans are ~3.7% of questions, close to the estimate, so the plan's conclusion
   (synthetic as primary plan supervision) stands (0046).
3. **Image identity had to become pixel-based** (0048). RefChartQA ships re-encoded ChartQA charts: **0 of
   4,000** matched by file bytes. Deduplication would have run, reported clean, and double-counted exactly as
   `PLAN.md` 3.3 warns — while looking like success.
4. **Held-out charts reach training through the split label** (0049). 4 RefChartQA rows labelled `train` use
   ChartQA val/test charts, and **15 of ChartQA's own train images are pixel-identical to held-out ones**.
   Every `split` field reads "train". Guarded at ingest with a count, and asserted at mixture time.

## One process failure worth stating plainly

`src/chartqa_dt/data/` — the entire loaders package, nine files — was matched by an unanchored `data/`
rule in `.gitignore` and **never committed**. Local tests passed throughout because the files were on
disk; CI failed eight consecutive pushes with `ModuleNotFoundError`, and `check_ci.py` had been
reporting it the whole time. Kaggle pulls from GitHub, so Phase 5 training would have failed on a clean
checkout, and losing this machine would have lost the Phase 3 data layer (0050).

Fixed by anchoring the rule to `/data/`, adding `tests/test_repo_completeness.py` (which would have
failed on the first commit), and `scripts/preflight.sh`, which rebuilds CI's own torch-free venv and
runs all four CI steps before a push. The measurements and artefacts are unaffected — they came from
real code that was correct and present locally. What was wrong was the claim that it was in the repo.

## Open items carried into Phase 4

- **RefChartQA scaling ladder** (`PLAN.md` 3.4): the audit decides *whether*; the ladder at 4,000 / 10,000 /
  25,000 decides *how many*. Needs validation grounding measurements, so it runs once Phase 4 exists.
- **Line-chart grounding from ChartQA** deliberately unimplemented: annotations give segment boxes, so a
  point's position is recoverable but its box size is not. Revisit only with a measured marker size.
- **Disk**: ~16 GiB free. ChartQA is read from the zip and never extracted; RefChartQA is streamed.

## Next: Phase 4 — evaluation, built before training

The zero-shot baseline is the "before" number the project is measured against. `PLAN.md` builds evaluation
first so it cannot be shaped to flatter a result that already exists.
