# Status

**Phase 3 — Data. Complete except for the two mixture files, which are building.**
Phases 0, 1 and 2 are done. Cost so far: **USD 0** (everything local or free-tier Kaggle).

## Phase 3 results against its acceptance criteria

| criterion (`PLAN.md` 3) | status | evidence |
|---|---|---|
| Archives downloaded and hash-verified; `MANIFEST.json` written | **pass** | ChartQA 875,370,872 B, sha256 `1bf310e5a5110168…`, pinned revision `af8b6f5c08c9` |
| `--dev` mode works end to end without the full download | **pass** | `cdt-data dev`; RefChartQA streamed, never downloaded in full |
| Dedup merge count reported | **pass** | `DedupReport.summary()`; merges within a split, cross-split collisions reported not resolved |
| Audit complete, decision recorded, `refchartqa_audit.jsonl` written | **pass** | **200/200 acceptable, gate PASSED**; `DECISIONS.md` 0047 |
| Generator produces all chart types; box-verification test passes | **pass** | 8 types × 4 levels × 20 seeds = **640/640** verified; adversarial tests paired with every acceptance test |
| Plan yield reported by chart source, with the two rates | **pass** | human **15.41%**, machine **13.60%**, all **14.07%** over the full 28,299-question training split |
| Both mixture files written with composition breakdowns | **building** | sources cached: synthetic 6,000 generating, ChartQA local, RefChartQA 4,000 |
| **Zero val/test records in either mixture**, asserted in code | **pass** | `tests/test_mixture.py`; the check runs on the *inputs*, not the survivors |

530 tests pass; `ruff check src tests scripts` clean.

## The three findings that change what comes next

1. **ChartQA carries its own element boxes** (`DECISIONS.md` 0042). 80.8% of training charts, 12.7 boxes
   each, bar extent linear in the gold value at r² = 0.9999. The plan treats RefChartQA as the only source of
   real grounding supervision; it is not. Recorded *before* the audit ran.
2. **Mining yields 14.07%, not the estimated 5.7%** — but 73.6% of those plans are bare `lookup`, and every
   machine-generated one is. Compositional plans are ~3.7% of all questions, close to the original estimate,
   so the plan's conclusion (synthetic as primary plan supervision) stands (`DECISIONS.md` 0046).
3. **The plan's expected yield split is backwards, and the reason is informative.** It predicts human 1.9% /
   machine 16.5% from gold-table corruption. The corruption is real — human questions find no matching
   operation 5.3× as often — but human questions are also far less ambiguous, and the effects cancel.

## Open items carried into Phase 4

- **RefChartQA scaling ladder** (`PLAN.md` 3.4): the audit decides *whether* to use RefChartQA training rows;
  the ladder at 4,000 / 10,000 / 25,000 decides *how many*. It needs validation grounding measurements, so it
  runs once Phase 4's evaluators exist.
- **Line-chart grounding from ChartQA** is deliberately unimplemented: the annotations give segment boxes, so
  a point's position is recoverable but its box size is not. Revisit only with a measured marker size.
- **Disk**: 17 GiB free. ChartQA is read from the zip and never extracted; RefChartQA is streamed.

## Next: Phase 4 — evaluation, built before training

The zero-shot baseline is the "before" number the whole project is measured against. `PLAN.md` is explicit
that evaluation is built first, so it cannot be shaped to flatter a result that already exists.
