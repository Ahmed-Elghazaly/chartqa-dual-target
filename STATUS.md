# Status

**Phase 4 — Evaluation: COMPLETE.** Phases 0–3 done. Cost so far: **USD 0** (local, or free-tier Kaggle).

## Phase 4 acceptance criteria

| criterion (`PLAN.md` 4) | status | evidence |
|---|---|---|
| Metrics agree with **both** official evaluators on a shared prediction set | **pass** | On 11,690 **real** predictions: AP@0.5 differs by 0.000 / 0.068 / 0.036 pp (human/machine/pot); relaxed accuracy 0 of 423 disagreements; P@F1 0 of 40. Both evaluators vendored and hash-pinned. |
| Regression suite passes and is in CI | **pass** | `tests/test_metrics_regression.py`, all twelve `PLAN.md` 4.3 cases including the four it asks to *define*; runs in the fast CPU job |
| 32.83 reproduced, **or the discrepancy fully documented** | **documented** | It does not reproduce, and cannot: see below and `DECISIONS.md` 0052 |
| Stratified AP reporting works and reports the sub-token fraction | **pass** | Area buckets at one visual token; measured **24.8%** sub-token against `PLAN.md`'s predicted ~23.9% |
| `cdt-eval` runs end to end on `--dev` and writes structured results JSON | **pass** | No model, no network; writes `results.json` with intervals and strata |

**611 tests pass**; `ruff check src tests scripts` clean; **CI green**.

## The finding that changes the project's claim

**32.83 cannot be independently reproduced, because the artefacts needed do not exist.**
Running the byte-identical official evaluator on RefChartQA's released file gives:

| subset | published | official evaluator on the released file | delta |
|---|---:|---:|---:|
| human | **32.83** | **28.33** | −4.50 |
| machine | 59.28 | 71.25 | **+11.97** |
| pot | 39.32 | 59.66 | **+20.34** |

Deltas in both directions and up to +20 points are not a scoring error — they are a
different model's output. RefChartQA's own README calls that file *"an example file showing
the appropriate format"*; the repository has four files, publishes no per-model predictions,
and no checkpoints exist on the Hub. `PLAN.md` 4.4's premise is unsatisfiable, and `PLAN.md`
has been updated to say so.

**Consequence.** 32.83 is a **Level C** anchor — cited, not verified. The project's primary
claim moves to the **internal** comparison: the same backbone zero-shot versus fine-tuned,
both scored by us with the vendored official evaluator on the same sealed split. That is
reproducible end to end from this repository, and Phase 5 builds it.

## Three metric corrections, all in the official's favour (`DECISIONS.md` 0053)

1. **`relaxed_correctness`** — Appendix D strips commas and guards `t == 0` explicitly. The
   canonical pix2struct implementation does neither: it tests `target_float` for
   *truthiness*, so a gold `"0"` falls through to string comparison, and `float("1,234")`
   raises. Appendix D disagreed on **61 of 423** cases, every one in our favour.
2. **AP@0.5** — the official is COCO 101-point interpolation via `pycocotools`, not
   Appendix D's all-point rule. Appendix D was off by up to 0.009.
3. **P@F1 is not an F1** — the official helper computes COCO AP == 1.0 on one image. Every
   target must be matched *and* every false positive must rank after every true positive,
   so trailing extras are free and a leading one is fatal.

This sharpens `DECISIONS.md` 0014: P@F1 punishes **ordering**, AP punishes **count** — one
spurious box per image takes AP from 1.00 to 0.68 across a dataset.

## Open items carried into Phase 5

- **RefChartQA scaling ladder** (`PLAN.md` 3.4): the audit decided *whether*; the ladder at
  4,000 / 10,000 / 25,000 decides *how many*. It needs validation grounding numbers, which
  now exist as machinery.
- **One residual AP disagreement**, reported not fitted: 1 of 120 randomised scenarios
  differs by 0.0019. A float32 hypothesis was tried and made agreement *worse*, so it was
  reverted. No reported number depends on it — the official evaluator stays the scorer of
  record (`DECISIONS.md` 0003).
- **Line-chart grounding from ChartQA** remains deliberately unimplemented (segment boxes,
  no stated marker size).

## Next: Phase 5 — zero-shot baselines and pre-registration

The "before" number, measured with the ruler built above, and the pre-registration that
seals the test split until Phase 7.
