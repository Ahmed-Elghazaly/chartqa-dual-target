# Where the project stands

Updated 2026-08-27. Read this after any interruption; read `WORKING_AGREEMENT.md` for how to work.

## Is anything running? Does anything need stopping?

**No.** Kaggle kernels terminate themselves, nothing costs money (USD 0.00 spent, USD 20 contingency
untouched), and everything is committed and pushed. Local watcher processes only poll — killing them
does not stop the Kaggle job, and leaving them costs nothing.

```bash
python scripts/gpu_budget.py        # live GPU quota from Kaggle
python scripts/check_ci.py          # CI status for the CURRENT commit
python scripts/check_credentials.py # all three credentials, with a negative control
python scripts/kaggle_run.py --status   # poll a running kernel
python scripts/kaggle_run.py --logs     # fetch a finished kernel's output
```

## Phase status

| Phase | State |
|---|---|
| 0 — Re-verification | **complete** — ten claims verified, eleven further findings |
| 1 — Environment and repository | **complete** — all six acceptance criteria met |
| 2 — Backbone smoke test | **essentially complete** — see below; one confirmation run outstanding |
| 3 — Data | not started (gated on Phase 2), but heavily pre-verified |
| 4+ | not started |

## Phase 2 result

Measured on a single pinned Tesla T4, 100 optimizer steps, `hf_peft`, 512-pixel budget:

| gate | threshold | measured | |
|---|---|---|---|
| peak reserved memory | ≤ 13.5 GiB | **5.57 GiB** | pass |
| projected full run | ≤ 20 h (revised, 0034) | **~10 h** | pass |
| kill-and-resume verified | post-resume loss matches | **delta 0.0014–0.0053** vs 1e-2 | pass |
| LoRA on both sides | non-zero each | **7,208,960 / 17,432,576**, 0 unclassified | pass |
| loss decreasing, no NaN | — | 2.72 → 1.14, none | pass |
| vision tower unquantised | — | 104 full / 0 4-bit | pass |
| gradients alive | non-zero, finite | median 13.3, zero dead steps | pass |
| not sharded | single device | `{'cuda:0': 625}` | pass |

**Backbone selected: `Qwen/Qwen3-VL-2B-Instruct`, `hf_peft` backend, 512-pixel budget, batch 2 × accum 4.**
`unsloth` is unavailable at this model size, as `IDEA.md` §7 predicted.

## What Phase 3 already has, before it starts

* exact box extraction **proven against pixels** for bar, line, pie and scatter — each a different
  matplotlib path, each adversarially tested
* an objective ink pre-screen for the RefChartQA audit, validated on known ground truth
* pinned Hub commit SHAs for all four artifacts
* the gold-table formats read, and the wide-table flattening problem measured (it moves plan yield 3.4×)
* the leakage question analysed: question text is not identity, so the dedup key must be
  `(image_sha256, question)`
* download, `datasets` and `zipfile` APIs read rather than assumed

## Open items

1. One confirmation run (448 vs 512 timing) — does not change any decision, only confirms the model.
2. `PREREGISTRATION.md` does not exist yet; until it does, every test split is refused by code.
3. The revised compute gate (0034) must be written into `PREREGISTRATION.md` before Phase 7.
