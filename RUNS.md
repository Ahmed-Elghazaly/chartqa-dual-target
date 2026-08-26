# Run log

One row per GPU session. Free tiers get killed; this is how we know what actually ran, where the
artifact went, and what it cost. Appendix F requires it, and `scripts/gpu_budget.py` reads the wall
times in this table to track consumption against Kaggle's ~30 GPU-hours per week.

| # | Date | Host | What ran | Wall | Peak GB | Outcome | Notes |
|---|---|---|---|---|---|---|---|
| 1 | 2026-08-26 | Kaggle | 3 steps @512 | 0.5 min | — | **ERROR** | Code dataset never attached: Kaggle lowercases refs and `dataset_sources` had a mixed-case username. |
| 2 | 2026-08-26 | Kaggle | 3 steps @512 | 0.5 min | — | **ERROR** | Generated kernel had a `SyntaxError`: a `\n` in a non-raw string became a real newline. Generated code is now `compile()`d before pushing. |
| 3 | 2026-08-26 | Kaggle | 3 steps @512 | 0.5 min | — | **ERROR** | Expected `code.zip`, found it already expanded. Kaggle auto-extracts uploads; both layouts now handled. |
| 4 | 2026-08-26 | Kaggle | 3 steps @512 | 20 min | — | superseded | Slow; investigating led to decisions 0017/0018 (bf16 emulated on pre-Ampere). |
| 5 | 2026-08-26 | Kaggle **P100** | 3 steps @512 | 4 min | — | **ERROR** | Two findings: Kaggle assigned a P100 (sm_60) unusable by its own PyTorch build, and `torchao 0.10.0` is too old for transformers/peft. **Confirmed working:** Kaggle path resolution, seeding, backend registry, and decision 0012's quantisation skip. |
| 6 | 2026-08-26 | Kaggle **P100** | 3 steps @512 | 0.5 min | — | **ERROR (by design)** | P100 again — `machine_shape: "gpu_t4x2"` was accepted and silently ignored. The new architecture check caught it in twenty seconds. Correct values are `NvidiaTeslaT4` / `NvidiaTeslaP100` / `Tpu1VmV38` (decision 0020). |
| 7 | 2026-08-26 | Kaggle **T4** | 3 steps @512, no resume test | 3 min | 1.48 | **COMPLETE** | First fully working run. Path validated end to end; every guard fired correctly (dtype substitution, quantisation skip, LoRA both sides). |
| 8 | 2026-08-26 | Kaggle T4 | **100 steps** @512 + @native, with resume | 25 min | 1.48 | **partial** | 512 arm completed 100 steps and passed every gate; resume failed on `KeyError: 'exp_avg'`, native arm failed before training. Both bugs were ours — decision 0021. |
| 9 | 2026-08-26 | Kaggle T4 | 100 steps @512 + @native, with resume | _running_ | | | Re-run with both fixes. |

**Total GPU time so far: ~0.9 h** of the ~30 h weekly quota.

Runs 1–3 and 6 cost under a minute each because the failure was caught by a cheap gate rather than
discovered after a download. That is the entire argument for
`verification/preflight_checklist.md`: run 4 cost twenty minutes for a configuration mistake, and
run 8 cost twenty-five for two bugs that local tests now catch in milliseconds.

## Budget tracker

| Item | Spent | Cap |
|---|---:|---:|
| Paid compute | **USD 0.00** | USD 20.00 (interruption contingency only — rule 9) |
| Kaggle GPU hours this week | **~0.9 h** | ~30 h |
| Committed ahead (Phases 5–7) | ~17 h | — |

## Phase 2 gates (IDEA.md §14, PLAN.md Appendix F)

Measured on run 8, `hf_peft` backend, 512-pixel budget, 100 optimizer steps, Tesla T4:

| Gate | Threshold | Measured | Status |
|---|---|---|---|
| Peak reserved memory | ≤ 13.5 GiB | **1.482 GiB** | **pass**, with an order of magnitude to spare |
| Projected full run (3,000 steps) | ≤ 10 h | **7.22 h** | **pass** |
| LoRA trainable on both sides | non-zero by name and count | **7,208,960 vision / 17,432,576 language**, 0 unclassified | **pass** |
| No NaN | — | none over 100 steps | **pass** |
| Loss decreasing | — | **2.879 → 0.968** | **pass** |
| Vision tower not quantised | — | 104 full / 0 4-bit | **pass** |
| Checkpoint kill-and-resume | post-resume loss matches | _run 9_ | pending |
| Native-resolution arm | within gates | _run 9_ | pending |

**The binding constraint is time, not memory.** 1.48 GiB against a 13.5 GiB gate inverts the
assumption behind the Phase 2 fallback ladder, whose first two rungs (`batch 2→1`, `image 512→448`)
are memory levers. Memory is not the problem; 7.22 h against a 10 h ceiling is.
