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
| 9 | 2026-08-26 | Kaggle T4 | 100 steps @512 + @native, with resume | 20 min | — | **ERROR** | **Ran stale code.** Dataset version 10 uploaded at 17:56:05; kernel started 17:56:07 and Kaggle attached version 9. Reproduced two already-fixed bugs exactly. Fixed by a code fingerprint the kernel verifies before doing anything (decision 0024). |
| 10 | 2026-08-27 | Kaggle T4 ×2 | 100 steps @512 + @native, with resume | 25 min | 1.75* | **partial** | Fingerprint gate confirmed current code. 512 arm ran 100 steps; native arm crashed. Root cause: `device_map="auto"` sharded the model across **two** T4s — explaining the crash, a +52% step-time penalty, and a memory figure measuring only device 0 (decision 0025). *Peak is therefore an undercount. |
| 11 | 2026-08-27 | Kaggle T4 (pinned) | 100 steps @512 + @native, with resume | 59 min | 5.57 / 6.72 | **COMPLETE** | First clean single-card measurement. 512px 11.90 s/step → 9.92 h; native 21.27 s/step → 17.72 h (77% over gate). Resume failed as predicted — launched before the RNG fix. |
| 12 | 2026-08-27 | Kaggle T4 (pinned) | 60 steps @512 × batch{2,4,8}, with resume | 42 min | 5.65–10.87 | **COMPLETE** | **Resume now passes** (deltas 0.0053/0.0018/0.0014 vs 0.0438 before the RNG fix). Micro-batch grouping buys only 5.5% time for 92% more memory. |

GPU hours are **not** totalled here — Kaggle is the authority. Run `python scripts/gpu_budget.py` for the live figure.

Runs 1–3 and 6 cost under a minute each because the failure was caught by a cheap gate rather than
discovered after a download. That is the entire argument for
`verification/preflight_checklist.md`: run 4 cost twenty minutes for a configuration mistake, and
run 8 cost twenty-five for two bugs that local tests now catch in milliseconds.

## Budget tracker

| Item | Spent | Cap |
|---|---:|---:|
| Paid compute | **USD 0.00** | USD 20.00 (interruption contingency only — rule 9) |
| Kaggle GPU hours this week | see `scripts/gpu_budget.py` (live) | 30 h |
| Committed ahead (Phases 5–7) | ~17 h | — |

## Phase 2 gates (IDEA.md §14, PLAN.md Appendix F)

**Note:** the figures below come from run 8, which is now known to have been **sharded across two
T4s** (decision 0025). Memory is an undercount and step time carries a ~52% inter-GPU penalty. They
are retained because they establish that the pipeline trains at all; the authoritative single-card
measurement is run 11.

Measured on run 8, `hf_peft` backend, 512-pixel budget, 100 optimizer steps, Tesla T4:

| Gate | Threshold | Measured | Status |
|---|---|---|---|
| Peak reserved memory | ≤ 13.5 GiB | 1.482 GiB (device 0 only — undercount) | provisional |
| Projected full run (3,000 steps) | ≤ 10 h | 7.22 h, vs 10.94 h on run 10 | **unresolved — the two straddle the gate** |
| LoRA trainable on both sides | non-zero by name and count | **7,208,960 vision / 17,432,576 language**, 0 unclassified | **pass** |
| No NaN | — | none over 100 steps | **pass** |
| Loss decreasing | — | **2.879 → 0.968** | **pass** |
| Vision tower not quantised | — | 104 full / 0 4-bit | **pass** |
| Checkpoint kill-and-resume | post-resume loss matches | _run 9_ | pending |
| Native-resolution arm | within gates | _run 9_ | pending |

**The binding constraint is time, not memory.** 1.48 GiB against a 13.5 GiB gate inverts the
assumption behind the Phase 2 fallback ladder, whose first two rungs (`batch 2→1`, `image 512→448`)
are memory levers. Memory is not the problem; 7.22 h against a 10 h ceiling is.
