# Run log

One row per GPU session. Free tiers get killed; this is how we know what actually ran, where the
artifact went, and what it cost. Appendix F requires it.

| # | Date | Host | What ran | Wall | Peak GB | Outcome | Notes |
|---|---|---|---|---|---|---|---|
| 1 | 2026-08-26 | Kaggle T4 | Phase 2 smoke, 3 steps @512 (path validation) | ~30 s | — | **ERROR** | Code dataset never attached: Kaggle lowercases refs, `dataset_sources` had the mixed-case username. Fixed; kernel now prints `/kaggle/input` and exits with an explanation. |
| 2 | 2026-08-26 | Kaggle | 3 steps @512 | ~30 s | — | **ERROR** | Generated kernel had a `SyntaxError`: a `\n` in a non-raw string became a real newline. Generated code is now `compile()`d before pushing. |
| 3 | 2026-08-26 | Kaggle | 3 steps @512 | ~30 s | — | **ERROR** | Expected `code.zip`, found it already expanded. Kaggle auto-extracts uploads; both layouts now handled. |
| 4 | 2026-08-26 | Kaggle | 3 steps @512 | ~20 min | — | superseded | Slow; investigation found bf16 emulated on pre-Ampere and led to decisions 0017/0018. |
| 5 | 2026-08-26 | **Kaggle P100** | 3 steps @512, no resume test | ~4 min | — | **ERROR** | Two real findings: Kaggle assigned a **P100 (sm_60)** unusable by its own PyTorch build, and `torchao 0.10.0` is too old for transformers/peft. Both fixed (decision 0019). **Confirmed working:** Kaggle path resolution, seeding, backend registry, and decision 0012's vision-tower quantisation skip (`vision 104 full / 0 4-bit`). |

| 6 | 2026-08-26 | Kaggle P100 | 3 steps @512 | ~20 s | — | **ERROR (by design)** | Assigned a P100 again: `machine_shape: "gpu_t4x2"` was accepted and silently ignored. The new architecture check caught it in twenty seconds instead of running a meaningless benchmark. Correct values are `NvidiaTeslaT4` / `NvidiaTeslaP100` / `Tpu1VmV38` (decision 0020). |
| 7 | 2026-08-26 | Kaggle T4 (requested) | 3 steps @512, no resume test | _running_ | | | Survived the twenty-second architecture gate, so the accelerator request was honoured this time. |

Deliberately running 3-step validations before the real 100-step measurement. Row 1 is exactly why:
a full run would have burned a session to discover a one-word bug in a metadata field. Rows 5–6 are
the same argument at a larger scale — two Kaggle sessions to learn that the free tier hands out a
card its own PyTorch cannot use, and that asking for a different one requires a string the SDK
documents but does not validate.

## Budget tracker

| Item | Spent | Cap |
|---|---:|---:|
| Paid compute | **USD 0.00** | USD 20.00 (interruption contingency only — rule 9) |
| Kaggle GPU hours this week | ~0.01 | ~30 |

## Phase 2 gates (IDEA.md 14, PLAN.md Appendix F)

| Gate | Threshold | Status |
|---|---|---|
| Peak reserved memory | ≤ 13.5 GiB | _pending_ |
| Projected full run (3,000 optimizer steps) | ≤ 10 h | _pending_ |
| LoRA trainable on **both** vision and language | non-zero by name and count | verified on the real architecture at toy scale; pending on-GPU |
| No NaN, no OOM | — | _pending_ |
| Checkpoint kill-and-resume | post-resume loss matches | _pending_ |
