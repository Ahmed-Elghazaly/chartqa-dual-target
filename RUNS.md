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

Deliberately running a 3-step validation before the real 100-step measurement. Row 1 is exactly why:
a full run would have burned a session to discover a one-word bug in a metadata field.

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
