# Where the project stands

Updated 2026-08-26. Read this first after any interruption.

## Is anything running right now?

Possibly a Kaggle kernel. **Nothing needs to be stopped, and nothing costs money.**

* Kaggle kernels terminate themselves. A run of this project's kind takes 20–45 minutes and stops on
  its own; Kaggle's own hard ceiling is 12 hours.
* No paid compute is used anywhere in this project by design. **USD 0.00 spent**, and the USD 20
  contingency is untouched.
* Everything is committed and pushed. Closing the laptop loses nothing.

If you *want* to stop a running kernel: kaggle.com/code → your notebook → Stop. It is never
necessary — the only thing it saves is a few minutes of the weekly GPU quota.

To see what is happening at any time:

```bash
cd chartqa-dual-target
python scripts/gpu_budget.py      # GPU hours used and remaining
python scripts/check_ci.py        # CI status for the current commit
python scripts/check_credentials.py
```

## Phase status

| Phase | State |
|---|---|
| 0 — Re-verification | **complete** — all ten claims verified; eleven further findings recorded |
| 1 — Environment and repository | **complete** — all six acceptance criteria met |
| 2 — Backbone smoke test | **measurement in progress**; 512-pixel arm already passes every gate |
| 3 — Data | not started (gated on Phase 2) |
| 4+ | not started |

## What Phase 2 has already established

Measured on a Kaggle Tesla T4, 100 optimizer steps, `hf_peft` backend, 512-pixel budget:

| Gate | Threshold | Measured | |
|---|---|---|---|
| Peak reserved memory | ≤ 13.5 GiB | **1.482 GiB** | pass |
| Projected full run (3,000 steps) | ≤ 10 h | **7.22 h** | pass |
| LoRA on both sides | non-zero each | **7,208,960** vision / **17,432,576** language | pass |
| Loss over 100 steps | must fall | **2.879 → 0.968** | pass |
| NaN | none | none | pass |
| Vision tower excluded from 4-bit | — | 104 full / 0 quantised | pass |

Outstanding for Phase 2: the checkpoint kill-and-resume verification, and the native-resolution arm.
Both are what the current run is for.

## Budget

| | |
|---|---|
| Paid compute | **USD 0.00** of USD 20 contingency |
| Kaggle GPU hours used | **~0.9 h** of ~30 h per week (resets weekly) |
| Committed ahead (Phases 5–7) | ~17 h |

## If work stops unexpectedly

Nothing is in a fragile state. Concretely:

1. **No action required.** Close everything.
2. All work is in git and pushed to the private repository. `git log` is the record.
3. `DECISIONS.md` explains every choice made and why, in order.
4. `RUNS.md` records every GPU session and what it cost.
5. `verification/preflight_checklist.md` is what any long run must clear before starting.
6. `verification/measured_facts.json` is the single source of truth for every measured number.

To resume, the next task is whatever the Phase table above says is in progress.

## Known-good commands

```bash
pytest -q                                   # full test suite
ruff check src tests scripts                # lint
python scripts/kaggle_run.py smoke --steps 100 --resolutions 512,native
python scripts/kaggle_run.py --status       # poll a running kernel
python scripts/kaggle_run.py --logs         # fetch a finished kernel's output
```
