# Pre-flight checklist for any long GPU run

Kaggle allows roughly **30 GPU-hours per week**. Phase 6 alone is budgeted at 6–10 hours, so a long
run that turns out to have been misconfigured costs a meaningful fraction of a week and cannot
simply be repeated.

This checklist exists because seven Kaggle sessions were spent before one produced a usable
measurement, and **every one of those failures was cheap to detect locally and expensive to detect
remotely**. Nothing below requires a GPU except where marked.

---

## A. Before pushing anything to Kaggle (all local, seconds)

| # | Check | How | Status |
|---|---|---|---|
| A1 | Generated kernel code parses | `render_kernel_script()` calls `compile()` | automated |
| A2 | No credentials or dataset content staged | `assert_no_dataset_content`, explicit include list | automated |
| A3 | Every config loads and every override coerces | `pytest tests/test_config.py` | automated |
| A4 | Accelerator request uses a value the SDK documents | `test_machine_shape_values_match_the_sdk_contract` | automated |
| A5 | `torchao`, `peft`, `bitsandbytes` pinned before the model download | `test_torchao_is_pinned_before_the_model_download` | automated |
| A6 | Full test suite green, lint clean | `pytest -q && ruff check src tests scripts` | manual gate |

## B. In the first 60 seconds of the kernel (fail fast, before the 4.2 GB download)

| # | Check | Failure it prevents |
|---|---|---|
| B1 | A CUDA device exists | a silent CPU run that takes hours and measures nothing |
| B2 | Its `sm_XX` is in `torch.cuda.get_arch_list()` | a P100 (sm_60) that this PyTorch build cannot use at all |
| B3 | The code dataset is attached, contents listed | a kernel that starts with no code and fails three frames deep |
| B4 | `pyproject.toml` found; package installs | a partial upload |

## C. After the model loads, before training (seconds)

| # | Check | Failure it prevents |
|---|---|---|
| C1 | LoRA reaches **both** vision and language, by name and count | the documented Qwen3-VL bug where the vision tower silently never trains |
| C2 | Every declared LoRA target matches a real module | a guessed module name (`fc1` vs `linear_fc1`) attaching nothing |
| C3 | Vision tower is **not** 4-bit, verified on the loaded model | skip patterns that are accepted and match nothing |
| C4 | The resolved compute dtype is recorded | emulated bf16 on pre-Ampere inflating step time by an unknown factor |
| C5 | Visual-token geometry read from the processor | a factor-28 assumption on a factor-32 model |

## D. During the first ~10 steps (cheap, catches the silent killers)

| # | Check | Failure it prevents |
|---|---|---|
| D1 | Loss is finite | divergence |
| D2 | Loss decreases | a wrong objective, or a frozen model |
| D3 | **Gradient norm is non-zero and finite** | fp16 underflow with no scaler — loss sits flat, nothing errors |
| D4 | Supervised label count matches the answer length | training on the prompt as well as the answer |
| D5 | Peak reserved memory inside the gate | an OOM twenty minutes in |

## E. Before committing to the full run

| # | Check | Why |
|---|---|---|
| E1 | A ≤100-step run of the **exact same config** has passed A–D | the only reliable evidence is a rehearsal of the real thing |
| E2 | Kill-and-resume verified by comparing post-resume loss | a resume that has never been tested does not work |
| E3 | Projected wall time inside the gate **and** the weekly quota | `scripts/gpu_budget.py` |
| E4 | Checkpoints push to the Hub on every save | a killed session loses nothing |

---

## Known gaps, carried forward deliberately

Recorded here so they are not rediscovered as surprises.

| Gap | Where it matters | Why it is acceptable now |
|---|---|---|
| **No learning-rate scheduler** in the smoke loop | Phase 6 must add cosine + `warmup_ratio` from the config | The smoke test measures memory and step time; a scheduler changes neither. It must **not** be inherited by Phase 6 training. |
| **fp16 without a `GradScaler`** | Phase 6 | LoRA parameters are upcast to fp32 by `prepare_model_for_kbit_training`, so gradients are fp32 and underflow is unlikely — but "unlikely" is why D3 now gates on gradient norm. If Phase 6 shows dead gradients, add a scaler. |
| `device = next(model.parameters()).device` assumes a single device | Phase 6 if a model is ever sharded | A 2B model in 4-bit uses 1.48 GB of a 15 GB card; sharding will not occur. Revisit only if the backbone changes. |
| Resume test uses plain `AdamW`, not `AdamW8bit` | Phase 6 checkpointing | Optimizer *state* round-trip is what is being verified; the implementation difference does not affect that. Phase 6 must round-trip the real optimizer. |

## Why Phase 2 is not being re-run for the D3 instrumentation

Gradient-norm recording was added *after* the Phase 2 measurement started. Re-running to capture it
would cost about an hour of a 30-hour weekly quota to improve a diagnostic that **no Phase 2 gate
depends on** — the gates are peak memory, projected wall time, LoRA coverage, no NaN, and loss
decreasing, all of which the running job already records.

The instrumentation matters for Phase 6, where it will be present from the first step. Spending
quota to backfill a number into a phase that does not use it would be exactly the kind of avoidable
consumption this checklist exists to prevent.
