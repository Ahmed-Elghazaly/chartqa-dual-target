# Working agreement

Standing instructions from Ahmed and the operating rules derived from them. This is the file to read
first after any context loss — it is what would otherwise have to be re-learned by repeating mistakes.

Everything here was earned, not assumed. Where a rule exists because something went wrong, the
decision entry is cited.

---

## How Ahmed wants this done

| | |
|---|---|
| **Work in long sessions.** | Do not stop every few minutes to report. When waiting on a Kaggle run, work on something that does not depend on it. Re-prompting is expensive. |
| **Be right the first time.** | Read the authoritative source — API signature, library source, model config, the plan — *before* writing code against it. Never from memory. |
| **Verify, don't assume.** | Research anything unclear. Check whether tools, skills or MCP servers exist that would help. Do not build on old or general knowledge. |
| **Report briefly and plainly.** | At the end of a stretch: what was done or found, whether anything is wrong, whether anything is needed from Ahmed. Short. |
| **Test heavily.** | Many tests and verifications. Prove techniques against ground truth known by construction before depending on them. |
| **Delete what is dead.** | Redundant, superseded or wrong files are removed, not left to rot. Do not pollute the repository. |
| **The plan may be edited.** | `PLAN.md` and `IDEA.md` contain errors; when one is found, say so with evidence and propose the change rather than silently working around it. |
| **Recover from mistakes properly.** | Find them, fix them, record them — but the goal is not to generate corrections. |

## Standing facts about the environment

* **Kaggle** — account `nanonanite`. Token is a `KGAT_` **bearer** token in `~/.kaggle/access_token`,
  **not** the legacy `kaggle.json` username/key pair. Putting it in the wrong file fails every
  authenticated endpoint (`SETUP.md`).
* **Quota** — 30 GPU-hours per week, resetting weekly. Read it live with
  `python scripts/gpu_budget.py`; never keep a parallel tally.
* **GPU** — request `machine_shape: "NvidiaTeslaT4"` explicitly. Kaggle otherwise hands out a P100
  (`sm_60`) that its own PyTorch build cannot use (0019, 0020).
* **Hugging Face** — user `NanoPhotonic`, write token in `.env`. Private artifact repo
  `NanoPhotonic/chartqa-dt-artifacts`, verified end to end (0022).
* **GitHub** — `gh` CLI, account `Ahmed-Elghazaly`, scopes include `repo` and `workflow`. No separate
  token needed. Private repo `Ahmed-Elghazaly/chartqa-dual-target`.
* **Local disk** — about 11 GB free, so full-corpus work happens on Kaggle and local work uses `--dev`.
* **TLS** — the venv Python has no CA store; import `chartqa_dt.net` in any script that makes network
  calls or `urllib` will raise what looks like a rejected credential.
* **Skills/MCP** — searched for pytorch / transformers / VLM / LoRA / HF / Kaggle: **none exist.**

## The failure patterns this project keeps hitting

Every one of these produced a plausible, wrong result rather than an error.

1. **Acceptance is not compliance.** A request that is accepted may have no effect.
   `llm_int8_skip_modules=["visual"]` matched nothing (0012); `machine_shape="gpu_t4x2"` was ignored
   (0020); an uploaded code version was superseded by an older one (0024). **Verify the effect, never
   the acknowledgement.**
2. **A guard that cannot fail is not a guard.** `is_bf16_supported()` defaults to counting emulation,
   so it never fired on the hardware it existed for (0018). `torch.cuda.is_available()` was true on a
   GPU PyTorch could not use (0019). **Test the condition that causes the failure, not a nearby
   boolean.**
3. **Reading tells you what code says; running tells you what it does.** The official evaluator's
   behaviour reverses between one image and twenty (0014). **Execute it, at the scale you will use.**
4. **Writing a risk down is not assessing it.** A "known gap" accepted with a confident reason broke a
   session (0021). **Check it cheaply instead.**
5. **Defaults are decisions someone else made.** `device_map="auto"` split the model across two GPUs
   because it could, not because it needed to (0025).
6. **A plausible failure is more dangerous than an implausible one.** `resume delta 0.0456` invited
   widening the tolerance; the real cause was a missing checkpoint component (0026).
7. **Prose is not a control.** Rule 1 was in three documents and still violated (0031). Every
   invariant that actually holds in this project is an assertion.
8. **"It passes locally" can be true and useless.** `src/chartqa_dt/data/` — nine files, the whole
   loaders package — was excluded by an unanchored `.gitignore` rule and never committed. Local tests
   passed because the files were on disk; CI failed eight consecutive pushes with
   `ModuleNotFoundError` (0050). **The dev environment is not the environment.** Run
   `bash scripts/preflight.sh`, which reproduces CI's own venv, before every push.
9. **A tool that reports a failure nobody reads is no tool.** `check_ci.py` was written after an
   earlier instance of exactly this, and it had been printing `failure=8` the whole time. Checking is
   now a step in `preflight.sh`, not something to remember.
10. **A criterion valid for one geometry is not valid for another.** Displacement false-failed on
    adjacent bars (0038); relative tightness, correct for solid synthetic elements, failed RefChartQA
    at 84% and would have dropped the dataset — its boxes sit on printed numbers *inside* bars (0047).
    **Prove a criterion against ground truth you can see before you let it gate anything.**
11. **Identity has to match the question you are asking.** Hashing image *files* found 0 of 4,000
    cross-dataset duplicates; hashing *pixels* found 609 merges and 19 contaminated records (0048,
    0049). The first would have reported a clean deduplication while double-counting.

## Non-negotiables, and how each is now enforced

| rule | enforcement |
|---|---|
| Test splits are sealed | `chartqa_dt.splits` refuses by default; opening needs a committed, clean `PREREGISTRATION.md` **and** a logged reason |
| LoRA must reach vision **and** language | `assert_lora_on_both_sides`, fails the run, checks names not just counts |
| Invalid outputs count as failures | executor raises and is counted; never silently replaced |
| Never average two evaluator variants | official vendored byte-identical and hash-pinned; ours is a labelled diagnostic |
| No dataset content in git | `/data/` **anchored** in `.gitignore`, `assert_no_dataset_content` on every upload, CI history check |
| No source file silently untracked | `tests/test_repo_completeness.py` runs `git check-ignore` over `src`, `tests`, `scripts` |
| CI is green before a phase is reported done | `bash scripts/preflight.sh` (CI's own venv, all four steps) then `scripts/check_ci.py` |
| Held-out charts never reach training | `sealed_image_hashes` + `assert_no_held_out_images`; image-level, not split-label |
| Record every decision | `DECISIONS.md`, format enforced by `tests/test_docs_consistency.py` |

## Where things stand

`STATUS.md` has the current phase state. `verification/measured_facts.json` is the single source of
truth for every measured number — change it there and the tests will find every document that
disagrees.
