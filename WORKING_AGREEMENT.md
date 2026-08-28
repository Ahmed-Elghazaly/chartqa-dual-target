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

### Reporting (added 2026-08-27, standing)

Every report must say, in plain language:

1. **Is this good or bad for us?** Label each finding explicitly. A measurement is not
   self-explanatory — "AP is 0.68" means nothing to a reader until it is "worse than we
   hoped and here is what it costs us". Never leave the judgement implicit.
2. **What was done, simply.** Progress, in the fewest words that stay accurate.
3. **What was decided**, and why it went that way.
4. **What is needed from Ahmed**, or explicitly "nothing".
5. **Anything else worth knowing** — surprises, risks, wasted effort, cost.

Brief and concrete. The failure to avoid is a technically complete report that leaves the
reader unable to tell whether the project is going well.

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

## Before writing a non-trivial component: the design pass (added 2026-08-28)

Ahmed's objection, and it is correct: too many defects were *discovered by running* when
they were knowable beforehand. The prompt took four GPU iterations, and each one found
something already established — `PLAN.md` itself said "compact JSON" and the first version
was pretty-printed; that a model imitates its example's formatting is not a new result.

The diagnosis is not carelessness. It is **iterating where front-loading was possible**.
So, before writing any component that costs a run to test:

1. **Re-read what this project already decided.** `PLAN.md` for the section, `DECISIONS.md`
   for anything related, `measured_facts.json` for the numbers. Most of the prompt defects
   were already answered in text I had read once and not applied.
2. **Research the established practice**, from primary sources — library docs, the actual
   signature, the paper, the model card. Not memory. If a whole class of failure has a
   standard solution, find out *before* building around it (`DECISIONS.md` 0061 evaluates
   constrained decoding; it should have been evaluated before iteration one, not after
   four).
3. **Enumerate the failure modes on paper first.** For each: how would it show up, and
   what is the cheapest measurement that would reveal it? A component whose failure modes
   were listed in advance rarely surprises.
4. **Measure the direct thing, never a proxy.** `DECISIONS.md` 0060 claimed a limitation
   from *table length* when *evidence length in actual plans* was one query away and said
   the opposite. If a proxy is all that is available, say so in the claim.
5. **Only then write**, and pair each failure mode with the test that catches it.

The cost of this pass is minutes. The cost of skipping it has been GPU runs, wrong entries
in the decision log, and Ahmed reading a correction instead of a result.

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

## Two shell habits that have each caused a wrong report

1. **Never pipe a check whose exit status you then rely on.**
   `bash scripts/preflight.sh | tail -3 && git commit …` runs the commit even when
   preflight fails, because a pipeline's status is the *last* command's. That is how a
   red commit reached `main`. Run the check bare, read its output, then commit as a
   separate step.
2. **`git status` is silent about ignored files.** A source file excluded by
   `.gitignore` shows nothing at all — not "untracked", not "modified" — so a whole
   package can be missing from the repository while every local run passes
   (`DECISIONS.md` 0050, 0051). `tests/test_repo_completeness.py` runs `git check-ignore`
   over every source file and required artefact, and it is step 1 of preflight.
