# Dual-Target Grounded-Execute ChartQA

Fine-tune one small vision-language model so that, given a chart image and a question, it
(a) says whether the question is answerable, (b) draws boxes around the evidence it used,
(c) emits the required calculation as a **typed program**, and (d) gives an answer — then a
small deterministic CPU interpreter re-runs that program against the model's own evidence, so
the arithmetic is **checkable rather than asserted**.

> The interpreter currently *checks* the answer; it does not replace it. Every evaluation
> path scores the model's stated answer, and disagreement between the two is reported as a
> headline number (`DECISIONS.md` 0059). Whether scoring the executed value instead would do
> better is a real question with a cheap answer — `plans.roundtrip.answer_under` scores all
> three policies from one set of generations — and it is open (0096).

Measured on **two** official public benchmarks, each against the same untrained model:
**ChartQA** relaxed accuracy and **RefChartQA** grounding (AP@0.5 and P@F1).

> **Status: in development. No results yet.** Phases are built in order and their gates are
> hard. See `DECISIONS.md` for every choice made so far and why.

---

## Which file to open

Nine markdown files at the root is a lot; each answers a different question.

| if you want to know | read |
|---|---|
| **what this project is** and how to run it | this file, then `SETUP.md` |
| **how the system works today**, end to end | `ARCHITECTURE.md` |
| **the audit's verdict** — what was wrong, what changed, what to test next | `VERDICT.md` |
| **every finding in the brief's 15-field record** | `FINDINGS.md` |
| **why any choice was made** — the permanent record, ~100 entries, newest last | `DECISIONS.md` |
| **what is currently wrong** and how bad, ranked | `AUDIT.md` |
| **what the audit still has to do** and what is blocked | `AUDIT_PLAN.md` |
| **where the project stands right now** — measured numbers only | `STATUS.md` |
| **what was promised before the results existed** | `PREREGISTRATION.md` |
| **what each training run did** and what it cost | `RUNS.md` |
| **the rules this repo is built under** — non-negotiables | `WORKING_AGREEMENT.md` |

`DECISIONS.md` is the one to search. Every finding, every reversal and every measurement
that justified a change is recorded there with its numbers, including the ones that turned
out to be wrong and why. Code comments point into it by number.

### Where the code lives

| directory | holds |
|---|---|
| `src/chartqa_dt/` | the library — data readers, plan executor, training, evaluation |
| `scripts/` | things you run once: build mixtures, mine plans, check the environment |
| `audit/` | one-off measurement scripts, kept so a number can be reproduced |
| `tests/` | ~1,050 tests; `scripts/preflight.sh` runs the lot plus lint |
| `book/`, `presentation/` | explanatory writing, not part of the pipeline |

---

## The idea in one example

Question: *"How many more units shipped in 2019 than in 2018?"* over a bar chart.

```json
{
  "answerable": true,
  "evidence": [
    {"label": "2019", "value": 245, "unit": "millions", "bbox": [412, 180, 486, 742]},
    {"label": "2018", "value": 210, "unit": "millions", "bbox": [318, 265, 392, 742]}
  ],
  "plan": {"op": "difference",
           "args": [{"op": "lookup", "args": ["2019"]},
                    {"op": "lookup", "args": ["2018"]}]},
  "model_answer": "35"
}
```

The executor validates the record, resolves the lookups, computes `245 − 210` and returns **35**.
If the model had said `"34"`, the executor would still return 35 — which separates *did it read
the chart correctly* from *can it do arithmetic*. Both are measured.

## Why this task

Chart **answer accuracy** is close to saturated: Qwen3-VL-2B-Instruct already scores **79.1** on
ChartQA out of the box, and the best published gain over a comparable prompted model is +3.8
points obtained on 4×H100. Chart **evidence grounding** is not saturated: the strongest published
RefChartQA-human AP@0.5 is **32.83**, from a *larger* 3B model trained one epoch with boxes as
plain text.

So the project reports both, and is designed so the defensible measured win does not depend on
the saturated one.

## Results

<!-- PHASE 7 — every cell carries a bootstrap CI, and every comparison states whether it is matched. -->

| System | ChartQA relaxed (H / M / all) | RefChartQA AP@0.5 (H / M / PoT) | RefChartQA P@F1 |
|---|---|---|---|
| Untouched model (zero-shot) | _pending Phase 5_ | _pending Phase 5_ | _pending_ |
| Direct-answer LoRA (control) | _pending Phase 7_ | _pending Phase 7_ | _pending_ |
| **Grounded plan + executor** | _pending Phase 7_ | _pending Phase 7_ | _pending_ |
| RefChartQA published reference | — | 32.83 (human) / 59.28 / 39.32 | _pending Phase 4.4_ |

The published 32.83 is the **human subset only** — the official evaluator has no aggregate
(see `DECISIONS.md` 0002). Human-subset comparisons are against **n = 500**, which bounds how
small a difference can be called real.

## Install

Python **3.11**.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # CPU: executor, metrics, generator, tests
pip install -e ".[dev,eval,gpu]"   # add the official evaluators and the training stack
python scripts/check_credentials.py
```

See [`SETUP.md`](SETUP.md) for credentials — in particular the Kaggle token trap, where a valid
token in the wrong file fails on every authenticated endpoint while an unauthenticated endpoint
keeps returning 200 and makes it look like a permissions problem.

## Commands

```
cdt-data     download, hash-verify, audit, deduplicate, build mixtures
cdt-gen      generate synthetic charts with exact boxes, answers and typed plans
cdt-mine     mine typed plans from real ChartQA gold tables (uniqueness rule)
cdt-train    LoRA fine-tuning: smoke / stage1 / stage2 / control
cdt-eval     evaluate with the official ChartQA and RefChartQA evaluators
cdt-report   assemble LaTeX tables and figures from recorded results
```

Any config field is overridable by its dotted path:

```bash
cdt-train --config configs/stage1_grounding.yaml --train.lr 5e-5 --data.max_examples 2000
```

`cdt-<cmd> --list-fields` prints every field. The fully resolved config, the git SHA and a
dirty-tree flag are written to `<output_dir>/resolved_config.yaml` at the start of every run.

## Running on a free GPU

The package is installable and CLI-driven precisely so the same command works on Kaggle, Colab
and a rented box. `notebooks/kaggle_run.ipynb` and `notebooks/colab_run.ipynb` are ten-line
wrappers: clone, install, authenticate, call the CLI. Long jobs are resumable and push
checkpoints to a private Hugging Face repo on every save, because free sessions are killed
without warning.

## Ground rules

These are enforced in code and in CI, not by discipline:

1. **Test splits are sealed.** ChartQA test, RefChartQA test and ChartQAPro are opened once, at
   the end, after decisions are frozen in `PREREGISTRATION.md`. `tests/` asserts no validation or
   test record reaches a training mixture.
2. **LoRA must reach both the vision and the language sides.** Qwen3-VL issues #2016/#2079 make
   this fail *silently*; the run asserts non-zero trainable parameters on both sides by name and
   count, and dies if either is zero.
3. **Invalid outputs are counted as failures**, never quietly replaced by a fallback.
4. **Two evaluator variants are never averaged, swapped or relabelled.** The pinned official tool
   output is primary; anything corrected is reported separately with its disagreement count.
5. **No dataset content in git.** ChartQA is GPL-3.0 and RefChartQA is AGPL-3.0. Scripts, IDs,
   hashes, adapters and derived statistics only — CI fails the build if an image or archive ever
   appears in history.

## Repository map

```
configs/          YAML configs; every field CLI-overridable
src/chartqa_dt/
  env.py          Kaggle/Colab/local detection — the only place paths are decided
  config.py       typed config, strict key checking, resolved dump with git SHA
  hub.py          private HF push/pull, with the rule-7 upload guard
  data/           download, loaders, dedup, the 200-row audit, mixtures
  synth/          own matplotlib generator; boxes read from artists, never estimated
  plans/          schema, deterministic executor, plan mining
  vision/         coordinate and smart_resize mathematics
  modeling/       backends, prompts, the LoRA assertion, constrained decoding
  train/          stage1, stage2, control, resumable checkpointing
  eval/           metrics, official evaluator wrappers, stratification, oracle
verification/     Phase 0 evidence: every claim, source URL, date, verdict
book/notes/       teaching notes captured while building
DECISIONS.md      append-only decision log
```

## Limitations

Stated up front rather than in a footnote.

- The headline grounding comparison is on RefChartQA-human, **n = 500**. Differences smaller
  than a few points are not distinguishable from noise.
- Plan supervision is **predominantly synthetic by design** — only ~5.7% of real ChartQA
  questions yield a unique typed plan under the uniqueness rule. Whether that transfers to real
  charts is a hypothesis under test, not an assumption.
- Some RefChartQA questions have **several equally valid grounding regions**; a correct
  alternative box scores zero.
- ChartQA gold tables are **corrupt in 18.8% of human-sourced charts**.
- Asking for structured output has a real accuracy cost — published measurements on comparable
  models put it at 4.0 to 8.9 points, and this project asks for more than either. It is reported
  as a named result, not absorbed into the headline.
- Training runs **once** per configuration; evaluation runs across three seeds. Single-run
  results are labelled as single-run.

## Licences

Apache-2.0 for this code. The datasets are not redistributed: ChartQA is GPL-3.0, RefChartQA is
AGPL-3.0, ChartQAPro is MIT. Their obligations are reviewed in the Phase 10.4 publication
checklist before anything is made public.
