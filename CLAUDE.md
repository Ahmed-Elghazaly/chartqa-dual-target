# Working rules for this repository

Read this first, every session. It is short on purpose. The long-form reasoning lives in
`DECISIONS.md`; this is only what you must not get wrong.

---

## One command

```bash
bash scripts/preflight.sh
```

**Never commit without it passing, and never pipe it** — `preflight.sh | tail` reports
`tail`'s exit status, which is how a red commit reached main once already. It runs exactly
what CI runs. If CI fails and preflight passed, that gap is a bug in preflight and fixing it
is more urgent than whatever you were doing: preflight silently diverged from CI once and
main was red for a week across ~30 commits.

---

## The three mistakes that caused almost every bug here

These are not hypotheticals. Each is taken from a defect in `DECISIONS.md`.

### 1. Changing one end of a pipeline and not the others

A fallback was wired into the feed but not the mixture builder, so it was dead code that
passed 36 tests (0116). Provenance was tagged at three sources but not the legacy-cache
path, leaving 470 elements untagged (0126). A decode fix was wired into two functions but
given no CLI flag, so it could not be run at all (0114). A cache cap of 4,000 made a
scaling ladder at 10,000 and 25,000 *impossible*, not merely deferred (0115).

**Before editing any shared function, constant or field, list every caller and decide about
each one explicitly.**

```bash
grep -rn "the_symbol" --include="*.py" src/ scripts/ tests/ | grep -v __pycache__
```

### 2. Validating on aggregate counts instead of instances

Grounding-only targets were extended to ChartQA and the composition report showed ChartQA
going from 5 records to 4,944. That looked like a large win. Reading one actual target
showed *"Which year has the most crime?"* → answer 2014 → evidence: **all six years**. It
would have added 4,939 records that teach "point at everything" (0116).

**After any change to data generation or target building, print several real instances and
read them.** A total that moves in the direction you hoped is the least trustworthy
evidence available.

### 3. Starting long jobs before the changes settled

The synthetic corpus regeneration takes ~90 minutes and was started and killed four times,
each time because another generator bug surfaced after it began.

**Long jobs go last.** Finish the edits, run preflight, then start the job.

---

## Non-negotiable rules

1. **Never commit dataset content.** No chart images, questions, answers, values or tables
   in git — ChartQA is GPL-3.0, RefChartQA AGPL-3.0. Derived data goes to
   `~/.cache/chartqa_dt/data/`. The guardrail in preflight scans *all* history.
2. **Never rewrite git history.**
3. **Never weaken a gate to raise yield.** Supervision correctness beats dataset size,
   always. If a change increases yield, say explicitly why it is a correctness fix and not
   a relaxation — 0129 is the worked example of that distinction.
4. **Discard, never repair.** A record that fails a gate is dropped, not patched into
   passing.
5. **Check before changing a schema** (`ChartRecord`, the target format, stored artifacts),
   and handle migration for anything already cached.
6. **If an experiment cannot run here, do not invent a result.** Document the blocker and
   the exact command in `STATUS.md`.
7. **Measure before deciding.** Every claim in `DECISIONS.md` carries a number. A change
   with no measurement behind it is a guess, and should say so.

---

## How Ahmed wants this done

| | |
|---|---|
| **Work in long sessions.** | Do not stop every few minutes to report. When waiting on a long run, work on something that does not depend on it. |
| **Be right the first time.** | Read the authoritative source — API signature, library source, model config, the plan — *before* writing code against it. Never from memory. |
| **Verify, don't assume.** | Research anything unclear rather than building on general knowledge. |
| **Report briefly and plainly.** | What was done, whether anything is wrong, whether anything is needed. Short. |
| **Test heavily.** | Prove a technique against ground truth known by construction before depending on it. |
| **Delete what is dead.** | Superseded files are removed, not left to rot. |
| **The plan may be wrong.** | `PLAN.md` and `IDEA.md` contain errors. When you find one, say so with evidence and propose the change rather than silently working around it. |

### Reporting — the rule most often broken

Every report says, in plain language:

1. **Is this good or bad for us?** Label it. *"AP is 0.68"* means nothing until it is
   *"worse than we hoped, and here is what it costs"*. **Never leave the judgement
   implicit** — this is the single most repeated complaint about these reports.
2. **What was done**, in the fewest words that stay accurate.
3. **What was decided**, and why it went that way.
4. **What is needed from Ahmed** — or explicitly *nothing*.
5. **Anything else worth knowing** — surprises, risks, wasted effort, cost.

The failure to avoid is a technically complete report that leaves the reader unable to tell
whether the project is going well.

### Work ordering

Compute-heavy work that does **not** improve the final result waits until the end — three
training seeds and the scaling ladder *measure* the result rather than improving it. Work
that improves it runs first.

---

## Standing facts about the environment

Each of these cost hours to discover once.

* **Kaggle** — account `nanonanite`. The token is a `KGAT_` **bearer** token in
  `~/.kaggle/access_token`, **not** the legacy `kaggle.json` username/key pair. In the
  wrong file it fails every authenticated endpoint.
* **Quota** — ~30 GPU-hours per week per account, three accounts. Read it live with
  `python scripts/gpu_budget.py`; never keep a parallel tally.
* **GPU** — request `machine_shape: "NvidiaTeslaT4"` explicitly, or Kaggle hands out a
  P100 (`sm_60`) that its own PyTorch build cannot use (0019, 0020).
* **Hugging Face** — user `NanoPhotonic`, write token in `.env`, artifacts repo
  `NanoPhotonic/chartqa-dt-artifacts`.
* **GitHub** — `gh` CLI, account `Ahmed-Elghazaly`, repo `chartqa-dual-target`.
* **TLS** — the venv Python has no CA store. Import `chartqa_dt.net` in any script that
  makes network calls, or `urllib` raises what looks like a rejected credential.
* **Disk** — Ahmed has said space is not a constraint and to download what is needed.

---

## Two shell habits that have each caused a wrong report

1. **Never pipe a check whose exit status you then rely on.**
   `bash scripts/preflight.sh | tail -3 && git commit …` commits even when preflight
   fails, because a pipeline's status is the *last* command's. That is how a red commit
   reached `main`. Run the check bare, read it, then commit as a separate step.
2. **`git status` is silent about ignored files.** A source file excluded by `.gitignore`
   shows nothing — not untracked, not modified — so a whole package can be missing from
   the repo while every local run passes (0050, 0051). `tests/test_repo_completeness.py`
   runs `git check-ignore` over every source file, and it is step 1 of preflight.

---

## Where things are

| you want | file |
|---|---|
| why any decision was made | `DECISIONS.md` — append-only, numbered, never edit an old entry except to annotate |
| what was audited and found | `AUDIT.md`, `AUDIT_COVERAGE.md`, `FINDINGS.md`, `VERDICT.md`, `PROMPT_CHECKLIST.md` |
| where the project stands | `STATUS.md` |
| what cannot run here, and the command | `STATUS.md` |
| how the code is organised | `ARCHITECTURE.md` |
| what we committed to before seeing results | `PREREGISTRATION.md` — a research artifact; do not edit after the fact |

## Writing a decision

Every entry in `DECISIONS.md` needs `**Context.**`, `**Decision.**` and `**Consequences.**`
— a test enforces it. Number sequentially. Cite the measurement, not the intuition. When a
later finding contradicts an earlier decision, **write a new decision that says so** rather
than editing the old one.

## Tests

Roughly one line of test per line of source, and that ratio has paid for itself. When you
fix a bug, add the test that would have caught it, and then **check the test actually
fails** against the old behaviour — a test written after the fix that passes either way is
worse than none, because it looks like coverage.
