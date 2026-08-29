# Chapter 9 — Mixtures, and sealing the test split

**Files:** `data/mixture.py` (190), `splits.py` (241).

The last chapter. Assembling the training set, and the machinery that stops us cheating —
including by accident.

---

## 9.1 Two mixtures, and why one is not shuffled

```python
"""* **Stage 1 — grounding only.** Ordered `L1 -> L4` synthetic, then audited real boxes.
  Cap 12,000. The order is the point: difficulty is a curriculum, so stage 1 is *not*
  shuffled.
* **Stage 2 — joint box + plan + answer.** Shuffled, including ~2,000 exact synthetic
  replay examples. Cap 12,000."""
```

**Stage 1 teaches pointing.** Easy charts first, then harder, then real ones — a
**curriculum**. Shuffling would destroy the thing that makes it a curriculum, so the code
does not shuffle it, and the comment says why in case someone "fixes" it later.

**Stage 2 teaches everything jointly** and *is* shuffled, because there is no ordering
argument and shuffling reduces the chance that a run of similar examples pushes the model
around.

📘 **"Replay"** — 2,000 synthetic examples mixed into stage 2. Neural networks suffer
*catastrophic forgetting*: train on B after A and performance on A collapses. Keeping some
of stage 1's data in stage 2 is a standard defence. Here they are the examples whose plans
and boxes are exact by construction, so they are the ones worth not forgetting.

---

## 9.2 A mixture holds ids, not data

The mixture file records **record ids only** — no images, no questions. Rule 7 forbids
committing dataset content (ChartQA is GPL-3.0, RefChartQA AGPL-3.0), and a mixture must
stay reproducible without carrying any.

So training *rehydrates* every record from the sources using the ids.

⚠️ **That has a sharp edge.** The sources are sampled with caps. Build a mixture from 30,000
ChartQA rows while training rehydrates only 8,000, and the ids at the tail resolve to
nothing. The loader refuses on the mismatch — loudly, but **on the GPU, an hour into a run**.
The draw sizes are now shared constants both sides import, with a test forbidding a
hand-written number at either call site. `DECISIONS.md` 0072.

---

## 9.3 Refusing to filter silently

```python
"""Both are checked for validation/test records before they are written — `PLAN.md` 3.7
requires zero, and `build_mixture` raises rather than filtering silently. A mixture that
quietly dropped a leaked record would hide the fact that something upstream produced one."""
```

This is the same principle as Chapter 2's cross-split rule, and it is worth stating as a
general one.

If a leaked record reaches mixture construction, there are two responses:

- **filter it out** — the mixture is now clean, and nobody ever learns that a loader is
  producing leaked records;
- **raise** — the mixture is not written, and someone fixes the loader.

The first is more convenient and strictly worse. A pipeline that repairs its inputs cannot
tell you its inputs are broken.

---

## 9.4 Making rule 1 mechanical

```python
"""Rule 1 says: *"Never train on, tune on, or even inspect ChartQA test, RefChartQA
test, or ChartQAPro."* That sentence appears in the README, the pre-flight
checklist and the non-negotiable list — and it was still violated (`DECISIONS.md`
0031), because **a sentence is not a control**."""
```

⚠️ **The rule was written in three places and broken anyway.** That is the honest bit, and
the reason this module exists.

> Every invariant in this project that actually holds is enforced by an assertion: LoRA
> coverage, the quantisation skip, code freshness, device pinning, documentation
> consistency. Sealed-split access had only prose. This module gives it the same treatment.

📘 The general claim — *rules that hold are enforced; rules that are merely written are
eventually broken* — is worth taking out of this project. If something must never happen, an
assertion is the only version of it that survives a tired afternoon.

### The gate

> deliberately hard to pass by accident and easy to pass on purpose, once, at the point the
> plan intends

Three conditions, all required:

1. **`PREREGISTRATION.md` exists.**
2. **It is committed to git.** A file on disk can be written in the moment; a commit is a
   deliberate, timestamped, auditable act.
3. **It is clean in the working tree** — no uncommitted edits. Otherwise the committed
   pre-registration is not the one on disk, and *"frozen before the test split was opened"*
   would not be a true statement.

The default is **refusal**. Passing a flag is not enough — the flag *and* the gate must both
hold. A flag can be added by habit; a committed pre-registration cannot.

Every opening is logged with a reason, so the audit trail is produced rather than remembered.

### And one more condition

A generated pre-registration full of `TBD` placeholders would satisfy all three. So the gate
also rejects placeholder text:

```python
PREREGISTRATION_PLACEHOLDERS = ("TBD", "has not run yet", "has not run")
```

⚠️ **This nearly failed anyway.** The pre-registration reads its baseline numbers from result
files. A stale **12-question** smoke run was sitting at exactly the path it reads — 91.67%
accuracy, interval [75.0, 100.0]. Had it been picked up, the table would have filled with
real-looking numbers, no placeholders would remain, and the gate would have opened the test
split **on the strength of twelve questions**. The reader now takes the pre-registered slice
size and refuses anything smaller, rendering `TBD (n=12; needs n≥1,920)`.

📘 A placeholder should say **what is missing**. A bare `TBD` invites deletion;
`needs n≥1,920` does not.

---

## 9.5 Why any of this matters

If you look at test results and then change *anything* — a prompt, a threshold, a mixture —
your final test number stops being an estimate of performance on unseen data. It becomes an
estimate of how well you tuned against that particular test set. And it will still look like
the first thing to everyone reading it.

That failure is undetectable from the outside. Nobody can audit it from your results. The
only defence is machinery that made it impossible in advance, plus a document written before
the results existed saying what would count as success.

That is why a chapter about a training-set builder ends up being about honesty.

---

## 9.6 What to take from this chapter

1. **Stage 1 is ordered on purpose** — a curriculum, easy to hard — and stage 2 is shuffled
   with ~2,000 replay examples against catastrophic forgetting.
2. **Mixtures store ids, not data**, so nothing licensed is committed — and the source draw
   sizes must match on both sides or the ids fail to resolve on the GPU.
3. **A leaked record raises rather than being filtered.** A pipeline that repairs its inputs
   cannot report that its inputs are broken.
4. **Rule 1 was written in three documents and broken anyway.** A sentence is not a control.
5. **The seal needs a committed, clean, placeholder-free pre-registration**, and refuses by
   default. A flag alone cannot open it.
6. **A stale 12-question result nearly opened the seal.** The guard now requires the
   pre-registered sample size, and says what is missing rather than just `TBD`.

---

## Where this leaves you

Nine chapters, 4,869 lines. If a single thread runs through all of it, it is this:

> **Almost every defect in this project was silent.** The wrong visual-token factor, the
> evaluator discarding edge boxes, the question-only dedup key, the misspelled metadata
> field, the doubled braces, the fold-over-evidence interaction, the zero quirk, the AP
> ordering, the 12-question smoke result. Not one raised an error. Every one produced a
> plausible number.

Which is why the code is shaped the way it is: refuse rather than default, count every
discard, test with perfect input, measure before deciding, and write down the reason next to
the number.

**For the presentation**, chapters 3, 5, 6 and 7 are your slides 5, 8, 9 and 6. Chapter 1 is
slide 7. The rest is what to say when someone asks a follow-up.
