# Notes — the failure that counts itself and carries on

## What this note is about

Not a component. A *shape of bug* that this project produced five times in two days, in five
unrelated places, and that would have cost roughly fifty GPU hours between them if the last
one had not been caught by a number disagreeing with another number three lines above it.

The shape is this:

```python
try:
    example = build(record)
except SomeError as exc:
    stats.note_refusal(exc)     # counted, and honestly counted
    continue                    # and the run carries on
```

That code is not careless. It is what you write when you have already learned that one bad
record should not kill a ten-hour job, and it is the right instinct. The counter is real,
the reason string is real, and the summary at the end of the run tells the truth.

And yet: **from outside, an `except` that counts a failure and continues is
indistinguishable from there being no failures.**

## Why it exists — what breaks without it

Here is what the five looked like from the outside. All of them.

> The run started. It logged its step count. It finished on schedule. It reported 3,000
> optimizer steps and 24,000 presentations, and both numbers were true.

Here is what they were.

| what happened | data lost |
|---|---:|
| the synthetic reader wrote its element metadata under `meta["evidence"]`; the target builder reads `meta["elements"]` | **100%** of stage 1 |
| a plan meaning *"the mean of everything on the chart"* was handed only the labels it named, so the mean was one item and the difference was zero | **100%** of the compositional level |
| the round-trip check inherited the official evaluator's quirk that `"0" != "0.0"` | 512 records whose correct answer was zero |
| mixture slots filled with records that yield no target at all | **46%** of stage 2 |
| chart images opened from disk when they live inside a zip that is never extracted | **38%** of stage 2 |

Not one of them raised. Not one produced a wrong number anywhere. Each produced a *smaller
training set*, and a smaller training set is not an error condition — it is a Tuesday.

The subtlety worth sitting with: the *loss curve would have looked fine*. Training on a
quarter of your data gives you a perfectly respectable loss curve. It is a curve about a
quarter of your data. Nothing in it says so.

## The reasoning that led to the current design

The first instinct is "add more logging". That was already there. `FeedStats` recorded every
one of these, with reasons, ranked by frequency. The information existed and was correct and
was useless, because it sat in a summary at the end of a run that had already been paid for,
and nobody is *required* to read a summary.

The second instinct is "fix the five bugs". Necessary, and done, but it protects against the
five bugs that happened rather than the sixth.

What actually generalises is a change of category: **make the rate a gate rather than a
statistic.**

```
MixtureFeed.check_refusal_rate()
    fires once 200 records have been offered
    raises FeedRefusedTooMuch below a 90% usable floor
```

Two numbers, both chosen against measurements rather than taste:

* **200 offered** is about eight optimizer steps at effective batch 8 — under two minutes of
  GPU, against the ten hours it previously took to *not* find out. It is also enough that
  the rate is no longer noise.
* **90%** because the measured yield after the fixes is 99.5%, the residual half a percent
  being examples over the sequence limit; while the failures that actually occurred cost
  38%, 46%, 100% and 100%. Nothing real lives between 90% and 62%, so the floor can sit
  anywhere in that gap and 90% is the readable end of it.

And the exception carries the command that reproduces the same refusals on a CPU, because an
error message that tells you what broke is half a message.

## The general lesson

There is a common piece of advice — *fail loudly* — and it is usually taught against silent
`except: pass`. This project's `except` blocks were not silent. They were diligent. They
counted, categorised and reported.

The lesson is narrower and more useful than "fail loudly":

> **A component that can discard its input must publish a rate, and something must be
> willing to stop on that rate.** A count is a fact about the past. A rate with a floor is a
> claim about the present that can be false.

The corollary is about where to spend attention. Four of these five were found not by reading
code but by **asking a quantitative question of the pipeline before running it** — *of the
12,000 records in this mixture, how many actually become training examples?* The answer for
stage 1 was zero. That question costs nothing on a CPU and no amount of code review reliably
substitutes for it.

The fifth was found because a summary line said *"over limit: 78 of 200 (39.0%)"* three
lines below *"max 938 (limit 1,024)"*, and both could not be true. Which is its own small
lesson: **print the intermediate quantities, not only the conclusion.** A conclusion cannot
contradict itself. Two numbers can, and that contradiction is free evidence.

## Where this shows up in the repository

* `src/chartqa_dt/train/feed.py` — `check_refusal_rate`, `FeedRefusedTooMuch`,
  `MIN_USABLE_FRACTION`
* `scripts/measure_target_yield.py` — the pre-flight question, on CPU, with `--tokens` to
  run the real collator over real images
* `DECISIONS.md` 0071, 0072, 0073, 0074 — the five defects and the gate
* `tests/test_feed.py::TestRefusalRateGate` — including that the floor still tolerates the
  measured 0.5% over-length loss, so the gate cannot be quietly loosened
