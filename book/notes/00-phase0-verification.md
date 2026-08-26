# Notes — Phase 0: re-verification

## What this component is, in plain language

Before writing any code, we went back to every original source that the project's design rests on — the
dataset pages, the model pages, the papers, the evaluator source code — and checked that each number still
says what the design document claims it says. Ten specific claims, each with a URL, a date, and a verdict.

No neural networks are involved. It is closer to an accountant's reconciliation than to machine learning.

## Why it exists — what breaks without it

A research project is a tower of numbers stacked on each other. "We will beat 32.83" is only a sensible goal
if 32.83 is real, if it means what you think it means, and if it was produced under conditions you can match.
If any of those is wrong, everything built on top is wasted — and you find out at the end, when there is no
time left.

Datasets get re-uploaded with different row counts. Papers get v2 revisions with corrected tables. Model cards
change licences. Evaluators get patched, which silently changes what your score means. None of these
announce themselves.

The asymmetry is the whole argument: verification costs a few hours, and the failure it prevents costs the
entire project.

## What surprised me

**The biggest finding was not that a number had changed — none had — but that a number had been
under-specified.** `IDEA.md` says the published target is "32.83 AP@0.5". That is true. What it does not say
is that RefChartQA's evaluator never produces a single number: it splits the test set into human-written,
machine-generated and program-of-thought questions and scores all three separately. 32.83 is the *human*
subset. The same model scores 59.28 on machine and 39.32 on PoT.

Comparing a whole-test-set average against 32.83 would have looked like a huge win and would have been
meaningless. This is the exact failure mode `IDEA.md` §13 lists as "claiming a published win that is not
comparable" — and it would have been completely invisible from the dataset card. It only surfaced by reading
the evaluator's source code, function by function.

The second surprise came from the same reading. The evaluator accepts a predicted box only when all four
coordinates are between 0 and 999. Qwen3-VL emits coordinates between 0 and **1000**. A box touching the right
or bottom edge of a chart — which in a bar chart is extremely common — produces a coordinate of 1000, and the
evaluator throws the entire box away without raising an error, printing a warning, or leaving any trace. The
score just comes out lower. You would never find that by looking at your outputs, because your outputs would
look perfect.

## What I decided, and what I rejected

**Decided:** vendor the official evaluators verbatim and use them as the scorers of record, keeping the
project's own cleaner implementations as separately-named diagnostics.

**Rejected:** using our own implementation as primary. Ours is arguably better — for instance, the official
`relaxed_accuracy` has a subtle bug where a gold answer of `"0"` is *falsy* in Python, so the numeric-comparison
branch is skipped and it falls back to string matching. Ours handles that explicitly.

But "better" is not the goal. **Comparable** is the goal. A number is only a number relative to the procedure
that produced it, and the published 32.83 was produced by the official evaluator, bug and all. Scoring
ourselves with a fixed evaluator and comparing to their unfixed one would be measuring two different things
and calling the difference progress. The fix is to report both and count the disagreements — never to average
them, and never to quietly swap one for the other.

## Which concept a reader must understand first

**A benchmark is a procedure, not a dataset.** Beginners think a benchmark is a pile of test data with correct
answers, and that "scoring 40" is a property of your model. It is not. The score is a property of
(model + prompt + decoding settings + output format + parsing code + metric implementation + data version).
Change any one and the number moves, often by several points, usually without any error being raised.

This is why the project pins evaluator commits, freezes prompts before touching test data, and writes down the
decoding parameters. It is also why "we got 35, they got 32.83, we win" is a claim that has to be *earned* by
showing the two numbers came out of the same procedure — not merely asserted because 35 > 32.83.

The related idea, which the coordinate trap illustrates painfully: **silent failure is the default in this
field.** Nothing crashes when a box is dropped, when an evaluator flag is computed but never passed, or when
LoRA fails to attach to the vision tower. The code runs, produces a plausible number, and is wrong. Most of
the defensive machinery in this project — assertions on trainable parameter counts, round-trip coordinate
tests, counting invalid records rather than substituting fallbacks — exists because of this one property.

## Forward pointers

- The `always_use_exact_match` defect (ChartQAPro's evaluator computes a flag and then never passes it to the
  function that takes it) is the same species of bug as the coordinate clamp, found in someone else's code.
  It is a good worked example for the evaluation chapter.
- The `bbox` vs `bbox_2d` question — whether to name a field the way the design document says or the way the
  model was pretrained to emit it — is a nice small illustration of a general principle in fine-tuning: you
  are cheaper to train when you agree with the pretraining distribution. Deferred to Phase 5.1 with a
  measurement rather than settled by argument.
