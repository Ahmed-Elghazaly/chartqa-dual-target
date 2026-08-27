# Notes — building the ruler before the thing you measure

## What this component is, in plain language

Everything that turns a model's output into a number: how close an answer has to be to
count as right, how much a predicted box has to overlap the real one, and how much of the
resulting number is real rather than luck.

It was built before any training run existed, which is the whole point.

## Why it exists — what breaks without it

If you build evaluation after training, you will shape it — not dishonestly, just by
noticing that one reasonable choice gives 41% and another gives 38%, and finding the first
more convincing. Nobody writes down the moment they did that. Building the ruler first
removes the opportunity.

There is a second reason, less obvious and in this project more important. The number this
work is compared against was produced by someone else's evaluator. "We got 41%" only means
something if our 41% and their 32.83 were measured the same way — and there is no way to
know that except to run their code, on the same inputs, and check.

## What surprised me

**The published target could not be reproduced, and the reason is that the inputs were
never published.** The plan required confirming RefChartQA's 32.83 before spending compute
trying to beat it. Running their evaluator on their released file gave human **28.33** —
but also machine **71.25** against a published 59.28, and PoT **59.66** against 39.32.
Deltas in *both* directions, up to twenty points.

That pattern rules out a scoring bug. A wrong coordinate convention or an off-by-one in a
join makes everything worse, not one subset worse and two much better. What it looks like
is a different model's output — and the project's own README says so: the file is *"an
example file showing the appropriate format"*. The repository has four files. No
checkpoints are published. So the target is a number we can cite but cannot verify.

The consequence is not that the project stops. It is that the *claim* changes: the honest
anchor is the internal before-and-after, same backbone, same evaluator, same sealed split —
which is stronger evidence anyway, because anyone with the repository can re-run it.

**The official metric has a bug, and reproducing the bug is the correct thing to do.** The
canonical ChartQA `relaxed_correctness` guards its zero-division with
`if prediction_float is not None and target_float` — a *truthiness* test. A gold answer of
`"0"` is falsy, so the numeric comparison is skipped entirely and the answer is compared as
a string. That is almost certainly not what the author meant. Fixing it would change what
`"0"` vs `"0.0"` scores, and therefore make our numbers incomparable with every published
ChartQA result. The plan's rule — *the official one wins* — is right, and it is
uncomfortable in exactly the way good rules are.

**A metric named for F1 that does not compute F1.** RefChartQA's "P@F1" helper is
documented as "F_1 score = 1.0" and actually tests whether COCO average precision on the
single image equals 1.0. Those differ: a spurious box emitted *after* the correct ones is
free, because precision is interpolated right-to-left and recall has already reached 1.0.
The same box emitted *first* is fatal. One measured case scored F1 = 0.667 and *correct*
officially.

I only found this because the cross-check disagreed on one scenario out of forty and I
went to look at it instead of widening a tolerance.

**Extra boxes are free on one image and expensive across a dataset.** Following that thread
gave a number worth having: a model that emits one spurious box per image scores 1.00 on
any single image and **0.68** across the set, because the official evaluator ties every
prediction's score at 1.0, so one image's false positive is ranked among another image's
true positives. Two metrics, two different reasons, same instruction: few boxes, best
first.

## What I decided, and what I rejected

**Reproduce the official's quirks; normalise our own output instead.** The official metric
does not strip whitespace, so `" Yes "` fails against `"Yes"`. Rather than loosen a metric
the rest of the field shares, the pipeline strips the prediction before scoring — one
visible line, in our code, where anyone can see it.

**Report the residual instead of fitting it.** After the corrections, our AP matched the
official exactly on 119 of 120 randomised scenarios; the last differs by 0.0019 in whether
one recall threshold is included. I formed a hypothesis about float32 storage, implemented
it, and it made agreement *worse* — 112 of 120. So I reverted it and wrote down what
remains. Matching the last case without understanding it would have been curve-fitting a
test.

**Split strata by area, and filter predictions with them.** The first stratified
implementation restricted targets to a bucket but scored every prediction against them.
With *perfect* predictions it reported 78% and 94% per bucket while the overall figure was
100% — which would have been written up as a finding about small targets. It is now COCO's
area-range semantics, and the test that guards it simply asserts that a perfect prediction
set scores 100% in every stratum.

## Which concept a reader must understand first

**A metric is a convention, not a truth.** There is a better zero guard, a better
tolerance, a better way to handle commas. Choosing any of them makes your number
incomparable with everyone else's, which is a larger loss than the error it fixes. The
place to be more correct is your own pipeline, not the shared ruler.

Second: **agreement has to be measured on real data, not just constructed cases.**
Hand-picked cases test what the author already suspected. Randomised scenarios found the
P@F1 divergence; the 11,690-row real prediction set is what turned "our metrics agree" from
a claim into a measurement.

## Forward pointers

- `DECISIONS.md` 0052 — 32.83, why it does not reproduce, and what the claim becomes.
- `DECISIONS.md` 0053 — the three metric corrections, each in the official's favour.
- `DECISIONS.md` 0054 — area versus axis sub-token definitions, and the bucket-filtering bug.
- Phase 5 builds the zero-shot baseline this ruler was made to measure.
