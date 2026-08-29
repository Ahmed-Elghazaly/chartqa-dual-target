# Chapter 7 — What a score actually computes

**Files:** `eval/metrics.py` (480), `eval/runner.py` (206), `eval/stratified.py` (295).

This is slide 6. Nearly a thousand lines to compute three numbers, and every line of it is
there because scoring is where results quietly become wrong.

---

## 7.1 Relaxed accuracy — for answers

Gold `74`, model says `74.0`. Correct? Exact string matching says no. So the official metric
allows 5% relative error on numbers, and exact match otherwise.

```python
def relaxed_correctness(target, prediction, max_relative_change=0.05):
    prediction_float = to_float(prediction)
    target_float = to_float(target)
    if prediction_float is not None and target_float:
        return abs(prediction_float - target_float) / abs(target_float) <= max_relative_change
    return str(prediction).lower() == str(target).lower()
```

Five lines. The docstring above them is thirteen, and it is the more important half:

> Two behaviours look like bugs and are load-bearing, because published numbers were
> produced with them

### ⚠️ The zero quirk

```python
    if prediction_float is not None and target_float:
```

Read that guard carefully. The prediction is checked against `is not None`. **The target is
not.** It is used as a plain truthiness test — and in Python `0.0` is falsy.

So when the gold answer is zero, the condition is false, and the code falls to string
comparison. `"0"` vs `"0"` is correct; **`"0"` vs `"0.0"` is not.**

That is a bug. We reproduce it exactly, because every published ChartQA number was produced
with it. Changing it would make our numbers incomparable with the literature while looking
more correct.

📘 **The general principle, and it is not obvious:** when you re-implement a metric, you are
not implementing the *definition*, you are implementing the *program that produced the
published numbers*. Those differ, and where they do, fidelity beats correctness.

The plan's own Appendix D wrote the principled version (`if t == 0:` explicitly). It gives
a **different answer**. So it is kept as a separate function, used for nothing:

```python
def relaxed_correctness_appendix_d(...):
    """Appendix D's variant, kept **only** so the divergence stays measurable.
    Never used for a reported number."""
```

Measured on 423 borderline cases, the two disagree on **61** — *"all of them in our
favour"*. Which is precisely why using the flattering one would have been indefensible.

### No whitespace stripping

> `" Yes "` does not match `"Yes"`. Normalising a prediction is the pipeline's job, not the
> metric's

A separate `normalise_prediction` runs before scoring. Keeping cleanup out of the metric
means the metric stays byte-identical to the published one, and the cleanup is visible where
someone can audit it.

---

## 7.2 The one change we did make

```python
    """The one addition is coercing to `str` first, so a non-string input returns None
    instead of raising `AttributeError`. That changes no result the official can produce —
    it would have crashed."""
```

The bar for deviating: **the behaviour being changed is one the original cannot exhibit**.
The official code would crash; crashing is not a score, so no published number depends on
it. Stated explicitly rather than done quietly.

---

## 7.3 AP@0.5 — for boxes

📘 **IoU** — *intersection over union*. Two boxes: divide the area they share by the area
they jointly cover. Identical → 1.0. Non-overlapping → 0. **@0.5** means a predicted box
counts as a hit if IoU with a true box is ≥ 0.5.

📘 **AP** — *average precision*. For each prediction in confidence order, track **precision**
(what fraction of predictions so far are correct) and **recall** (what fraction of true
boxes found so far). That traces a curve; AP is the area under it. One number balancing
"found everything" against "did not invent things".

```python
def average_precision_coco(predictions, ground_truths, iou_threshold=0.5) -> float:
    """AP@IoU computed the way the **official** evaluator computes it.

    `PLAN.md` 4.2: where our metric and the official one disagree, the official one wins.
    They did disagree, and this function is the fix."""
```

Two differences from the plan's version, both measured.

**101-point interpolation.** COCO samples the precision curve at 101 evenly spaced recall
values rather than integrating over every point. On a test case: all-point gave **0.6787**,
official **0.6815**.

**⚠️ Deterministic ordering under ties — the serious one.**

> The official evaluator assigns every prediction a score of 1.0, so the ranking is entirely
> decided by sort stability. Appendix D's implementation sorted the caller's list, which made
> the result depend on whether predictions arrived grouped by image or interleaved — **1.0
> versus 0.6787 for the *same* predictions**.

AP requires ranking predictions by confidence. Our model emits no confidence, so the official
evaluator gives everything 1.0 — and then *every* prediction is tied. A stable sort preserves
input order, so the score depended on the order the caller happened to iterate. Same
predictions, same model: 1.0 or 0.68.

The fix: group detections by image in first-seen order, then sort stably — matching
`pycocotools`. Input order can no longer change the score.

**And the finding that shaped the whole output design:**

> one spurious box per image took AP from 1.0 to 0.68

A perfect system that also emits one extra box per image loses a third of its score. That is
the measurement behind "emit few boxes, best first" (`DECISIONS.md` 0014) — and behind
Chapter 4's *"`maxItems: 8` is a hazard, not an allowance"*.

---

## 7.4 Confidence intervals, and one that cannot be computed

```python
def bootstrap_ci(per_item_scores, n_resamples=10_000, alpha=0.05, seed=0):
    """Resample per-item scores with replacement; report the percentile interval."""
```

📘 **Bootstrapping.** You measured 50% on 200 questions. How sure are you? Draw 200 scores
*from your own 200, with replacement*, and recompute. Do it 10,000 times. The middle 95% of
those results is your confidence interval. It needs no assumption about the distribution —
only that your sample is representative.

The docstring's warning matters more than the method:

> Requires *per-item* scores. A metric that only exists at dataset level — **AP is one**,
> because it depends on the ranking across items — cannot be bootstrapped this way

Accuracy is a mean of per-item 0/1 scores, so resampling those is valid. **AP is not a
mean.** It depends on how every prediction ranks against every other, so there is no
"per-item AP" to resample. `bootstrap_ci_of` takes a *callable* and recomputes AP from
scratch on each resample — far slower, and correct.

Getting this wrong produces confidence intervals that look fine and mean nothing.

---

## 7.5 Two implementations, and why

```python
"""`PLAN.md` 4.2: where our metric and the official one disagree, the official one wins."""
```

**Reported numbers come from the vendored official code**, whose SHA-256 is recorded.

**Ours exists for what the official cannot do:** confidence intervals, and breakdowns by box
size, chart type and question kind (`stratified.py`).

Validated against each other on **11,690 real predictions**:

| | agreement |
|---|---|
| AP@0.5 human / machine / PoT | 0.000 / **0.068** / 0.036 percentage points |
| relaxed accuracy, 423 borderline cases | **0 disagreements** |
| P@F1 | **0 disagreements** |

⚠️ One subtlety in `stratified.py` worth knowing, because the obvious implementation is
wrong and *visibly* so. To score AP for small boxes only, you might restrict the ground
truths to small ones and score all predictions against them. With **perfect** predictions
that reported 78% and 94% for two buckets while the overall score was 100% — because a
prediction correctly matching a *large* target became a false positive in the small bucket.
The fix follows COCO's area-range semantics: keep the targets in the bucket, keep the
predictions that matched them, and keep an unmatched prediction only if its own area falls in
the bucket.

📘 **How that bug was caught: feed the metric a perfect prediction set and check it returns
1.0.** Any decomposition of a perfect system must also be perfect. A metric that scores 78%
on flawless input is broken, and you do not need real predictions to see it.

---

## 7.6 What to take from this chapter

1. **Re-implementing a metric means reproducing the program, not the definition.** The zero
   quirk is a bug we keep on purpose, because published numbers contain it.
2. **The principled variant gives different answers** and is kept, unused, only so the
   divergence is measurable — 61 of 423 cases, all favouring us, which is exactly why we
   don't use it.
3. **Deviations are allowed only where the original could not produce a result at all** —
   `str()` coercion replaces a crash, not a score.
4. **AP is decided by ranking, and everything is tied at 1.0**, so input order silently moved
   the score between 1.0 and 0.68 until ordering was pinned to pycocotools' behaviour.
5. **One spurious box per image costs a third of AP.** That single measurement drives the
   whole "emit few boxes" design.
6. **AP cannot be bootstrapped from per-item scores** because it is not a mean. Doing it
   anyway yields intervals that look right and mean nothing.
7. **Test a metric with perfect input.** Anything less than a perfect score is a bug in the
   metric.

**Next:** Chapter 8 — what we actually ask the model for, and how we repair what comes back.
