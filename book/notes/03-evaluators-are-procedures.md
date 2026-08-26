# Notes — characterising an evaluator by running it

## What this component is, in plain language

Before writing any of our own scoring code, we took the official scoring script that the paper's
authors released, fed it predictions whose correct answers we already knew, and wrote down what it
did. Not read it — *ran* it, on cases we constructed specifically to expose behaviour.

## Why it exists — what breaks without it

Because we had already been burned by reading.

We *read* the evaluator carefully and concluded, in a written decision, that "because all prediction
scores are constant, emitting extra speculative boxes can only reduce precision." That conclusion
happened to be right. The reasoning was wrong, and the reasoning is what we would have built on.

When we finally ran it, the single-image result said the opposite: appending three false positives
to a correct prediction left the score at a perfect 1.0000. Free. If we had tested only that — the
obvious test — we would have concluded that extra boxes cost nothing and happily emitted eight of
them.

Then we ran the same thing over twenty images:

| | one image | twenty images |
|---|---:|---:|
| correct box only | 1.0000 | 1.0000 |
| correct box + 3 extras | 1.0000 | **0.3243** |

The score collapses. Not because anything changed about the predictions, but because
`compute_AP_50` calls `update()` per image and `compute()` **once**, pooling every prediction in the
dataset into a single precision–recall curve. On one image there is nothing for the extras to be
ranked below. Across twenty, sixty false positives interleave with twenty true ones, and precision
falls everywhere.

## What surprised me

**Three things, and the first two were only visible from execution.**

**1. The metric has no notion of confidence, so emission order *is* the ranking.** Every predicted
box is assigned `score = 1.0`. That sounds harmless. It means the precision–recall curve is built in
whatever order you happened to emit boxes, and AP works out to exactly `1 / (rank of the first
correct box)`:

```
[correct]                -> 1.000
[bad, correct]           -> 0.500
[bad, bad, correct]      -> 0.333
[bad, bad, bad, correct] -> 0.250
```

One wrong guess placed first halves your score for that image. The same set of boxes in a different
order scores twice as well.

**2. The two headline metrics disagree about the same behaviour.** `P@F1` is described in our plan
as requiring "the full predicted grounding set to be correct." Measured, that is not what it does:
trailing false positives leave it at 1.0000, and only a false positive placed *before* a true one
breaks it. So extras are free under P@F1 and catastrophic under dataset AP. A report that presented
those two numbers as measuring the same thing would be wrong in a way no reader could detect.

**3. The bug we were most worried about was inherited, not introduced.** The relaxed-accuracy
function has a subtle flaw: the guard reads `if prediction_float is not None and target_float`, and
that second clause is a *truthiness* test. A gold answer of `"0"` parses to `0.0`, which is falsy in
Python, so the numeric comparison is skipped entirely and the code silently falls back to string
matching. Answer `"0.0"` against gold `"0"` is marked wrong.

We assumed this was RefChartQA's mistake. It is not. It comes from pix2struct, and we confirmed the
two implementations agree on all 216 target/prediction pairs we tried. Which means **every published
ChartQA number was computed with this behaviour in it** — including the 79.1 our own baseline is
measured against.

## What I decided, and what I rejected

**Decided:** adopt the quirk. **Rejected:** fixing it.

This feels backwards and is not. The goal is not to compute the most defensible possible metric; it
is to compute *the same* metric that the numbers we are comparing against were computed with. A
fixed evaluator produces a number on a different scale, and "we got 35, they got 32.83" then compares
two things that were never the same measurement. The fix is to report both and count the
disagreements — never to average them, and never to quietly swap one for the other.

**Decided:** emit few boxes, best-first, and filter the evidence list before scoring.
**Rejected:** letting the model fill the schema's eight evidence slots. Under this evaluator, a model
that helpfully lists eight plausible regions scores close to zero on the headline metric while
looking thorough. The schema's generous `maxItems: 8` is a hazard, not an allowance.

## Which concept a reader must understand first

**A benchmark is a procedure, not a dataset.**

Beginners picture a benchmark as a pile of test questions with an answer key, and a score as a
property of the model. It is not. A score is a property of the whole pipeline:

> model + prompt + decoding settings + output format + parsing code + metric implementation + data
> version

Change any one and the number moves, often by several points, usually with nothing raised and
nothing logged. Here, three separate elements of that list each moved the grounding score by more
than the entire improvement this project hopes to demonstrate:

- **parsing code** — a coordinate of exactly 1000 makes the parser discard the whole box, silently;
- **output format** — emission order changes AP by a factor of two;
- **metric implementation** — a truthiness test flips every zero-valued answer.

This is why the project pins evaluator commits by hash, freezes prompts before touching test data,
and writes down decoding parameters. It is also why "we beat 32.83" is a claim that has to be
*earned* by showing both numbers came out of the same procedure — not asserted because 35 > 32.83.

The corollary, which took two mistakes to learn properly: **reading code tells you what it says;
running it tells you what it does.** For anything whose output becomes a number in your report, run
it, on cases where you already know the answer, at the scale you will actually use it. The
one-image-versus-twenty-images divergence here would not have been found any other way.

## Forward pointers

- The "emit few, best-first" rule becomes a constraint on prompt design and on the JSON schema, and
  the filter that implements it is a parameter frozen before test evaluation.
- The same execute-don't-read discipline found a bug in our own plan's executor: a bare string
  argument meant "evidence label" in some operations and "numeric literal" in others, so
  `mean(["2019","2018"])` returned 2018.5 instead of the mean of the values.
- The zero-handling quirk propagates into the answer normaliser, which must emit bare `"0"` — and
  18.8% of human-sourced ChartQA tables contain a zero, so this is not an edge case.
