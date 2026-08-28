# Notes — building the training loop before there is anything to train

## What this component is, in plain language

Everything between a mixture file and a fine-tuned model: choosing which example comes
next, turning it into the exact string the model should produce, deciding which tokens
carry loss, running the optimiser, saving enough state to survive a crash, and knowing when
to stop.

None of it was written while a GPU was running. All of it was written, measured and tested
first — and that ordering found four defects that would each have cost a ten-hour run and
produced a *plausible* failure rather than an error.

## Why it exists — what breaks without it

The obvious version of this component is fifty lines: load a batch, forward, backward, step,
save every hundred steps. That version trains something. Whether it trains the *right*
thing, on the *right* data, in a way you can *resume*, is a different question, and none of
the answers show up in the loss curve.

That is the difficulty specific to training code. A wrong learning rate announces itself. A
truncated target does not. A model that never learns to stop does not. A resume that
silently restarts the epoch does not. Each of those produces a curve that looks like
learning and a model that is subtly wrong, and the diagnosis arrives hours later as "the
method didn't work".

## What surprised me

**The prompt did not fit.** The zero-shot prompt is 980 tokens by the real tokenizer. Add
247 visual tokens, a target, and the chat template, and a training example is 1,363 to 1,498
tokens against a limit of 1,024. Every single example would have been truncated by 300 to
470 tokens.

I had estimated this earlier at 3.7 characters per token and been sixty tokens optimistic —
which is its own lesson, since I had written the rule "measure the direct thing, not a
proxy" an hour before violating it.

Raising the limit was the obvious fix and the wrong one: 1,536 tokens implies at least 14.9
hours for three thousand steps, against a ten-hour budget, and that is a lower bound because
attention is quadratic. The right fix was a *shorter prompt for training* — 117 tokens —
which costs nothing and is better on its own terms. After fine-tuning the output format
lives in the weights. Spending 980 tokens per example to restate instructions the model has
already learned from thousands of examples is waste as well as overflow.

**The targets did not reproduce their own answers.** The project's central claim is that an
emitted plan recomputes the emitted answer. It had never occurred to me to check whether the
*training targets* satisfied that — and when I did, four separate defects fell out.

The worst: RefChartQA records carry boxes but no per-element values, and my first version
filled them with `null` and a lookup plan. **Every one of 800 sampled targets failed the
round-trip.** A quarter of stage two would have been actively teaching the model to emit
plans that cannot run, on precisely the metric the project exists to move.

The most instructive: evidence was selected as "the first eight boxes". For a twelve-bar
chart whose plan references the tenth, the referenced label was simply absent, and the
executor refused with *"lookup of unknown evidence label: 'Indonesia'"*. One record in 636
produced a usable target. Selecting by *the labels the plan needs* fixed it — and happens to
be the behaviour the grounding metric rewards anyway.

The most embarrassing: two functions built "the same" record differently. One stored the
per-element labels; the copy the pipeline actually used stored only a count. Duplicated
construction logic diverges, and the wrong copy is always the one in the hot path.

**A model is not taught to stop unless you teach it.** The chat template closes an assistant
turn with `<|im_end|>`. The Phase 2 smoke harness concatenated prompt and answer and never
supplied it — which was harmless there, because it only measured loss. In real training it
would mean a model that never emits a stop token and generates until it hits the cap, every
single time. The tokenizer's padding token and its end-of-turn token are different ids here
(151643 and 151645); if they had coincided, masking padding would have masked every stop
token, and nothing would have said so.

**The plan's own stopping rule was unaffordable.** `PLAN.md` says to stop when validation AP
stops improving. AP requires generation, and generation costs: on an affordable slice the
confidence interval is ±8.7 points, which cannot detect "has not improved". Stopping on it
means stopping on noise, and a spurious early stop ends a run that was still improving while
leaving no trace in the curve.

Validation *loss* costs one forward pass, has hundreds of supervised positions per example
instead of one binary outcome, and — because the target contains the boxes — responds
directly to grounding quality. The deviation is from the plan's mechanism, not its intent.

## What I decided, and what I rejected

**Refuse rather than fill in.** A target whose plan cannot be derived honestly is not
emitted. `PLAN.md` says a question without a unique plan is "never given an invented plan";
extending that from operations to *values* is what took RefChartQA's usable fraction from
0% to 52%, and what makes the remaining targets trustworthy.

**Test control flow against a stand-in model.** Which learning rate a stage uses, where
checkpoints land, whether early stopping fires, whether a dead gradient is visible — none of
that needs a vision-language model, and all of it is expensive to discover on a GPU.

**Write the negative control.** The resume test runs the real loop, kills it at step three,
resumes, and asserts step four matches. That test would pass even if the RNG state were
never saved — so there is a second test asserting that a resume *without* it genuinely
diverges and is caught. A resume test without a negative control is a test that a resume
happened, not that it was correct.

## Which concept a reader must understand first

**In training code, the loss curve is not evidence.** It falls for a truncated target, a
half-masked prompt, a model that never stops, a resume that restarts the epoch, and a run
with LoRA on one side only. Every invariant that matters has to be checked somewhere other
than the curve — before the run, in a test, or by an assertion that fires before the first
forward pass.

Second: **cost and precision are the same decision.** "Evaluate AP every 500 steps" sounds
like a monitoring choice and is really a statistical one. Working out what a measurement can
resolve, before running it, changed the design here twice.

## Forward pointers

- `DECISIONS.md` 0064 — the sequence budget, and why raising it was rejected.
- `DECISIONS.md` 0067 — four join defects between the data pipeline and the model.
- `DECISIONS.md` 0069 — why the stopping signal is loss rather than AP.
- Phase 6 runs this code; Phase 7 opens the sealed split and reports.
