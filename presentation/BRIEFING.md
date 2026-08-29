# Presenter's briefing — everything through Phase 5

Written for **you**, not for your audience. The slides are what people see; this is what
you need to know so that you can present them, and answer the question after.

Every number here is measured and lives in `verification/measured_facts.json`. Where a
number could be misread, I say which population it was measured on.

**How to read this.** Parts I–II are the idea and the data. Part III is measurement, which
is most of what week 1 actually was. Part IV is the discipline that makes the later results
believable. Part V is the baseline. Part VI is the questions you will get.

---

# Part I — The problem and the idea

## 1. What the task is

**ChartQA** is a dataset of chart images with questions about them.

> *[a bar chart of five regions]* — "What is the difference between West and South?" → `74`

A model reads the image and the question, and produces the answer. That is the whole task
as normally defined, and it is the task the published numbers measure.

There are two flavours of question in ChartQA, and they matter later:

- **human** questions, written by people. Varied, messy, often needing reasoning.
- **machine** questions, generated automatically from the chart's underlying table.
  More formulaic.

They are reported separately because models do noticeably better on machine questions.

## 2. What a vision-language model is, in plain terms

A **language model** predicts the next word (strictly, the next *token*) given everything
before it. Train it on enough text and "predict the next token" turns into something that
can answer questions, because answering is just continuing the text plausibly.

A **vision-language model (VLM)** does the same thing, except the "text so far" can contain
an image. The trick is that the image is converted into things that look, to the model,
exactly like words.

Concretely, in our model:

1. The image is resized to a standard size (we use **512 pixels** on the long side).
2. It is cut into small square patches.
3. Each patch is passed through a vision encoder that turns it into a vector — the same
   *kind* of object the model uses to represent a word.
4. Those vectors are placed into the sequence in front of the question.

So the model literally sees something like:

```
[patch] [patch] [patch] ... [patch]  What  is  the  difference  between  West  and  South ?
```

and then predicts what comes next. **Everything the model knows about the picture arrives
through those patch vectors.** That single fact explains most of Part III.

**The model we use** is `Qwen3-VL-2B-Instruct`: an open-weights VLM with about **1.45
billion parameters**. "2B" is the marketing name; "Instruct" means it has already been
tuned to follow instructions rather than just continue text.

## 3. What "grounding" means, and why it matters

A normal chart model gives you only the answer. If it says `74`, you cannot tell whether it:

- read the West and South bars correctly,
- read the wrong bars and happened to get 74 anyway,
- or ignored the chart and guessed from the phrasing of the question.

**A correct answer and a lucky guess look identical from the outside.** That is the problem.

**Grounding** means making the model also point at *where* it got its information —
literally, giving pixel coordinates of a box around each thing it used.

**RefChartQA** is the dataset that adds this: the same kind of chart questions, but each
one also comes with the boxes a correct answer should point at. It is what lets us *score*
whether the model pointed at the right things.

## 4. Our design: a record, and a calculator that checks it

We ask the model to produce a single structured record instead of a bare answer:

```json
{
  "answerable": true,
  "evidence": [
    {"label": "West",  "value": 144, "unit": null, "bbox": [412, 180, 468, 640]},
    {"label": "South", "value": 70,  "unit": null, "bbox": [330, 240, 386, 640]}
  ],
  "plan": {"op": "difference", "args": ["West", "South"]},
  "model_answer": "74"
}
```

Four parts:

| field | what it is |
|---|---|
| `answerable` | whether the question can be answered from this chart at all |
| `evidence` | what it read: a label, a value, a unit, and a **box** saying where |
| `plan` | the arithmetic, as a small tree of operations over the evidence labels |
| `model_answer` | what it says the answer is |

Then — and this is the part that makes the design worth anything — **a small ordinary
Python program re-runs the plan over the evidence** and checks it produces `model_answer`.

`difference(West, South)` → `144 − 70` → `74` → matches. ✓

**Why bother.** The executor is not a model. It is a few hundred lines of deterministic
code that cannot hallucinate. So when the executor and the model disagree, we know
something is wrong **without needing a human or a gold answer**. That gives us a signal
that works at test time, on questions nobody has labelled.

**The honest caveat, which you should be ready to say:** this checks *internal
consistency*, not truth. A model that misreads both bars and then correctly subtracts its
own wrong numbers will pass the check. It catches reasoning and formatting failures, not
perception failures. What catches perception failures is the box score, which is why we
need both.

**Terminology.** A `plan` is what the literature calls **program-of-thought**: instead of
reasoning in prose ("chain-of-thought"), the model emits a small program, and something
else runs it. The advantage is that a program can be executed and checked; prose cannot.

---

# Part II — The data

## 5. The two datasets

| | ChartQA | RefChartQA |
|---|---:|---:|
| training | 28,299 | 55,789 |
| validation | 1,920 | 6,223 |
| test | 2,500 | 11,690 |
| has answers | yes | yes |
| marks every element in the chart | **yes** | — |
| marks what each *question* needs | no | **yes** |

RefChartQA is built *on top of* ChartQA's images. Both mark regions, but they mark
different things, and the distinction matters: **ChartQA labels every element of the chart**
(each bar, each pie slice) as a description of its structure, with no link to any particular
question. **RefChartQA labels, for each question, the regions a correct answer should point
at.** So RefChartQA can be scored directly for grounding; for ChartQA we have to *derive*
per-question evidence by selecting the elements a question's calculation actually uses. That
overlap matters and we measured it: **99.9% of the RefChartQA training images we cached also
appear in ChartQA's training set**. Which means a naive setup could easily train on an
image and then "test" on the same image through the other dataset. We check for that by
image content, not by filename (see §12).

Every archive is pinned by a **SHA-256 fingerprint** recorded before download, so we always
know precisely which version of the data produced a given number. ChartQA is 875 MB;
RefChartQA's parquet files are 2.88 GB.

**Licences matter here and you should know this one.** ChartQA is GPL-3.0 and RefChartQA is
AGPL-3.0. We never commit their images or content to our repository — only IDs, hashes,
scripts and derived statistics. It is also why every figure in your deck is drawn from our
*own* generated charts.

## 6. What auditing the data actually found

We did not trust the dataset descriptions. We read **2,500 ChartQA annotations** directly
and measured what is in them.

**Finding: box coverage is very uneven.**

| chart type | share of ChartQA | have element boxes |
|---|---:|---:|
| vertical bar | 54.6% | 96.8% |
| horizontal bar | 29.3% | 91.5% |
| line | 12.9% | **0.0%** |
| pie | 3.2% | 54.8% |
| | | overall **80.8%** |

Line charts have **no** element box annotations at all. So ChartQA's own annotations cannot
teach a model to point at anything on a line chart. Charts that *do* have boxes carry
**12.7** of them on average.

**What these annotations are for.** ChartQA has no per-question grounding — it labels every
element of a chart, not the ones a given question needs. We derive the link: plan mining
(§7) works out that a question is, say, `West − South`, which tells us the answer depends on
the labels "West" and "South", and we then take *those two* elements' boxes. A ChartQA
question thereby becomes a fully grounded training example. Without the element annotations
it could only teach the answer and the reasoning, never the pointing.

⚠️ **Do not confuse 12.7 with the 8-item evidence cap.** They count different things: 12.7 is
elements *on a chart*, 8 is the most evidence items *one answer* may contain. A chart with 12
bars whose question needs two of them uses two. Measured on RefChartQA questions, **76.5%
need exactly one box and 92.5% need two or fewer**; only **2%** need more than eight. Pointing
at everything would be actively harmful — one spurious box per image drops the grounding
score from 1.00 to 0.68, and three drop it to 0.32. The cap does bite on aggregate questions
("what is the average?"), which genuinely need every element: on a chart with more than eight
we refuse the record rather than truncate, because dropping bars would change the average and
the example would no longer reproduce its own answer.

**Finding: the boxes are geometrically sound where they exist.** We checked whether a bar's
box height actually tracks its value in the gold table, across 1,290 series. The median
r² was **0.9999** for vertical bars and **1.0** for horizontal bars. So where annotations
exist, they are trustworthy — the problem is coverage, not quality.

**Finding: RefChartQA's boxes pass an independent quality audit.** We sampled **200** rows
and checked each box actually contains chart ink and marks a *region* rather than the whole
chart. **200 of 200 acceptable**, across all three question subsets. That was a gate: had it
failed, we would have had to build our own grounding data instead.

## 7. Plan mining — and why the yield is only 14%

Our design needs the model to emit a **plan**. To teach it plans, we need training examples
where we know the correct plan. ChartQA does not provide one.

But ChartQA *does* provide the chart's underlying data table. So we can search: **is there
exactly one simple operation over this table that reproduces the gold answer?**

For "What is the difference between West and South?" with answer 74, we try every operation
over every combination — `difference(West, South) = 74`. Found, and nothing else gives 74.
So we record that plan.

We ran this over all **28,299** ChartQA training questions:

| | |
|---|---:|
| gave exactly one operation that explains the answer | **14.07%** |
| of those, plans that are just "read one number" (`lookup`) | **73.6%** |
| questions that teach genuine multi-step reasoning | **3.7%** |

**Why so low, and why we did not loosen it.** The obvious way to raise the yield is to
accept a plan that gets *close* to the answer. We refused, for a measured reason: ChartQA's
own scoring allows 5% tolerance, and 5% of the year "2014" is a window of ±100 years. Under
that tolerance, mining accepted `difference → 2096` as the explanation for a gold answer of
`2019`. That is not a plan, it is an arithmetic coincidence, and training on it teaches
arithmetic that is wrong. So we require the operation to reproduce the answer at the
precision the answer was written to, and we reject a question if **more than one** operation
explains it — ambiguity is a rejection, not a guess.

**The consequence is the reason §8 exists:** real data can barely teach reasoning at all.

## 8. The generator, and how a box gets verified

If real data cannot supply enough reasoning examples, we generate them — because in a chart
*we* draw, we know the answer and the boxes by construction.

The generator produces **8 chart types** (vertical bar, horizontal bar, grouped bar, line,
multi-line, pie, scatter, area) at **4 difficulty levels**:

| level | what it teaches | example |
|---|---|---|
| L1 | read one value | `lookup(West)` |
| L2 | compare two | `difference(West, South)` |
| L3 | aggregate over everything | `mean()` |
| L4 | combine — an operation *inside* another | `difference(West, mean())` |

**24,000 examples.**

**The part that makes them trustworthy.** Knowing where we drew a bar is not the same as
knowing the box is right — a box can be computed correctly and still be wrong because of a
stroke width, a marker edge, or a wedge's geometry. So each box is *measured against the
rendered image*: we compute how well the box overlaps the region where the element's ink
actually is.

A box is kept only if that overlap is at least **0.70**. Across the 640 examples we
verified in depth, correct boxes scored **0.84–0.99**. Figure 4 on your slide shows the
check running: the same box correct (0.95, kept) and moved 26 pixels (0.43, rejected).

**Why this matters more than it sounds:** every synthetic example that reaches training has
had its box confirmed against pixels, not merely computed. That is what makes "we know the
right answer because we drew it" an actual guarantee rather than a hope.

---

# Part III — Measurement

This part is most of what week 1 was. It is also the least intuitive, so it gets the most
space.

## 9. What the metrics actually mean

Three numbers get reported in this field. You should be able to explain each in one
sentence.

### Relaxed accuracy — for answers

You cannot score chart answers with exact string matching. The gold answer might be `74`
and the model might say `74.0`, or `74%`, or `73.8` after reading the bar slightly low.

**Relaxed accuracy** counts an answer correct if it is a number within **5%** of the gold
number; and falls back to exact string match for non-numeric answers.

> One quirk worth knowing because it bit us: in the official implementation, when the gold
> answer is **zero**, the 5% relative test cannot be computed (dividing by zero), so it
> falls back to exact string matching — which means `"0"` and `"0.0"` count as *different*.
> We reproduce this behaviour exactly when reporting scores, because a reported number must
> match what the benchmark's own code produces. We do *not* use it internally when asking
> "does this plan reproduce its own answer", where a correct result of zero is correct.

### AP@0.5 — for boxes

This is the standard object-detection measure, and the name unpacks like this.

**IoU** ("intersection over union") measures how well two boxes overlap: the area they share
divided by the area they cover together. Identical boxes give 1.0; boxes that do not touch
give 0.

**@0.5** means a predicted box counts as a hit if its IoU with a true box is at least 0.5 —
roughly, "more than half right".

**AP** ("average precision") summarises the trade-off between finding all the true boxes and
not emitting spurious ones, as a single number between 0 and 1.

The practical thing to remember: **extra boxes are expensive.** We measured it — a system
with perfect boxes scores 1.00, and the same system emitting *one* spurious box per image
drops to **0.68**. So a model that hedges by pointing at several things is heavily
penalised. This is why our design says "point at what the answer needs and nothing else".

### P@F1 — a stricter box measure

"Perfect at F1": the fraction of questions where the model's boxes are **exactly** right —
every true box found, no extra ones. It is all-or-nothing per question, so it is always
lower than AP and it moves in bigger jumps.

## 10. Why we implemented the scoring twice

This is slide 6, and it is the most defensible thing in the deck.

**The problem.** If we score our own model with our own code, and later report an
improvement, we have no way to tell a real improvement from a bug in our scoring. Scoring
code is easy to get subtly wrong — the zero quirk above is one of several.

**What we did.** Both datasets publish their official scoring programs. We took them
verbatim, recorded their SHA-256, and use them as the scorer of record. Then we wrote our
*own* implementation as well, and compared the two on **11,690 real predictions**:

| | agreement |
|---|---|
| AP@0.5, human subset | difference of **0.000** percentage points |
| AP@0.5, machine subset | **0.068** pp |
| AP@0.5, PoT subset | **0.036** pp |
| relaxed accuracy, over 423 borderline cases | **0 disagreements** |
| P@F1 | **0 disagreements** |

**Why have two at all, if the official one is the scorer of record?** Because the official
program produces one number for a whole set. It cannot give confidence intervals, and it
cannot break results down by box size or chart type — both of which we need for the
analysis in weeks 3 and 4. So: **official code for every reported headline; our
implementation for the breakdowns, validated against it to 0.07 points.**

That is the sentence to say if a TA asks why we wrote our own.

## 11. Visual tokens — why box size is the whole problem

Back to §2: the image reaches the model as patches. Each patch becomes one **visual token**.

For our model, one visual token corresponds to a **32 × 32 pixel** block of the resized
image. That number is not a guess — it is derived from the model's own configuration
(patch size 16, spatial merge 2 → 32).

At our 512-pixel input, a typical chart becomes about **247 visual tokens**.

**Now the consequence.** If the thing the model must point at is *smaller than one block*,
the model has no representation that isolates it. The information is averaged in with
whatever else shares that block. It is not that the model finds it hard — it is that the
input does not contain the distinction.

We measured how often this happens, on **7,158 boxes from RefChartQA's training split** at
512 px:

| | |
|---|---:|
| targets narrower than one block on at least one axis | **66.7%** |
| targets smaller than one block in total area | **24.8%** |

On the validation split, measured separately on 1,045 boxes, the by-axis figure is
**53.2%**. Both are correct; they are different populations. *Say which split you mean.*

**Why this shapes the project:**

1. It predicts that grounding scores will be low and that a large part of the error is not
   fixable by training.
2. It gives a concrete thing to test in week 3: raise the input resolution and see whether
   the gain lands specifically on the small-target group. If it does, we have identified a
   mechanism rather than just tuned a number.
3. It is why we chose 512 pixels rather than 448. We measured the trade: 448 uses 29% fewer
   visual tokens (176 vs 247) and trains faster (9.07 vs 11.90 seconds per step), but it
   pushes the sub-token fraction up by 9.5 points. We took the slower, better-resolved
   option.

---

# Part IV — The discipline

Nothing in this part produces a number. All of it exists so that the numbers in weeks 3–4
are believable.

## 12. Sealed test splits

**The rule:** we never train on, tune on, or even look at the test splits. Validation is for
choosing settings. Test is opened once, at the end, after every decision is frozen.

**Why it is not just good manners.** If you look at test results and then change something —
a prompt, a threshold, a learning rate — your final test number is no longer an estimate of
how the system performs on unseen data. It is an estimate of how well you tuned against
that particular test set. The number stops meaning what everyone will assume it means.

**How we enforce it rather than just intending it.** Any code path that touches a sealed
split calls a guard that refuses unless a completed pre-registration is committed to git and
unmodified. Passing a flag is not enough.

**Contamination is checked by image content, not filename.** Because RefChartQA reuses
ChartQA's images, the same chart can appear under two different names *and in a different
file format*. We hash the *decoded pixels* of every image and compare.

Comparing **file bytes** finds **0 of 4,000** matches. Comparing **pixel content** finds
**99.9%**. The same chart, re-encoded, is a different file and the same picture — and only
the second comparison notices. That difference is the whole reason we hash pixels.

**2,563** image hashes are sealed off. **15** ChartQA training images turned out to be
pixel-identical to a held-out chart, which removed **31 training questions** from the
mixtures — a small number, and one nobody would have found by comparing filenames.

## 13. Pre-registration

Before opening any test split, we write and commit a document recording every decision the
results could depend on: the model and why, the exact prompt text, the decoding settings,
the answer normaliser, both evaluators and their hashes, the training mixtures with exact
counts, all hyperparameters, the early-stopping rule, what counts as success, and how much
of each test split we will evaluate.

**The point** is that a decision recorded before the results exist cannot be adjusted after
them. It converts "we got a good number" into "we said in advance what would count as a
good number, and here is what happened".

It also records, explicitly, **the zero-shot baselines the trained system must beat** — so
the bar cannot move later. The document currently reads `TBD` for those, and the seal stays
shut until the baseline runs finish. That is deliberate: the guard rejects a
pre-registration that still contains placeholders.

## 14. The compute budget

Everything runs on **free** GPUs — Kaggle's free tier, a Tesla T4, **30 hours per account
per week**. The hard rule is to stop rather than spend money.

Measured on that hardware, at 512 px, 4-bit:

| | |
|---|---:|
| peak memory used | **5.57 GB** of 13.5 available |
| time per optimizer step | **11.9 s** |
| trainable parameters | **24.6 M** of 1.45 B (**1.7%**) |
| projected full training run | **~10 hours** |

**Why only 1.7% of parameters are trainable** — this is **LoRA**, and it is worth being able
to explain. Instead of updating all 1.45 billion weights, LoRA freezes them and inserts small
"adapter" matrices next to the big ones, training only those. You get most of the benefit of
fine-tuning at a fraction of the memory, and the result is a **49 MB** adapter file rather
than a 4 GB model.

**And 4-bit** — the frozen weights are stored at 4 bits each instead of 16, which is what
makes a 1.45-billion-parameter model fit in 5.57 GB at all. LoRA on top of a 4-bit base is
called **QLoRA**.

**Cost to date: $0.**

---

# Part V — The baseline

## 15. Prompts, and why they are frozen

A VLM does what you ask it, and *how* you ask changes the answer. So the prompt is a
setting like any other, and it has to be fixed before results are measured — otherwise you
end up tuning the prompt against your own test set.

We use three prompts, and each one's exact text is hashed and recorded:

| prompt | length | used for |
|---|---:|---|
| **plain** | 27 tokens | the published baseline condition — copied verbatim from the model's own technical report, so our number is comparable to theirs |
| **structured** | 980 tokens | asking an untrained model for the full record: schema, limits, and worked examples |
| **training** | 117 tokens | what the fine-tuned model will see — short, because a trained model does not need the examples explained |

**One measured lesson from designing them.** Our first version asked for pretty-printed
JSON. Pretty-printing costs **80% more tokens** than compact for identical content (253 vs
141), and a third of outputs were being cut off at the token limit before they finished.
Demanding compact JSON cut the median output **2.6×** and raised valid-JSON from 58% to 75%.

**A lesson about ourselves, worth telling if asked what went wrong.** We iterated the
prompt three times on samples of 12, 20 and 24 questions. At those sizes the confidence
interval on a 50% rate is roughly ±20 points — wider than any effect we were claiming to
see. We were tuning on noise. We now size a measurement by what it can actually resolve
before running it: the slice that produced our baseline is **200** questions, which
resolves about 10 points.

## 16. Which model variant, and why

Qwen3-VL-2B ships in two variants: **Instruct** and **Thinking**. Thinking generates
internal reasoning before answering and scores higher on the published ChartQA benchmark
(86.6 vs 79.1), so it looks like the obvious choice.

We wrote three gates **before** measuring, and required all three:

| gate | threshold |
|---|---|
| accuracy gain | at least 2 points better |
| valid JSON | at least 90% |
| latency | at most 2× Instruct's |

Thinking failed two of them: **3.1× the latency**, and only **30%** valid JSON — its
reasoning text runs on and the structured record never closes properly. So we selected
**Instruct**.

**The point to make:** the higher published number was not the deciding factor, because the
published number measures a different task from ours. Ours requires a parseable structured
output, and a model that reasons at length is worse at producing one.

## 17. Where the untouched model actually starts

Measured on a frozen 200-question slice of ChartQA validation, with the structured prompt,
no training at all:

| | | of what |
|---|---:|---|
| relaxed accuracy | **50.0%** | all 200 questions |
| output is valid JSON | **66.5%** | all 200 questions |
| output satisfies our schema | **35.5%** → **46.5%** after repair | all 200 questions |
| plan runs without error | **94.4%** | the 71 usable records |
| **plan reproduces its own answer** | **69.0%** | the 71 usable records |

**Read those denominators carefully — this is the easiest thing to state wrongly.** The last
two are conditional on the model having produced a usable record at all. As a share of all
200 questions, the plan reproduces the answer only **24.5%** of the time. Both framings are
honest; say which one you mean.

**What the numbers say.** The model can already answer half the questions. What it cannot
do is produce a well-formed record — two-thirds of the loss is outputs that never parse or
never satisfy the schema. And even among the records that *are* usable, its own stated
arithmetic fails to reproduce its own answer three times in ten.

**That gap is the project.** We are not primarily trying to make the model better at chart
reading. We are trying to make its stated reasoning actually support its answer, so that
the answer becomes checkable.

**One honest note about the 50%.** With the *plain* prompt the same model scores far higher
— the published figure for this checkpoint on ChartQA test is 79.1. Asking for the full
structured record costs a lot. Measuring that cost precisely is itself one of the
project's planned results, not an embarrassment: published work on comparable models
reports 4–9 points of cost for merely requesting JSON, and we are asking for considerably
more than JSON.

---

# Part VI — Questions you will be asked

Short answers you can give as-is.

**"Why is 50% accuracy so low? Isn't the published number 79?"**
> Different task. 79.1 is the model answering with a plain prompt on the test split. Our 50%
> is the same model asked for a full structured record — boxes, a plan, and an answer — on
> validation, with no training. Measuring what that structure costs is one of our planned
> results.

**"Why did you write your own scoring if an official one exists?"**
> We use the official one for every reported number. Ours exists for the breakdowns the
> official code cannot produce — confidence intervals, results by box size, by chart type.
> We validated ours against the official on 11,690 predictions: at most 0.07 points apart,
> zero disagreements on answers.

**"Why generate synthetic charts instead of using real ones?"**
> Because real data can barely teach reasoning. Only 14% of ChartQA questions have a single
> operation that clearly explains the answer, and three-quarters of those are just reading
> one number — so about 4 in 100 teach real multi-step reasoning. In a chart we draw, we
> know the answer and the boxes by construction.

**"How do you know the generated boxes are correct?"**
> We measure each box against the rendered image — how well it overlaps where the element's
> ink actually is — and keep it only above 0.70. Correct boxes score 0.84–0.99; a box moved
> 26 pixels scores 0.43 and is thrown away.

**"Will a model trained on synthetic charts work on real ones?"**
> That is the assumption the training set rests on, and we do not assume it — we measure it.
> Plan quality is reported separately for real and generated charts, and the gap between
> them is a planned result.

**"Two-thirds of targets are too small — doesn't that sink the whole project?"**
> It bounds how precise the boxes can be at this resolution, and we say so rather than
> hiding it. It also gives us a specific thing to test: raise the resolution and check that
> the gain lands on exactly the small-target group. If it does, we have found a mechanism
> rather than tuned a number.

**"Why this model? Why not something bigger?"**
> Everything runs on free GPUs — 30 hours a week on a T4. A 2-billion-parameter model in
> 4-bit with LoRA fits in 5.6 GB and trains in about ten hours. Cost so far: $0.

**"What did you get wrong?"**
> We iterated our prompt three times on samples of 12 to 24 questions. At that size the
> uncertainty is about ±20 points — wider than the effects we thought we were seeing. We
> were tuning on noise. We now compute what a measurement can resolve before running it.

**"Where is the trained model?"**
> Week 2. Week 1 was deliberately data, measurement and baselines: everything the later
> weeks depend on being trustworthy.

---

## If you remember five things

1. **A right answer and a lucky guess look identical** — that is the problem we are solving.
2. **We built the scoring before the model**, and checked it against the official scoring on
   11,690 predictions to within 0.07 points.
3. **Two-thirds of the things the model must point at are smaller than one block of what it
   sees** — a mechanism, not a mystery.
4. **Only 4 in 100 real questions teach multi-step reasoning**, which is why we generate our
   own charts and verify every box against pixels.
5. **The untouched model answers half the questions, but its own arithmetic backs its own
   answer only 69% of the time** — closing that gap is the project.
