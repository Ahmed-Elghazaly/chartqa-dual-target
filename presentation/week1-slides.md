# Week 1 — slide content

Audience: course peers and TAs. Language kept plain; every number below is measured and
traceable to `verification/measured_facts.json`.

10 slides, ~1 minute each.

---

## Slide 1 — Title

**Grounded Chart Question Answering**
Making a model show *where* it looked and *how* it calculated

Team: [names]  ·  Week 1 of 4

---

## Slide 2 — The problem

A chart question-answering model reads a chart and answers a question.

**But it only ever gives you the answer.**

> *"What is the difference between 2019 and 2018?"* → **"35"**

You cannot tell whether it:
- read the right two bars, or
- read the wrong bars and got lucky, or
- ignored the chart and guessed from the question

A right answer and a lucky guess look identical.

*Visual: a bar chart with a question and a bare "35" answer, question mark over the chart.*

---

## Slide 3 — Our idea

Make the model produce **three things instead of one**:

| what | example |
|---|---|
| **where it looked** | boxes around the 2019 and 2018 bars |
| **what it did** | `difference(2019, 2018)` |
| **the answer** | 35 |

Then a small program **re-does the arithmetic** from the boxes and checks it gives the
same answer.

**If they disagree, we know the answer is unreliable — without needing a human to check.**

*Visual: same chart, now with two boxes drawn, the expression underneath, and a tick.*

---

## Slide 4 — The four-week plan

| week | what |
|---|---|
| **1** | **Get the data. Build the scoring. Measure the starting point.** ← *we are here* |
| 2 | Train the model |
| 3 | Evaluate it properly |
| 4 | Analyse what worked, and write it up |

**Week 1 is deliberately not about training.** Everything in weeks 2–4 depends on having
data we trust and scoring we trust.

---

## Slide 5 — Week 1 · The data, and what is actually inside it

We collected two public datasets and checked every file against a fingerprint so we know
exactly which version we have.

| dataset | training | validation | test |
|---|---:|---:|---:|
| ChartQA | 28,299 | 1,920 | 2,500 |
| RefChartQA | 55,789 | 6,223 | 11,690 |

Then we **looked inside 2,500 annotations** rather than trusting the description:

- Bar charts are well annotated — **97%** and **92%** of charts have boxes
- Pie charts: only **55%**
- **Line charts: 0%** — no box annotations at all
- Charts that do have boxes average **12.7** of them

**Why it matters:** we cannot teach the model to point at line charts using this data.

---

## Slide 6 — Week 1 · We built the ruler before the thing we measure

Before writing any model code, we wrote the **scoring code** — and then checked it.

We took **11,690 real predictions** and scored them twice: once with the official scoring
program published with the dataset, once with ours.

| measure | biggest difference |
|---|---|
| Box-accuracy score | **0.07 percentage points** |
| Answer correctness (423 tricky cases) | **0 disagreements** |

**Why it matters:** any improvement we report later is a real improvement, not a bug in
our own scoring.

---

## Slide 7 — What we learned #1: the targets are tiny

The model does not see pixels. It sees the chart as a grid of small blocks.

**Two-thirds of the things it has to point at are thinner than one block.**

- 66.7% of target boxes are narrower than one block on at least one side
- 24.8% are smaller than one block in total area

*Visual: a chart with the block grid overlaid, and a thin bar sitting inside a single block.*

**Why it matters:** at normal input size the model physically cannot point at these
precisely. This is not something more training fixes — it tells us to test higher
resolutions later.

---

## Slide 8 — What we learned #2: there is very little to teach with

To teach the model *how* to calculate, we need questions where we know the calculation.

We searched all **28,299** training questions for one where exactly one arithmetic step
reproduces the correct answer.

- **14%** of questions gave a clear, single calculation
- Of those, **74% were just "read one number off the chart"**
- So only about **4 in 100** questions teach real multi-step reasoning

**Why it matters:** the real data can barely teach reasoning at all.

---

## Slide 9 — So we generate our own charts

We built a chart generator where **we already know the right answer and the right boxes**,
because we drew them.

- **8 chart types** × **4 difficulty levels** (read one value → compare → aggregate → combine)
- **24,000 examples**

**How we know the boxes are correct:** we re-draw each chart in test colours and check the
box actually contains the right coloured shape.

| | overlap score |
|---|---:|
| Correct boxes | **0.84 – 0.99** |
| Deliberately wrong boxes | never above **0.38** |

The check clearly separates right from wrong, so no bad example reaches training.

---

## Slide 10 — Where we start, and what is next

**The untouched model, before any training:**

| | |
|---|---:|
| Answers correct | **50%** |
| Produces a usable structured output | **47%** |
| Its own calculation matches its own answer *(when usable)* | **69%** |

That last number is the interesting one — even when it answers correctly, its stated
reasoning often does not support its answer. **That is the gap we are trying to close.**

**Next:** week 2 — training.

**Cost so far: $0.** Everything runs on free GPUs.
