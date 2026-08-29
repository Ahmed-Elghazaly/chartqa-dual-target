# Week 1 — slide outline (15 slides, ~15 minutes)

Written for a general audience: **what we did, what we measured, what we found.** No
implementation detail. The deck is built by `build_deck.js`, which is authoritative.

Each content slide carries a tag — **BUILT**, **MEASURED** or **FOUND**.

| # | tag | slide | the one thing it says |
|---|---|---|---|
| 1 | — | Title | Grounded Chart Question Answering · Week 1 of 4 |
| 2 | — | The problem | A model gives only an answer, so a right answer and a lucky guess look the same |
| 3 | — | Our idea | Ask it to mark what it used and state the calculation, then have a program redo the calculation and check |
| 4 | — | The four-week plan | W1 data & measurement ← here · W2 train · W3 evaluate · W4 analyse |
| 5 | — | **What week 1 delivered** | Four things built, four measured, five found |
| 6 | BUILT | The data foundation | ~86,000 chart questions. ChartQA marks every element in a chart; RefChartQA marks what each question needs. Test data sealed from day one |
| 7 | FOUND | The annotations are patchy | Bar charts 92–97%, pie 55%, **line charts 0%**. Where they exist, they are accurate |
| 8 | MEASURED | Is the grounding data usable? | **200 / 200** sampled annotations acceptable. Pass mark set before looking |
| 9 | FOUND | The datasets overlap | Comparing files: **0 of 4,000**. Comparing the pictures: **99.9%** |
| 10 | BUILT | Scoring we can trust | Official scoring for every number, ours for the breakdowns; **11,690** predictions scored twice, largest gap **0.07 points** |
| 11 | FOUND | Targets are too small | **67%** are smaller than one block the model can see; **25%** in every direction |
| 12 | FOUND | Little teaches reasoning | 28,299 questions → **14%** have one clear calculation → **74%** of those are "read one number" → **4 in 100** |
| 13 | BUILT | We generate our own charts | 8 styles × 4 levels = **24,000**; **6,000** are the multi-step kind real data barely provides |
| 14 | MEASURED | Where the model stands today | **50%** correct · **47%** properly structured · **69%** of those have reasoning that matches the answer |
| 15 | — | What is next | W2 train · W3 evaluate · W4 analyse. Test data stays sealed. **Cost so far: $0** |

## Figures

All four are drawn from **our own generated charts** — never from the public datasets,
whose licences do not allow redistributing their images. Regenerate with
`python scripts/make_presentation_figures.py`.

| figure | on slides | shows |
|---|---|---|
| `fig1_ungrounded.png` | 2 | a chart, a question, a bare answer, a question mark |
| `fig2_grounded.png` | 1, 3 | the same chart with the two bars marked, and the check that they give the answer |
| `fig3_subtoken.png` | 11 | the blocks the model sees, and a target smaller than one of them |
| `fig4_verification.png` | 13 | a correct region kept at 95%, the same region nudged and rejected at 43% |

## Three things to say carefully out loud

- **Slide 14's 69%** is of the answers that came out properly structured, not of all 200
  questions. Of all 200 it is 24.5%. Both are honest — just say which you mean.
- **Slide 11's 67%** is measured on the training portion. The validation portion gives 53%.
- **Slide 7's 0% for line charts** is a decision, not missing data: their annotations mark
  the lines *between* points, so a region around a point would have to be invented, and we
  refuse to put invented data into training.

## Likely questions

**"Why is 50% low — isn't the published number 79?"** Different task. 79 is the model
answering plainly. Ours is the same model asked for a full structured answer with regions
and reasoning, with no training. Measuring what that structure costs is one of our planned
results.

**"Why write your own scoring?"** We report the official one. Ours exists for the
breakdowns and error bars the official code cannot produce, and we checked the two agree.

**"Why generate charts instead of using real ones?"** Because only 4 real questions in 100
teach multi-step reasoning. In a chart we draw, we know the answer for certain.

**"Doesn't the small-target finding sink the project?"** It bounds how precise we can be at
this view size, and it tells us exactly what to test in Week 3 — a sharper view — rather
than guessing.
