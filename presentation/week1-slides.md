# Week 1 — slide outline (20 slides, ~20 minutes)

The deck is built by `build_deck.js`, which is authoritative. This is the same content in
readable form. Every number is measured; sources are in `BRIEFING.md` §"traceable".

Each content slide carries a tag: **BUILT** (we made it), **MEASURED** (we measured it),
**FOUND** (what it told us).

| # | tag | slide | the one thing it says |
|---|---|---|---|
| 1 | — | Title | Grounded Chart Question Answering · Week 1 of 4 |
| 2 | — | The problem | A model gives only an answer, so a right answer and a lucky guess look identical |
| 3 | — | Our idea | Ask for evidence boxes + the arithmetic + the answer, then re-run the arithmetic to check |
| 4 | — | The four-week plan | W1 data & measurement ← here · W2 train · W3 evaluate · W4 analyse |
| 5 | — | **What week 1 delivered** | Five things built, five measured, four found — the summary slide |
| 6 | BUILT | The data pipeline | Two datasets, 875 MB + 2.88 GB, hash-pinned; 18,317 charts, 28,299 questions; 12.7 boxes per annotated chart |
| 7 | FOUND | Inside the annotations | Bars 96.8% / 91.5%, pie 54.8%, **line 0.0%**; where boxes exist they track values at r² 0.9999 |
| 8 | MEASURED | Are the boxes any good? | RefChartQA audit: **200 / 200 acceptable**, all three question types. A gate set before looking |
| 9 | FOUND | The same chart under two names | Comparing files: **0 of 4,000** matches. Comparing pixels: **99.9%** |
| 10 | BUILT | The measuring instrument | Official scoring vendored and hashed; our own for intervals and breakdowns. Official wins on conflict |
| 11 | MEASURED | Do they agree? | **11,690** predictions · largest gap **0.07 points** · **0** disagreements on 423 borderline answers |
| 12 | FOUND | The targets are tiny | **67%** thinner than one 32-px block on some side; **25%** smaller than one block in area |
| 13 | BUILT | The plan miner | Try every operation over the gold table; accept only if exactly one explains the answer |
| 14 | FOUND | Little teaches reasoning | 28,299 → **14%** have one clear calculation → **74%** of those are "read one number" → **4 in 100** |
| 15 | BUILT | The chart generator | 8 types × 4 levels = **24,000**; **6,000** are level 4 — the kind real data supplies 4 in 100 of |
| 16 | MEASURED | Proving the boxes | Overlap against rendered ink; kept only above **0.70**; correct boxes score **0.84–0.99** |
| 17 | BUILT | Prompts and the parser | Three prompts, hashed and frozen. Parser may drop and unwrap, **never add** |
| 18 | MEASURED | Where the model starts | 50.0% correct · 66.5% valid JSON · 46.5% schema-valid · 94.4% runs · **69.0%** reproduces its own answer |
| 19 | FOUND | The gap | It answers half correctly, but its own arithmetic backs its answer only 69% of the time |
| 20 | — | What is next | W2 train · W3 evaluate on sealed test · W4 analyse. **Cost so far: $0** |

## Figures

All four are drawn from **our own generated charts**, never from ChartQA or RefChartQA
images (GPL-3.0 / AGPL-3.0 — a chart with boxes drawn on it is a derivative work).
Regenerate with `python scripts/make_presentation_figures.py`.

| figure | on slides | shows |
|---|---|---|
| `fig1_ungrounded.png` | 2 | a chart, a question, a bare answer, a question mark |
| `fig2_grounded.png` | 1, 3 | the same chart with evidence boxes and `144 − 70 = 74 ✓` |
| `fig3_subtoken.png` | 12 | the 32×32 grid the model sees, with a target of 0.47 × 0.45 blocks |
| `fig4_verification.png` | 16 | the box check running: 0.95 KEPT vs 0.43 REJECTED |

## Denominators to state correctly

- Slide 18's last two rows are **of the 71 usable records**, not of all 200. As a share of
  all 200, the plan reproduces the answer **24.5%** of the time. Both are honest; say which.
- Slide 12's 67% is the **training** split. Validation gives 53.2% on the same definition.
- Slide 7's 0.0% for line charts is a **decision**, not missing data: their boxes are the
  segments *between* points, so a marker size would have to be invented.
