# Week-1 presentation

| file | what it is |
|---|---|
| `ChartQA-Week1.pptx` | **the deck** — 10 slides, open in PowerPoint, Keynote or Google Slides |
| `week1-slides.md` | the slide content in plain text, the source the deck is built from |
| `figures/*.png` | the four figures |
| `build_deck.js` | rebuilds the `.pptx` — `node build_deck.js` |
| `check_layout.py` | geometry QA: off-slide content, stretched images, text spilling past a card |

## Rebuilding

```
node build_deck.js
python check_layout.py ChartQA-Week1.pptx
```

## The figures

`scripts/make_presentation_figures.py` draws all four from this project's **own generated
charts**, and every number they print is measured rather than typed — the overlap scores
come from `synth/verify.ink_bbox_iou`, the block sizes from the same `smart_resize` port
the model's processor uses.

**They are deliberately not built from ChartQA or RefChartQA images.** ChartQA is GPL-3.0
and RefChartQA is AGPL-3.0, and a chart image with boxes drawn on it is a derivative of
that image (rule 7; `eval/figures.write_figure` refuses it in code). Our generated charts
are our own work, so they can go in a deck that is handed around — and they demonstrate
the generator while they are at it.

## Every number is traceable

Nothing in the deck is rounded from memory. Sources:

| claim | where it comes from |
|---|---|
| split sizes | `verification/measured_facts.json` → `datasets` |
| box-annotation coverage by chart type | `phase3.element_box_coverage_pct` |
| 11,690 predictions, 0.07 pp, 0 disagreements | `phase4.crosscheck_vs_official` |
| 67% / 25% sub-token | `phase4.stratification` |
| 14% mining yield, 74% lookup | `phase3.mining_yield_pct`, `mining_lookup_share_pct` |
| 8 × 4 types, 24,000 examples, 0.84–0.99 | `phase3.synth_*` |
| 50% / 47% / 69% zero-shot | `phase5.variant_selection_5_2` |

The 69% is **of the records that were usable**, not of all 200 questions — say it that way
if asked.
