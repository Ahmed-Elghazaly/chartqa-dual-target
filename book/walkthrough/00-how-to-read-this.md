# Code walkthrough — how to read this

## What this is

A line-by-line explanation of every piece of code Phases 0–5 rest on: **4,869 lines**
across 18 files. Written so that you can read a chapter, then read the file, and recognise
every line.

Nothing is assumed except Python and school maths. Where a machine-learning concept is
needed, it is taught at the point it becomes necessary, not before and not by reference.

## The order, and why it is this order

Each chapter depends only on the ones before it. That is the whole reason for the ordering —
you should never meet a term that has not been defined.

| # | chapter | files | lines |
|---|---|---|---:|
| 1 | Coordinates and visual tokens | `vision/coords.py` | 305 |
| 2 | The record, and knowing when two things are the same | `data/records.py`, `data/dedup.py` | 325 |
| 3 | Reading ChartQA off the disk | `data/chartqa.py` | 348 |
| 4 | The plan language: schema, executor, round-trip | `plans/schema.py`, `executor.py`, `roundtrip.py` | 540 |
| 5 | Mining plans out of real data | `plans/mining.py` | 302 |
| 6 | Generating charts, and proving the boxes | `synth/curriculum.py`, `generator.py`, `verify.py` | 920 |
| 7 | Metrics: what a score actually computes | `eval/metrics.py`, `runner.py`, `stratified.py` | 981 |
| 8 | Prompts, and repairing what comes back | `prompting/prompts.py`, `parsing.py` | 463 |
| 9 | Mixtures, and sealing the test split | `data/mixture.py`, `splits.py` | 431 |

**Chapter 1 is not optional.** Chapters 3, 6, 7 and 9 all silently assume it. If the old
briefing lost you around its §6, this is why: it used "visual token", "0–1000 space" and
"sub-token" as though they had been defined, and they had not.

## Conventions

- Code is quoted **exactly** as it is in the file. If a quote is trimmed, it says so.
- `DECISIONS.md 00NN` refers to a numbered decision record. Every one of them exists
  because something went wrong or a choice had a real alternative.
- Numbers are measured. Where a number could be misread, the population it was measured on
  is named.
- **Boxes** are marked 📘 for a concept explained from zero, and ⚠️ for a mistake that was
  actually made in this project.
