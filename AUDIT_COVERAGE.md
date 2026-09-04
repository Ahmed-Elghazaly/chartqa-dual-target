# Task 1's 99 items, one by one

`Prompt.md` Task 1 enumerates 99 things to review. `AUDIT_PLAN.md` tracks them by *subsystem*;
this tracks them by *item*, because the two are not the same and the difference is where a gap
hides.

**Depth is marked honestly, not uniformly.** Ticking 99 boxes would be worth nothing.

| mark | means |
|---|---|
| **M** | **measured** — a number on real data, with a decision record |
| **V** | **verified in code** — read end to end, contract confirmed, no defect found |
| **L** | **limitation recorded** — a real weakness, stated rather than fixed |
| **I** | **inspected** — read, nothing found, no separate measurement made |

---

| # | item | depth | evidence |
|---:|---|:---:|---|
| 1 | Overall architecture | **V** | `ARCHITECTURE.md`, ten subsystems in the six-part form |
| 2 | Repository organization | **M** | 330 → 298 tracked files; one mining path; outputs gitignored |
| 3 | Phase structure and dependencies | **V** | `STATUS.md`; the two-stage split traced (0088) |
| 4 | ChartRecord | **M** | `boxes` means 3 things (C2); identity design recorded (0107) |
| 5 | ChartQA adapter | **M** | colour never read (0087); series dropped (0083) |
| 6 | RefChartQA adapter | **M** | boxes ARE ChartQA elements, 98.9% IoU ≥ 0.9 (0077) |
| 7 | Synthetic adapter / conversion | **M** | elements = operands, not the chart (0098) |
| 8 | Dataset merging | **M** | the merge is discarded before training (H2) |
| 9 | Deduplication | **M** | separated from fusion; every consumer enumerated (0108) |
| 10 | Image identity | **V** | `image_content_sha256` over decoded pixels, not file bytes |
| 11 | Decoded-pixel hashing | **M** | caught 15 train images pixel-identical to held-out charts (0049) |
| 12 | Same-image matching | **M** | 86.9% of RefChartQA shares a ChartQA image |
| 13 | Same-question matching | **M** | only 42.1% shares a question — why fusion beats a merge (0105) |
| 14 | Cross-source matching | **M** | IoU ≥ 0.90 with a 0.50 margin, greedy one-to-one, all-or-nothing (0077) |
| 15 | Train/val/test split handling | **V** | `assert_train_only` on the **inputs**, before any filtering |
| 16 | Record-level leakage | **V** | same guard; checked before filtering so a leak cannot hide |
| 17 | Image-level leakage | **M** | `assert_no_held_out_images` — RefChartQA ships "train" rows using test charts (0049) |
| 18 | Sealed-test logic | **V** | `sealed_image_hashes`; synthetic holdout by style and data seed |
| 19 | Table representation | **L** | two shapes: `{columns, rows}` and `{labels, values}` (0098) |
| 20 | Chart-element representation | **M** | series and colour now carried (0083, 0087) |
| 21 | Labels | **M** | non-unique on 22.6% of charts, not 74.2% — that sample was biased (0083) |
| 22 | Values | **M** | two parsers 100× apart; four call sites wrong (0082, 0089) |
| 23 | Units | **M** | unit suffixes (`'26.29 t'`) on 0.8% of charts; `check_units` verified |
| 24 | Bounding boxes | **M** | 0–1000 anisotropic; the official evaluator discards a box at exactly 1000 |
| 25 | Evidence representation | **L** | derived, not stored; four defects lived in the derivation (0108) |
| 26 | `meta` | **M** | `ELEMENTS_KEY` means two things by source (0098) |
| 27 | Label/value/bbox associations | **M** | 9.2% described different marks (C1, 0075) |
| 28 | Coordinate systems | **V** | anisotropic by axis; ported to match `smart_resize` exactly |
| 29 | Coordinate normalization | **V** | `clamp_for_official_evaluator`; G1 re-verified |
| 30 | Image loading | **V** | PIL from the zip member; size read from the decoded image, never assumed |
| 31 | Image resizing | **V** | the processor owns it; no double resize (G1) |
| 32 | Qwen3-VL preprocessing | **M** | Idea 12 — verified correct against the official processor, no change |
| 33 | Visual patch / token behaviour | **M** | patch 16 × merge 2 = factor 32, derived from the loaded processor, never hard-coded |
| 34 | Resolution choices | **M** | 512px → native; 11.9 points of sub-token targets (0095) |
| 35 | Target construction | **M** | grounding-only targets: 56.6% → 98.5% supervisable (0104) |
| 36 | Structured output schema | **M** | admitted three operations the executor refuses (0109) |
| 37 | Output serialization | **M** | JSON kept; short keys save 0 tokens/item on this tokenizer (0094) |
| 38 | Invalid-output handling | **V** | drop / unwrap / never add, every repair counted (0064) |
| 39 | Typed reasoning DSL | **M** | 93.3% of a random corpus expressible; human questions need more (0081, 0090) |
| 40 | Operator set | **M** | `within` added on measured demand; six more requested with numbers (0090) |
| 41 | Operator semantics | **M** | each judged against 60 real questions (0081) |
| 42 | Nested / compositional plans | **M** | depth ≤ 4, arity ≤ 4; median depth 1–2 in practice (0102) |
| 43 | Executor implementation | **M** | one shared parser after four defects (0082, 0089) |
| 44 | Numerical tolerance | **V** | the answer's own precision, not 5% — 5% of the year 2014 is a century (0045) |
| 45 | Unit behaviour | **V** | `check_units` refuses a mixed-unit aggregate |
| 46 | Executor errors | **V** | `ExecutorError` never swallowed; every occurrence counted |
| 47 | Round-trip verification | **M** | it cannot catch wrong evidence (0075) or a coincidence (0097, 0106) |
| 48 | Plan mining | **M** | direction was the constraint, not search (0085) |
| 49 | Candidate generation | **M** | `candidate_sets` flattenings measured; `union` scored worse than `all_cells` |
| 50 | Operand generation | **M** | 22.6% of unique operations had several operand pairs (0106) |
| 51 | Candidate ambiguity | **M** | 53.9% refused; `lookup` vs extremum is 26.6% of all rows (H4) |
| 52 | **Concrete-program ambiguity** | **M** | **22.6%, the case the brief flags for special attention (0106)** |
| 53 | Semantic correctness of mined plans | **M** | 100 records hand-judged across two seeded samples |
| 54 | Mining tolerances | **V** | confirmed correct; not the binding constraint (0045, 0085) |
| 55 | Numeric-looking categories | **M** | `answer_is_a_category` — 8.5% of mined plans had a gold answer that is a row label |
| 56 | Percentage handling | **M** | 21.4% of charts all-percent with a bare answer; the 100× defect (0082) |
| 57 | Question semantics in mining | **M** | the miner never reads the question — the whole finding (0085, 0086) |
| 58 | Synthetic curriculum | **M** | L1→L4 balanced; `balance_by_level` stops L3/L4 being cut by the cap (0066) |
| 59 | Synthetic value generation | **I** | seeded per example (`data_seed`); reproducible; no defect found |
| 60 | Synthetic question generation | **M** | templated — which is why pattern matching works on it and not on humans (0086) |
| 61 | Synthetic rendering | **V** | boxes from matplotlib artists, never a formula; proven against pixels |
| 62 | Synthetic chart diversity | **M** | 25% were types ChartQA lacks, now dropped; density 2.8× under (0091, 0098) |
| 63 | Synthetic style diversity | **I** | `style_seed` over font size, dark mode, grid; three seeds sealed for the robustness test |
| 64 | Synthetic geometry extraction | **V** | `artist_box`, `point_box`, `scatter_point_box`; degenerate boxes rejected |
| 65 | Synthetic bbox verification | **V** | 640/640 verified across 8 chart types × 4 levels × 20 seeds |
| 66 | Real-vs-synthetic domain gap | **M** | three mismatches: type, operation mix, density (0091, 0098, 0101) |
| 67 | Data-quality gates | **M** | five gates; discard never repair; the value/box gate (0075) |
| 68 | Target-yield checks | **M** | before/after on identical records throughout |
| 69 | Supervision provenance | **M** | complete and unread; report before weighting (0105) |
| 70 | Confidence tracking | **M** | `match_iou` and `match_margin` per element; `gates_passed` per plan (0105) |
| 71 | Source balancing | **M** | absent chart types dropped; levels stay balanced (0091) |
| 72 | Training mixtures | **M** | the 12,000 cap is the compute budget backwards (0092) |
| 73 | Training objective | **M** | boxes 35.6% of the loss, the answer 3.7% (0096) |
| 74 | Standard autoregressive SFT | **V** | prompt masked, target supervised, end-of-turn included |
| 75 | Token masking | **V** | the prompt boundary is measured **with the image**, never by text count |
| 76 | Possible auxiliary losses | **L** | considered and not adopted — the answer policy must settle first (0096) |
| 77 | Curriculum strategy | **M** | stage 1 is a curriculum stage, so its distribution matters (0101) |
| 78 | Evaluation methodology | **M** | per subset, never aggregated — they differ by 30 points (0002) |
| 79 | ChartQA answer evaluation | **M** | byte-faithful to the official parser, quirks included |
| 80 | RefChartQA grounding evaluation | **M** | ours matches the official evaluator to **0.068 pts** on 11,690 predictions (0093) |
| 81 | Plan validity evaluation | **M** | schema validity tracked separately from JSON validity (0058) |
| 82 | Plan execution evaluation | **M** | executable rate reported separately from agreement (0059) |
| 83 | Semantic plan evaluation | **M** | what arithmetic cannot catch, now detected (0097, 0106) |
| 84 | Round-trip consistency metrics | **M** | a headline number since 0059 |
| 85 | Malformed-output rate | **M** | `ParseStats` counts every repair by kind (0064) |
| 86 | Failure analysis | **M** | refusal profiles by cause throughout, never a bare total |
| 87 | Logging | **I** | `logging_utils.py`; run records written per phase; no defect found |
| 88 | Reproducibility | **V** | every sample seeded and named; ids stable across processes |
| 89 | Random seeds | **M** | RNG state in checkpoints — its absence cost 0.0438 loss on resume (0026) |
| 90 | Config management | **M** | the pixel budget was restated at a call site and would have drifted (0095) |
| 91 | Tests | **M** | 1,200 → see `tests/`; expanded specifically against this audit's defect classes |
| 92 | CLI behaviour | **V** | six entry points; `cdt-mine` repointed from the retired miner (0088) |
| 93 | Error handling | **V** | typed errors, never swallowed; every refusal names its cause |
| 94 | Performance | **M** | 11.903 s/step at 512px, 21.267 native — measured on the real GPU (0060) |
| 95 | Memory cost | **M** | 5.572 GB peak at 512px, 6.723 native; an earlier 1.482 was a sharded misread (0060) |
| 96 | Scalability | **I** | the archive is read lazily by member; `qa_rows` materialises one split at a time |
| 97 | Documentation accuracy | **M** | the README's central claim was false and is corrected (0096) |
| 98 | Research defensibility | **M** | 32.83 is not in the paper; six published baselines instead (0093) |
| 99 | Anything else discovered | **M** | 0106 operand ambiguity; 0109 unexecutable operators offered to the model |

---

## Where the depth is thinnest

Four items are marked **I** — inspected, nothing found, no separate measurement. They are the
honest weak spots of this audit:

* **59 — synthetic value generation.** Seeded and reproducible, but the *distribution* of
  generated values was never compared to real charts. Given that chart **density** turned out
  to be badly mismatched (0098), value distribution is a plausible place for a similar gap.
* **63 — synthetic style diversity.** Font size, dark mode and grid vary by seed; whether that
  spans the visual variation of real Statista charts is unmeasured.
* **87 — logging.** Adequate for the phases run so far; no failure has been traced to a
  missing log, which is weak evidence.
* **96 — scalability.** Nothing here has run at the full 55,789-row RefChartQA split. Lazy
  reads and per-split materialisation suggest it holds; it has not been shown.

Items 59 and 63 fold naturally into the L3–L4 regeneration (0101), which is already scheduled
and already has to measure a distribution.
