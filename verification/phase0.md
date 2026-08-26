# Phase 0 — Re-verification of load-bearing facts

**Date performed:** 2026-08-26
**Performed by:** automated checks against primary sources (HF Hub API, HF datasets-server API, GitHub API, raw source files, arXiv PDF).
**Source of claims:** `IDEA.md` (written 2026-08-24/25/26) and `PLAN.md` Phase 0 table.

**Verdict: all ten claims CONFIRMED. Claim 0.3 is confirmed but was under-specified in `IDEA.md`; see finding F1 — it changes how results must be reported, not whether the project is viable.**

Method note: where possible I queried structured APIs that return the dataset/model as it exists right now
(`huggingface.co/api/...`, `datasets-server.huggingface.co/...`, `api.github.com/...`) rather than reading a
rendered card, because cards can lag the artefact. Numbers below are what those APIs returned today.

---

## 1. Claim-by-claim results

| # | Claim | Source URL | Checked | Status | What I found / did |
|---|---|---|---|---|---|
| 0.1 | ChartQA splits are 28,299 / 1,920 / 2,500 | `https://datasets-server.huggingface.co/size?dataset=ahmed-masry/ChartQA` | 2026-08-26 | **confirmed** | Exactly `train=28299`, `val=1920`, `test=2500`. Split key is `val`, not `validation`. |
| 0.2 | RefChartQA splits are 55,789 / 6,223 / 11,690, download ~2.88 GB | `https://datasets-server.huggingface.co/size?dataset=omoured/RefChartQA` ; `https://github.com/moured/RefChartQA` | 2026-08-26 | **confirmed** | Exactly `train=55789`, `validation=6223`, `test=11690` (repo README states the same, total 73,702). Parquet bytes measured `2,157,086,739 + 251,258,803 + 476,158,160 = 2,884,503,702` (2.88 GB). `IDEA.md` quotes 2,886,453,081 — a 0.07% difference, consistent with repo-total vs parquet-only accounting. Immaterial. |
| 0.3 | **RefChartQA published AP@0.5 target is 32.83; per-model prediction files released** | `https://arxiv.org/html/2503.23131v2` Table 2 ; `https://github.com/moured/RefChartQA/tree/main/evaluation` | 2026-08-26 | **confirmed, with refinement** | 32.83 is reported for **Qwen2.5-VL (3B, native resolution)** on the **RefChartQA-H (human) subset**. Same model scores 59.28 (machine) and 39.32 (PoT). Training was **1 epoch** for Qwen2.5-VL (5 epochs for the other models) — confirms `IDEA.md`'s "one epoch" claim. A prediction file `filtered_results.jsonl` (11,698 rows, 1.5 MB) is released. See **F1** and **F8**. |
| 0.4 | RefChartQA evaluator present and runnable | `https://raw.githubusercontent.com/moured/RefChartQA/main/evaluation/evaluate.py` | 2026-08-26 | **confirmed** | `evaluate.py` (12,589 bytes) downloaded and read in full; `requirements.txt` = `torch, torchmetrics[detection], pillow, pandas, datasets`. Runnable. It imposes a specific output string format and coordinate range — see **F2**, **F3**, **F4**. Target level stays **B**. |
| 0.5 | `Qwen/Qwen3-VL-2B-Instruct` and `-Thinking` exist, Apache-2.0 | `https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct` (+ `-Thinking`) | 2026-08-26 | **confirmed** | Both exist, `license=apache-2.0`, `gated=false`. Instruct: 2,630,627 downloads, 451 likes. Thinking: 73,657 downloads, 115 likes. BF16 `model.safetensors` = **4,255,140,312 bytes** — matches `IDEA.md` exactly. |
| 0.6 | `unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit` exists | `https://huggingface.co/api/models/unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit` | 2026-08-26 | **confirmed** | Exists, `license=apache-2.0`, ungated, 171,556 downloads. `model.safetensors` = **2,406,693,586 bytes** — matches `IDEA.md` exactly. |
| 0.7 | Qwen3-VL-2B ChartQA scores are 79.1 (Instruct) / 86.6 (Thinking) | `https://arxiv.org/pdf/2511.21631` Table 4 | 2026-08-26 | **confirmed** | Table 4 ("Performance of small-sized Qwen3-VL models and GPT-5-nano on visual benchmarks"), row `ChartQA_test`: `86.6  79.1  88.8  84.6  88.6  89.6  52.1  48.6` under headers `Qwen3-VL 2B (thinking, instruct) | 4B (thinking, instruct) | 8B (thinking, instruct) | GPT-5 nano (high, minimal)`. So **2B-Thinking = 86.6, 2B-Instruct = 79.1**. Exact match. |
| 0.8 | ChartQAPro evaluator still drops `always_use_exact_match` | `https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/utils/chartqapro.py` ; `https://github.com/vis-nlp/ChartQAPro/blob/main/evaluate_predictions.py` | 2026-08-26 | **confirmed — defect still live in BOTH** | VLMEvalKit: line 326 computes `always_use_exact_match = True if split in ['Fact Checking', 'Multi Choice'] else False`; line 327 calls `relaxed_correctness_chartqapro(gt, pred, year_flags=year_flags_per_row)` — the flag is never passed, so the parameter keeps its default `False`. Official ChartQAPro repo: identical defect at lines 140/141. |
| 0.9 | CURV reachable at pinned commit `2e79668d…`, Apache-2.0 | `https://api.github.com/repos/xhguo7/CURV` | 2026-08-26 | **confirmed** | `license=Apache-2.0`, 2 stars, 6.6 MB. Pinned commit `2e79668d840cac7cf58102cd6bc441431a0ea3fc` resolves (dated 2026-08-05, message "update readme") and is repository HEAD. Secondary path only. |
| 0.10 | Unsloth Qwen3-VL notebook coverage by model size | `https://api.github.com/repos/unslothai/notebooks/contents/nb` | 2026-08-26 | **confirmed** | Vision notebooks exist for **Qwen3-VL 8B** (incl. a Kaggle variant), **Qwen2.5-VL 7B**, **Qwen2-VL 7B**, **Qwen3.5 0.8B / 2B / 4B**. **No Qwen3-VL-2B notebook.** Exactly as `IDEA.md` §7 states. The risk stands and Phase 2 must resolve it by measurement. |

---

## 2. New findings that were not in `IDEA.md`

These came out of reading primary sources rather than cards. Each is recorded because it changes what
gets built or how results get reported.

### F1 — 32.83 is the **human subset**, not an aggregate (materially changes reporting)

The RefChartQA evaluator does not produce one number. `load_datasets_by_source()` partitions the test set
into `human` / `machine` / `pot` by the `type` column and `evaluate_all_datasets()` reports
`accuracy`, `AP_50`, `P@F1` **for each of the three separately**. There is no combined row.

Paper Table 2, Qwen2.5-VL (3B, native res): **AP@0.5 = 32.83 (H) / 59.28 (M) / 39.32 (PoT)**.
`IDEA.md`'s "other scored models span roughly 18.30–27.81" is exactly the human column
(UniChart 18.30 … TinyChart 27.81), so `IDEA.md` was implicitly using the human subset throughout
but never said so.

**Consequence:** every grounding number this project reports must be a triple (H / M / PoT), and the
comparison against 32.83 is a comparison **on RefChartQA-H only**. Reporting a single aggregate AP and
comparing it to 32.83 would be exactly the "claiming a published win that is not comparable" failure in
`IDEA.md` §13.3.

**Secondary consequence:** the human test subset is small — **500 rows**, measured directly against the live
dataset (see F8). Bootstrap CIs on the headline grounding comparison will be correspondingly wide (order ±4
points at that n), and that has to be stated rather than hidden.

### F2 — The evaluator's required output format, and a silent coordinate trap

`evaluate.py` expects each prediction's `model_answer` to be a single string of the form:

```
<box>x1,y1,x2,y2</box><box>…</box><grounding-sep>ANSWER_TEXT
```

- `extract_bounding_boxes()` only accepts a box when **all four coordinates satisfy `0 <= v <= bins-1`, i.e. `0..999`**.
  A coordinate of exactly **1000 causes the entire box to be silently discarded** — no error, no warning, it
  just stops existing and AP drops. Qwen3-VL's native output range is 0–1000 **inclusive**, so this will
  happen unless we clamp. Everything we emit for scoring must be clamped to `[0, 999]`.
- `eval_is_element_correct()` and `compute_P_at_FI()` both require `model_answer.split("<grounding-sep>")`
  to yield **exactly 2 parts**. Any other count scores 0 for accuracy and is skipped for P@F1 **while still
  counting in the denominator**. Template compliance is therefore itself a measured quantity.
- Ground-truth boxes arrive as absolute-pixel `{x, y, w, h}` and are converted by
  `transform_bbox_to_quantized()` → clip to image → xyxy → divide by w/h → `min(int(v*1000), 999)`.

**Consequence:** we need an adapter that serialises our JSON record into this exact string, with clamping,
and a test that a coordinate of 1000 survives the round trip as 999.

### F3 — The official `relaxed_accuracy` differs from `PLAN.md` Appendix D in three ways

Official (RefChartQA `evaluate.py`, copied from pix2struct):

```python
if text.endswith("%"): return float(text.rstrip("%")) / 100.0
else: return float(text)
...
if prediction_float is not None and target_float:   # <-- truthiness, not `is not None`
```

Differences from Appendix D:
1. **No comma stripping.** Official `_to_float("1,234")` raises → returns `None` → falls back to string
   comparison. Appendix D does `.replace(",", "")`.
2. **No whitespace stripping.** Appendix D does `text.strip()`.
3. **`and target_float` is a truthiness test.** When the target parses to `0.0` it is falsy, so the numeric
   branch is skipped entirely and the comparison falls through to case-insensitive string equality. Appendix D
   instead guards explicitly with `if t == 0: return p == 0`. These agree on `("0","0")` and `("0","0.1")`
   but **disagree on `("0", "0.0")`** — official says incorrect, Appendix D says correct.

**Consequence:** per non-negotiable rule 5 and Phase 4.2, the **official implementation is primary**. I will
vendor it verbatim as the scorer of record and keep the Appendix D version as a separately-named diagnostic,
reporting the per-question disagreement count. These three cases go into the Phase 4.3 regression suite.

### F4 — Official AP@0.5 is structurally different from Appendix D's AP

`compute_AP_50()` uses `torchmetrics.detection.MeanAveragePrecision(iou_thresholds=[0.5])` and takes
`result["map"]`. Every predicted box is assigned **`score = 1.0`** (`torch.ones(len(bboxes))`), so there is
no confidence ranking at all — the precision-recall curve degenerates toward a single operating point, and
torchmetrics applies COCO 101-point interpolation. Appendix D's `average_precision_at_iou` sorts by score and
uses all-point interpolation.

**Consequence:** these two will not agree in general, and that is structural, not a bug in either. Official is
primary; Appendix D's is for stratified analysis only, exactly as `PLAN.md` already says. Since scores are
constant, emitting *more* boxes can only hurt precision — the model should emit the boxes it believes, not a
speculative ranked list.

### F5 — Qwen3-VL's native grounding key is `bbox_2d`, and the 0–1000 claim is confirmed at source

Qwen3-VL technical report, §3: *"Different from Qwen2.5-VL, we adopt a normalized coordinate system scaled to
the range [0, 1000]"*. The official 2D-grounding cookbook emits
`{"bbox_2d": [x1, y1, x2, y2], "label": "..."}` and post-processes with `coord / 1000 * width`.

`IDEA.md`'s claim that the model natively emits 0–1000 boxes is **confirmed at source**, and this is a real
advantage over the Qwen2.5-VL-3B fallback, which emits absolute pixel coordinates.

**Open point for Phase 5.1:** `PLAN.md` Appendix A names the field `bbox`, but the model was pretrained to
emit `bbox_2d`. That mismatch plausibly costs zero-shot box quality — which is the "before" number the whole
project is measured against. Resolve it by measuring both on validation during prompt design. Recorded as an
open decision, not settled here.

### F6 — ChartQA gold tables are NOT in the HF parquet; plan mining needs the zip

The `ahmed-masry/ChartQA` parquet/viewer version exposes only `imgname, query, label, type, image`. The gold
data tables and chart-element annotations that Appendix E plan mining depends on live in a separate file in
the same repo: **`ChartQA Dataset.zip`, 875,370,872 bytes** — which is exactly the "~875 MB full archive"
figure in `IDEA.md` §6.1, now located precisely.

**Consequence:** Phase 3.1 must fetch that zip via `hf_hub_download(repo_type="dataset")`, not `load_dataset()`.

### F7 — Confirmed dataset schemas (de-risks Phase 3)

- **RefChartQA:** `id` (e.g. `RefChartQA_human_val_0`), `image`, `query`, `response`, `label` (gold answer),
  `grounding_bboxes` (list of `{x,y,w,h}` float32, **absolute pixels**), `type` ∈ {`human`,`machine`,`pot`}.
- **ChartQA (parquet):** `imgname`, `query`, `label`, `type` ∈ {`human`,`machine`}, `image` (binary).

### F8 — The released prediction file has 8 more rows than the test split

`filtered_results.jsonl` = **11,698** rows, all ids unique, composed of 508 `human` / 1,032 `machine` /
10,158 `pot`. The test split is **11,690**.

The composition of the actual test split was then measured directly
(`datasets-server.huggingface.co/filter?...&where="type"='<t>'`):

| `type` | test split | prediction file | difference |
|---|---:|---:|---:|
| `human` | **500** | 508 | **+8** |
| `machine` | 1,032 | 1,032 | 0 |
| `pot` | 10,158 | 10,158 | 0 |
| total | 11,690 | 11,698 | +8 |

So the surplus is **entirely in the human subset**, and machine and PoT align exactly. The evaluator's
`pd.merge(test_dataset, result_df, on="id", how="left")` is left-joined on the test set, so the 8 surplus ids
are dropped and scoring proceeds over 500 human rows. The file is described only as "an example … showing the
appropriate format", with no statement of which model produced it.

**Consequence 1 — statistical.** The headline grounding comparison is against **500** test items, not 11,690.
At n=500 the 95% bootstrap CI on an AP-like score in the low 30s is roughly ±4 points. A trained-vs-zero-shot
gap smaller than that is not distinguishable from noise, and the report must say so rather than presenting a
bare point estimate. This is now the single tightest statistical constraint in the project and it was not
visible anywhere in `IDEA.md`.

**Consequence 2 — Phase 4.4.** Carried forward as an open item. Re-scoring this file is the Level-B
reproduction attempt; if it yields ≈32.83 on the human subset then it is the Qwen2.5-VL-3B prediction file and
the target is independently confirmed. Max coordinate observed in the file is 984, consistent with the 0..999
convention.

### F9 — The prompt that produced the 79.1 figure is documented

Qwen3-VL report, appendix of evaluation prompts, for `DocVQA | InfoVQA | ChartQA_TEST`:

```
<image>
{question}
Answer the question using a single word or phrase.
```

**Consequence:** this is the prompt to use for the structured-output-**off** arm in Phase 8.1 and for the
matched plain baseline, so that our "structured output costs N points" measurement is anchored to the same
elicitation that produced the published 79.1.

### F10 — ChartQAPro size confirmed exactly

`test = 1,948` rows, `193,053,989` bytes — byte-for-byte the figure in `IDEA.md` §6.4. Licence `mit`.

---

## 3. Licence re-confirmation

| Artefact | Declared licence (checked 2026-08-26) | Matches `IDEA.md`? |
|---|---|---|
| `ahmed-masry/ChartQA` (HF dataset) | `gpl-3.0` | yes |
| `vis-nlp/ChartQA` (GitHub) | `GPL-3.0` | yes |
| `omoured/RefChartQA` (HF dataset) | `agpl-3.0` | yes |
| `moured/RefChartQA` (GitHub) | `GPL-3.0` | yes — the card/repo split noted in `IDEA.md` §6.3 is real and still present |
| `ahmed-masry/ChartQAPro` (HF dataset) | `mit` | yes |
| `Qwen/Qwen3-VL-2B-Instruct` / `-Thinking` | `apache-2.0` | yes |
| `unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit` | `apache-2.0` | yes |
| `xhguo7/CURV` | `Apache-2.0` | yes |

The strictest obligation in the set remains RefChartQA's **AGPL-3.0**. Non-negotiable rule 7 (no dataset
content in git) and rule 8 (private repository) stand unchanged.

---

## 4. Items deferred to later phases

| Item | Why not resolved now | Resolved in |
|---|---|---|
| Does `filtered_results.jsonl` re-score to 32.83? | Needs the 2.88 GB dataset for image sizes and GT boxes | Phase 4.4 |
| The 8-row surplus in the prediction file | Needs the ids of the 500 human test rows to diff against | Phase 4.4 |
| Does RefChartQA `train` contain any ChartQA `test` question? | Needs both datasets locally | Phase 3.3 (dedup) + `tests/test_no_test_split_leakage.py` |
| `bbox` vs `bbox_2d` in the output schema | Must be decided on measured validation evidence | Phase 5.1 |

---

## 5. Phase 0 acceptance criteria

- [x] `verification/phase0.md` exists with one row per claim: claim, source URL, date checked, status, action taken.
- [x] **0.3 confirmed** — the only hard blocker. Confirmed, with the human-subset refinement recorded as F1.
- [x] Every changed/refined fact propagated into `DECISIONS.md` (entries 0001–0005).

**Phase 0 is complete. Entry gate for Phase 1 is met.**

---

## 6. F11 — CORRECTION: Qwen3-VL's visual token is **32×32 px**, not 28×28 (added 2026-08-26, after §1–§5)

This was found while writing `pyproject.toml`, by reading the model's own `config.json` rather than the paper.
It contradicts a stated fact in both `IDEA.md` and `PLAN.md`.

### The claim as written

- `IDEA.md` §5.2: *"A faithful port of Qwen's `smart_resize` … (factor 28 — patch size 14 × spatial merge 2)"*
- `IDEA.md` §17 glossary: *"For Qwen, one visual token corresponds to a **28×28 pixel** region (patch size 14, spatially merged 2×2)"*
- `PLAN.md` Appendix C: `FACTOR = 28  # Qwen: patch 14 x spatial merge 2`

### What is actually true

`https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/main/config.json`:

```json
"vision_config": { "patch_size": 16, "spatial_merge_size": 2, "depth": 24, "hidden_size": 1024 }
```

`https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/main/preprocessor_config.json`:

```json
{ "image_processor_type": "Qwen2VLImageProcessorFast", "patch_size": 16, "merge_size": 2,
  "size": { "longest_edge": 16777216, "shortest_edge": 65536 } }
```

`transformers/models/qwen2_vl/image_processing_qwen2_vl.py` (the class this model uses):

- line 226: `factor=patch_size * merge_size`
- `resize()`: `min_pixels=size.shortest_edge`, `max_pixels=size.longest_edge`
- `patchify()`: `grid_h, grid_w = resized_height // patch_size, resized_width // patch_size`, then merged 2×2

Therefore for **Qwen3-VL-2B**: `factor = 16 × 2 = 32`, `min_pixels = 65,536`, `max_pixels = 16,777,216`.
**One visual token covers 32×32 pixels, not 28×28.** 28 is the Qwen2-VL / Qwen2.5-VL value (patch 14 × merge 2)
and was carried over unchanged.

`PLAN.md` Appendix C is additionally wrong on the pixel bounds (`min_pixels=4*28*28=3,136`,
`max_pixels=16384*28*28=12,845,056`), and places the `max(factor, …)` guard on the initial rounding rather
than inside the downscale branch, where `transformers` puts it. Ours should be a port of what actually runs.

### Measured impact

Re-derivation of `IDEA.md` §5.2 on **800 RefChartQA validation images / 1,045 boxes**
(validation, not test — rule 1). Script: `scripts/measure_subtoken.py`, `scripts/measure_resolution_ladder.py`.
Image sizes are dominated by 800×557 (599/800). "sub-token (axis)" = box smaller than one token on at least one
axis; "(area)" = box area below one token's area.

| Configuration | median visual tokens | sub-token (axis) | sub-token (area) |
|---|---:|---:|---:|
| A — `PLAN.md` App. C assumption: f=28, min 3,136, max 12,845,056 | **580** | 35.1% | 18.9% |
| B — **real Qwen3-VL-2B**: f=32, min 65,536, max 16,777,216 | **425** | **41.3%** | **20.6%** |
| C — f=28 with real bounds (isolates the factor change) | 580 | 35.1% | 18.9% |
| D — **real f=32 at the planned 512-px budget** | **247** | **53.2%** | **24.7%** |

Configuration A reproduces `IDEA.md`'s *other* preprocessing figure exactly — its resolution note quotes
"median visual tokens from about **580**", and A gives **580**. That is strong evidence this reconstruction of
their method is faithful and that the only substantive difference is the factor. (`IDEA.md`'s headline 23.9%
sits between our 18.9% area rule and 35.1% axis rule; their sample was 787 boxes of unstated split, ours is
1,045 validation boxes, so exact agreement was not expected.)

### Resolution ladder (same sample, f=32)

| `max_pixels` cap | med. tokens | sub-token (axis) | | longest-edge resize | med. tokens | sub-token (axis) |
|---|---:|---:|---|---|---:|---:|
| 448² | 176 | 60.5% | | 448 | 140 | 65.0% |
| **512²** (planned) | **247** | **53.2%** | | 512 | 176 | 60.4% |
| 640² | 368 | 43.1% | | 640 | 280 | 48.3% |
| 768² | 425 | 41.3% | | 768 | 408 | 42.0% |
| 896² / 1024² | 425 | 41.3% | | 896 | 532 | 38.0% |
| native | 425 | 41.3% | | 1024 | 704 | 28.6% |

### Why this matters

1. **The sub-token problem is worse than stated, not better.** A larger token means more targets fall below it.
   At the planned 512-px setting, **53.2%** of validation targets are sub-token on at least one axis. This
   *strengthens* the project's central mechanistic claim — chart grounding is hard because targets are smaller
   than the model's smallest visual unit — but the honest number is roughly double the one in `IDEA.md`.
2. **Every "one visual token" boundary must move from 28 to 32.** That includes the Phase 4.5 stratification
   bucket, which is the axis the report's main grounding story is told along.
3. **`IDEA.md`'s resolution-cost fear is overstated for this model.** It warns that doubling resolution takes
   median tokens ~580 → ~2,280 (≈4×). In fact these charts are only ~800 px wide, so `max_pixels` caps at or
   above 768² are already **native** — 425 tokens. Going from the planned 512 to native costs
   **247 → 425 tokens (1.7×)** and cuts sub-token targets from 53.2% to 41.3%. That is a large grounding gain
   for a small, bounded memory cost, and it was invisible under the factor-28 assumption.

### Proposed change (not applied unilaterally)

- Set `FACTOR = 32` and read `patch_size`/`merge_size`/`shortest_edge`/`longest_edge` from the loaded
  processor config rather than hard-coding any of them, so a backbone switch cannot silently reintroduce this
  bug. Port `smart_resize` from `transformers`, not from Appendix C, and unit-test the two against each other.
- Correct the 23.9% figure to a measured number in the report, with the definition stated.
- **Add input resolution as a measured variable in the Phase 2 smoke test** (512² vs native/768²), so the base
  resolution is chosen on measured memory evidence *before* pre-registration. This stays inside Phase 2's
  existing remit — its declared fallback ladder already treats image size as a tunable lever
  ("image 512 → 448") — and it does not touch any test split.

Recorded as `DECISIONS.md` entry 0008. Raised with Ahmed on 2026-08-26 before proceeding past Phase 1.

### F11 addendum — the source of the error, confirmed by loading both processors

Added 2026-08-26 during Phase 2, with `transformers` 5.16.0 installed. Both processors were loaded
and their geometry read by `VisualGeometry.from_processor`:

```
Qwen/Qwen3-VL-2B-Instruct
   patch=16 merge=2 -> factor=32  min_pixels=65,536     max_pixels=16,777,216
   800x557 -> (544, 800) = 425 visual tokens

Qwen/Qwen2.5-VL-3B-Instruct
   patch=14 merge=2 -> factor=28  min_pixels=3,136      max_pixels=12,845,056
   800x557 -> (560, 812) = 580 visual tokens
```

**`PLAN.md` Appendix C is Qwen2.5-VL's geometry, not Qwen3-VL's — every constant matches exactly.**
Appendix C hard-codes `FACTOR = 28`, `min_pixels = 4*28*28 = 3,136` and
`max_pixels = 16384*28*28 = 12,845,056`. Those are, to the digit, Qwen2.5-VL-3B's real values as
printed above. And Qwen2.5-VL-3B's median token count on these images is **580** — precisely the
figure `IDEA.md` quotes.

So this was not a typo in one constant. The appendix was written against the fallback model's
preprocessing and applied to the primary model. Three independent numbers agreeing to the digit
rules out coincidence.

**Two consequences worth noting.**

1. It vindicates deriving the geometry from the loaded processor rather than hard-coding it
   (decision 0008). If Phase 2 falls back to Qwen2.5-VL-3B, the factor becomes 28 automatically and
   correctly, with no code change — and the 580-token figure becomes the right one to quote.
2. Our `smart_resize` port is now cross-checked against
   `transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize` across 10 image shapes at
   both factors, and matches on every one (`tests/test_coords.py::test_our_smart_resize_matches_transformers`,
   marked `official`, runs in CI).


---

## 7. Phase 0.4 completed properly — the evaluator was executed, not just located

Added 2026-08-26. The original 0.4 check confirmed `evaluate.py` exists and read it. That is not the
same as confirming it runs, and reading it is how `DECISIONS.md` 0003 reached a conclusion whose
*reasoning* later proved wrong (see 0014). The evaluator has now been executed against synthetic
predictions whose correct answers are known by construction.

Reproduce with `python scripts/characterise_official_evaluator.py`; pinned by
`tests/test_official_evaluator_contract.py` (25 tests, `official` marker, runs in CI).

**Confirmed by execution:**

| Behaviour | Result |
|---|---|
| A coordinate of exactly 1000 | discards the **whole box**, silently — decision 0004 validated |
| Ground-truth boxes | also capped at 999, so clamping matches the GT convention exactly |
| `relaxed_accuracy("0.0", "0")` | **False** — the predicted divergence from Appendix D, now measured |
| `relaxed_accuracy("1234", "1,234")` | **False** — official does not strip commas |
| `relaxed_accuracy("0.5", "50%")` | **True** — percent divided by 100 on both sides |
| `relaxed_accuracy("Yes.", "Yes")` | **False** — no punctuation normalisation |
| Answer template | requires **exactly one** `<grounding-sep>`; anything else scores 0 |

**The finding that was not predictable from reading (decision 0014):**

AP@0.5 equals `1 / (rank of the first correct box)` — so a single wrong box placed ahead of a correct
one halves the score, and P@F1 goes to zero. And extra boxes behave in opposite directions at the two
scales:

| | one image | twenty images |
|---|---:|---:|
| correct box only | 1.0000 | 1.0000 |
| correct box + 3 extras | 1.0000 | **0.3243** |

`compute_AP_50` calls `metric.update()` per item and `metric.compute()` once, so all predictions
land in a single precision–recall curve; because every score is tied at 1.0, extras cannot be ranked
away and simply depress precision globally.

`PLAN.md` describes P@F1 as requiring "the **full** predicted grounding set to be correct". Measured,
that is not what it does: trailing false positives leave it at 1.0000, and only a false positive
*before* a true one breaks it.

**Consequence:** the output schema's `maxItems: 8` on `evidence` is a hazard, not a generous
allowance. A model that helpfully lists eight plausible regions would score near zero on the headline
grounding metric while appearing thorough. See decision 0014.

---

## 8. Audit of the code `PLAN.md` supplies verbatim

Added 2026-08-26. `PLAN.md`'s code policy is: *"Where a subtle mistake would silently corrupt
results, this document gives you the exact code — copy it."* Since Appendix C turned out to contain
a serious error (F11), the other three code appendices were audited the same way — by executing
them — before the phases that depend on them begin.

| Appendix | What it supplies | Verdict |
|---|---|---|
| **A** — output schema | strict JSON Schema for the record | **Sound.** Valid Draft 2020-12; `IDEA.md` §1's worked example validates against it; what it accepts matches its own documented "validation rules beyond the schema" exactly. |
| **B** — executor | typed-tree interpreter | **Bug found.** A bare string argument means "evidence label" in `argmin`/`argmax`/`check_units` and "numeric literal" in `sum`/`mean`/`difference`/`ratio`. See `DECISIONS.md` 0016. |
| **C** — coordinates | `smart_resize`, box maths | **Bug found** (F11): factor 28 is Qwen2.5-VL's, not Qwen3-VL's. See `DECISIONS.md` 0008. |
| **D** — metrics | relaxed accuracy, IoU, AP, P@F1, bootstrap | **Sound.** AP tracks the official evaluator to within 0.007 across six scenarios — the expected gap between COCO 101-point and all-point interpolation. Fit for stratified analysis, as the plan intends. |
| **E** — plan mining | uniqueness rule | **Sound.** Verified below. |

### Appendix B — measured

Evidence `2019=245, 2018=210`:

| plan | result |
|---|---:|
| `argmax(["2019","2018"])` | `"2019"` (strings treated as **labels**) |
| `mean(["2019","2018"])` | **2018.5** (strings treated as **numbers**) |
| `mean(lookup("2019"), lookup("2018"))` | 227.5 (the intended answer) |

The failure profile is the dangerous one: a *non-numeric* label raises loudly
(`sum(["a","b"])` → `ExecutorError`), while a *numeric-looking* label — years, counts, quantities,
i.e. what chart categories overwhelmingly are — silently returns a plausible wrong number.

Depth accounting was checked at the same time and is correct: depth 4 executes, depth 5 raises.

### Appendix D — measured

AP@0.5 over 20 synthetic images, Appendix D versus the official evaluator:

| scenario | official | Appendix D | delta |
|---|---:|---:|---:|
| all correct, one box each | 1.0000 | 1.0000 | 0.0000 |
| all correct + 3 extras each | 0.3243 | 0.3176 | −0.0067 |
| all correct, one wrong first | 0.5000 | 0.5000 | 0.0000 |
| 60% correct only | 0.5941 | 0.6000 | +0.0059 |
| 60% correct + extras | 0.2179 | 0.2127 | −0.0051 |
| all wrong | 0.0000 | 0.0000 | 0.0000 |

Maximum divergence 0.0067. Small, structural, and in both directions — so stratified figures from
Appendix D and headline figures from the official evaluator will not be exactly consistent, and the
report says so rather than presenting them as one number.

### Appendix E — measured

| case | result |
|---|---|
| `IDEA.md` §4 worked example (2018=10, 2019=20, answer 10) | ops `{difference, lookup, min}` → **ambiguous, rejected** ✓ |
| clean case (2018=210, 2019=245, answer 35) | ops `{difference}` → **unique** ✓ |
| all-zero corrupt table, answer 0 | 9 ops match → **ambiguous, rejected** ✓ |

The last row is the important one: it confirms `IDEA.md` §5.3's claim that *"the uniqueness filter
already absorbs most of this: a corrupted table simply fails to produce any plan"*. Measured, not
assumed.

Minor note: `IDEA.md` §4 illustrates the ambiguity with `{difference, lookup, ratio, mean}`. The
actual enumeration yields `{difference, lookup, min}` — `ratio` and `mean` would need operands the
table does not contain. The illustration is loose; its conclusion is right.


### Appendix A — measured

The schema parses as valid JSON Schema Draft 2020-12, and the worked example in `IDEA.md` §1
validates against it unchanged.

What it **rejects** on its own: a coordinate above 1000, a ninth evidence item, an unknown operation,
any extra top-level key (`additionalProperties: false`), a missing required field.

What it **accepts**, which the "validation rules beyond the schema" must therefore catch:

| accepted by schema | must be caught by |
|---|---|
| inverted box (`x2 < x1`) | explicit rule: `x1 < x2` and `y1 < y2` |
| zero-area box | explicit rule: positive area |
| a coordinate of exactly 1000 | clamp to 999 before scoring (`DECISIONS.md` 0004) |
| plan depth 6 | computed depth ≤ 4 — "compute it; do not trust the model" |
| `lookup` of a label absent from `evidence` | the executor, which raises and is counted (rule 4) |
| **8 evidence items** | **`DECISIONS.md` 0014** — filter before scoring; at dataset scale this is close to a zero |

That list is exactly the plan's own stated extra rules, plus the two hazards found by measurement in
this project. So Appendix A is internally consistent; it simply delegates more than it looks.

---

## 9. Image geometry measured **separately by question subset**

Added 2026-08-26. The earlier sub-token measurement (F11) sampled 800 consecutive validation rows,
which turned out to be dominated by one subset. Since the published 32.83 target is the **human**
subset alone, its geometry is what the headline comparison depends on — not a blend.

Measured across 1,000 validation rows sampled at ten offsets spanning the split
(`scripts/` ad-hoc; reproducible from the datasets-server rows endpoint):

| subset | n | median W×H | already below 512² | median visual tokens | sub-token native | @512² | @448² |
|---|---:|---|---:|---:|---:|---:|---:|
| human | 392 | 800×557 | 6% | 425 | **40.6%** | **49.1%** | 55.6% |
| machine | 108 | 800×557 | 0% | 425 | **61.1%** | **73.1%** | 84.3% |
| pot | 500 | 800×557 | 7% | 425 | 39.8% | 49.1% | 64.2% |

Modal image size is **800×557 in all three subsets**, so the blended figure in F11 was not
misleading about size. Two things it did obscure:

1. **The machine subset is far harder to ground.** At the planned 512-pixel budget, **73.1%** of its
   targets are sub-token, against 49.1% for human. Its boxes are systematically smaller. Any
   cross-subset comparison of grounding scores must account for this rather than reading it as the
   model being worse at machine-generated questions.
2. **A small minority of images are already below the budget** (6–7% for human and PoT). For those,
   raising the pixel cap changes nothing at all, because no downscaling was happening. The
   resolution ablation's effect is therefore concentrated on the ~93% that are large enough to be
   downscaled, and the report should say so rather than implying a uniform treatment.

This also confirms `SOURCE_IMAGE_W × SOURCE_IMAGE_H = 800×557` as the right synthetic size for the
smoke test: it is the modal real size in every subset, so the benchmark exercises production shapes.
