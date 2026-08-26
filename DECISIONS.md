# Decision log

Append-only. One entry per decision, in the format of `PLAN.md` Appendix H. Entries are never edited after
they are written — if a decision changes, a new entry is added that names the entry it supersedes.

---

## 0001 — 2026-08-26 — Phase 0 re-verification: all ten claims confirmed, proceed to Phase 1

**Context.** `PLAN.md` Phase 0 requires independent re-verification of the ten load-bearing facts in `IDEA.md`
before any code is written, because four weeks of work must not rest on a number that moved. `IDEA.md` was
written 2026-08-24/25/26; this check ran 2026-08-26.

**Options.** (a) Trust `IDEA.md` and start building. (b) Re-verify everything against primary sources.
(c) Re-verify only the hard blocker 0.3.

**Decision.** (b). Full re-verification against structured APIs (HF Hub API, HF datasets-server, GitHub API)
and raw source files rather than rendered cards, plus direct extraction from the Qwen3-VL PDF.

**Evidence.** All ten claims confirmed — see `verification/phase0.md` for the per-claim table. Exact matches
included ChartQA `28299/1920/2500`, RefChartQA `55789/6223/11690`, Qwen3-VL-2B ChartQA `86.6` thinking /
`79.1` instruct (technical report Table 4), base BF16 checkpoint `4,255,140,312` bytes, Unsloth 4-bit
`2,406,693,586` bytes, ChartQAPro `1,948` rows / `193,053,989` bytes. Ten new findings (F1–F10) recorded that
were not in `IDEA.md`.

**Consequences.** Phase 1 entry gate is met. No claim in `IDEA.md` needs retracting. Five of the ten new
findings change what gets built and are recorded as separate entries below.

---

## 0002 — 2026-08-26 — The 32.83 grounding target is the RefChartQA **human subset**; all grounding results will be reported per-subset

**Context.** `IDEA.md` treats "32.83 AP@0.5" as *the* published RefChartQA target without stating which
partition of the test set it refers to. Reading the official evaluator and the paper's Table 2 shows the
benchmark has no aggregate number at all.

**Options.** (a) Report one aggregate AP@0.5 over the full 11,690-row test set and compare it to 32.83.
(b) Report the three subsets separately, as the official evaluator does, and compare only human-to-human.
(c) Report both an aggregate and the subsets.

**Decision.** (b), with (c) as a labelled convenience: the primary reported figures are the H / M / PoT triple
produced by the official evaluator, and the comparison against 32.83 is stated explicitly as being on
**RefChartQA-H only**. Any aggregate we compute is a separately-named diagnostic and is never compared to 32.83.

**Evidence.** `evaluate.py::load_datasets_by_source` partitions on the `type` column into `human`/`machine`/`pot`
and `evaluate_all_datasets` scores each independently; there is no combined path. Paper Table 2, Qwen2.5-VL
(3B, native resolution): AP@0.5 = **32.83 (H) / 59.28 (M) / 39.32 (PoT)**. `IDEA.md`'s stated competitor range
"18.30–27.81" is precisely the human column (UniChart 18.30 … TinyChart 27.81), confirming the human subset was
the intended referent all along.

**Consequences.** Makes the headline comparison genuinely matched, which non-negotiable rule 6 requires.
Costs statistical power: the released prediction file contains only **508** human test rows, so bootstrap CIs
on the headline grounding comparison will be wide and must be reported as such rather than quietly omitted.
Also means the project has three grounding results to move, not one — the machine and PoT subsets are extra
evidence at no extra inference cost, since the evaluator computes them in the same pass.

---

## 0003 — 2026-08-26 — The official evaluators are vendored verbatim as the scorers of record; `PLAN.md` Appendix D becomes a labelled diagnostic

**Context.** Reading `evaluate.py` in full revealed that the official metric implementations differ from the
reference implementations given in `PLAN.md` Appendix D in ways that change scores, not just rounding.

**Options.** (a) Use Appendix D as primary because it is cleaner and its zero-guard is more defensible.
(b) Use the official implementation as primary and keep Appendix D as a diagnostic. (c) Average them.

**Decision.** (b). (c) is forbidden outright by non-negotiable rule 5. Appendix D is retained for stratified
analysis (AP by box area), exactly as `PLAN.md` already scopes it, and every divergence is reported as a
per-question disagreement count rather than smoothed over.

**Evidence.** Three concrete divergences in `relaxed_accuracy`:
1. official does **not** strip commas — `_to_float("1,234")` returns `None` and falls back to string equality;
2. official does **not** strip whitespace;
3. official tests `if prediction_float is not None and target_float:` — a **truthiness** test, so a target of
   `0.0` is falsy and the numeric branch is skipped entirely. Official and Appendix D therefore disagree on
   `(target="0", pred="0.0")`: official says incorrect, Appendix D says correct.

And one structural divergence in AP@0.5: the official `compute_AP_50` uses
`torchmetrics.MeanAveragePrecision(iou_thresholds=[0.5])` with **every predicted box assigned `score = 1.0`**,
so there is no confidence ranking and torchmetrics applies COCO 101-point interpolation; Appendix D sorts by
score and uses all-point interpolation. These cannot agree in general.

**Consequences.** Phase 4.2's cross-check will show disagreements — that is now expected and pre-declared,
not a bug to be chased. `PLAN.md` Phase 4.2 already says "the official one wins", so this is compliance, not
deviation. Second-order consequence worth acting on: because all prediction scores are constant, emitting
extra speculative boxes can only reduce precision, so the model should emit the evidence it believes rather
than a ranked candidate list.

---

## 0004 — 2026-08-26 — Everything emitted for official grounding scoring is clamped to integer `[0, 999]`

**Context.** Qwen3-VL natively emits normalised coordinates in `[0, 1000]` **inclusive**. The official
evaluator accepts a box only when every coordinate satisfies `0 <= v <= 999`.

**Options.** (a) Emit the model's raw range and accept the loss. (b) Clamp to `[0, 999]` in the serialisation
layer that feeds the official evaluator. (c) Change the model's target range during training to `[0, 999]`.

**Decision.** (b) now, unconditionally, as a property of the scoring adapter — with a unit test asserting that
a coordinate of `1000` survives the round trip as `999`. (c) is deferred: training targets will be authored in
`[0, 999]` too, so training and scoring share one convention, but the clamp stays in place regardless as a
defensive guard.

**Evidence.** `extract_bounding_boxes()` in `evaluate.py`:
`if all(0 <= elem <= bins - 1 for elem in bbox_floats): bboxes.append(bbox_floats)` with `bins=1000`.
There is no `else` — a box containing a `1000` is **silently discarded**, with no error and no warning. The
ground-truth path independently applies `min(int(value * bins), bins - 1)`, so GT is already capped at 999.

**Consequences.** Removes an entire class of silent AP loss that would have been extremely hard to diagnose
after the fact — a right-edge or bottom-edge box is exactly the case that produces a 1000, and edge-touching
boxes are common in charts. Costs at most one unit of coordinate resolution out of a thousand.

---

## 0005 — 2026-08-26 — ChartQA gold tables come from `ChartQA Dataset.zip`, not the HF parquet

**Context.** Appendix E plan mining requires the gold data tables. The obvious `load_dataset("ahmed-masry/ChartQA")`
path does not expose them.

**Options.** (a) `load_dataset()` and reconstruct tables some other way. (b) Download the archive file that
contains them. (c) Use the `vis-nlp/ChartQA` GitHub repository.

**Decision.** (b) — `hf_hub_download(repo_id="ahmed-masry/ChartQA", filename="ChartQA Dataset.zip", repo_type="dataset")`.

**Evidence.** The parquet/viewer schema is only `imgname, query, label, type, image` — no tables, no
annotations. The same HF repo contains `ChartQA Dataset.zip` at **875,370,872 bytes**, which is exactly the
"~875 MB full archive" figure `IDEA.md` §6.1 quotes; the repo README documents that this archive carries the
tables and the (noisy) chart-element annotations. Using the HF repo rather than GitHub keeps every download on
one host with one auth path and one hash manifest.

**Consequences.** Phase 3.1 fetches one 875 MB file plus the RefChartQA parquet, rather than mixing
`load_dataset()` and file downloads. Plan mining becomes possible at all — without this it would silently have
had no tables to mine.

---

## 0006 — 2026-08-26 — OPEN: `bbox` vs `bbox_2d` in the output schema, to be settled on validation evidence in Phase 5.1

**Context.** `PLAN.md` Appendix A specifies the evidence-box field as `bbox`. Qwen3-VL was pretrained to emit
`bbox_2d`. The zero-shot number produced under our prompt is the "before" baseline the entire project is
measured against, so a gratuitous mismatch with pretraining could understate it.

**Options.** (a) Keep `bbox` as specified. (b) Switch the schema to `bbox_2d` to match pretraining.
(c) Measure both on validation during Phase 5.1 prompt design and commit to the better one before
pre-registration.

**Decision.** (c). Not settled now — recorded here so the question cannot be quietly forgotten, and so that
whichever way it goes it is a measured choice rather than an accident. Whatever wins is frozen in
`PREREGISTRATION.md` before any test split is opened.

**Evidence.** Qwen3-VL technical report §3: *"Different from Qwen2.5-VL, we adopt a normalized coordinate
system scaled to the range [0, 1000]"*. Official 2D-grounding cookbook emits
`{"bbox_2d": [x1, y1, x2, y2], "label": "..."}` and rescales with `coord / 1000 * width`. No measurement of
the two field names against each other exists yet.

**Consequences.** Deferring is safe because nothing before Phase 5 depends on the field name, and the
generator/dataset builders will be written to take the key as a single configurable constant so switching it
is a one-line change rather than a rewrite.

---

## 0007 — 2026-08-26 — The headline grounding comparison has n=500, and that is pre-declared as the project's tightest statistical constraint

**Context.** Entry 0002 fixed the headline grounding comparison to the RefChartQA **human** subset. That entry
cited 508 rows, taken from the released prediction file. The live test split was then measured directly to get
the real figure.

**Options.** (a) Report the human-subset point estimate and move on. (b) Pre-declare the sample size and the
resulting resolution limit now, before any result exists, so the size of a "win" cannot be rationalised after
the fact. (c) Broaden the headline to the full test set to gain power.

**Decision.** (b). (c) is ruled out by entry 0002 — a broader set is not comparable to 32.83. The
pre-registration will state the sample size and the minimum gap that is distinguishable from noise, and the
report will carry bootstrap CIs on the point estimate rather than the point estimate alone.

**Evidence.** Measured against the live dataset via the HF datasets-server filter endpoint, RefChartQA test
composition by `type` is **human = 500**, **machine = 1,032**, **pot = 10,158** (sum 11,690 ✓). The released
prediction file holds 508 / 1,032 / 10,158 = 11,698, so its 8-row surplus is entirely in the human subset and
is discarded by the evaluator's left join. At n=500 the 95% bootstrap CI on a score in the low 30s is on the
order of ±4 points.

**Consequences.** Sets an honest expectation: a 2-point improvement on RefChartQA-H is not a result, it is
noise, and no amount of seed averaging changes that because the seeds share the same 500 items. Three
mitigations follow, all cheap and all pre-declared here rather than discovered later:
1. The machine (1,032) and PoT (10,158) subsets are reported alongside and have far more power — they cannot
   be compared to 32.83, but they can be compared to our own matched zero-shot baseline, which is the
   mandatory result anyway.
2. Area-stratified AP within the human subset splits 500 rows further and will be reported with explicit
   per-bucket counts, because a bucket of 40 items carries essentially no information and must not be
   presented as if it does.
3. The validation split (6,223 rows) carries the tuning decisions, so no test-set power is spent on choices.

---

## 0008 — 2026-08-26 — CORRECTION: the visual-token factor is 32, not 28; all token geometry is read from the processor config

**Context.** `IDEA.md` §5.2 and §17, and `PLAN.md` Appendix C, all state that one Qwen visual token covers
28×28 pixels (patch 14 × spatial merge 2), and Appendix C hard-codes `FACTOR = 28`. Reading
`Qwen/Qwen3-VL-2B-Instruct`'s own `config.json` while writing `pyproject.toml` shows `patch_size: 16`,
`spatial_merge_size: 2`. 28 is the Qwen2-VL / Qwen2.5-VL value, carried over to a model that changed it.

**Options.** (a) Keep `FACTOR = 28` as written. (b) Hard-code `FACTOR = 32`. (c) Derive the factor and the
pixel bounds from the loaded processor config at runtime, hard-coding nothing, and port `smart_resize` from
`transformers` rather than from Appendix C.

**Decision.** (c). `FACTOR = patch_size * merge_size` and `min_pixels`/`max_pixels` = the processor's
`size.shortest_edge`/`size.longest_edge`, all read from the model that is actually loaded, with a unit test
asserting our port matches `transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize` across a grid
of shapes. (b) is rejected for the same reason (a) failed: a hard-coded constant is exactly what breaks
silently when the backbone changes, and this project has a live fallback ladder that may change the backbone
in Phase 2.

**Evidence.**
- `config.json` → `vision_config: {patch_size: 16, spatial_merge_size: 2}`.
- `preprocessor_config.json` → `{patch_size: 16, merge_size: 2, size: {longest_edge: 16777216, shortest_edge: 65536}}`.
- `transformers` `image_processing_qwen2_vl.py`: `factor=patch_size * merge_size` (line 226);
  `min_pixels=size.shortest_edge`, `max_pixels=size.longest_edge`; `grid_h = resized_height // patch_size`
  then merged 2×2. So factor = 16 × 2 = **32**, and one visual token = 32×32 px.
- Appendix C's pixel bounds (`3,136` / `12,845,056`) are also wrong for this model (`65,536` / `16,777,216`),
  and it places the `max(factor, …)` guard on the initial rounding rather than in the downscale branch.

Measured on 800 RefChartQA **validation** images / 1,045 boxes (`scripts/measure_subtoken.py`):

| Configuration | med. tokens | sub-token (axis) | sub-token (area) |
|---|---:|---:|---:|
| Appendix C assumption (f=28) | **580** | 35.1% | 18.9% |
| Real Qwen3-VL-2B (f=32), native | 425 | 41.3% | 20.6% |
| Real (f=32) at the planned **512-px** budget | 247 | **53.2%** | 24.7% |

The factor-28 configuration reproduces `IDEA.md`'s independently-quoted "median visual tokens ≈ **580**"
exactly, which corroborates that the reconstruction is faithful and the factor is the substantive difference.

**Consequences.**
- The sub-token headline gets *worse*, not better: at the planned 512-px setting, **53.2%** of validation
  targets are smaller than one visual token on at least one axis, against `IDEA.md`'s 23.9%. This strengthens
  the project's central mechanistic claim while making its stated number wrong; the report must carry the
  measured figure with its definition, not the inherited one.
- Every "one visual token" boundary moves from 28 to 32 — including the Phase 4.5 stratification bucket, which
  is the axis the grounding story is told along.
- `IDEA.md`'s fear that raising resolution quadruples the token budget (580 → 2,280) does not hold for these
  images: RefChartQA charts are ~800 px wide, so a `max_pixels` cap of 768² or more is already native. Moving
  from the planned 512 to native costs **247 → 425** median tokens (1.7×) and cuts sub-token targets
  53.2% → 41.3%.
- Follows that input resolution should be a *measured* variable in the Phase 2 smoke test rather than a
  constant inherited from the plan. Proposed to Ahmed rather than applied unilaterally; it stays within
  Phase 2's remit, since its declared fallback ladder already treats image size as a tunable lever.

**Supersedes.** Nothing. This is the first correction to a stated fact in `IDEA.md`.

---

## 0009 — 2026-08-26 — Vendored evaluators are hash-pinned and excluded from every automated rewrite

**Context.** Within minutes of the first commit, git silently corrupted the vendored official
RefChartQA evaluator. `.gitattributes` carried the conventional `* text=auto eol=lf`; upstream's
`evaluate.py` uses CRLF line endings; git normalised them on commit. The working copy hashed to
`d0c9f87d…` (matching upstream) while the blob git actually stored hashed to `5ab767f5…`.

**Options.** (a) Accept it — it is only whitespace. (b) Mark vendored paths `-text` so git never
touches them. (c) (b) plus a recorded hash and a test that fails on any drift.

**Decision.** (c). Vendor directories are marked `-text` in `.gitattributes`, every vendored file's
SHA-256 and byte count are recorded in a `PROVENANCE.json` alongside it, and
`tests/test_vendored_integrity.py` fails the build if any byte changes. Vendored paths are also
excluded from `ruff` (the official evaluator has 66 lint violations; fixing them is precisely the
wrong thing to do).

**Evidence.** Working copy `sha256 d0c9f87d68d999da7963ea655935a7140fc35f245ad2c26c53e28e4f651c0dd8`,
matching a fresh download from
`https://raw.githubusercontent.com/moured/RefChartQA/main/evaluation/evaluate.py`. Committed blob
before the fix: `5ab767f5fbd493b98c6e3229fba80db4bfd8d3f4dca004f71e5aabdd859ede0f`. 374 CRLF line
endings in the file. After the fix the committed blob hashes to `d0c9f87d…` again.

**Consequences.** Decision 0003 makes the official evaluator the scorer of record — "we ran the
official evaluator" is only a true statement if the bytes are the official ones, and a
whitespace-only diff is still a diff that would have to be defended in a report. Option (a) was
tempting and wrong for exactly the reason this project keeps running into: the corruption was
silent, automatic, and produced no error. Generalises to a standing rule — **anything vendored is
excluded from every automated rewrite (line endings, formatters, linters, import sorters) and
pinned by hash.**

---

## 0010 — 2026-08-26 — Input resolution becomes a measured variable in the Phase 2 smoke test

**Context.** Decision 0008 established that the visual-token factor is 32, not the 28 assumed in
`IDEA.md`/`PLAN.md`, and that `IDEA.md`'s stated cost of raising resolution (median tokens 580 → 2,280,
roughly 4×) was computed under the wrong factor and does not hold for these images. The plan fixes the
input budget at 512 px. That number was chosen using the incorrect analysis.

**Options.** (a) Keep 512 px as written and revisit only in the Phase 8.3 resolution study, which is
gated behind two other extensions and therefore lands after training is already done. (b) Measure peak
memory and step time at both 512² and native (768² cap) during the Phase 2 smoke test, and choose the
base resolution on that evidence before pre-registration. (c) Switch to native now without measuring.

**Decision.** (b). Approved by Ahmed on 2026-08-26. The smoke test already measures peak reserved
memory and seconds per optimizer step; it now does so at two input budgets instead of one. Whichever
wins is frozen in `PREREGISTRATION.md` before any test split is opened. (c) is rejected — the whole
problem with 512 was that it was chosen without measurement, and choosing native the same way repeats
the error in the other direction.

**Evidence.** Measured on 800 RefChartQA **validation** images / 1,045 boxes
(`scripts/measure_resolution_ladder.py`):

| Input budget | median visual tokens | targets sub-token (either axis) |
|---|---:|---:|
| 448² | 176 | 60.5% |
| **512²** (planned) | **247** | **53.2%** |
| 640² | 368 | 43.1% |
| 768² / native | **425** | **41.3%** |

Native costs 1.7× the tokens of 512², not 4×, and recovers ~12 percentage points of targets from the
sub-token stratum. The saturation at 768² is not a coincidence: these charts are only ~800 px wide, so
any cap at or above 768² is already native and there is nothing further to gain without upscaling.

**Consequences.** Stays inside Phase 2's remit — its declared fallback ladder already treats image size
as a tunable lever ("image 512 → 448"), so resolution was always a Phase 2 variable; this makes the
choice measured rather than inherited. Touches no test split. Costs one extra configuration in a smoke
test already being run. If native breaches the 13.5 GB memory gate, the Phase 2 fallback ladder applies
unchanged and 512 is used — in which case we will have paid one measurement to confirm the plan was
right, which is a good trade.

---

## 0011 — 2026-08-26 — Credentials are verified by a script with a negative control, never by "the command worked"

**Context.** Setting up the three credentials produced a wrong diagnosis that survived two rounds of
testing. Kaggle's `GET /api/v1/datasets/list` returns HTTP 200 to unauthenticated requests. Because
`dataset_list(mine=True)` returned 200 while `kernels_list` and `competitions_list` returned 401, the
evidence appeared to show a valid token missing a "kernels" scope. It showed nothing of the kind: the
token was being rejected everywhere, and the 200 was simply an endpoint that never checks.

**Options.** (a) Test credentials ad hoc when something breaks. (b) A checked-in script that probes only
endpoints requiring authentication. (c) (b) plus a mandatory negative control — the same call with a
junk token, which must be rejected, or the check is declared meaningless.

**Decision.** (c), as `scripts/check_credentials.py`, with the setup gotchas written up in `SETUP.md`.
The control runs *after* the real check, because Kaggle throttles a client that has just presented bad
credentials and that throttle would otherwise corrupt the real result.

**Evidence.** Verbatim probes. `datasets/list` with **no** `Authorization` header → **200, 82,233 bytes**;
with deliberately invalid credentials → **200, 82,233 bytes**; with the real token → **200, 82,233 bytes**.
Three different credential states, byte-identical responses. Root cause of the actual failure: Kaggle now
issues `KGAT_`-prefixed **bearer** access tokens belonging in `~/.kaggle/access_token` or
`KAGGLE_API_TOKEN`, while `kaggle.json` is for the **legacy** 32-hex key sent as HTTP Basic
`username:key`. Moving the token to the right file fixed every endpoint at once. A second false signal
came from TLS: the python.org 3.11 venv has no CA trust store, so raw `urllib` raised
`CERTIFICATE_VERIFY_FAILED`, which reads as a rejected credential. All network calls now use `requests`,
which bundles `certifi`.

**Consequences.** Generalises the Phase 0 lesson to infrastructure: **a check that cannot fail is not a
check.** The same reasoning already governs the evaluator regression suite (assert the cases that *must*
be wrong, not only the ones that should be right) and `assert_lora_on_both_sides` (fail loudly rather
than warn). Also a correction to my own reasoning worth recording: I reported "the token is missing a
scope" to Ahmed with confidence, from evidence that could not distinguish that hypothesis from "the
token is entirely rejected". The control is what separates them, and it should have been the first
thing run, not the last.

---

## 0012 — 2026-08-26 — The vision tower is kept out of 4-bit, and the skip patterns are full module paths

**Context.** QLoRA stores the frozen base weights in 4 bits. Applying that to the *visual encoder*
degrades exactly the capability this project's second headline measures, so `BitsAndBytesConfig`
is given `llm_int8_skip_modules` to hold the vision tower at higher precision. (Despite the `int8`
name, `quantizer_bnb_4bit.py` reads the same field for the 4-bit path — verified by test.)

The first implementation passed `["visual", "vision_tower", "lm_head"]` and **did nothing at all**
for the vision tower, while a code comment asserted that it worked.

**Options.** (a) Quantise everything, accept the loss, save a few hundred MB. (b) Skip the vision
tower using bare module names. (c) Skip it using full module paths, with a test that pins the
matching rule against the real Qwen3-VL module names.

**Decision.** (c). `VISION_SKIP_PATTERNS` now holds full paths (`model.visual`,
`model.vision_tower`, `model.vision_model`, plus a top-level `visual` for architectures that place
it there, and `lm_head`), and `tests/test_quantisation_skip.py` asserts that every real vision
module path is excluded while the language modules are still quantised.

**Evidence.** `transformers.quantizers.quantizers_utils.should_convert_module` matches with:

```python
re.match(f"{key}\.", full_name) or re.match(f"{key}", full_name) or full_name.endswith(key)
```

`re.match` is anchored at the start of the string. Measured directly against real module names:

| pattern list | `model.visual.blocks.0.attn.qkv` | `model.language_model...q_proj` |
|---|---|---|
| `["visual", "vision_tower", "lm_head"]` | **QUANTISED** | QUANTISED |
| `["model.visual", "model.vision_tower", "lm_head"]` | kept fp16 | QUANTISED |

`"visual"` fails all three tests against `"model.visual.blocks.0.attn.qkv"`: it is not a prefix,
not a full match, and not a suffix.

**Consequences.** Costs a few hundred MB of VRAM against the 13.5 GiB gate, which the Phase 2
measurement will price exactly. Worth it: this is the difference between measuring a model whose
visual encoder is intact and one whose visual encoder was silently degraded. The general lesson is
the same one this project keeps relearning — **a configuration option that is accepted without
error is not the same as a configuration option that took effect.** The only reliable check is to
run the library's own matching function against the real names, which is what the test does.

Found while a Kaggle kernel was already running with the broken version. That run is a 3-step path
validation, not the measurement, so nothing has to be discarded — but had it been the real 100-step
run, the memory figure would have been for a fully-quantised model and would have been wrong in the
flattering direction.

---

## 0013 — 2026-08-26 — `hf_peft` loads the BF16 checkpoint and quantises it here; the two backends are therefore not memory-comparable

**Context.** Two checkpoints exist for the primary backbone: `Qwen/Qwen3-VL-2B-Instruct` (BF16,
4,255,140,312 bytes) and `unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit` (2,406,693,586 bytes,
pre-quantised). Decision 0012 requires the vision tower to stay out of 4-bit; a pre-quantised
checkpoint has already made that choice and it cannot be revisited at load time.

**Options.** (a) Both backends load the pre-quantised checkpoint — smallest download, fastest start.
(b) `hf_peft` loads BF16 and applies our own quantisation config; `unsloth` uses Unsloth's
checkpoint, which is what its loader expects. (c) Both load BF16.

**Decision.** (b). `hf_peft` downloads 4.2 GB and quantises locally with `VISION_SKIP_PATTERNS`, so
what is and is not quantised is a property of this repository and is verified at runtime by
`summarise_quantisation` on the loaded model. `unsloth` uses Unsloth's checkpoint because forcing it
onto BF16 would be testing a configuration nobody ships. (c) is rejected because it would break the
Unsloth path for no gain.

**Evidence.** Unsloth's checkpoint declares a **63-entry** `llm_int8_skip_modules` list:

| entry kind | count |
|---|---:|
| `model.visual.blocks.N.attn` / `.mlp`, N = 0..23 | 48 |
| other vision (`visual`, `vision_tower`, …) | 3 |
| non-vision (`embed_tokens`, `lm_head`, `merger`, `router`, and selected language layers such as `model.language_model.layers.3.mlp`) | 12 |

Two things follow. First, this **independently corroborates decision 0012**: Unsloth, who do this
professionally, keep the entire vision tower out of 4-bit, and they express it with **full module
paths** (`model.visual.blocks.23.attn`) — the very form that was missing from our first attempt.
Their list does also contain the bare `visual` and `vision_tower` entries, which as measured in 0012
match nothing; the full paths are what actually does the work.

Second, they additionally hold a handful of *language* layers at higher precision — dynamic
quantisation driven by their own calibration, which we neither have nor can reproduce.

**Consequences.** The two backends will carry different quantisation profiles, so **their peak-memory
figures are not directly comparable and the Phase 2 table must say so**. Comparing them as if they
were would be a like-for-like claim that is not like-for-like — the same error non-negotiable rule 6
forbids for published baselines, applied internally. What *is* comparable within each backend is the
512-versus-native resolution arm of decision 0010, since only the input budget changes there.

Also worth stating plainly: the `hf_peft` path costs a 4.2 GB download per cold session rather than
2.4 GB. On a free tier that is minutes, not money, and it buys control over the one quantisation
decision that touches the headline grounding metric.

---

## 0014 — 2026-08-26 — Grounding output is emitted **minimally and best-first**; the evidence list is filtered before scoring

**Supersedes** the closing paragraph of entry 0003, which reasoned from first principles that
"because all prediction scores are constant, emitting extra speculative boxes can only reduce
precision". The conclusion was right; the reasoning was not, and the reasoning would have led us
astray — the per-image behaviour appears to say the opposite.

**Context.** The official evaluator assigns **every** predicted box `score = 1.0`
(`torch.ones(len(bboxes))`), so there is no confidence ranking. Before Phase 4 depends on it, the
evaluator was executed directly against synthetic predictions to characterise what it rewards.

**Options.** (a) Emit every evidence box the model produces, up to the schema's `maxItems: 8`.
(b) Emit boxes minimally, ordered best-first, and filter the evidence list before submitting it to
the grounding evaluator. (c) Emit one box only.

**Decision.** (b). Concretely: the `evidence` array in the JSON record may carry several items —
the typed plan's `lookup` operands need them — but **what is submitted to the grounding evaluator is
a filtered, confidence-ordered subset**, not the whole list. The exact filter is a pre-registered
parameter fitted on validation only. (c) is rejected because questions with genuinely multiple
grounding regions would lose all recall.

**Evidence.** Measured by running the official `compute_AP_50` and `compute_P_at_FI`.

*Single image, one ground-truth box — AP equals `1 / (rank of the first correct box)`:*

| prediction | AP@0.5 | P@F1 |
|---|---:|---:|
| `[correct]` | 1.0000 | 1.0000 |
| `[correct, bad, bad, bad]` | 1.0000 | 1.0000 |
| `[bad, correct]` | 0.5000 | 0.0000 |
| `[bad, bad, correct]` | 0.3333 | 0.0000 |
| `[bad, bad, bad, correct]` | 0.2500 | 0.0000 |

*Dataset of 20 images — the same extras that were free per-image are devastating:*

| strategy | AP@0.5 | P@F1 |
|---|---:|---:|
| all correct, one box each | **1.0000** | 1.0000 |
| all correct **+ 3 extra wrong each** | **0.3243** | 1.0000 |
| all correct, one wrong box **first** | 0.5000 | 0.0000 |
| 60% correct only, 40% nothing | 0.5941 | 0.6000 |
| 60% correct **+ extras**, 40% nothing | **0.2179** | 0.6000 |

**Three rules follow, all measured rather than argued:**

1. **Every extra box is a global false positive.** `compute_AP_50` calls `metric.update()` per item
   and `metric.compute()` once, so torchmetrics pools all predictions into a single
   precision–recall curve. With scores tied at 1.0, the extra boxes from every other image
   interleave with true positives and depress precision everywhere. Three extras per image took a
   perfect 1.0000 down to 0.3243 — a 68% relative loss for boxes that looked free in isolation.
2. **Order is decisive.** One wrong box ahead of a correct one halves AP and zeroes P@F1 for that
   image. The evidence list must be emitted best-first.
3. **The two metrics disagree about extras**, and the report must not present them as measuring the
   same thing. P@F1 is computed per image and is completely insensitive to boxes appended after a
   correct one (1.0000 either way); dataset AP is ruined by them. `PLAN.md` describes P@F1 as
   requiring "the **full** predicted grounding set to be correct" — measurement shows that is not
   what it does: trailing false positives do not break it, only a false positive *before* a true
   one does.

**Consequences.** The schema's `maxItems: 8` on `evidence` is a live hazard rather than a generous
allowance: a model that helpfully lists eight plausible regions would score close to zero on the
headline grounding metric while looking thorough. Prompt design (Phase 5.1) must therefore push
toward few, ordered boxes, and the submission filter must be frozen in `PREREGISTRATION.md` before
any test split is opened. This also removes any temptation to "improve recall" by padding — under
this evaluator that is strictly self-harm.

Found by executing the evaluator on synthetic inputs rather than by reading it, which is the only
reason the per-image/aggregate divergence surfaced at all.
