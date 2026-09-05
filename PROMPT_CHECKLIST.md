# `Prompt.md`, line by line — what is done, what is not

Ahmed asked for a checklist rather than prose. Every section of `Prompt.md` is below with a
status and the evidence. **Nothing is ticked because it was read; it is ticked because
something was measured, changed, or explicitly decided against with a reason.**

| mark | means |
|:---:|---|
| ✅ | **done** — measured or implemented, with a decision record |
| 🟡 | **partly done** — the honest state is written in the row |
| ❌ | **not done** — and the row says why, and what it would take |
| ⛔ | **blocked** — needs a GPU, or an API run, or Ahmed's call |

---

## The framing sections

| § | requirement | status | evidence |
|---|---|:---:|---|
| CORE MINDSET | treat prior decisions as hypotheses, not commitments | ✅ | 0112, 0115, 0117, 0118, 0120 each overturn an earlier decision of mine |
| | don't defend the repo because you built it | ✅ | 0116 and 0118 are both "my own change was wrong" |
| | don't accept Ahmed's ideas merely because suggested | ✅ | 0113 refutes the coordinate-quantisation idea; 0121 refutes executor-replaces-answer; 0120 finds "close values" already harder than real |
| | if both are weak, find a third approach | ✅ | 0118: neither "accept the density" nor "spend compute" — the cause was a 12-item tuple |
| | if evidence is inconclusive, design an experiment | ✅ | 0121 leaves a pre-registered prediction rather than a guess |
| PRIMARY OBJECTIVE | correctness > novelty > size | ✅ | 0116 rejected 4,939 extra records because they were wrong |
| SOURCE OF TRUTH | code and measurement, not memory | ✅ | every decision here carries a number |
| PROTECT THE REPOSITORY | never commit dataset content | ✅ | rule-7 guardrail now runs in preflight too, after a week of red CI |
| | never rewrite history | ✅ | none rewritten; the CI fix was an allow-list change |

---

## TASK 1 — the 99-item audit

✅ **All 99 items covered.** Tracked individually in `AUDIT_COVERAGE.md` with honest depth
marks (**M** measured / **V** verified / **L** limitation recorded / **I** inspected) —
ticking 99 boxes uniformly would be worth nothing.

---

## TASK 2 — external research

✅ Done, with the 14-point justification record per finding. Primary sources only; the
notable negative result is that **32.83 is not in the RefChartQA paper** and our evaluator
matches the official one to 0.068 points across 11,690 predictions.

---

## The 15 specific ideas

| # | idea | status | what actually happened |
|---:|---|:---:|---|
| 1 | reconsider `ChartRecord` | ✅ | Audited (0098, 0108), the flagged assumption **fired** in 0116, and the structural change is now **done** (0124): `elements` and `evidence` are first-class fields, `evidence=None` means "unknown". Yields identical — 87.8% at rung 4,000, before and after. |
| 2 | distinguish elements from evidence | ✅ | Identity: qualified labels, not opaque ids, with the trade-off table (0107). Container: ELEMENTS and EVIDENCE separated (0124). Synthetic now keeps the whole chart instead of only the operands. |
| 3 | connect RefChartQA grounding to ChartQA elements | ✅ | `scripts/align_refchartqa.py`. 98.9% of boxes match a ChartQA element at IoU ≥ 0.9 (0077). Re-run at full scale today: **3,405 → 48,770 aligned (87.9%)** (0115). |
| 4 | reconsider the ChartQA ↔ RefChartQA merge | ✅ | Merge separated from deduplication; every consumer enumerated (0108, 0109). |
| 5 | evidence should have one clear meaning | ✅ | TABLE / ELEMENTS / EVIDENCE / PLAN / ANSWER is what the code does; the one gap (EVIDENCE not first-class) is recorded, not hidden (0108). |
| 6 | target builder | ✅ | Five gates; discard, never repair. Grounding-only targets added then correctly restricted (0116). **Duplicate values** — the brief's own item — now refuse a tied `argmax` (0127), and truncated operands rejoin (0129). |
| 7 | plan mining, two approaches | ✅ | Deterministic search retired after measuring it must refuse 53.9%; LLM mining is the single path (0088). |
| 8 | (spurious programs / execution filtering) | ✅ | Spurious-program detector measured; the check it names by name was run. |
| 9 | synthetic data | 🟡 | **Density ceiling removed (0118)** and **value distribution fitted to ChartQA (0120)**. Remaining items listed in the table below. |
| 10 | DSL + executor | ✅ | Every edge case in the brief's list tested: division by zero, NaN/Infinity, missing evidence, depth limits, `mean([])` — all handled or refused explicitly. Duplicate labels with contradictory values now refused (0128). |
| 11 | round-trip verification | ✅ | Seven notions kept distinct. Executor-replaces-answer **measured and rejected** (−6.9 points); agreement kept as a label-free confidence signal, +15.9 points (0121). |
| 12 | Qwen3-VL preprocessing | ✅ | Verified against the official processor; we do not pre-resize. No change needed. |
| 13 | model output format | ✅ | Audited; no change. |
| 14 | training objective | ✅ | Loss masking verified end to end (prompt, padding and image tokens all masked, with a `supervised` count check). Composition measured — boxes 35.6%. Quantisation tested and refuted (0113). |
| 15 | supervision provenance / confidence | ✅ | Two closed vocabularies — grounding and value — tagged **per element** by every source; kept out of the model, as the brief requires (0126). |

---

## Idea 9 in detail — every sub-item it lists

**DATA**

| item | status | note |
|---|:---:|---|
| category distributions | ✅ | 5 → 11 pools, up to 50 labels (0118) |
| value distributions | ✅ | lognormal fitted to ChartQA, mean 1.45 sd 1.33 (0120) |
| extreme values | ✅ | 21.8% of charts above 1,000, against 20.5% real |
| close values | ✅ | **no change needed** — synthetic is already harder than real (49.5% vs 36.8% within 5%) |
| decimal values | ✅ | precision follows magnitude in both directions |
| negative values | ✅ | 1.6%, against 1.7% real |
| percentages | ✅ | 7.7% of charts sum to exactly 100, against 7.4% real |
| formatting | ✅ | thousands separators now rendered; the model had never seen one |
| units | 🟡 | 6 units, not measured against ChartQA's distribution |
| correlations | ❌ | values are i.i.d. within a chart; real series trend over time |
| realistic table structure | ❌ | flat `{labels, values}`; ChartQA has multi-column tables |

**CHART TYPES**

| item | status | note |
|---|:---:|---|
| vbar, hbar, grouped bar, line, multi-line, scatter, pie, area | ✅ | all eight |
| stacked bars | ❌ | not implemented |
| area/scatter in the mixture | ✅ | dropped by selection — ChartQA contains 0.0% of both (0091) |

**STYLE**

| item | status | note |
|---|:---:|---|
| font sizes, backgrounds, gridlines, legends, tick rotation, titles, value labels, aspect ratio, DPI, colours | ✅ | all randomised in `Style.sample` |
| axis formatting | ✅ | thousands separators (0120) |
| font *family* | ❌ | one family only |
| tick density, margins, subtitles, annotation placement, clutter, realistic imperfections | ❌ | none of these are varied |

**LANGUAGE**

| item | status | note |
|---|:---:|---|
| nested reasoning | ✅ | L4 |
| ambiguity avoidance | ✅ | colliding labels refused, not guessed (0083) |
| varied operand order | ✅ | multiple orderings per operation (0122) |
| template diversity, paraphrases, naturalness, ChartQA-like wording | ✅ | 40+ templates; median 7 → 10 words, past tense 0% → 51.3%, entity nouns instead of "category" (0122) |
| referring expressions, distractors | ❌ | need chart geometry and colour at question-build time, not just the series (0122) |

**GROUNDING**

| item | status | note |
|---|:---:|---|
| evidence count | ✅ | now varies with density (0118) |
| geometry verification | ✅ | every box checked on a recoloured render |
| false-positive boxes | ✅ | verification rejects them |
| line/scatter markers | ✅ | own box functions |
| grouped chart identity | ✅ | qualified labels (0083, 0107) |
| tiny elements | ❌ | rejected by `MIN_BOX_SIDE_PX`, but real charts contain them |
| overlapping elements | ❌ | not generated |
| legend association | ❌ | not generated as a task |

**DOMAIN GAP**

| item | status | note |
|---|:---:|---|
| chart-type resemblance | ✅ | fixed by selection (0091) |
| mark-density resemblance | ✅ | p50 4 → 10, max 7 → 40 (0118) |
| value resemblance | ✅ | fitted (0120) |
| operation-mix resemblance | ❌ | `difference` 13.8× over-represented; fixable by selection, **not done** |
| "unrealistically clean" | 🟡 | no imperfections, no clutter |
| other rendering engines | ❌ | matplotlib only; not investigated |

---

## Cross-cutting requirements

| requirement | status | evidence |
|---|:---:|---|
| DATA QUALITY > QUANTITY | ✅ | 0116 discarded 4,939 records rather than accept wrong ones |
| MANUAL SEMANTIC AUDIT SET | ✅ | `scripts/build_semantic_audit.py` — 16 strata, seeded, short strata left short; 237 rows awaiting judgement. Found the truncated-operand join bug on its first run (0129). |
| PRIORITIZATION (15-point record) | ✅ | `FINDINGS.md` |
| EMPIRICAL VALIDATION | 🟡 | every CPU-measurable claim has a number; the GPU ones are listed as blocked below |
| TESTING REQUIREMENTS | ✅ | 2,100+ tests; new work mutation-checked (8 mutations on the decoder, 4 on the generator) |
| — quality statistics before/after for data changes | ✅ | 0118 and 0120 both carry before/after tables |
| — successful / rejected / ambiguous / regression cases | ✅ | e.g. `test_synth_density.py`, `test_grounding_only_fallback.py` |
| WORK ORDER phases 1–9 | 🟡 | 1–6 done; 7 partial; 8–9 gated on GPU runs — every blocked one has a runnable command in `BLOCKED.md` |

---

## The FINAL EXPECTATION questions

| question | answered in |
|---|---|
| which decisions remain strong | `VERDICT.md` |
| which were wrong | 0112, 0115, 0117, 0118, 0120 — and 0116, which was mine, twice |
| which are outdated | 0112 (a starting point that became a ceiling), 0117 (a ratio that stopped being one) |
| which are merely uncertain | `SYNTHETIC_REPLAY`; the answer policy |
| what should change immediately | done, and listed above |
| what should be experimentally tested | the ⛔ rows below |
| what should remain unchanged | Idea 12, 13; `ChartRecord`'s identity scheme (0107) |
| how the changes interact | 0114 ↔ 0121 (both bear on decode); 0115 → 0116 → 0119 (a chain) |
| expected measurable benefit | stated per decision |
| how we will know it is better | pre-registered predictions, e.g. the agree-rate in 0121 |

---

## What is NOT done — the honest list

**Not done, and doable without a GPU:**

2. 🟡 **Question language** (Idea 9). Templates, tense, length and entity nouns done (0122). **Referring expressions and distractors are not** — they need geometry and colour at question-build time.
3. 🟡 **Operation mix** (0091). L3 weighted toward ChartQA (`argmax`/`argmin` 7.3% → 10.8% corpus-wide) (0123). The rest is a **level-proportion** question — `difference` is 25% because L2 and L4 are half the corpus — and that is a mixture-time selection decision needing your call on what stage 1 is for.
4. ❌ **Stacked bars**, tiny/overlapping elements, legend-association tasks, clutter and imperfections.
5. ❌ **Within-chart correlations** and multi-column tables.
6. ❌ Two ChartQA record constructors still exist (0119, 0124) — a merge with no measurement behind it yet.
7. ❌ **Distractor-aware spurious-program check.** Newly *possible* — synthetic records now keep the whole chart (0124) — and not yet built.
8. ❌ **Judge the 237-row semantic audit set.** Built and sampled (0129); the judging is manual and not done.

**Blocked:**

9. ⛔ **LLM plan mining at volume.** ChartQA contributes **5 records of 22,947** to stage 1 today. Everything is built and ready to run; it needs the API run.
10. ⛔ **RefChartQA scaling ladder** (4,000 / 10,000 / 25,000). Unblocked as of 0115 — needs GPU.
11. ⛔ **Re-run the zero-shot baseline with `close_evidence=True`** (0114), or the reported fine-tuning gain will include a truncation fix.
12. ⛔ **Three training seeds**, Phase 8–9 verification and reporting.


---

## Where the blocked work is written down

`BLOCKED.md` — six experiments, each with why it is blocked, a copy-pasteable command, and
what result would change a decision already recorded. `Prompt.md` requires exactly that:
*"clearly document the blocked experiment and exactly how it should be run later."*

Reading the prompt line by line for that section found that the decode fix from 0114 had
**no CLI flag** — it existed and could not be run, the same shape as the scaling ladder
whose cache held 3,996 rows. `run_zeroshot.py --close-evidence` now exists.
