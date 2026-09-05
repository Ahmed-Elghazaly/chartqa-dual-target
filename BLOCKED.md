# Blocked experiments, and the exact command for each

`Prompt.md` requires that anything which cannot run here is *"clearly documented"* with
*"the exact command/procedure required to finish it later"* rather than guessed at. This is
that list. Everything else — every claim in `DECISIONS.md` — was measured on this machine.

Each entry says what is blocked, **why**, the command, and what result would change a
decision already recorded.

---

## 1. LLM plan mining, at volume

**Why blocked:** needs Anthropic API spend. Nothing else is missing — the pipeline is
built, tested, and costed.

**Why it matters most:** ChartQA contributes **5 records of 22,947** to stage 1 today,
because only 2 carry a mined plan. This is the single largest gap between the system as
designed and the system as built.

```bash
# 1. write the batches (no API calls; prints the cost estimate)
python scripts/mine_plans.py --limit 30000 --kind all --write-batches

# 2. run them through the Message Batches API (half price)
python scripts/mine_plans.py --limit 30000 --kind all --api --model claude-opus-5

# 3. verify every returned plan through the five gates and write the cache
python scripts/mine_plans.py --score outputs/mining/replies.json
```

**Pre-registered checks.** Verified yield should land near **62.5%** (0088); if it is far
below, the teacher prompt is wrong rather than the questions being hard. And plan **depth**:
every plan mined from real data so far is depth 1 (12,667 of 12,667), so if LLM-mined plans
show a non-trivial share at depth 3+, decision **0125** is wrong and L4 should be extended.

---

## 2. The RefChartQA scaling ladder

**Why blocked:** three training runs. **Unblocked as of 0115** — the cache held 3,996 rows,
so rungs 2 and 3 had no data and the ladder could not have been run at all. It now holds
55,486.

```bash
for CAP in 4000 10000 25000; do
  python scripts/build_mixtures.py --refchartqa-cap "$CAP" --suffix "_ladder$CAP"
  python -m chartqa_dt.cli.train stage1 --mixture data/mixture_stage1_ladder$CAP.json
  python -m chartqa_dt.cli.evaluate --split validation --metric grounding
done
```

**What to do with it:** `PLAN.md` 3.4 says keep the point where the curve flattens, and
report the curve. `REFCHARTQA_CAP` moves only on that result (0115).

---

## 3. Re-run the zero-shot baseline with the decode guard

**Why blocked:** GPU. **This is a correctness blocker on the headline claim, not an
improvement.**

26.0% of structured generations hit the token cap and **every one of them scored zero**
(0114). Training targets average 3.04 evidence items, so fine-tuning fixes that
incidentally — which means comparing a fine-tuned model against **48.70%** would credit
fine-tuning with repairing a truncation bug. Both arms must decode under the same rule.

```bash
python scripts/run_zeroshot.py chartqa --close-evidence --tag closed
```

**Expected:** 25.9% of the eval set moves from a guaranteed zero to scoreable. The
"structured output costs 23.80 points" figure should fall to roughly **5–8** points.

---

## 4. Three training seeds, and the reported result

**Why blocked:** GPU hours.

```bash
for SEED in 0 1 2; do
  python -m chartqa_dt.cli.train stage1 --seed "$SEED"
  python -m chartqa_dt.cli.train stage2 --seed "$SEED" --resume-from stage1
  python -m chartqa_dt.cli.train control --seed "$SEED"     # PLAN.md 6.4
done
python -m chartqa_dt.cli.evaluate --seeds 0,1,2 --bootstrap
```

**Pre-registered check** from 0121: agreement between the executed plan and the emitted
answer is 76.3% zero-shot, and targets enforce it in 100% of cases. If fine-tuning does not
raise that rate, the model has learned the output format without learning to use its own
evidence — which is the failure this project would most want to know about.

---

## 5. `SYNTHETIC_REPLAY`, which is the one unmeasured constant

**Why blocked:** needs the stage-2 numbers to read.

Stage 2 is **46.9% synthetic** today where the comment claimed one sixth, and the same
constant gives 4% once the ladder runs (0117). The value was left alone deliberately:
guessing a second time is how the first guess got there.

```bash
for REPLAY in 0 1000 2000 4000; do
  python scripts/build_mixtures.py --replay "$REPLAY" --suffix "_replay$REPLAY"
  python -m chartqa_dt.cli.train stage2 --mixture data/mixture_stage2_replay$REPLAY.json
done
```

**How to read it:** if schema validity collapses in stage 2 the replay is too low; if
stage-2 accuracy lags the control it may be too high. Both are visible in the Phase 6
numbers.

---

## 6. Whether the mixture's *level proportions* should match ChartQA

**Why blocked:** a training run, and a decision from Ahmed about what stage 1 is for.

L3's operation mix now matches ChartQA (0123), but corpus-wide `difference` is still 25%
against a real 1.8%, because L2 and L4 are half the corpus by construction. Closing that
means making L1–L2 rare, which 0101 rejected on curriculum grounds. It is a real trade and
should be settled by measurement, not argument.

```bash
python scripts/build_mixtures.py --synthetic-stage1 6000 --suffix _uniform
# and a variant weighting the level draw toward L1/L3, then train both and compare
```

---

## What is *not* blocked

Everything else in `DECISIONS.md` 0112–0126 ran on this machine, on CPU, against the real
datasets. Where a result is an estimate rather than a measurement — the recovery rate in
0114, the extreme tail in 0120 — the decision says so in the row.
