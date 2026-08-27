# Notes — reading a dataset before believing what it is for

## What this component is, in plain language

The data half of the project: fetching ChartQA and RefChartQA, turning them into one record type, merging
duplicates, checking the boxes are trustworthy, and mining typed plans out of the gold tables.

Very little of it is machine learning. Most of it is finding out what is actually inside two zip files.

## Why it exists — what breaks without it

Everything downstream is a function of these records. If the boxes are in the wrong coordinate convention,
every grounding number is wrong. If duplicates are counted twice, the training set is smaller than believed
and every comparison is confounded. If a test-split question leaks into training, the headline result is
meaningless and no amount of care elsewhere recovers it.

None of those failures announce themselves. They all produce numbers.

## What surprised me

**You can read a zip file over the internet without downloading it.** A zip stores its table of contents at
the *end*, and each entry records its own byte offset. So an HTTP Range request can fetch the directory, and
then any single member, of an 875 MB archive for the cost of that member. The entire ChartQA layout, the
schema of its annotations, and the fact that bar boxes are an exact linear function of the gold table values
were all established for a few megabytes — before deciding whether to download it at all. On a machine with a
few gigabytes free, that is not a trick, it is the difference between the phase running and not.

**ChartQA has grounding boxes, and the project plan does not know it.** The design treats RefChartQA as the
only source of real box supervision, with synthetic charts as the fallback if RefChartQA's audit fails. But
ChartQA's own annotation files carry per-datapoint boxes: 80.8% of training charts, 12.7 boxes each, and the
bar extent tracks the gold value at r² = 0.9999. This was written down *before* running the audit, precisely
so it could not look like a convenient discovery afterwards.

**The plan's expected mining yields are backwards, and the reason is interesting.** `PLAN.md` predicts ~1.9%
for human-written questions and ~16.5% for machine-generated ones, and explains the gap as the signature of
known corruption in the human charts' gold tables. Measured on all 28,299 training questions: human **15.41%**,
machine **13.60%**.

The corruption is real — human questions fail to find *any* matching operation 5.3× as often (18.6% vs 3.5%),
which is exactly what a wrong table does. But it does not depress the yield, because human questions are also
far *less ambiguous* (31.7% vs 61.4%), and the two effects nearly cancel.

**And the headline yield is the wrong number to care about.** 73.6% of mined plans are a bare `lookup` — read
this cell — and every single machine-generated one is. Templated questions have templated answers. The
compositional plans, the ones that teach anything about expression trees, come almost entirely from the human
subset: about 3.7% of all questions. Which lands close to the original 5.7% estimate, and means the plan's
conclusion survives even though its number did not.

**A 5% tolerance is enormous when the answer is a year.** Mining accepted a plan when one operation reproduced
the gold answer within ChartQA's 5% relaxed tolerance. For a gold answer of 2014 that admits anything from
1913 to 2115 — so `difference → 2096` was accepted as the answer to "Which year contains the higher point on
the graph?". The tolerance is right for *scoring a model reading a chart by eye*; it is badly wrong for
*computing from a gold table*, where the right test is whether the operation reproduces the answer to the
precision the answer was printed at.

## What I decided, and what I rejected

**Merge duplicates within a split; report collisions across splits, never resolve them.** The first
implementation dropped the second record on a cross-split collision — which silently resolves a train/test
leak, the one failure the whole rule exists to prevent. A leak must be loud.

**Check for leakage on the inputs, not on the survivors.** A test-split record with no curriculum level would
be dropped by stage 1's level grouping; a check that ran afterwards would see a clean mixture and report
nothing. That is the worst possible outcome: a real leak, silently absorbed.

**Refuse to read a bare 4-sequence as a box.** RefChartQA ships `{x, y, w, h}`; this project uses
`[x1, y1, x2, y2]`. Both are four numbers. Accepting a list and guessing would, on the day someone passed the
wrong one, shrink every box toward the origin and produce a merely-bad grounding score with nothing pointing
at the cause. The loader raises instead.

**Audit in two layers and report them separately.** Measured criteria can be applied to all 200 rows but are
only *necessary* conditions — they cannot see whether a well-formed box is on the element the question asks
about. Actually rendering examples and looking at them is the layer that answers the question, and pretending
the automated pass rate covers it would defeat the purpose of auditing.

## Which concept a reader must understand first

**A tolerance is a claim about what counts as the same answer, and it belongs to a purpose, not to a
project.** ChartQA's 5% exists so a model that reads 48.6 as 48.2 off a chart is not punished. Reusing it to
decide whether an arithmetic operation *explains* a gold table entry is a category error, and it manufactures
plans out of coincidence. Same number, different question, wrong answer.

Second: **when a measurement disagrees with the plan, the interesting output is the explanation, not the
number.** "Human yield is 15.4%, not 1.9%" is a correction. "The corruption is real but shows up as
unmatchable questions rather than as low yield, because ambiguity falls at the same time" is knowledge, and it
is what tells you the plan's *conclusion* still holds.

## Forward pointers

- `DECISIONS.md` 0042 — ChartQA's element boxes, recorded before the audit ran.
- `DECISIONS.md` 0045, 0046 — the tolerance defect, and what the yield split actually means.
- `DECISIONS.md` 0047 — the audit, and the criterion that nearly failed it wrongly.
- Phase 5's scaling ladder decides how many RefChartQA rows to keep; the audit only decides *whether*.
