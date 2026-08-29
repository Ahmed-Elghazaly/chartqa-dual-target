# Chapter 5 — Mining plans out of real data

**File:** `plans/mining.py` — 302 lines.

Chapter 4 defined what a plan is. ChartQA does not provide one. This chapter is how we
recover it — and why the answer is *"only 14% of the time"*, the finding on slide 8.

---

## 5.1 The idea, and the rule that makes it trustworthy

ChartQA gives us the chart's **gold data table**. So for a question with a known answer, we
can search: **which operation over this table produces that answer?**

Question: *"What is the difference between West and South?"* Answer: `74`.
Table has `West = 144`, `South = 70`. Try everything. `difference(West, South) = 74`. Found.

The rule that makes this more than guesswork is in the docstring:

> accept a plan **only when exactly one operation type reproduces the recorded answer**

If two different operations both produce 74, we do not know which one the question was
asking for. **We refuse.** Ambiguity is a rejection, never a coin flip.

> Ambiguity is the normal case, not an edge case

That sentence sets expectations correctly. Most questions are rejected. That is the design
working, not failing.

---

## 5.2 How close is "reproduces the answer"?

```python
def gold_tolerance(target) -> float:
    """The granularity the answer was written to, not a fixed percentage. A gold answer of
    ``"48.6"`` was rounded to one decimal, so anything within 0.05 rounds to it; ``"2014"``
    was written to the unit, so the window is 0.5."""
    text = str(target).strip().replace(",", "").replace("$", "").rstrip("%").strip()
    frac = text.split(".")[1] if "." in text else ""
    digits = len(frac.rstrip())
    return 0.5 * (10.0 ** -digits)
```

The tolerance is derived from **how the answer was written**.

- `"48.6"` — one decimal digit → `0.5 × 10⁻¹` = **0.05**. Anything from 48.55 to 48.65
  rounds to 48.6.
- `"2014"` — no decimals → `0.5 × 10⁰` = **0.5**.
- `"3.142"` — three decimals → **0.0005**.

📘 **Why not use ChartQA's own 5% tolerance?** Because 5% of `2014` is a window of **±100
years**. Under that rule, mining accepted:

> `difference → 2096` as the explanation for a gold answer of `2019`

That is not a plan. It is an arithmetic coincidence, and training on it teaches arithmetic
that is wrong. The 5% rule exists to score a model *reading a chart by eye*, where a small
misread should be forgiven. Mining computes from the exact gold table, so there is nothing
to forgive.

**A second guard, for the same failure in a different disguise:**

```python
    if answer_is_a_category(target, rows):
        # Arithmetic cannot produce a category. Whatever matched was a coincidence.
        return MinedPlan(status="category_answer", flattening=flattening)
```

*"Which year had the most crime?"* is answered by `2014` — a **label**, not a quantity. No
arithmetic legitimately produces a category. If some operation happens to output 2014, it is
coincidence. Year answers are the dangerous case precisely because they look numeric.

---

## 5.3 ⚠️ Which cells count as candidates

Here is the decision the plan left open, and it is a genuinely hard one.

Appendix E assumes a flat list of `(label, value)` pairs. That is unambiguous for a
two-column table. **28% of ChartQA tables have 3–9 columns**, and then it is undefined: is a
"candidate set" a row? a column? every cell?

It matters enormously, because uniqueness decides everything:

| flattening | yield |
|---|---:|
| per row | 4.2% |
| all cells (union) | 14.2% |

**A 3.4× spread from a choice the plan did not make.**

📘 **Why more cells means more accepted plans, not fewer.** With a small candidate set,
several operations tend to hit the answer by luck, so uniqueness fails and we reject.
Widening the set gives each operation more chances to be *the only* one that lands exactly —
the tolerance is tight, so extra candidates mostly add near-misses rather than new ties.

The file's response is not to pick quietly:

> the flattening is an explicit parameter, every mode is measurable, and the yield is
> reported as a range rather than a point

```python
def candidate_sets(rows, mode) -> list[list[tuple[str, float]]]:
    """Returns a LIST of sets because ``per_row`` and ``per_column`` produce several,
    and a plan is unique only if it is unique across all of them."""
```

**"Unique across all of them"** is the strict reading. If a plan is unique within one row but
a different operation explains it in another, that is still ambiguity, and it is rejected.

⚠️ **A bug lived in this signature.** `rows` **includes the header**. A script that stripped
the header first passed body-only rows, silently shifting every label by one. `lookup` went
from 2% to 74% of mined operations once fixed — the labels had been wrong, so almost nothing
matched.

---

## 5.4 Mining, and then checking the mine

```python
def mine_plan(rows, target, *, flattening="union") -> MinedPlan:
    """Accept a plan only when exactly one operation type reproduces the answer."""
    if to_number(target) is None:
        return MinedPlan(status="non_numeric", flattening=flattening)
    if answer_is_a_category(target, rows):
        return MinedPlan(status="category_answer", flattening=flattening)
    sets = candidate_sets(rows, flattening)
    if not sets:
        return MinedPlan(status="none", flattening=flattening)
```

Note the **statuses**: `non_numeric`, `category_answer`, `none`, and later `ambiguous` and
`unique`. Not a boolean. Every rejection records *why*, which is what produces the breakdown
on slide 8 — and what let us notice that ambiguity, not absence, is the dominant cause.

The last step is the one that turns a hint into training data:

> Appendix E returns an operation *type*. A trainable example needs a concrete tree, so the
> operands that produced the match are recovered and the tree is **verified by executing
> it** — a mined plan that does not reproduce its own answer is discarded.

So a mined plan is run through Chapter 4's executor before being accepted. Mining and
verification are separate steps, and the second does not trust the first.

---

## 5.5 The numbers, and what they mean

Over all **28,299** ChartQA training questions:

| | human | machine | all |
|---|---:|---:|---:|
| unique plan found | 15.4% | 13.6% | **14.07%** |
| ambiguous | 31.7% | 61.7% | |
| answer not numeric | 30.5% | 15.1% | |

Of the plans found, **73.6% are bare `lookup`** — "read one number off the chart". So
questions teaching genuine multi-step reasoning are **3.7%** of the total: about **4 in
100**.

**What that means for the project.** A `lookup` plan teaches the output *format* but almost
nothing about reasoning. The real supply of compositional supervision is tiny, and no amount
of care in this file changes that — the information is not in the data.

**This is the argument for Chapter 6.** The generator is not a shortcut around doing the
real thing; it is the response to a measurement showing the real thing cannot supply what is
needed.

---

## 5.6 What to take from this chapter

1. **A plan is accepted only if exactly one operation explains the answer.** Ambiguity is a
   rejection, and it is the *common* case.
2. **Tolerance comes from how the answer was written**, not a fixed percentage — because 5%
   of a year is a century, and that rule accepted `2096` as an explanation for `2019`.
3. **Category answers are refused outright.** Arithmetic cannot produce a label, so a match
   is coincidence.
4. **The plan under-specified which cells are candidates**, and the choice moves yield 3.4×.
   The response was to make it a parameter and report a range, not to pick silently.
5. **Every rejection records its reason.** That is what showed ambiguity, not absence, is the
   binding constraint.
6. **A mined plan is verified by executing it**, so mining's output is checked by something
   that does not trust mining.
7. **14.07% yield, 73.6% of them bare lookups → ~4 in 100 questions teach real reasoning.**

**Next:** Chapter 6 — generating charts where the answer and the boxes are known by
construction, and proving the boxes against pixels.
