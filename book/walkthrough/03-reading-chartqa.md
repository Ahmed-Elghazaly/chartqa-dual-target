# Chapter 3 — Reading ChartQA off the disk

**File:** `data/chartqa.py` — 348 lines.

This chapter is where slide 5's finding comes from — *bars 92–97% annotated, pie 55%, line
0%*. That number is not quoted from a paper. It is what this loader measured while being
written.

---

## 3.1 What is actually in the archive

ChartQA ships as one 875 MB zip:

```
ChartQA Dataset/{train,val,test}/
    png/          {imgname}.png
    tables/       {stem}.csv          gold data table
    annotations/  {stem}.json         chart type, axis labels, ELEMENT BOXES
    {split}_human.json                [{imgname, query, label}, ...]
    {split}_augmented.json            the machine-generated questions
```

Two facts from the docstring worth holding on to.

**18,317 images against 28,299 QA rows** in train — several questions per chart. So charts
and questions are not one-to-one, which is why Chapter 2's identity key needs both.

**`question_kind` follows which file a row came from.** The archive calls the second file
`augmented`; the upstream parquet release calls the same rows `machine`. We use `machine`,
so that the two ways of getting the same data describe it identically. A small thing, but
"augmented" and "machine" being the same set is exactly the kind of detail that produces two
incompatible result tables later.

**Why we read the archive rather than the convenient parquet:**

> The annotations are the reason this loader reads more than the parquet does. They carry
> per-datapoint bounding boxes in absolute-pixel `{x, y, w, h}`, the same form as
> RefChartQA, aligned index-for-index with the series values.

The parquet has images, questions and answers. The zip additionally has **where each bar
is**. We need that, so we read the harder format.

📘 The archive layout above was established by **range-reading the zip's central directory**
before downloading anything. A zip file keeps an index at its end; an HTTP range request can
fetch just that index. So we learned the layout by downloading a few kilobytes, instead of
875 MB followed by the discovery that the layout was not what we assumed.

---

## 3.2 The gold table

```python
def parse_table(text: str) -> dict[str, Any]:
    """A gold CSV as ``{"columns": [...], "rows": [[...], ...]}``.

    Values stay as written. Coercing them here would quietly change what the gold answer
    is compared against, and the executor already handles numeric parsing where it
    matters.
    """
    rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
    if not rows:
        raise ChartQAError("empty table")
    return {"columns": rows[0], "rows": rows[1:]}
```

The **table** is the chart's underlying data — the numbers that were plotted. Chapter 5 uses
it to recover what calculation a question is asking for.

**"Values stay as written" is a real decision.** The tempting thing is to convert `"1,234"`
to `1234` and `"45%"` to `0.45` here. We do not, because the gold answer is compared as
*text* by the official metric. If we normalise `"1,234"` to `1234.0` and the gold answer says
`"1,234"`, we have silently changed what a match means — in the loader, far from the metric,
where nobody would look for it.

The filter `if any(c.strip() for c in r)` drops blank rows. `csv.reader` yields `[]` or
`['', '']` for them, and an empty row would become a column of `None`s downstream.

---

## 3.3 Chart types store boxes differently

```python
#: Measured over 4,000 random train charts:
#: v_bar and h_bar pair one box per datapoint (97.5% / 98.6% of series), line stores
#: SEGMENTS between consecutive points (85.6% of series have len(bboxes) == len(y) - 1),
#: and pie uses a per-wedge layout with its own keys.
ELEMENT_LAYOUTS = {"v_bar": "series", "h_bar": "series", "line": "segments", "pie": "wedges"}
```

This table is the heart of the file, and note that **every claim in the comment has a
measurement attached**. Not "line charts seem to store segments" — *85.6% of line series
have exactly one fewer box than they have y-values*, over 4,000 charts. That is what
establishes it as the format rather than a quirk of the chart someone happened to open.

Three layouts, three functions.

### Bars — `_series_elements`

```python
    boxes = model.get("bboxes") or []
    xs, ys = model.get("x") or [], model.get("y") or []
    if len(boxes) != len(ys) or (xs and len(xs) != len(boxes)):
        return []
```

Bars store three **parallel arrays**: the boxes, the category labels, the values. Entry *i*
of each describes the same bar.

The guard is the interesting line. If the lengths disagree, the function returns **nothing**
rather than zipping to the shorter one.

> A model whose lengths disagree is skipped rather than zipped short — a silent
> misalignment would attach boxes to the wrong values, and nothing downstream could detect
> it.

📘 Python's `zip` stops at the shortest input. Here that would pair box *i* with value *i*
for as far as they both go — producing records where the box says "the March bar" and the
value says February's number. Every such record would look completely normal. Refusing the
whole chart costs a few charts; accepting them poisons training data invisibly.

### Pie — `_wedge_element`

```python
    """Of 538 wedge models sampled, 268 carry ``bboxes``, 251 carry a possibly-null
    ``bbox``, and 19 carry neither a label nor a value. Only wedges with a label, a
    value and a usable box are returned; the rest are counted as uncovered rather than
    filled in with a guess."""
```

Pie charts have **three different shapes in the same dataset** — roughly half use `bboxes`
(plural), half `bbox` (singular, sometimes null), and 19 have no usable content at all. The
code tries both spellings and returns `None` when neither works.

**That `None` is why pie coverage is 55% and not higher.** The missing wedges are *counted
as uncovered*, not filled in. An honest 55% is more useful than a fabricated 100%.

### Lines — deliberately excluded

```python
    """Bars and pie wedges only. **Line charts are deliberately excluded**: their `bboxes`
    are the segments *between* consecutive points, so a point's position is recoverable
    but a point's box size is not — the annotation never states a marker size. Inventing
    one would put a fabricated box into training data, which is precisely the failure the
    RefChartQA audit gate exists to catch. Lines are 12.9% of ChartQA against 83.9% for
    bars, so the coverage lost is small and the alternative is unverifiable."""
```

**This is the source of the 0% on slide 5, and it is a decision rather than a gap.**

A line chart's annotation gives boxes for the *segments joining* the points, not the points.
From a segment you can work out where a point is — it is an endpoint — but not **how big a
box around it should be**. A data marker has some visual size, and the annotation never says
what.

We could invent one. Pick 8 pixels, or scale it to the chart. It would look right. And it
would be a **fabricated box in training data** — the model would be taught to reproduce a
number we made up. That is exactly the failure the RefChartQA quality audit exists to catch,
so doing it deliberately in our own loader would be incoherent.

The cost is stated honestly: 12.9% of ChartQA is line charts, against 83.9% bars. We lose
the smaller share and keep the guarantee that every box in training came from an annotation.

---

## 3.4 `_norm_or_none` — the smallest function, doing two jobs

```python
def _norm_or_none(box, image_w, image_h):
    try:
        norm = xywh_to_norm1000(box, image_w, image_h)
    except ValueError:
        return None
    return norm if norm[2] > norm[0] and norm[3] > norm[1] else None
```

**Job one:** convert `{x, y, w, h}` (a corner plus a size, in pixels) into the
`[x1, y1, x2, y2]` 0–1000 form from Chapter 1. Malformed input raises, and is caught.

**Job two:** the return line rejects **degenerate** boxes — where the right edge is not
strictly right of the left, or the bottom not below the top. A zero-width or inverted box is
not a region. It would pass every later type check, and IoU against it is zero or undefined,
so it would silently cost score at evaluation time (Chapter 7).

📘 The pattern — *convert, and return `None` rather than something malformed* — recurs
throughout the data layer. It is why `boxes` is `None`-able in Chapter 2. Every layer either
produces something valid or produces nothing, and never produces something that merely looks
valid.

---

## 3.5 What the loader measured

The coverage numbers on slide 5 are the output of running this loader over 2,500 annotations
and counting what came back:

| chart type | share of ChartQA | element boxes recovered |
|---|---:|---:|
| v_bar | 54.6% | 96.8% |
| h_bar | 29.3% | 91.5% |
| line | 12.9% | **0.0%** (§3.3) |
| pie | 3.2% | 54.8% (§3.3) |
| | | overall **80.8%** |

And a second measurement checks the boxes are *right*, not merely present: a bar's box
height should track its value in the gold table. Across 1,290 series the median r² is
**0.9999** for v_bar and **1.0** for h_bar.

📘 **r²** ranges 0 to 1 and says how much of one quantity's variation is explained by
another. 1.0 is a perfect straight-line relationship. So: taller bar, proportionally taller
box, essentially exactly.

**Why that check matters.** Coverage says the annotations *exist*. r² says they *mean what
we think*. Without the second, a loader with an off-by-one or a flipped y-axis would report
97% coverage and produce entirely wrong boxes.

---

## 3.6 What to take from this chapter

1. **We read the 875 MB zip, not the convenient parquet**, because only the zip has element
   boxes — and we learned its layout by range-reading its index first.
2. **Table values stay as written.** Normalising them in the loader would silently change
   what the metric compares against.
3. **Every format claim in this file has a measurement behind it** — 4,000 charts for the
   layout table, 538 models for the pie shapes.
4. **Misaligned arrays cause the whole chart to be refused**, because `zip` would silently
   pair a box with the wrong value and nothing downstream could tell.
5. **Line charts are excluded on principle.** Their boxes are segments, so a marker size
   would have to be invented — and inventing box data is the exact failure our own audit
   gate exists to catch.
6. **Coverage and correctness are separate checks.** 80.8% coverage, and r² ≈ 1.0 that the
   covered boxes track their values.

**Next:** Chapter 4 — what a plan *is*: the schema that constrains it, the executor that
runs it, and the round-trip check that makes an answer verifiable.
