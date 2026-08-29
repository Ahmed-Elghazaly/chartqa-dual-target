# Chapter 6 — Generating charts, and proving the boxes

**Files:** `synth/curriculum.py` (192), `synth/generator.py` (514), `synth/verify.py` (214).

Chapter 5 established that real data supplies almost no reasoning supervision. So we draw
our own charts, where the answer and the boxes are known by construction.

The interesting half is not the drawing. It is **proving the boxes are right**.

---

## 6.1 Four levels of difficulty

`curriculum.py` builds a question at one of four levels, each teaching a different shape of
plan:

| level | plan shape | example |
|---|---|---|
| L1 | `lookup(X)` | "What is the value for West?" |
| L2 | two named operands | "What is the difference between West and South?" |
| L3 | fold over everything | "What is the mean value?" |
| L4 | an operation **inside** another | "How far is West from the average?" |

**L4 is the one that matters.** Chapter 5 measured that real data gives roughly 4 questions
in 100 that teach multi-step reasoning. The generator produces 6,000 L4 examples on demand.

Each level builds the question text, the plan, and the list of labels the plan needs — all
three together, so they cannot disagree.

```python
def build_question(level, series, rng, *, unit=None, quantity="value") -> SynthQuestion | None:
    if len(series) < 2:
        return None
```

Returning `None` rather than raising is the pattern from Chapter 3: some data cannot support
some questions (you cannot ask for a difference with one bar), and that is normal, not an
error.

---

## 6.2 Drawing, and why the boxes are not simply "where we put things"

The generator draws with matplotlib, so in principle it knows where every bar is. But
knowing where you *asked* for a bar is not the same as knowing where the ink *is*:

- a bar's outline has a **line width**, so the drawn shape is slightly larger than the
  rectangle requested;
- a scatter marker's size is specified in points, not pixels, and depends on the figure's
  resolution;
- a pie **wedge** is not a rectangle at all — its bounding box contains a great deal of
  non-wedge;
- layout adjusts after drawing: labels rotate, the axes shrink, everything shifts.

So a computed box is a *hypothesis*. `verify.py` tests it against the rendered pixels.

```python
"""`PLAN.md` 3.5 requires it: *"For a sample of generated charts, re-render the box
onto the image and assert the pixels inside it actually contain the intended
element."* A generator whose boxes are subtly wrong is very hard to detect later,
so this runs on every generation batch rather than on request."""
```

**"On every generation batch rather than on request"** — the plan asked for a sample. The
file checks everything, because a subtly wrong generator is exactly the failure that would
survive a sample.

---

## 6.3 ⚠️ Two checks, because one is not enough

```python
"""The check is deliberately two-sided. "Does the box contain the element" passes
trivially for a box that is far too large, so the element must also **fill** the
box, and a box displaced by most of its own width must **fail**."""
```

This sentence is the design, and it took three attempts to get right.

**Check one — containment.** *What fraction of the element's ink is inside the box?*

```python
def containment(img, box, colour, tol=12) -> float:
    """Fraction of ALL pixels of `colour` in the image that fall inside `box`."""
    mask = (np.abs(img.astype(int) - np.array(colour)) <= tol).all(axis=-1)
```

📘 `img` is a height × width × 3 array of colour values. The expression builds a **mask** —
a true/false grid marking pixels of the target colour. `np.abs(img - colour) <= tol` compares
each channel with tolerance 12 (antialiasing means edge pixels are blends, not exact
matches); `.all(axis=-1)` requires all three channels to match.

Containment answers "did we miss part of the element?" — but it passes trivially for a box
covering the whole image. Hence check two.

**Check two — `ink_bbox_iou`.** *How well does the box match where the ink actually is?*

```python
    ys, xs = np.nonzero(mask)
    ink = (ox + xs.min() - slack_px, oy + ys.min() - slack_px,
           ox + xs.max() + 1 + slack_px, oy + ys.max() + 1 + slack_px)
```

Find every matching pixel, take the tightest rectangle around them — that is where the
element *is* — then compute IoU (Chapter 7, §7.3) between that and the proposed box.

The docstring calls this *"the decisive test, and the only one that is genuinely
shape-independent"*: rather than compare fill to a constant that depends on the shape, it
compares the box to the ink.

Three details, each fixing a real false failure:

**`expand(box)` before searching.** Ink is collected from a window slightly larger than the
box, not the whole image, so that several elements sharing a colour do not confuse each
other. *"Neighbours sit far outside that window."*

**`slack_px = 1`.**
> an antialiased edge blends into the background and falls outside the match tolerance, so
> raw ink reads about a pixel small in every direction

Without it, every box fails by a pixel — a systematic bias mistaken for bad boxes.

**Floor the near edges, ceil the far ones:**
```python
    x1, y1 = math.floor(box[0]), math.floor(box[1])
    x2, y2 = math.ceil(box[2]), math.ceil(box[3])
```
> a pixel the box only partly covers is still inside it. Rounding both ways instead drops a
> boundary row or column, which on a thin bar is a few percent of its ink and read as a bad
> box.

### Two rejected designs

**Displacement** — "a box moved by most of its width must fail". It false-failed on
**adjacent bars**: shift a bar's box sideways and it lands on the neighbouring bar, which is
also ink, so it looked fine. The test could not tell "correct" from "one bar over" — the most
likely real error.

**Relative tightness** — the ratio of ink area to box area. Scale-invariant, so it is blind
to a box that is uniformly too large.

`ink_bbox_iou` is immune to both: a shifted box has poor overlap with *its own* element's
ink, and an oversized box has a large union and therefore low IoU.

---

## 6.4 Thresholds per geometry

```python
GEOMETRY_THRESHOLDS = {
    "rect":        {"min_containment": 0.98, "min_ink_iou": 0.70},
    "wedge":       {"min_containment": 0.98, "min_ink_iou": 0.70},
    "disc_unique": {"min_containment": 0.98, "min_ink_iou": 0.70},
    "disc_shared": {"min_containment": None, "min_ink_iou": 0.70},
}
```

Four geometry classes, because different shapes admit different checks.

**`disc_shared` has `min_containment: None`** — containment is *disabled*. These are line and
scatter markers, where several points share one colour, so "the fraction of this colour
inside the box" would count every other marker as missing ink. The check is skipped rather
than loosened, and `ink_bbox_iou` still applies.

```python
    try:
        t = GEOMETRY_THRESHOLDS[geometry]
    except KeyError:
        raise ValueError(f"unknown geometry {geometry!r}; expected one of {sorted(...)}")
```

> Raises on an unknown geometry rather than falling back to a default: a silent default
> would apply a rectangle's assumptions to a wedge, which is exactly the mistake this table
> exists to prevent.

⚠️ **That mistake was made.** Pie wedges were first checked with a rectangle-derived fill
threshold and failed at **41.8–77.8%** fill — because a wedge inscribed in its bounding box
can occupy at most π/4 ≈ 78.5% of it. Nothing was wrong with the boxes; the threshold was
geometrically impossible.

---

## 6.5 Verifying without disturbing what is verified

The generator has a problem: to check a box contains *this* element, it must find that
element's colour. But a palette can repeat, and neighbouring bars can be similar.

The solution:

1. temporarily **recolour** every element to a distinct sentinel colour;
2. render, and run the checks against that render;
3. **restore** the real colours and save the real image.

> Colour does not move any artist, so the geometry verified here is exactly the geometry
> shipped.

That last clause is the justification: changing colour does not change position, so a box
verified on the sentinel render is valid for the shipped one.

⚠️ Two more real defects, both found this way. A **palette wrap** gave two elements the same
colour — fixed by shifting lightness by 60. And **marker boxes omitted the stroke**, scoring
97.2% containment instead of ~100%; the box now pads by half the line width.

---

## 6.6 The result

- **8 chart types × 4 levels = 24,000 examples.**
- Correct boxes score **0.841–0.990** ink-IoU; deliberately wrong ones never exceed **0.377**.
- A box is kept only above **0.70** — a threshold sitting in the empty gap between those two
  ranges, not at an arbitrary round number.

Slide 9's figure shows exactly this: the same box correct (0.95, kept) and moved 26 pixels
(0.43, rejected).

---

## 6.7 What to take from this chapter

1. **Knowing where you drew something is not knowing where its ink is** — stroke widths,
   marker sizing and layout adjustment all move it. A computed box is a hypothesis.
2. **The check is two-sided by necessity.** Containment alone passes an enormous box; IoU
   against the ink's own bounding box catches both displacement and oversize.
3. **Two designs were rejected on measurement:** displacement false-failed on adjacent bars;
   relative tightness is scale-invariant and blind to oversized boxes.
4. **Thresholds are per geometry, and an unknown geometry raises** — because applying a
   rectangle's assumptions to a wedge is the exact mistake that happened (π/4 ≈ 78.5%).
5. **Verification uses a recoloured render** and restores the real colours, because colour
   does not move geometry.
6. **The 0.70 threshold sits in a measured gap** between 0.841–0.990 and ≤0.377, not at a
   round number someone liked.

**Next:** Chapter 7 — what a score actually computes, and why we implemented every metric
twice.
