# Chapter 1 — Coordinates and visual tokens

**File:** `src/chartqa_dt/vision/coords.py` — 305 lines.

This is the first chapter because four later chapters silently assume it. Everything about
*why grounding is hard* lives in this file.

---

## 1.1 The question this file answers

Our model has to output a **box**: four numbers marking a rectangle around something in a
chart. Two questions follow immediately, and neither has an obvious answer.

1. **In what units?** A box on a 800×600 chart and the same box on a 1600×1200 chart are
   the same *region* but different *numbers*. What numbers should the model emit?
2. **How small can a box be before the model cannot possibly get it right?**

The second is the important one, and answering it requires knowing what the model actually
receives when you hand it an image.

---

## 1.2 📘 From pixels to visual tokens, from zero

A language model does not process letters. It processes **tokens** — chunks of text, each
mapped to a vector of numbers. The sentence "What is the difference" might become five
tokens, each a vector of (say) 2,048 numbers. The model's whole world is a *sequence of
vectors*.

A vision-language model feeds images into that same sequence. To do that it must turn a
picture into vectors. The standard recipe, and the one Qwen3-VL uses:

**Step 1 — resize.** The image is resized to a standard size. Not a fixed size; a size that
satisfies certain constraints (§1.3).

**Step 2 — cut into patches.** The resized image is cut into a grid of small squares. For
Qwen3-VL each square is **16×16 pixels**. This number is called the **patch size**.

**Step 3 — embed each patch.** Each 16×16×3 patch (3 for red, green, blue = 768 numbers) is
multiplied by a learned matrix, producing one vector of the model's working width. That
vector is a "patch embedding".

**Step 4 — merge neighbouring patches.** Qwen3-VL then groups patches in **2×2** blocks and
merges each group into a single vector. This number is the **spatial merge size**. It exists
purely to cut cost: a 512×512 image is 32×32 = 1,024 patches, which after 2×2 merging becomes
16×16 = 256 vectors. Four times fewer things for the model to attend over.

**Step 5 — put them in the sequence.** Those merged vectors are inserted into the token
sequence in front of the question, and from that point the model treats them exactly like
word tokens.

### The consequence that matters

After merging, the smallest region the model has a *separate vector for* is:

```
patch_size × merge_size = 16 × 2 = 32 pixels on a side
```

We call that a **visual token**, and 32 is the **factor**.

Two different things inside the same 32×32 block do not get separate representations. They
are averaged into one vector. The model can still infer something about them from context —
but the input no longer contains the distinction.

**So: a chart element smaller than 32×32 pixels in the resized image cannot be pointed at
precisely, no matter how well the model is trained.** That single sentence is the reason
this file exists, and it is the finding on slide 7 of your deck.

Here is how the file records that geometry:

```python
QWEN3VL_PATCH_SIZE = 16
QWEN3VL_MERGE_SIZE = 2
QWEN3VL_FACTOR = QWEN3VL_PATCH_SIZE * QWEN3VL_MERGE_SIZE  # 32

# Qwen2-VL / Qwen2.5-VL geometry, i.e. what PLAN.md Appendix C assumed.
QWEN2VL_FACTOR = 14 * 2  # 28
```

Why the second constant is there is §1.5.

---

## 1.3 `smart_resize` — how big the image actually becomes

Before patching, the image is resized. Not to a fixed size, because charts have wildly
different shapes and squashing them would distort the very geometry we are trying to
measure. Instead the resize satisfies three constraints at once:

1. both dimensions must be **divisible by the factor** (32) — otherwise the patch grid does
   not tile the image evenly;
2. the **total pixel count** must fall between a minimum and a maximum — the cost budget;
3. the **aspect ratio** must be preserved as closely as possible.

Here is the whole function:

```python
def smart_resize(height, width, factor, min_pixels, max_pixels):
    if min(height, width) <= 0:
        raise ValueError(f"image dimensions must be positive, got {height}x{width}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(...)
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)
```

Line by line.

### The two guards

The first two lines reject a non-positive dimension and an aspect ratio above 200:1. Both
would make the arithmetic below produce nonsense rather than raise, and the upstream
implementation refuses them identically — our job here is to *predict what the real
preprocessing does*, so we match it including its refusals.

### Rounding to the grid

```python
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
```

📘 **This idiom — divide, round, multiply back — appears throughout the file, so read it
once carefully.** `height / factor` says "how many whole tokens tall is this, as a
fraction". `round(...)` picks the nearest whole number of tokens. `* factor` converts back
to pixels.

With `height = 600, factor = 32`: `600/32 = 18.75` → `round` → `19` → `19 × 32 = 608`.

So a 600-pixel-tall image becomes 608 tall: the nearest height that is a whole number of
tokens. `h_bar` and `w_bar` ("bar" as in the maths convention for an adjusted quantity) are
the grid-aligned dimensions.

At this point the aspect ratio is very slightly changed — by at most half a token on each
axis — and the pixel budget has not been considered at all. The two branches below handle
the budget.

### Branch one: too many pixels

```python
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
```

📘 **Why a square root.** We need to shrink the image so its *area* fits the budget. If we
divide both the height and the width by some number β, the area shrinks by β². We want:

```
(height × width) / β²  =  max_pixels
```

Solving for β:

```
β = sqrt( (height × width) / max_pixels )
```

So β is exactly the factor to divide *each side* by. If the image has 4× too many pixels,
β = 2, and each side shrinks by half.

```python
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
```

Same divide-round-multiply idiom, with two changes:

- **`floor` instead of `round`.** We are enforcing an upper bound. Rounding *up* could push
  the area back over `max_pixels`, which would defeat the branch. Flooring guarantees we
  land at or under.
- **`max(factor, ...)`.** After flooring, a very thin image could reach zero — a 0-pixel
  dimension, which is not an image. This clamps to one token minimum.

### Branch two: too few pixels

The exact mirror. β is now what to *multiply* by — note the ratio is inverted,
`min_pixels / area` — and `ceil` replaces `floor` because we are enforcing a *lower* bound
and must not land under it. No `max(factor, …)` is needed: we are enlarging.

`elif` matters. An image cannot be both over the maximum and under the minimum, and two
separate `if`s would let the second branch act on values the first just set.

### Worked example

Budget `max_pixels = 512 × 512 = 262,144`, factor 32, image 800×600:

```
h_bar = round(600/32)*32 = 608          w_bar = round(800/32)*32 = 800
608 × 800 = 486,400 > 262,144           → branch one
beta  = sqrt(800*600 / 262144) = 1.353
h_bar = floor(600/1.353/32)*32 = 416    w_bar = floor(800/1.353/32)*32 = 576
```

Final **576 × 416** — inside the budget, both sides divisible by 32. The token grid is
18 × 13 = **234 visual tokens**. That is what "the model sees the chart as a grid of blocks"
means concretely: this chart arrives as 234 vectors.

---

## 1.5 ⚠️ The mistake: 28 versus 32

`PLAN.md` Appendix C — this project's own plan, written before any code — specified:

```python
FACTOR = 28  # Qwen: patch 14 x spatial merge 2
```

Correct for **Qwen2-VL** (patch 14 × merge 2). Wrong for **Qwen3-VL**, which is 16 × 2 = 32.

**Why it mattered.** The factor is the denominator in every "how big is this box in tokens"
calculation. Using 28 makes every box look **(32/28)² = 1.31×** larger in token terms than
it is. The sub-token finding on slide 7 would have been understated by about a third — and
nothing would have crashed.

The file keeps Appendix C's version verbatim as `smart_resize_appendix_c`, used by nothing
except a test that runs both and shows where they diverge. The principle in its docstring is
worth stealing: **a documented deviation should be demonstrable, not asserted.** If someone
later doubts the plan was wrong, they run the test rather than trusting a note.

Recorded as `DECISIONS.md` 0008.

---

## 1.6 `VisualGeometry` — never guess the factor again

Having been wrong once about the factor, the file makes it structurally hard to be wrong
again. The rule: **read the geometry off the model that is actually loaded.**

```python
@dataclass(frozen=True)
class VisualGeometry:
    factor: int
    min_pixels: int
    max_pixels: int
    patch_size: int
    merge_size: int
    source: str = "explicit"
```

📘 **`@dataclass(frozen=True)`** generates `__init__`, `__repr__` and `__eq__` from the
field list, and `frozen=True` makes instances immutable — assigning to a field raises. That
matters here: geometry is a property of a model, and code that quietly mutated it would
produce measurements that disagree with the model actually running.

`source` records *where the numbers came from*, so a printed geometry can be traced.

### Reading it from the processor

```python
    @classmethod
    def from_processor(cls, processor: Any) -> VisualGeometry:
        ip = getattr(processor, "image_processor", processor)
        patch = getattr(ip, "patch_size", None)
        merge = getattr(ip, "merge_size", None)
        if patch is None or merge is None:
            raise ValueError(
                "processor exposes no patch_size/merge_size; refusing to guess the "
                "visual-token factor (DECISIONS.md 0008 exists because it was guessed once)"
            )
```

📘 A **processor** is the object that converts raw inputs into model inputs — a tokenizer
for text, an image processor for pixels. `AutoProcessor` returns the pair and keeps the
image half at `.image_processor`; the first line accepts either.

**The rest is the important part of the file.** If the processor does not state its
geometry, the code **refuses** rather than defaulting. A default of 32 would be correct
today and silently wrong the moment the backbone changes — which is exactly the shape of
0008. The error message says so, so whoever hits it understands why they are stopped.

(The following ten lines try three different places `transformers` versions keep the pixel
budget, and raise if none of them has it. Same principle, less interesting.)

### `with_max_pixels`

Because the class is frozen, "the same model at a smaller input budget" must be a *new*
object. This is what let us ask "what would the sub-token fraction be at 448 pixels instead
of 512?" without touching the real geometry — the measurement behind choosing 512.

---

## 1.7 The measurement everything else quotes

```python
    def n_visual_tokens(self, height, width) -> int:
        h, w = self.resize(height, width)
        return (h // self.factor) * (w // self.factor)
```

Resize, then count grid cells. `//` is integer division; after `smart_resize` both
dimensions are exact multiples of the factor, so this is exact.

For our worked 800×600 chart: `(416//32) × (576//32)` = `13 × 18` = **234**.

```python
    def box_in_tokens(self, bbox_px, img_h, img_w) -> tuple[float, float]:
        h, w = self.resize(img_h, img_w)
        sx, sy = w / img_w, h / img_h
        x1, y1, x2, y2 = bbox_px
        return ((x2 - x1) * sx / self.factor, (y2 - y1) * sy / self.factor)
```

**This is the function that produces slide 7's number.** Four steps:

1. `self.resize(...)` — what size does this image actually become?
2. `sx, sy` — the scale factors. Our 800×600 became 576×416, so `sx = 576/800 = 0.72` and
   `sy = 416/600 = 0.693`. **Note they differ**: `smart_resize` rounds each axis to the grid
   separately, so the aspect ratio shifts very slightly. Using one scale for both axes would
   introduce a small error, always in the same direction.
3. `(x2 - x1) * sx` — the box's width in *resized* pixels.
4. `/ self.factor` — that width expressed in visual tokens.

The return is a *pair* of floats, not one number: a box can be many tokens wide and a
fraction of a token tall, and that is exactly the common case for a chart bar.

```python
    def is_sub_token(self, bbox_px, img_h, img_w, *, rule: str = "axis") -> bool:
        tw, th = self.box_in_tokens(bbox_px, img_h, img_w)
        if rule == "axis":
            return min(tw, th) < 1.0
        if rule == "area":
            return tw * th < 1.0
        raise ValueError(f"unknown rule {rule!r}; use 'axis' or 'area'")
```

📘 **Two definitions of "too small", and they answer different questions.**

- **`axis`** — under one token on *either* axis. A bar 6 tokens tall and 0.4 tokens wide is
  sub-token by this rule. And rightly so: its left and right edges fall inside the same
  block, so the model cannot represent where it starts and stops horizontally.
- **`area`** — the box's total area is under one token. The same bar has area
  `6 × 0.4 = 2.4` tokens, so it is *not* sub-token by this rule.

The `axis` rule is stricter and is the mechanistically meaningful one, which is why it is
the default. This is why the two numbers on slide 7 differ so much: **66.7% by axis, 24.8%
by area**, on the same 7,158 boxes. They are not inconsistent — they measure different
things, and both are reported so nobody can pick the flattering one after the fact.

The `raise` on an unknown rule is the same principle as everywhere else in this file: a
typo'd `rule="Axis"` fails loudly instead of silently taking the `area` branch.

---

## 1.8 The 0–1000 coordinate space

Back to the first question in §1.1: what numbers should the model emit?

**Not pixels** — a model emitting pixels would need to know the image size, and the same
region would get different numbers on a rescaled copy of the same chart.

**Instead:** divide x by the width and y by the height, and multiply by 1000.

```python
def px_to_norm1000(bbox_px, img_w, img_h):
    x1, y1, x2, y2 = bbox_px
    return [1000.0 * x1 / img_w, 1000.0 * y1 / img_h,
            1000.0 * x2 / img_w, 1000.0 * y2 / img_h]
```

Note **x uses width, y uses height**, separately. So the mapping is *anisotropic*: on a
non-square chart, one unit of x and one unit of y are different physical distances. That is
a real property of the space, not an oversight, and Chapter 7 returns to it. It is the space
the model emits in, the datasets annotate in, and the official evaluator scores in — so we
use it throughout rather than converting. `norm1000_to_px` is the exact inverse.

---

## 1.9 ⚠️ The silent failure: why 1000 is not allowed

Here is the most consequential twenty lines in the file.

```python
OFFICIAL_MAX_COORD = 999


def clamp_for_official_evaluator(bbox):
    """Integer box in 0..999, which is the only range the official evaluator accepts.

    ``extract_bounding_boxes()`` in the released evaluator does::

        if all(0 <= elem <= bins - 1 for elem in bbox_floats):
            bboxes.append(bbox_floats)

    with ``bins = 1000`` and **no else branch**. A coordinate of exactly 1000 —
    which is what the model emits for a box touching the right or bottom edge —
    makes the entire box vanish with no error and no warning.
    """
    return [int(max(0, min(OFFICIAL_MAX_COORD, round(v)))) for v in bbox]
```

Read the quoted evaluator code carefully:

```python
if all(0 <= elem <= bins - 1 for elem in bbox_floats):
    bboxes.append(bbox_floats)
```

`bins = 1000`, so `bins - 1 = 999`. The condition says: *append this box only if every one
of its four coordinates is between 0 and 999 inclusive.*

**There is no `else`.** A box with a coordinate of exactly 1000 is not clamped, not warned
about, not counted as an error. It is simply never appended. It disappears.

**Why that is dangerous specifically for charts.** The model emits coordinates in 0–1000
*inclusive*. A bar that runs to the bottom of the plot area, or a legend flush against the
right edge, produces a coordinate of exactly 1000 — and chart elements touch edges
constantly. So a correct prediction about an edge-touching element scores as *no prediction
at all*, and the score drops for a reason invisible in the output.

**The fix, and its cost.** Clamp every coordinate into 0–999 before handing it over:

```python
return [int(max(0, min(OFFICIAL_MAX_COORD, round(v)))) for v in bbox]
```

Read the inner expression outward for one coordinate `v`:

- `round(v)` — the evaluator wants integers.
- `min(999, ...)` — no coordinate may exceed 999.
- `max(0, ...)` — none may fall below 0.
- `int(...)` — `round` returns an int for a float argument in Python 3, but this makes the
  type explicit and handles the case where `v` arrives as a `numpy` scalar.

The cost is one part in a thousand of positional resolution — far below the precision any
box annotation has. In exchange, a whole class of silent score loss disappears.

**Recorded as `DECISIONS.md` 0004.** It is the clearest example in the project of why we
read other people's evaluation code line by line instead of calling it: the bug produces no
error, no warning, and a plausibly-lower number.

---

## 1.10 One more function, in passing

`remap_crop_box_to_original` maps a box predicted *inside a crop* back to the original
image, for a Phase 8.2 experiment that is not part of week 1. It is three steps — scale the
box by the crop's real size, add the crop's position, renormalise — and it is unit-tested
against a hand-computed answer because, in its docstring's words, *"getting this wrong
silently destroys the ablation rather than failing."* A wrong remap puts boxes in
plausible-looking but systematically offset places, and the experiment then reports a
conclusion about our arithmetic as a conclusion about the technique.

---

## 1.11 What to take from this chapter

1. **A visual token is a 32×32-pixel block of the resized image** — the smallest region the
   model represents separately. `factor = patch_size × merge_size`.
2. **A box smaller than one token cannot be localised precisely**, however well the model is
   trained. `box_in_tokens` measures this; `is_sub_token` applies the strict `axis` rule
   (66.7%) or the looser `area` rule (24.8%). Both are reported so nobody can pick the
   flattering one afterwards.
3. **Boxes live in 0–1000 normalised space**, anisotropically — x by width, y by height.
4. **Two real defects shaped this file.** The plan specified the wrong factor for our model
   (0008); the official evaluator silently discards any box touching an edge (0004). Neither
   raises an error, and both move reported numbers.
5. **The recurring principle:** read a value from the thing that actually runs; if you
   cannot, refuse rather than default. `from_processor` raises instead of assuming 32,
   *because assuming it was the bug once already*.

**Next:** Chapter 2 — how one training example is represented, and how we know when two of
them are secretly the same chart.
