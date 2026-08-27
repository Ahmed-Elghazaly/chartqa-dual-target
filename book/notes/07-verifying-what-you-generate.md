# Notes — the synthetic generator, and how to prove a box is right

## What this component is, in plain language

A program that draws charts — bars, lines, pies, scatters — and, for each one, writes down four things it
knows *by construction*: the underlying table, the question, the exact answer, and a rectangle around every
chart element the answer depends on.

The rectangles are the interesting part. When you draw a bar with matplotlib you get back an object, and that
object can tell you exactly which pixels it occupies. So the box is not estimated, not annotated, not
predicted. It is read off the drawing itself.

Which raises the question this note is really about: if the box comes from the drawing, how would you ever
know it was wrong?

## Why it exists — what breaks without it

The project trains a model to point at chart evidence and then compute from it. That needs examples of
correct pointing. Real datasets have some, but the questions that come with an unambiguous *plan* are rare —
about 5.7% by the original estimate, 14% as measured, and most of those turn out to be trivial lookups.

Synthetic charts have no such problem: we chose the numbers, so we know everything. The catch is that a bug
in the generator produces confidently wrong training data, in bulk, with no error message. A model trained on
boxes that are systematically ten pixels too high learns to be systematically ten pixels too high, and you
find out in Phase 9.

So the generator ships with a verifier, and the verifier is the component that actually matters.

## What surprised me

**Three verification designs failed before one worked, and each failed for a different reason.**

The first idea was *displacement*: slide the box sideways and check the fill collapses. If the box is really on
the bar, moving it off should show. It reported failures on perfectly exact boxes — because on a bar chart,
sliding a box sideways lands it on **the next bar**, which is the same colour and just as full. The check was
measuring "is there a bar here", and there was.

The second was *tightness*: grow the box and check the ink density drops. This one is elegant. Expanding a
box by a factor `f` multiplies its area by `(1 + 2f)²` while the ink inside stays the same, so an exact box
loses `1 - 1/(1+2f)²` — 65.4% at `f = 0.35` — regardless of what shape it is. Measurement agreed beautifully:
60.6% to 65.6% across bars, wedges and markers.

And it is useless for the thing that matters most, for exactly the reason it is elegant. Being scale-free
means it cannot see a box that is simply *too big*: a pie wedge box grown to 1.8× still loses the same
fraction, because all the ink is still inside both. It passed.

The third works: compare the box to **the tight extent of the element's own ink nearby**. Instead of asking
"does this box have the right amount of ink in it", ask "is this box where the ink actually is". Exact boxes
score 0.84–0.99; boxes shifted, shrunk or grown score 0.31–0.38. Nothing about that comparison depends on the
element being a rectangle.

**Fill fraction cannot be a threshold, and this took a while to accept.** A bar fills its bounding box almost
completely. A circular marker inscribed in its bounding square fills π/4 = 78.5% and no more. A thin pie
sliver's *tight* bounding box is mostly empty — one 10° wedge measured 21% — and that is correct geometry,
not a bad box. There is no single number, and the first version of the threshold table rejected the sliver
for having the fill a sliver must have.

**The verifier found three real bugs I would not have found by reading the code.** A five-colour palette
wrapped to a seventh category and gave two bars the same colour. Marker boxes omitted the stroke, which
matplotlib centres on the path so half of it lies *outside* the marker — 97.2% containment instead of 100%.
And on area charts the translucent fill was painted over the markers, changing their colour.

**And the same class of mistake nearly cost us a whole dataset.** The tightness idea was later reused to audit
RefChartQA's boxes, and it failed that dataset at 84% — below the 90% gate, which would have removed
RefChartQA from training entirely. Rendering the rejected examples and looking at them showed the boxes were
fine: RefChartQA often grounds on the *printed number inside a bar*, and growing a box that sits on a number
inside a bar captures more bar. A criterion valid for one geometry, applied to another. The same error, twice,
three weeks apart in project time.

## What I decided, and what I rejected

**Verify on a recoloured render, not the delivered image.** Pixel matching has a tolerance, and a muted style
colour can fall inside that tolerance of antialiased text — one near-grey palette produced an element colour
where 48 of its 686 "matched" pixels were actually letters. So before checking, every element is repainted in a
saturated sentinel colour that appears nowhere else on a chart; then the real colours are restored before
saving. Colour moves no artist, so the geometry checked is the geometry shipped. The alternative — banning
muted palettes — would have thrown away realistic charts to dodge a measurement artefact.

**Pair every acceptance test with an adversarial one.** A verifier that accepts everything is worse than no
verifier, because it certifies. Every test that asserts an exact box passes also asserts that shifted, shrunk,
grown and far-away boxes fail.

**Exclude line charts from the real-data box extraction.** ChartQA's annotations turn out to contain element
boxes — a genuinely useful discovery — but for line charts the boxes are the *segments between* points, so a
point's position is recoverable while its box size is stated nowhere. Inventing a marker size would put
fabricated boxes into training data. Lines are 13% of the dataset; the alternative is unverifiable.

## Which concept a reader must understand first

**A tight bounding box does not imply a high fill.** That implication holds only for shapes that are already
rectangles, and almost nothing on a chart is. Once that is clear, why the threshold table has four geometry
classes — and why fill is measured but never used to decide — stops looking like over-engineering.

Second: **the difference between a necessary condition and a sufficient one.** "The box contains ink" is
necessary. It is nowhere near sufficient — it cannot tell whether the box is on the element the *question* is
about. Most verification mistakes in this project came from treating one as the other, and the fix was never a
stricter threshold. It was looking at the data.

## Forward pointers

- `DECISIONS.md` 0038, 0039, 0040 — the three verification designs, with the measurements that killed two.
- `DECISIONS.md` 0047 — the same tightness idea failing on real data, and being caught by looking.
- Phase 9.5 uses the sealed holdout style seeds, which is why `is_holdout` is the only place that decides.
