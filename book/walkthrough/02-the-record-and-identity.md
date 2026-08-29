# Chapter 2 — The record, and knowing when two things are the same

**Files:** `data/records.py` (152 lines), `data/dedup.py` (173 lines).

We have four data sources: ChartQA, RefChartQA, ChartQAPro and our own generator. Each
stores things differently. Chapter 2 is about the one shape everything is converted into,
and the harder question that follows: **when are two records secretly the same example?**

---

## 2.1 `ChartRecord` — one shape for everything

```python
@dataclass(frozen=True)
class ChartRecord:
    record_id: str
    source: Source
    split: Split
    image_path: str
    image_sha256: str        # of the DECODED PIXELS — see `image_content_sha256`
    question: str
    answer: str | None
    question_kind: QuestionKind
    table: dict | None = None
    boxes: list[list[float]] | None = None       # 0-1000 normalised [x1,y1,x2,y2]
    plan: dict | None = None                     # typed tree, only when known exactly
    meta: dict[str, Any] = field(default_factory=dict)
```

Most fields are obvious. Four are worth stopping on.

**`boxes` is `None`, not `[]`, when there are none.** These mean different things: `None` is
"no annotation exists for this record" — a ChartQA line chart, say, or a chart whose
annotation file is missing — while `[]` would be "annotated, and there are genuinely zero
regions". Collapsing them would make a missing annotation indistinguishable from a chart
with nothing to point at.

📘 Both datasets populate this field, but they mean different things by it. **ChartQA**
annotates *every element of the chart* — each bar, each slice — independently of any
question. **RefChartQA** annotates, per question, *the regions that question's answer needs*.
So RefChartQA is directly scorable for grounding, while ChartQA's boxes are raw material from
which per-question evidence is selected (Chapter 5 recovers which elements a question uses).

**`plan` carries the comment "only when known exactly".** A plan is the arithmetic (Chapter
4). We only fill it when we can prove it — never a guess. Chapter 5 is entirely about that
proof.

**`meta` is a free dictionary**, because sources carry things the others do not: chart type,
image size, difficulty level, per-element details.

⚠️ **The free dictionary caused the worst bug in the project.** `build_target` joins a plan's
labels against `meta["elements"]`. The synthetic reader wrote the identical data under
`meta["evidence"]`. Same content, same shape, one different word — so every synthetic
record fell through to a fallback that labelled its evidence `item1, item2, …`, the plan
then referenced labels that matched nothing, and the record was refused. **All 12,000
stage-1 training targets, silently.** Nothing raised; the training set was simply empty.

The fix is the constant on the line above:

```python
ELEMENTS_KEY = "elements"
```

Both readers and the target builder now import it, and a test fails on any hand-spelled
`"elements"` in those files. Recorded as `DECISIONS.md` 0071.

📘 **The general lesson:** a string key shared across modules is an untyped interface. If
two sides must agree on a name, put the name in one place and import it.

---

## 2.2 Hashing the image — pixels, not bytes

```python
    image_sha256: str        # of the DECODED PIXELS
```

📘 **A hash** is a function turning any data into a fixed-length fingerprint. Same input →
same fingerprint; different input → almost certainly different. SHA-256 gives 64 hex
characters. It lets us ask "are these two things identical?" by comparing 64 characters
instead of megabytes.

The obvious way to hash an image is to hash the file. **That is wrong here**, and the
comment is emphatic about why.

RefChartQA is built from ChartQA's images — but re-saved. Re-encoding a PNG changes the
bytes (different compression, different metadata) while the *picture* is unchanged. So:

- comparing **file bytes**: **0 of 4,000** RefChartQA images match ChartQA;
- comparing **decoded pixels**: **99.9%** match.

Since the whole point is to catch the same chart appearing in training and in test, a check
that finds nothing is worse than no check — it produces false confidence.

So the hash is taken over the decoded pixel array:

```python
    digest.update(array.tobytes())
```

with the image's dimensions mixed in first, so that two images with the same pixel values in
different shapes cannot collide.

---

## 2.3 Question identity

```python
def normalise_question(q: str) -> str:
    q = unicodedata.normalize("NFKC", q).strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q.rstrip(" ?.!").strip()
```

Four normalisations, each removing a difference that should not count as a difference:

- `unicodedata.normalize("NFKC", …)` — folds unicode variants to one form. A curly
  apostrophe and a straight one, a full-width character and its ASCII twin, become the same.
- `.lower()` — case.
- `re.sub(r"\s+", " ", q)` — any run of whitespace becomes one space.
- `.rstrip(" ?.!")` — trailing punctuation.

So *"What is the median value?"* and *"what is the median value"* are one question.

---

## 2.4 ⚠️ Why identity needs the image too

```python
def dedup_key(image_sha256: str, question: str) -> str:
    qh = hashlib.sha256(normalise_question(question).encode("utf-8")).hexdigest()[:16]
    return f"{image_sha256[:16]}:{qh}"
```

The key is **image hash + question hash**, and the docstring says the image half is
load-bearing. It was measured, not assumed:

> generic questions such as "what is the median value" appear on many different charts —
> one of them on three separate ChartQA test charts

Keying on question text alone would call those three *the same example*. They are three
different charts with the same wording. The consequence would have been **phantom leakage**:
the pipeline reporting that training data had leaked into test, when nothing had. A team
that trusts that alarm throws away good data; a team that learns to ignore it has disabled
its leak detector.

Recorded as `DECISIONS.md` 0028.

**`record_id`** is built the same way, with one addition:

```python
def make_record_id(source, split, image_sha256, question, index=None):
```

`index` disambiguates one chart carrying two questions that normalise identically — rare,
but it happens in ChartQA. Everything is hashed, so the id is deterministic: same input,
same id, on any machine, in any run.

---

## 2.5 Merging duplicates

Because RefChartQA is derived from ChartQA, mixing them naively counts the same question
twice. The module docstring states the stake plainly:

> a training set that is 15% smaller than believed is a silent confound on every result
> that follows.

The rule is **merge, not drop and not double-count**: keep the answer, union the boxes, keep
any exact plan.

Why merging rather than dropping one: the two copies carry *different information*. ChartQA
has the gold answer and the data table; RefChartQA has the boxes. Dropping either loses
something the other did not have.

When answers disagree, a fixed priority decides:

```python
SOURCE_PRIORITY = {"chartqa": 0, "chartqapro": 1, "refchartqa": 2, "synthetic": 3}
```

ChartQA wins because its labels are what the official metric scores against.

Boxes are unioned with a tolerance:

```python
BOX_EPSILON = 1.0
```

Two boxes within 1.0 unit (on the 0–1000 scale — so one part in a thousand) are the same
box. Without this, two annotations of the same bar differing in the seventh decimal would
both be kept, and the model would be trained to emit a duplicate.

---

## 2.6 Two properties that are easy to get wrong

The docstring names both, which is the sign of someone having thought about them rather than
discovered them later.

**Merging is order-independent.** Records arrive from different loaders in whatever order a
mixture iterates. If merge order changed the outcome, two runs of the same pipeline would
produce different training sets. The merge is deliberately commutative, and the test
**shuffles the input** and checks the result is unchanged — testing the property, not one
example of it.

**Splits are never merged across.**

> A train record and a test record that share a key are a *leak*, not a duplicate. Merging
> them would hide exactly the thing rule 1 exists to prevent.

This one is subtle and worth re-reading. The dedup code exists to *remove* duplicates. But
if a training record and a test record are the same example, silently merging them makes the
leak vanish from the report while remaining entirely present in the data. So a cross-split
collision is **reported, never resolved**. The tool that cleans data must not be allowed to
clean away the evidence of contamination.

---

## 2.7 What to take from this chapter

1. **One record type for four sources**, with `None` meaning "not annotated" and `[]`
   meaning "annotated as empty".
2. **Images are hashed by decoded pixels, not file bytes.** Bytes find 0 of 4,000 matches;
   pixels find 99.9%. The wrong check would have produced confident silence.
3. **Identity is image + question**, because generic questions repeat across charts and a
   question-only key invents leaks that do not exist (0028).
4. **Duplicates are merged, not dropped** — the two copies carry different halves of the
   information.
5. **The dedup step is forbidden from resolving a cross-split collision**, because that is a
   leak and hiding it is the one thing it must never do.

**Next:** Chapter 3 — reading ChartQA off the disk, and where the annotation-coverage
finding on slide 5 comes from.
