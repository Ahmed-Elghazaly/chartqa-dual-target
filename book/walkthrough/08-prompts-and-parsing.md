# Chapter 8 — Prompts, and repairing what comes back

**Files:** `prompting/prompts.py` (205), `prompting/parsing.py` (258).

Two sides of one interface: what we ask the model for, and what we accept back.

---

## 8.1 Three prompts, each hashed

| prompt | length | used for |
|---|---:|---|
| `PLAIN_PROMPT` | 27 tokens | the published-baseline condition |
| `STRUCTURED_PROMPT` | 980 tokens | asking an *untrained* model for the full record |
| `TRAINING_PROMPT` | 117 tokens | what the *fine-tuned* model sees |

```python
PLAIN_PROMPT = "{question}\nAnswer the question using a single word or phrase."
```

**Copied verbatim from the model's own technical report.** That is what makes our
plain-prompt number comparable to their published one — a different wording would be a
different measurement.

Each prompt's exact text is SHA-256 hashed and recorded in the pre-registration, because a
prompt is a setting like a learning rate. Editing one silently would change results with no
trace.

**Why the training prompt is 8× shorter than the structured one.** The structured prompt
explains the schema, the limits and two worked examples — an untrained model needs all of
it. A fine-tuned model has seen the format 24,000 times; the explanation is pure cost.

⚠️ And it was not merely cost. With the 980-token prompt plus ~250 visual tokens plus the
target, a training example measured **1,363–1,498 tokens** against a limit of **1,024**. Every
example would have been silently truncated — the model trained on chopped-off targets, and
nothing would have raised. The 117-token prompt leaves 389 tokens of headroom.
`DECISIONS.md` 0064.

⚠️ A smaller one worth telling because it is so easy to make: `TRAINING_PROMPT` was copied
from a string that had been `.format()`ed, so its braces were doubled — `{{` instead of `{`.
The placeholder `{question}` was never substituted. The prompt was a *literal* string
containing the word "question".

---

## 8.2 One measured lesson about prompt design

Our first structured prompt asked for pretty-printed JSON. Measured:

| | pretty | compact |
|---|---:|---:|
| tokens for the identical record | 253 | **141** |

**Pretty-printing costs 80% more tokens for the same content**, and a third of outputs were
being cut off at the generation limit before finishing. Demanding compact JSON cut the median
output **2.6×** and raised valid-JSON from 58% to 75%.

⚠️ **And a lesson about method.** Those three prompt iterations ran on **12, 20 and 24**
questions. At n=24 the confidence interval on a 50% rate is roughly **±20 points** — wider
than any effect being claimed. We were tuning on noise, and one "improvement" (a fourth
version) made things measurably worse when finally tested properly. Sample sizes are now
computed from what they can resolve *before* running. `DECISIONS.md` 0062.

---

## 8.3 The parser's one rule

```python
"""**Non-negotiable rule 3: invalid outputs count as failures.** That is the whole design
constraint here. It would be easy to write a parser that always returns *something*: pull
the last number out of the text, guess a box, fall back to an empty plan. Every one of
those choices converts a model failure into a silently plausible number, and the resulting
score would measure the parser rather than the model."""
```

**"The score would measure the parser rather than the model."** That sentence is the entire
chapter.

A lenient parser is tempting because it raises every number you report. It also makes those
numbers meaningless — a sufficiently clever parser could score well with a model that emits
nothing useful.

So three categories, explicitly separated:

**1. Recovery from transport noise** — a code fence, a sentence before the JSON, a trailing
comma, smart quotes. *The model produced the record; the transport was untidy.* Legitimate,
and **every recovery is counted** so the rate is visible.

**2. Invention** — supplying a field the model did not produce. **Never.**

> A record missing `model_answer` is a failure

**3. Discarding what the schema cannot hold** — an evidence item with no `bbox`, or the
ninth item when the cap is eight. Allowed, because it *adds nothing*:

> the alternative is to throw away an otherwise good record entirely. The model is
> instructed to order evidence most-important-first, so keeping the first eight respects its
> own ranking, and `DECISIONS.md` 0014 wants fewer boxes anyway.

**The rule across all three: we may drop, we may unwrap, we never add.**

Measured on 200 real generations, dropping and capping moved schema validity from **35.5% to
46.5%**, and usable records from 49 to 61. Real recovery, no invention.

---

## 8.4 What the repairs actually are

```python
STRAY_QUOTE_AFTER_ARRAY_RE = re.compile(r"(\]\s*)\"(\s*[},])")
SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
REQUIRED_FIELDS = ("answerable", "evidence", "plan", "model_answer")
```

The stray-quote pattern is oddly specific, and that is the point — it was written after
*seeing* the failure. The model emits `"bbox":[1,2,3,4]"` — a quote after the closing
bracket, 11–21 times in a single affected record. The regex matches `]`, then a quote, then
`}` or `,`, and deletes just the quote.

📘 It is deliberately narrow. A general "remove suspicious quotes" pass would corrupt records
that were fine. A repair should fix the failure that was observed and nothing else.

Repairs are tried in **combination**:

```python
        if STRAY_QUOTE_AFTER_ARRAY_RE.search(body):
            cleaned = STRAY_QUOTE_AFTER_ARRAY_RE.sub(r"\1\2", body)
            extended.append((cleaned, [*repairs, "removed stray quote after an array"]))
            if TRAILING_COMMA_RE.search(cleaned):
                extended.append((TRAILING_COMMA_RE.sub(r"\1", cleaned), [...]))
```

A record can be broken two ways at once, and fixing one exposes the other. Each candidate
carries the **list of repairs applied**, so a parsed record always knows how much of it was
ours.

```python
        missing = [f for f in REQUIRED_FIELDS if f not in obj]
        if missing:
```

Four required fields. Missing any is a failure — *"supplying one would mean scoring our
default instead of the model's output."*

---

## 8.5 Two numbers, not one

```python
class ParseResult:
    """What came back, and exactly how much of it was ours rather than the model's."""
```

`ParseStats` tracks **`valid`** (parsed as JSON with all four fields) and **`schema_valid`**
(also satisfies every schema constraint) separately.

They differ a lot, and the gap is the diagnosis. On the 200-question baseline: **66.5%
valid, 35.5% schema-valid.** So roughly a third of outputs parse cleanly and are still
unusable — `args` too long, duplicate labels, an operation name that does not exist
("average" for `mean`).

Reporting only "valid JSON" would have said the format was mostly working. Reporting both
showed the binding constraint was the schema, and that is what the prompt was then changed
to address.

---

## 8.6 What to take from this chapter

1. **A prompt is a setting.** All three are hashed and pre-registered; the plain one is
   copied verbatim from the model's own report so the comparison is honest.
2. **Pretty-printed JSON costs 80% more tokens** than compact for identical content, and
   compaction was worth 17 points of valid-JSON rate.
3. **The 980-token prompt did not fit the training budget** — examples were 1,363–1,498
   tokens against a 1,024 limit, silently truncating everything.
4. **Three prompt iterations ran on 12–24 samples**, where the interval is ±20 points. We
   were tuning on noise, and one "improvement" was measurably worse.
5. **A lenient parser scores the parser, not the model.** Drop, unwrap, never add.
6. **Repairs are narrow, combinable, and counted** — written after seeing the actual failure,
   never a general "clean it up" pass.
7. **Valid and schema-valid are tracked separately**, and the 31-point gap between them was
   the finding that redirected the prompt work.

**Next:** Chapter 9 — assembling training mixtures, and the machinery that keeps the test
split sealed.
