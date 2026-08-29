# Chapter 4 — The plan language

**Files:** `plans/schema.py` (178), `plans/executor.py` (198), `plans/roundtrip.py` (164).

This is the idea the project is named after. A **plan** is the arithmetic, written down as
data, so that something other than the model can run it.

---

## 4.1 What a plan is

```json
{"op": "difference", "args": ["West", "South"]}
```

An **operation** and its **arguments**. Arguments are either:

- a **string** — always the label of an evidence item ("West" means "the value of the
  evidence item labelled West"), or
- another **plan** — which is what makes it a *tree* rather than a single step.

```json
{"op": "difference", "args": ["West", {"op": "mean", "args": []}]}
```

*"West minus the mean of everything."* The nesting is the point: it can express multi-step
reasoning, and each step is separately inspectable.

📘 **Why "typed expression tree" and not prose.** Chain-of-thought reasoning writes the steps
in English. English cannot be executed, so nothing can check it — a model can write plausible
reasoning and a wrong answer and no automatic process notices. A tree can be executed. That
is the entire trade: less expressive, verifiable.

---

## 4.2 The executor

```python
def execute(node, evidence, *, _depth_checked=False):
    if not _depth_checked:
        d = plan_depth(node)
        if d > MAX_DEPTH:
            raise ExecutorError(f"plan depth {d} exceeds {MAX_DEPTH}")
    if not isinstance(node, dict):
        return node
    op = node.get("op")
    if op not in OPS:
        raise ExecutorError(f"unknown op: {op!r}")
```

**The depth guard is first**, before anything is evaluated.

```python
MAX_DEPTH = 4

def plan_depth(node) -> int:
    """Depth of a typed tree. Computed, never trusted from the model."""
    if not isinstance(node, dict):
        return 0
    args = node.get("args") or []
    return 1 + max([plan_depth(a) for a in args], default=0)
```

📘 A **recursive** function: the depth of a tree is 1 plus the depth of its deepest child. A
leaf (not a dict) has depth 0. `default=0` handles `max([])`, which otherwise raises.

The docstring's "never trusted from the model" matters. The model emits this structure. A
deeply nested plan — accidental or otherwise — would recurse until Python's stack limit and
crash the process. We measure the depth ourselves and refuse above 4.

`_depth_checked=True` is passed on recursive calls, so the walk happens once rather than at
every node.

**Argument resolution** is where the design shows:

```python
    def resolve(a):
        """A bare string is ALWAYS an evidence label (decision 0016)."""
        if isinstance(a, dict):
            return execute(a, evidence, _depth_checked=True)
        if isinstance(a, str):
            if a not in by_label:
                raise ExecutorError(f"lookup of unknown evidence label: {a!r}")
            return by_label[a].value
        return a
```

**"A bare string is ALWAYS an evidence label."** The alternative — sometimes a label,
sometimes a literal value — would mean `"5"` is ambiguous: the number five, or an item
labelled "5"? Chart categories are frequently years and numbers, so this is common, not
exotic. The rule removes the ambiguity by fiat, and an unknown label **raises** rather than
defaulting to zero.

📘 That raise is load-bearing. If an unknown label returned 0, a plan referencing something
the model never put in evidence would still produce a number — a plausible one — and the
round-trip check below would sometimes accept it. The executor's refusals are what make its
agreements meaningful.

### Operations that fold over everything

```python
    if op in ("sum", "mean", "median", "min", "max"):
        check_units(args)
        values = numbers(args) if args else all_values()
```

With arguments, operate on those. **With empty arguments, operate on all the evidence.** So
`{"op": "mean", "args": []}` means "the mean of everything on the chart".

That compact form exists because the schema caps `args` at 4 items, and "the mean of these
eleven bars" would not fit. Recorded as `DECISIONS.md` 0041.

⚠️ **It also caused a 100%-data-loss bug.** Evidence for a training target is normally
selected by the labels the plan names. But `difference("Alpha", mean-of-everything)` names
only one label — so the evidence list held one item, so the mean was that item, so the
difference was exactly **zero**. Every one of the 6,000 level-4 records failed, silently.
Two individually-correct decisions, fatal in combination. `DECISIONS.md` 0071.

```python
    def check_units(seq):
        units = {by_label[a].unit for a in seq if isinstance(a, str) and ...}
        if len(units) > 1:
            raise ExecutorError(f"unit mismatch: {sorted(units)}")
```

Adding a percentage to a dollar amount is refused. The model can emit that plan; it cannot
get a number out of it.

---

## 4.3 The schema, and what it deliberately does not do

```python
"""Auditing it in Phase 0 confirmed it is sound and that it deliberately delegates:
it rejects a coordinate above 1000, a ninth evidence item, an unknown operation, an
extra key and a missing field, but it accepts inverted boxes, zero-area boxes, a
coordinate of exactly 1000, plans deeper than four, and a ``lookup`` of a label that
is not in ``evidence``."""
```

📘 **A JSON Schema** checks *shape*: types, required fields, allowed values, list lengths. It
cannot check *relationships between fields* — that a plan's label appears in the evidence, or
that `x2 > x1`. Those five gaps are not schema bugs; they are the limit of what the format
expresses.

So this file states them explicitly and implements them alongside. The value is that the
list is **written down**. An unlisted gap is one nobody has thought about.

Two of the five carry measured hazards:

**A coordinate of exactly 1000** is silently discarded by the official evaluator (Chapter
1, §1.9). The schema allows it; we flag it as a validation finding rather than clamping
quietly, so it appears in a report rather than being fixed invisibly.

**`maxItems: 8` on evidence is "a hazard, not an allowance".** This phrasing is precise and
worth absorbing. The schema *permits* eight boxes. That does not mean emitting eight is
safe:

> dataset-level AP collapses from 1.0000 to 0.3243 when three extra boxes are appended per
> image (`DECISIONS.md` 0014)

Perfect boxes plus three spurious ones per image cost **two-thirds of the score**. So the
prompt tells the model to emit as few as possible, and the limit is a ceiling, not a target.

---

## 4.4 The round-trip check

```python
def check_record(record):
    """Run one record's plan against its own evidence and compare."""
    plan = record.get("plan")
    stated = str(record.get("model_answer", ""))
    if not isinstance(plan, dict) or not plan.get("op"):
        return RoundTrip("no_plan", stated=stated)
    evidence = [EvidenceItem(str(e.get("label")), e.get("value"), e.get("unit")) ...]
    try:
        got = execute(plan, evidence)
    except Exception as exc:
        return RoundTrip("raises", stated=stated, error=f"{type(exc).__name__}: {exc}")
    return RoundTrip("agrees" if answers_agree(stated, got) else "disagrees", ...)
```

**This is the check that makes an answer verifiable.** Take what the model said, run its own
plan over its own evidence, and see whether the two match.

**Four outcomes, deliberately distinct:**

| outcome | meaning |
|---|---|
| `no_plan` | no plan emitted at all |
| `raises` | the plan is malformed — bad label, unit clash, too deep |
| `disagrees` | the plan ran and produced a *different* number |
| `agrees` | the plan reproduces the stated answer |

Collapsing these to pass/fail would lose the diagnosis. `raises` is a formatting problem;
`disagrees` is a reasoning problem. They call for different fixes.

**No gold answer is involved.** That is what makes this usable at test time on unlabelled
questions — and it is the 69% figure on slide 10.

⚠️ **The comparison had a bug worth knowing.** It originally used the official
`relaxed_correctness`. In the published implementation, a gold answer of **zero** cannot be
compared by relative error (division by zero), so it falls back to string equality — making
`"0"` and `"0.0"` different. Correct to reproduce *when reporting a benchmark score*. Wrong
here: this asks whether a plan reproduces its own answer, and a correct result of zero is
correct. It discarded **512** valid training records, every one a `difference` whose two
operands were equal. `answers_agree` now compares numerically with a symmetric 5% tolerance.
`DECISIONS.md` 0071.

---

## 4.5 What to take from this chapter

1. **A plan is an operation and arguments**, where a bare string is *always* an evidence
   label and a nested dict is a sub-plan. That rule exists because chart labels are often
   numbers.
2. **The executor refuses rather than defaults** — unknown label, unknown op, mismatched
   units, depth over 4. Its refusals are what make its agreements mean something.
3. **Empty args means "fold over all evidence"**, a compaction forced by the 4-argument cap
   — and the interaction of that with label-based evidence selection cost 100% of the
   compositional training data.
4. **The schema checks shape; five relationships it cannot express are listed and checked
   separately.** Writing the list down is the point.
5. **`maxItems: 8` is a hazard, not an allowance** — three extra boxes cost two-thirds of AP.
6. **The round-trip check needs no gold answer**, which is what makes it usable at test time,
   and it reports four distinct outcomes because they have different causes.

**Next:** Chapter 5 — where plans for *real* charts come from, and why only 14% of questions
yield one.
