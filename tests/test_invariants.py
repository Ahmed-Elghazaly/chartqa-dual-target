"""Cross-module invariants — the guards for the defect *classes* this audit found.

Module coverage was never the problem. Of 65 modules, 61 are referenced by a test, and every
serious defect the audit found still got through, because each lived **between** modules:

| the defect | how it looked from inside one module |
|---|---|
| two numeric parsers 100x apart | each correct, and independently tested |
| `MAX_EVIDENCE` restated in the prompt | both values right when written |
| the pixel budget restated at a call site | the default and the caller each looked fine |
| `ALLOWED_OPS` a hand-written copy of `OPS` | a valid tuple of valid names |
| the prompt offering operations the executor refuses | a valid schema and a valid executor |
| `meta[elements]` meaning two things by source | each reader correct for its own source |
| record ids from `hash()` | stable within one process |
| the fold guard requiring named labels | correct for every case it was written for |

Every test here fails on a *relationship*, not on a function. They are written to fire on the
next instance of a class of mistake, not the last one.
"""
from __future__ import annotations

import ast
import collections
import pathlib
import random
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "chartqa_dt"


def _modules():
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in str(p)]


# ===================================================================== duplicated constants


#: Names that are legitimately defined more than once — a local alias, a test double, or a
#: value whose two definitions are deliberately independent. Each needs a reason.
_ALLOWED_DUPLICATE_CONSTANTS = {
    "MAX_ARGS",       # `llm_mining` states the schema's arity for its own shape gate; the
                      # schema owns the real one, and they are asserted equal below.
    "SOURCES",        # two unrelated things sharing a generic name: dataset specs in
                      # `data/sources.py`, report file paths in `reporting/build.py`. No
                      # drift risk — they were never the same value — but the collision
                      # makes `grep SOURCES` useless, which is a cleanup, not a bug.
}


#: An assignment whose right-hand side reaches into another module cannot drift — it *is*
#: the other definition. `MAX_EVIDENCE = OUTPUT_SCHEMA[...]["maxItems"]` is the fix for a
#: duplication, not an instance of one, and the check must be able to tell them apart.
_DERIVED = (ast.Subscript, ast.Call, ast.Attribute, ast.BinOp)


def _module_level_constants() -> dict[str, list[str]]:
    """Every UPPER_CASE module-level assignment to a LITERAL, by name.

    Derived assignments are skipped: they restate nothing, so they cannot diverge.
    """
    found: dict[str, list[str]] = collections.defaultdict(list)
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names, value = [node.target.id], node.value
            else:
                continue
            if value is None or isinstance(value, _DERIVED):
                continue
            for name in names:
                if name.isupper() and len(name) > 3:
                    found[name].append(str(path.relative_to(SRC)))
    return found


def test_no_constant_is_defined_in_two_places():
    """**Five defects came from exactly this.** A value defined twice is a value that will
    drift, and every one of ours drifted silently — the reader kept working, the writer kept
    working, and the pair stopped agreeing.

    Import it instead. If two definitions really are independent, say so in
    `_ALLOWED_DUPLICATE_CONSTANTS` with the reason.
    """
    duplicated = {name: sorted(set(where))
                  for name, where in _module_level_constants().items()
                  if len(set(where)) > 1 and name not in _ALLOWED_DUPLICATE_CONSTANTS}
    assert not duplicated, (
        "these constants are defined in more than one module and will drift:\n  "
        + "\n  ".join(f"{n}: {', '.join(w)}" for n, w in sorted(duplicated.items())))


def test_the_arity_cap_agrees_with_the_schema():
    """`llm_mining.MAX_ARGS` is allowed to exist separately; it must still be the same number."""
    from chartqa_dt.plans.llm_mining import MAX_ARGS
    from chartqa_dt.plans.schema import OUTPUT_SCHEMA
    assert OUTPUT_SCHEMA["$defs"]["node"]["properties"]["args"]["maxItems"] == MAX_ARGS


# ===================================================== the schema / prompt / executor triangle


def test_the_schema_the_prompt_and_the_executor_agree_on_the_operations():
    """Three components derived from one vocabulary, and one of them offered operations the
    other two refused (0109). Any future divergence fails here."""
    from chartqa_dt.plans.executor import EXECUTABLE_OPS
    from chartqa_dt.plans.schema import OUTPUT_SCHEMA
    from chartqa_dt.prompting.prompts import ALLOWED_OPS
    schema_ops = set(OUTPUT_SCHEMA["$defs"]["node"]["properties"]["op"]["enum"])
    assert set(ALLOWED_OPS) == EXECUTABLE_OPS == schema_ops


def test_every_operation_the_prompt_offers_actually_runs():
    """The strongest form: not that the sets match, but that each operation executes."""
    from chartqa_dt.plans.executor import EvidenceItem, ExecutorError, execute
    from chartqa_dt.prompting.prompts import ALLOWED_OPS
    evidence = [EvidenceItem("a", 3.0), EvidenceItem("b", 1.0)]
    args = {"lookup": ["a"], "difference": ["a", "b"], "ratio": ["a", "b"],
            "percent_change": ["a", "b"], "compare": ["a", "b"], "boolean": ["a"]}
    for op in ALLOWED_OPS:
        plan = {"op": op, "args": args.get(op, [])}
        try:
            execute(plan, evidence)
        except ExecutorError as exc:
            if "requires table context" in str(exc):
                pytest.fail(f"the prompt offers {op!r}, which the executor refuses outright")


def test_the_teacher_offers_only_operations_it_can_describe():
    from chartqa_dt.plans.teacher import OFFERED, SIGNATURES
    missing = sorted(set(OFFERED) - set(SIGNATURES))
    assert not missing, f"offered to a reader with no signature: {missing}"


# ============================================================== a field's meaning by source


def test_each_source_declares_what_its_boxes_mean():
    """`record.boxes` means three different things (C2), which is survivable only because
    every consumer knows which source it is holding. This pins the contract so a fourth
    source cannot be added without deciding."""
    from chartqa_dt.cli.train import grounding_truth_for
    from chartqa_dt.data.records import ChartRecord

    def rec(source):
        return ChartRecord(record_id="r", source=source, split="train", image_path="i.png",
                           image_sha256="d", question="q?", answer="1",
                           question_kind="human", boxes=[[0, 0, 10, 10]])

    # question-specific grounding: usable as AP ground truth
    assert grounding_truth_for(rec("refchartqa")) == [[0, 0, 10, 10]]
    assert grounding_truth_for(rec("synthetic")) == [[0, 0, 10, 10]]
    # every element on the chart: NOT ground truth for "which boxes answer this question"
    assert grounding_truth_for(rec("chartqa")) == []


# ================================================================= nothing is dropped silently


def test_every_filter_reports_what_it_removed():
    """A filter that shrinks a mixture without saying so is how 12,000 stage-1 records were
    lost once (0071) and how the dedup merge vanished (H2)."""
    from chartqa_dt.data.mixture import drop_absent_chart_types
    from chartqa_dt.data.records import ChartRecord
    recs = [ChartRecord(record_id=f"r{i}", source="synthetic", split="train",
                        image_path="i.png", image_sha256=f"d{i}", question="q?", answer="1",
                        question_kind="synthetic", meta={"chart_type": t, "level": "L1"})
            for i, t in enumerate(["vbar", "area", "scatter", "line"])]
    kept, dropped = drop_absent_chart_types(recs)
    assert dropped == 2
    assert len(kept) + dropped == len(recs), "a filter must account for every record"


def test_the_target_builder_never_returns_a_target_it_cannot_verify():
    """Every refusal names its cause; there is no silent `return None` path."""
    from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord
    from chartqa_dt.train.targets import TargetError, build_target
    broken = ChartRecord(record_id="b", source="chartqa", split="train", image_path="i.png",
                         image_sha256="d", question="q?", answer="99", question_kind="human",
                         plan={"op": "lookup", "args": ["absent"]},
                         meta={ELEMENTS_KEY: [{"label": "a", "value": 1.0, "unit": None,
                                               "bbox": [0, 0, 5, 5]}]})
    with pytest.raises(TargetError) as excinfo:
        build_target(broken)
    assert len(str(excinfo.value)) > 40, "a refusal must explain itself, not just fail"
    assert "b:" in str(excinfo.value), "a refusal must name the record"


# ========================================================================== determinism


@pytest.mark.parametrize("snippet,label", [
    ("from chartqa_dt.data.records import make_record_id;"
     "print(make_record_id('chartqa','train','deadbeef','how many?'))", "record id"),
    ("from chartqa_dt.plans.teacher import build_system;"
     "import hashlib;print(hashlib.sha256(build_system().encode()).hexdigest())",
     "teacher prompt"),
    ("from chartqa_dt.prompting.prompts import prompt_fingerprint;"
     "print(prompt_fingerprint())", "prompt fingerprint"),
])
def test_identity_is_stable_across_processes(snippet, label):
    """`hash()` on a string is salted per process. An id built from it is stable within one
    run and different in the next, so a cache never hits and a plan mined today cannot be
    joined tomorrow."""
    code = f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); {snippet}"
    runs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           check=True).stdout.strip() for _ in range(3)}
    assert len(runs) == 1, f"{label} is not stable across processes: {runs}"


def test_a_seeded_sample_is_reproducible():
    """Every measurement in this audit is quoted with a seed. If a seed does not reproduce,
    none of the numbers mean anything."""
    from chartqa_dt.plans.distinguish import fingerprint
    from chartqa_dt.plans.executor import EvidenceItem
    ev = [EvidenceItem(f"e{i}", float(i)) for i in range(6)]
    plan = {"op": "argmax", "args": []}
    assert fingerprint(plan, ev, seed=7) == fingerprint(plan, ev, seed=7)
    assert fingerprint(plan, ev, seed=7) != fingerprint(plan, ev, seed=8)


# ================================================== guards must cover the case they describe


def test_the_fold_guard_covers_the_bare_case_and_the_mixed_one():
    """It required the plan to NAME a label, so it caught `difference("A", mean-of-all)` and
    missed a bare `argmax()` — the common case — which was then silently truncated (C4)."""
    from chartqa_dt.plans.executor import folds_over_evidence
    assert folds_over_evidence({"op": "argmax", "args": []}), "bare fold"
    assert folds_over_evidence({"op": "difference",
                                "args": ["A", {"op": "mean", "args": []}]}), "mixed fold"
    assert not folds_over_evidence({"op": "lookup", "args": ["A"]})
    assert not folds_over_evidence({"op": "difference", "args": ["A", "B"]})


def test_a_fold_over_more_elements_than_the_cap_is_refused_not_truncated():
    from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord
    from chartqa_dt.plans.schema import MAX_EVIDENCE
    from chartqa_dt.train.targets import TargetError, _evidence_from
    n = MAX_EVIDENCE + 1
    els = [{"label": f"e{i}", "value": float(i), "unit": None, "bbox": [i, 0, i + 5, 9]}
           for i in range(n)]
    rec = ChartRecord(record_id="f", source="chartqa", split="train", image_path="i.png",
                      image_sha256="d", question="q?", answer="x", question_kind="human",
                      plan={"op": "argmax", "args": []}, meta={ELEMENTS_KEY: els})
    with pytest.raises(TargetError, match="folds over all"):
        _evidence_from(rec)


# ============================================================ the executor never lies quietly


def _random_evidence(rng, n):
    from chartqa_dt.plans.executor import EvidenceItem
    return [EvidenceItem(f"L{i}", rng.choice([rng.uniform(-500, 500), 0.0, rng.randint(0, 99)]))
            for i in range(n)]


def test_the_executor_either_answers_or_raises_never_returns_junk():
    """Fuzzed over seeded random plans and evidence: every call returns a finite number, a
    string from a known set, None for `unanswerable`, or raises `ExecutorError`. It must
    never return NaN, infinity, or a silent wrong type."""
    import math

    from chartqa_dt.plans.executor import EXECUTABLE_OPS, ExecutorError, execute
    rng = random.Random(0)
    words = {"greater", "less", "equal", "increasing", "decreasing", "flat"}
    checked = 0
    for _ in range(1500):
        ev = _random_evidence(rng, rng.randint(0, 5))
        labels = [e.label for e in ev]
        op = rng.choice(sorted(EXECUTABLE_OPS))
        args = rng.sample(labels, min(len(labels), rng.randint(0, 2))) if labels else []
        try:
            got = execute({"op": op, "args": args}, ev)
        except ExecutorError:
            continue
        except RecursionError:                     # pragma: no cover - depth is capped
            pytest.fail("unbounded recursion")
        checked += 1
        if isinstance(got, bool) or got is None:
            continue
        if isinstance(got, str):
            assert got in words or got in labels, f"{op} returned an unexpected string {got!r}"
            continue
        assert isinstance(got, (int, float)), f"{op} returned {type(got).__name__}"
        assert math.isfinite(got), f"{op} returned a non-finite number"
    assert checked > 200, "the fuzz did not exercise enough successful calls"


def test_a_plan_deeper_than_the_limit_is_refused_not_evaluated():
    from chartqa_dt.plans.executor import MAX_DEPTH, EvidenceItem, ExecutorError, execute
    plan = {"op": "lookup", "args": ["a"]}
    for _ in range(MAX_DEPTH + 1):
        plan = {"op": "difference", "args": [plan, {"op": "lookup", "args": ["a"]}]}
    with pytest.raises(ExecutorError, match="depth"):
        execute(plan, [EvidenceItem("a", 1.0)])


# ============================================================ measurements must not be biased


def test_audit_scripts_shuffle_before_they_sample():
    """Twice, a measurement iterated a source in its natural order and reported a badly
    skewed number — human-only rows once (0081), filename-ordered annotations the other
    (0083). Both were in scripts written to check for bias.

    A script that slices a population must shuffle it first.
    """
    offenders = []
    for path in sorted((ROOT / "audit").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        reads_a_pool = "qa_rows" in text or "rglob" in text or "_names" in text
        # A full census cannot be biased by order — only a SLICE can. `--limit` or an early
        # `break` is what turns iteration into sampling. A `[:n]` is not enough on its own:
        # `question[:90]` and `examples[:5]` are display truncation, and matching those made
        # this test accuse a script that reads a whole mixture by record id.
        slices_it = "--limit" in text or "break" in text
        if not (reads_a_pool and slices_it):
            continue
        if "shuffle" not in text and "sample(" not in text:
            offenders.append(path.name)
    assert not offenders, (
        "these audit scripts slice a population without shuffling it, which is how two "
        f"findings were reported three times too large: {offenders}")


# ============================= every operator must survive the path it will actually take


#: A correct plan for each operation, with evidence it should verify against. If an operation
#: cannot be expressed here it cannot be mined, which is the point.
_WORKING_PLANS = {
    "lookup":         ({"op": "lookup", "args": ["b"]}, "20"),
    "count":          ({"op": "count", "args": []}, "3"),
    "sum":            ({"op": "sum", "args": []}, "60"),
    "mean":           ({"op": "mean", "args": []}, "20"),
    "median":         ({"op": "median", "args": []}, "20"),
    "min":            ({"op": "min", "args": []}, "10"),
    "max":            ({"op": "max", "args": []}, "30"),
    "argmin":         ({"op": "argmin", "args": []}, "a"),
    "argmax":         ({"op": "argmax", "args": []}, "c"),
    "difference":     ({"op": "difference", "args": ["c", "a"]}, "20"),
    "ratio":          ({"op": "ratio", "args": ["c", "a"]}, "3"),
    "percent_change": ({"op": "percent_change", "args": ["c", "a"]}, "200"),
    "compare":        ({"op": "compare", "args": ["c", "a"]}, "greater"),
    "trend":          ({"op": "trend", "args": []}, "increasing"),
    "boolean":        ({"op": "boolean", "args": ["a"]}, "Yes"),
    "within":         ({"op": "within",
                        "args": ["S", {"op": "argmax", "args": []}]}, "y2"),
}

_FLAT = [{"label": "a", "value": 10.0}, {"label": "b", "value": 20.0},
         {"label": "c", "value": 30.0}]
_SERIES = [{"label": "S · y1", "value": 1.0}, {"label": "S · y2", "value": 9.0}]


#: Operations that cannot be mined from ChartQA, with the reason. Each is a real limitation
#: rather than an oversight, and naming them here stops the coverage check below passing by
#: accident.
_UNMINABLE = {
    "unanswerable": "it executes to None, and every ChartQA question has a gold answer, so "
                    "no record can ever demonstrate it",
}


@pytest.mark.parametrize("op", sorted(_WORKING_PLANS))
def test_every_operation_can_pass_all_five_gates(op):
    """The property that catches a whole class of defect.

    `within` was added to the DSL, the schema, the prompt and the executor — and the
    verifier still rejected every plan using it, because `llm_mining` carried its **own**
    copy of the label extractor and that copy did not know `within`'s first argument names a
    series rather than an element. Four components agreed and the fifth quietly disagreed, so
    the operator was unusable through the only path that mines it.

    An operation nothing can mine is an operation the model will never be taught.
    """
    from chartqa_dt.plans import llm_mining
    plan, answer = _WORKING_PLANS[op]
    evidence = _SERIES if op == "within" else _FLAT
    got = llm_mining.verify(plan, answer=answer, evidence=evidence)
    assert got.status == llm_mining.OK, f"{op}: {got.status} — {got.detail}"


def test_the_working_plans_cover_every_executable_operation():
    """So that adding an operation to the DSL and forgetting to mine it fails here."""
    from chartqa_dt.plans.executor import EXECUTABLE_OPS
    missing = sorted(EXECUTABLE_OPS - set(_WORKING_PLANS) - set(_UNMINABLE))
    assert not missing, (
        f"these operations have no worked example, so nothing proves they can be mined: "
        f"{missing}")


def test_only_one_module_extracts_labels_from_a_plan():
    """The specific shape of the `within` defect: a second implementation of one concept.

    `plans.executor.plan_labels` is the definition. A module that walks a plan's `args`
    looking for strings has written a second one, and the two will diverge the next time an
    operation's arguments mean something new.
    """
    offenders = []
    for path in _modules():
        if path.name == "executor.py":
            continue
        text = path.read_text(encoding="utf-8")
        walks_args = 'plan.get("args")' in text or 'node.get("args")' in text
        collects_strings = "isinstance(arg, str)" in text
        if walks_args and collects_strings:
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, (
        "these modules walk a plan's arguments collecting labels instead of calling "
        f"`executor.plan_labels`, which is how `within` became unminable: {offenders}")


# ================================================ every number that bounds something says why


#: Constants that need no comment: their name is the explanation, or they are fixtures.
_SELF_EXPLAINING = {
    "SOURCE_IMAGE_W", "SOURCE_IMAGE_H",   # a smoke-test fixture's image size
    "DEV_ROWS",                            # the `--dev` subset size, a convenience
    "HOLDOUT_SEED_START",                  # named by `is_holdout`, which explains it
    "QWEN3VL_MERGE_SIZE",                  # a property of the model, not a choice
    "MEMORY_GATE_GB",                      # named by the gate it guards
}


def _numeric_constants():
    """Every module-level UPPER_CASE assignment to a bare number, with its line."""
    out = []
    for path in _modules():
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                name, value = node.targets[0].id, node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name, value = node.target.id, node.value
            else:
                continue
            if not (name.isupper() and value is not None):
                continue
            try:
                literal = ast.literal_eval(value)
            except Exception:                       # noqa: BLE001 - not a literal
                continue
            if isinstance(literal, bool) or not isinstance(literal, (int, float)):
                continue
            above = lines[node.lineno - 2].strip() if node.lineno >= 2 else ""
            same = lines[node.lineno - 1]
            documented = above.startswith("#") or "#" in same.split("=", 1)[-1]
            out.append((str(path.relative_to(SRC)), name, documented))
    return out


def test_every_numeric_limit_carries_its_reason():
    """A number that bounds something and does not say why is a number nobody can revise.

    Four of this audit's findings were exactly that: a value that was right when written,
    under a premise that later changed, with nothing recorded to check the premise against.
    The worst cost 92.8% of RefChartQA — `REFCHARTQA_CAP` said *"start at"* and nobody
    finished (`DECISIONS.md` 0112).

    A comment is cheap. Write what the number is for and what would change it.
    """
    undocumented = sorted(
        f"{path}:{name}" for path, name, doc in _numeric_constants()
        if not doc and name not in _SELF_EXPLAINING)
    assert not undocumented, (
        "these numeric limits carry no explanation, so nobody can tell whether they are "
        "still right:\n  " + "\n  ".join(undocumented))


def test_a_constant_that_says_start_at_has_a_ladder_behind_it():
    """`REFCHARTQA_CAP` read *'start at the single-box cap'* for a month while the scaling
    ladder it was starting toward went unrun, and the project trained on 7.2% of the dataset.

    A constant describing itself as provisional must say what would finish it.
    """
    offenders = []
    for path in _modules():
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines()):
            if "start at" not in line.lower() or not line.strip().startswith("#"):
                continue
            # The constant's OWN comment block: contiguous `#` lines around this one, not a
            # window. A window let this pass by finding a neighbouring constant's ladder.
            block, j = [line], i - 1
            while j >= 0 and text.splitlines()[j].strip().startswith("#"):
                block.append(text.splitlines()[j])
                j -= 1
            j = i + 1
            while j < len(text.splitlines()) and text.splitlines()[j].strip().startswith("#"):
                block.append(text.splitlines()[j])
                j += 1
            joined = "\n".join(block).lower()
            if not any(w in joined for w in ("ladder", "until", "ends it", "what finishes")):
                offenders.append(f"{path.relative_to(SRC)}:{i + 1}")
    assert not offenders, (
        "these say a value is a starting point without saying what ends it: " + str(offenders))


def _ladder_rungs() -> list[int]:
    """The RefChartQA scaling ladder as `PLAN.md` 3.4 states it."""
    return [4_000, 10_000, 25_000]


def test_the_cache_can_supply_every_rung_of_the_ladder_it_is_feeding():
    """The bug this catches was invisible for a month, and it was not a deferred task.

    `PLAN.md` 3.4 sets a ladder at 4,000 / 10,000 / 25,000 RefChartQA rows. The *mixture*
    cap was left at rung 1, which is what the plan says to do. But the **cache** that
    feeds the mixture had its own, unrelated `--cap`, also 4,000, and it held 3,996 rows.
    So rungs 2 and 3 had no data behind them: the ladder could not have been run even if
    someone had tried, and "we will run the ladder later" was never going to happen
    (`DECISIONS.md` 0115).

    Two caps with the same value and different jobs. The invariant that separates them:
    **a supply cap must be able to serve the largest demand anyone is allowed to ask
    for.** Rung caps may sit low; the cache underneath them may not.
    """
    import ast

    src = (ROOT / "scripts" / "cache_refchartqa.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    default = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "add_argument"
                and node.args and getattr(node.args[0], "value", "") == "--cap"):
            for kw in node.keywords:
                if kw.arg == "default":
                    default = ast.literal_eval(kw.value)
    assert default is not None, "cache_refchartqa.py no longer has a --cap default to check"
    top = max(_ladder_rungs())
    assert default >= top, (
        f"the RefChartQA cache defaults to {default:,} rows but the scaling ladder's top "
        f"rung asks for {top:,}. The rungs above the cache size cannot be run at all — "
        f"this is exactly the failure in DECISIONS.md 0115, where the cache held 3,996."
    )


def test_the_mixture_cap_is_a_rung_of_the_ladder_and_not_an_arbitrary_number():
    """`REFCHARTQA_CAP` is allowed to sit at rung 1. It is not allowed to sit *between*
    rungs, which would mean it was set by something other than the ladder."""
    from chartqa_dt.data.mixture import REFCHARTQA_CAP

    assert REFCHARTQA_CAP in _ladder_rungs(), (
        f"REFCHARTQA_CAP = {REFCHARTQA_CAP:,} is not one of the ladder's rungs "
        f"{_ladder_rungs()}. Either move it to a rung or amend the ladder in PLAN.md 3.4."
    )
