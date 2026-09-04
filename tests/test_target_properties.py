"""Properties a built target must satisfy, over seeded random records.

`_evidence_from` and `build_record` are where four separate defects lived — the first-eight-
boxes selection (0067), the elements-key mismatch (0071), the value/box disagreement (0075)
and the silently truncated fold (0082). Each was found by a specific failing record, and each
would have been caught by a property.

Every case here is generated from a seed, so a failure names a reproducible record rather
than a shape. No `hypothesis` dependency: the repo's own convention is that a measurement is
quoted with its seed, and these follow it.
"""
from __future__ import annotations

import json
import random

import pytest

from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord
from chartqa_dt.plans.executor import EvidenceItem, execute
from chartqa_dt.plans.roundtrip import check_record
from chartqa_dt.plans.schema import MAX_EVIDENCE, validate_record
from chartqa_dt.prompting.parsing import parse_record
from chartqa_dt.train.targets import TargetError, build_target


def a_chart(rng, *, n=4, series=None, collide=False):
    """A chart of `n` marks. `collide` repeats a label across two series."""
    labels = [f"20{10 + i}" for i in range(n)]
    if collide and series:
        # HALF as many distinct labels as marks, so each one really is drawn once per
        # series. Using `n` labels for `n` marks made `i % len(labels)` the identity and the
        # test skipped every seed -- it passed twenty times and checked nothing.
        distinct = max(1, n // len(series))
        return [{"label": labels[i % distinct], "series": series[i // distinct % len(series)],
                 "value": round(rng.uniform(1, 500), 2), "unit": None,
                 "bbox": [i * 7, 10, i * 7 + 6, 90]} for i in range(n)]
    return [{"label": labels[i], "series": series[0] if series else None,
             "value": round(rng.uniform(1, 500), 2), "unit": None,
             "bbox": [i * 7, 10, i * 7 + 6, 90]} for i in range(n)]


def a_record(elements, plan, answer, *, rid="r", table=None):
    return ChartRecord(record_id=rid, source="chartqa", split="train", image_path="i.png",
                       image_sha256="d", question="q?", answer=str(answer),
                       question_kind="human", table=table, plan=plan,
                       boxes=[e["bbox"] for e in elements], meta={ELEMENTS_KEY: elements})


def _value_of(elements, label):
    return next(e["value"] for e in elements if e["label"] == label)


# ============================================================= the invariant that matters most


@pytest.mark.parametrize("seed", range(40))
def test_a_built_target_always_reproduces_its_own_answer(seed):
    """The property every one of those four defects violated. If `build_target` returns at
    all, the target's own plan run against the target's own evidence must give the target's
    own answer — and if it cannot, it must raise instead."""
    rng = random.Random(seed)
    elements = a_chart(rng, n=rng.randint(1, 6))
    label = rng.choice(elements)["label"]
    plan = {"op": "lookup", "args": [label]}
    record = a_record(elements, plan, _value_of(elements, label), rid=f"s{seed}")

    try:
        text = build_target(record)
    except TargetError as exc:
        assert str(exc).startswith(f"s{seed}:"), "a refusal must name the record"
        return
    trip = check_record(json.loads(text))
    assert trip.outcome == "agrees", f"seed {seed}: {trip.outcome} — {trip.executed!r}"


@pytest.mark.parametrize("seed", range(30))
def test_a_fold_target_reproduces_its_answer_or_refuses(seed):
    """A fold reads the whole chart, so truncating its evidence changes the answer. This is
    the property the bare-`argmax` defect broke, and it broke it silently."""
    rng = random.Random(1000 + seed)
    elements = a_chart(rng, n=rng.randint(2, MAX_EVIDENCE + 3))
    op = rng.choice(["max", "min", "argmax", "argmin", "mean", "sum", "count"])
    truth = execute({"op": op, "args": []},
                    [EvidenceItem(e["label"], e["value"]) for e in elements])
    record = a_record(elements, {"op": op, "args": []}, truth, rid=f"f{seed}")

    try:
        text = build_target(record)
    except TargetError as exc:
        assert "folds over all" in str(exc) or f"f{seed}:" in str(exc)
        return
    assert check_record(json.loads(text)).outcome == "agrees"


# ================================================================ schema and shape properties


@pytest.mark.parametrize("seed", range(25))
def test_a_built_target_parses_and_satisfies_the_schema(seed):
    rng = random.Random(2000 + seed)
    elements = a_chart(rng, n=rng.randint(1, 5))
    label = rng.choice(elements)["label"]
    record = a_record(elements, {"op": "lookup", "args": [label]},
                      _value_of(elements, label), rid=f"p{seed}")
    try:
        text = build_target(record)
    except TargetError:
        return
    parsed = parse_record(text)
    assert parsed.ok, parsed.reason
    assert not parsed.repairs, "a target we built should need no repair"
    assert validate_record(parsed.record).ok


@pytest.mark.parametrize("seed", range(25))
def test_evidence_never_exceeds_the_cap(seed):
    rng = random.Random(3000 + seed)
    elements = a_chart(rng, n=rng.randint(1, MAX_EVIDENCE + 5))
    label = rng.choice(elements)["label"]
    record = a_record(elements, {"op": "lookup", "args": [label]},
                      _value_of(elements, label), rid=f"c{seed}")
    try:
        got = json.loads(build_target(record))
    except TargetError:
        return
    assert len(got["evidence"]) <= MAX_EVIDENCE


@pytest.mark.parametrize("seed", range(25))
def test_every_evidence_entry_carries_a_usable_box(seed):
    """A box is the grounding half of the metric; an entry without one is dead weight the
    parser would later have to repair away."""
    rng = random.Random(4000 + seed)
    elements = a_chart(rng, n=rng.randint(1, 5))
    label = rng.choice(elements)["label"]
    record = a_record(elements, {"op": "lookup", "args": [label]},
                      _value_of(elements, label), rid=f"b{seed}")
    try:
        got = json.loads(build_target(record))
    except TargetError:
        return
    for entry in got["evidence"]:
        box = entry["bbox"]
        assert len(box) == 4 and box[2] > box[0] and box[3] > box[1], box
        assert all(0 <= v <= 1000 for v in box), box


@pytest.mark.parametrize("seed", range(20))
def test_a_plan_only_ever_names_evidence_the_target_carries(seed):
    """`lookup of unknown evidence label` is what 0067 produced 635 times out of 636."""
    from chartqa_dt.plans.executor import plan_labels
    rng = random.Random(5000 + seed)
    elements = a_chart(rng, n=rng.randint(2, 6))
    two = rng.sample([e["label"] for e in elements], 2)
    plan = {"op": "difference", "args": two}
    answer = _value_of(elements, two[0]) - _value_of(elements, two[1])
    try:
        got = json.loads(build_target(a_record(elements, plan, answer, rid=f"n{seed}")))
    except TargetError:
        return
    carried = {e["label"] for e in got["evidence"]}
    assert set(plan_labels(got["plan"])) <= carried


# ============================================================== grouped charts and collisions


@pytest.mark.parametrize("seed", range(20))
def test_a_collision_is_qualified_or_refused_never_resolved_by_position(seed):
    """Before 0083 the target builder kept the FIRST match and the executor the LAST, so a
    plan pointed at one bar and stated another's number."""
    rng = random.Random(6000 + seed)
    elements = a_chart(rng, n=6, series=["Democratic", "Republican"], collide=True)
    names = [e["label"] for e in elements]
    assert len(set(names)) < len(names), "the generator must actually produce a collision"
    from chartqa_dt.data.records import qualified_labels
    qualified = qualified_labels(elements)
    assert len(set(qualified)) == len(qualified), "qualification must produce unique names"
    chosen = rng.choice(qualified)
    value = next(e["value"] for q, e in zip(qualified, elements) if q == chosen)
    try:
        got = json.loads(build_target(
            a_record(elements, {"op": "lookup", "args": [chosen]}, value, rid=f"q{seed}")))
    except TargetError:
        return
    assert check_record(got).outcome == "agrees"


# ==================================================================== refusals are informative


@pytest.mark.parametrize("seed", range(15))
def test_every_refusal_names_the_record_and_the_cause(seed):
    """A refusal that does not say why sends the reader to the wrong bug — the truncated fold
    was reported as *the plan does not reproduce its answer* for exactly that reason."""
    rng = random.Random(7000 + seed)
    elements = a_chart(rng, n=rng.randint(1, 4))
    record = a_record(elements, {"op": "lookup", "args": ["definitely-absent"]}, "1",
                      rid=f"x{seed}")
    with pytest.raises(TargetError) as excinfo:
        build_target(record)
    message = str(excinfo.value)
    assert message.startswith(f"x{seed}:")
    assert len(message) > 50, f"refusal too terse to act on: {message!r}"


def test_a_record_with_no_answer_is_refused_before_anything_else():
    rng = random.Random(0)
    elements = a_chart(rng, n=2)
    record = ChartRecord(record_id="na", source="chartqa", split="train", image_path="i.png",
                         image_sha256="d", question="q?", answer=None,
                         question_kind="human", plan={"op": "count", "args": []},
                         meta={ELEMENTS_KEY: elements})
    with pytest.raises(TargetError, match="no answer"):
        build_target(record)
