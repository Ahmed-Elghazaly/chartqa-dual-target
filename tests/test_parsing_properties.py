"""Properties of reading a model's output, over seeded random and adversarial text.

`parse_record` is the boundary between a language model and everything deterministic. Every
string it is given was produced by a model that may be wrong in any way at all — truncated
mid-token, wrapped in prose, fenced in markdown, or simply not JSON.

The rule it must never break is `DECISIONS.md` 0064's: **drop, unwrap, never add.** A parser
that invents a field to make a record valid manufactures supervision, and a parser that
crashes on malformed input turns a bad generation into a dead run.

These tests fuzz it rather than listing cases, because the interesting inputs are the ones
nobody thought to write down.
"""
from __future__ import annotations

import json
import random
import string

import pytest

from chartqa_dt.plans.schema import MAX_EVIDENCE
from chartqa_dt.prompting.parsing import answer_of, coerce_boxes, parse_record

GOOD = {"answerable": True,
        "evidence": [{"label": "2019", "value": 245, "unit": None,
                      "bbox": [412, 180, 486, 742]}],
        "plan": {"op": "lookup", "args": ["2019"]},
        "model_answer": "245"}


def dumps(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# ================================================================ it never crashes


@pytest.mark.parametrize("seed", range(20))
def test_arbitrary_text_never_raises(seed):
    """A malformed generation must become a refusal, not an exception. One crash here ends a
    ten-hour evaluation run."""
    rng = random.Random(seed)
    alphabet = string.printable + "…—·«»​\U0001f600"
    for _ in range(120):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 300)))
        got = parse_record(text)
        assert isinstance(got.ok, bool)
        if not got.ok:
            assert got.reason, "a refusal must say why"


@pytest.mark.parametrize("cut", range(1, 60))
def test_a_truncated_record_is_refused_not_half_read(cut):
    """The failure `max_seq_len` produces. A record cut off mid-way must never parse into a
    partial one that then trains as if complete."""
    text = dumps(GOOD)
    got = parse_record(text[:-cut])
    assert not got.ok or got.record.get("model_answer") == GOOD["model_answer"]


@pytest.mark.parametrize("text", ["", "   ", "null", "[]", "[1,2,3]", "true", '"a string"',
                                  "{}", "{,}", '{"a":', "```json\n```", "NaN"])
def test_degenerate_inputs_are_refused_with_a_reason(text):
    got = parse_record(text)
    assert not got.ok
    assert got.reason


# =========================================================== drop, unwrap, never add


@pytest.mark.parametrize("seed", range(15))
def test_parsing_never_invents_a_field(seed):
    """The rule that keeps the pipeline from authoring its own supervision."""
    rng = random.Random(100 + seed)
    for _ in range(60):
        record = {k: v for k, v in GOOD.items() if rng.random() > 0.35}
        got = parse_record(dumps(record))
        if got.ok:
            assert set(got.record) <= set(GOOD), "a field appeared that was not sent"
            for key in got.record:
                if key != "evidence":
                    assert key in record, f"{key} was invented"


@pytest.mark.parametrize("seed", range(15))
def test_every_repair_is_counted(seed):
    """A silent repair is a measurement lost: the malformed-output rate is a headline
    number and it is only true if every fix is recorded."""
    rng = random.Random(200 + seed)
    for _ in range(60):
        n = rng.randint(0, MAX_EVIDENCE + 6)
        record = dict(GOOD)
        record["evidence"] = [
            {"label": f"e{i}", "value": i,
             **({"bbox": [i, 0, i + 5, 9]} if rng.random() > 0.25 else {})}
            for i in range(n)]
        got = parse_record(dumps(record))
        if not got.ok:
            continue
        kept = got.record.get("evidence") or []
        if len(kept) != n:
            assert got.repairs, f"{n} evidence items became {len(kept)} with no repair noted"


def test_evidence_beyond_the_cap_is_dropped_and_said_so():
    record = dict(GOOD)
    record["evidence"] = [{"label": f"e{i}", "value": i, "bbox": [i, 0, i + 5, 9]}
                          for i in range(MAX_EVIDENCE + 5)]
    got = parse_record(dumps(record))
    assert got.ok
    assert len(got.record["evidence"]) == MAX_EVIDENCE
    assert any("first" in r for r in got.repairs)


def test_the_model_s_own_ordering_is_respected_when_truncating():
    """The prompt asks for most-important-first, so keeping the first N honours its ranking
    rather than imposing ours."""
    record = dict(GOOD)
    record["evidence"] = [{"label": f"e{i}", "value": i, "bbox": [i, 0, i + 5, 9]}
                          for i in range(MAX_EVIDENCE + 3)]
    got = parse_record(dumps(record))
    assert [e["label"] for e in got.record["evidence"]] == \
        [f"e{i}" for i in range(MAX_EVIDENCE)]


# ============================================================ unwrapping what models do


@pytest.mark.parametrize("wrap", [
    "{body}",
    "```json\n{body}\n```",
    "```\n{body}\n```",
    "Here is the record:\n{body}",
    "{body}\n\nLet me know if you need anything else.",
    "Sure!\n```json\n{body}\n```\nThat is the answer.",
])
def test_a_record_is_found_inside_the_prose_models_wrap_it_in(wrap):
    got = parse_record(wrap.format(body=dumps(GOOD)))
    assert got.ok, got.reason
    assert got.record["model_answer"] == "245"


def test_the_first_record_wins_here_and_the_last_one_wins_for_the_teacher():
    """The same-looking text, read by two parsers under opposite rules — on purpose.

    A **fine-tuned** model emits one record and nothing else, so a second one is drift and
    the first is the answer. A **chat** model asked to mine a plan routinely restates the
    format before answering, so there the last block is the answer. Pinning both here makes
    the divergence visible; it is the shape of defect this audit found five times, and it is
    only safe while it is deliberate (`DECISIONS.md` 0111).
    """
    from chartqa_dt.plans.teacher import parse_proposal

    second = dumps({**GOOD, "model_answer": "999"})
    text = f"```json\n{dumps(GOOD)}\n```\nactually:\n```json\n{second}\n```"
    assert parse_record(text).record["model_answer"] == "245", "generation: first wins"

    plans = ('```json\n{"op":"sum","args":[]}\n```\n'
             'on reflection:\n```json\n{"op":"max","args":[]}\n```')
    assert parse_proposal(plans).plan == {"op": "max", "args": []}, "teacher: last wins"


# ======================================================================= box coercion


@pytest.mark.parametrize("seed", range(12))
def test_coerce_boxes_returns_only_well_formed_boxes(seed):
    """A malformed box must be dropped, not passed to the evaluator as a real prediction --
    one spurious box takes AP from 1.00 to 0.68."""
    rng = random.Random(300 + seed)
    for _ in range(80):
        evidence = []
        for i in range(rng.randint(0, 6)):
            shape = rng.choice(["good", "short", "long", "string", "none", "nested"])
            box = {"good": [i, 0, i + 5, 9], "short": [i, 0], "long": [i, 0, 1, 2, 3],
                   "string": "not a box", "none": None,
                   "nested": [[i], 0, 1, 2]}[shape]
            evidence.append({"label": f"e{i}", "value": i, "bbox": box})
        for box in coerce_boxes({"evidence": evidence}):
            assert isinstance(box, list) and len(box) == 4
            assert all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box)


def test_coerce_boxes_on_a_record_with_no_evidence_is_empty_not_an_error():
    assert coerce_boxes({}) == []
    assert coerce_boxes({"evidence": []}) == []
    assert coerce_boxes({"evidence": None}) == []


# ============================================================================ the answer


@pytest.mark.parametrize("value,expected", [
    ("245", "245"), (245, "245"), (245.0, "245.0"), (True, "True"), (None, ""),
])
def test_answer_of_always_returns_a_string(value, expected):
    """It feeds `relaxed_correctness`, which calls `.endswith`. A non-string would crash the
    metric on a generation rather than scoring it wrong."""
    got = answer_of({"model_answer": value})
    assert isinstance(got, str)
    assert got == expected


def test_answer_of_a_record_without_one_is_empty_not_an_error():
    assert answer_of({}) == ""
