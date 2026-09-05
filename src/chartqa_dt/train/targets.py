"""Turning a `ChartRecord` into the exact string the model is trained to emit.

This is the join between the data pipeline and the model, and it is the easiest place in
the project to introduce a defect that nothing catches. A target that differs from what the
parser accepts — by one key order, one float where an integer belongs, one space — teaches
the model to produce output our own evaluator then rejects, and the only symptom is a
disappointing score.

So the module has one invariant, asserted on every target it builds:

    parse_record(build_target(record)) must succeed AND satisfy OUTPUT_SCHEMA

Three conventions are load-bearing and each was chosen to match something that already
exists rather than invented here:

* **Boxes are integers 0–999**, via `clamp_for_official_evaluator` — the same function the
  inference path uses, because the official evaluator silently discards anything outside
  that range (`DECISIONS.md` 0004).
* **Compact separators**, matching the prompt's worked examples. The model imitates the
  format it is shown, and a pretty-printed target would undo the 2.6× token reduction that
  compaction bought (`DECISIONS.md` 0058).
* **At most `MAX_EVIDENCE` items, best-first.** More is invalid under the schema, and extra
  boxes cost AP (`DECISIONS.md` 0014).
"""

from __future__ import annotations

import json
from typing import Any

from chartqa_dt.data.records import (
    ELEMENTS_KEY,
    ChartRecord,
    fold_for_matching,
    qualified_labels,
)
from chartqa_dt.eval.metrics import to_float
from chartqa_dt.plans.executor import (
    folds_over_evidence,
    parse_numeric,
    plan_labels,
)
from chartqa_dt.plans.roundtrip import check_record
from chartqa_dt.plans.schema import MAX_EVIDENCE, validate_record
from chartqa_dt.prompting.parsing import parse_record
from chartqa_dt.vision.coords import clamp_for_official_evaluator

#: `json.dumps` separators that produce exactly the compact form the prompt demonstrates.
COMPACT = (",", ":")


class TargetError(ValueError):
    """A record could not be turned into a valid training target."""


class NoPlanAvailable(TargetError):
    """The record has gold boxes and a gold answer, and no plan — nothing is *wrong*.

    Separated from the other refusals because it is the only one that says "this record
    is incomplete" rather than "this record is inconsistent". A plan that fails to
    reproduce its answer, or points at a label with no box, is evidence that something is
    wrong and the record should go. A missing plan is evidence of nothing.

    Stage 1 is grounding only (`PLAN.md` 6.1), so it can still use such a record —
    `build_grounding_only_target` emits the boxes and the answer and omits the plan. The
    distinction has to be in the *type* rather than the message, or the feed would be
    matching on prose that any later edit could change.
    """



def _table_values(record: ChartRecord) -> dict[Any, Any]:
    """Label to value, from the gold table — the source the mined plan was verified on.

    Two kinds of key, because a bare label is not enough on a grouped chart:

    * `"2019"` — the row's first numeric cell. Correct on a two-column table, which is
      74.7% of ChartQA, and what every existing caller expects.
    * `(folded_column, "2019")` — that specific column's cell. On a three-column table the
      bare key silently returned the FIRST series' number, so an element belonging to the
      second series was given another series' value. The column is folded through
      `fold_for_matching` because 14.4% of charts spell the annotation's series name
      differently from the table's column heading — Cyrillic homoglyphs, a stray letter,
      scrambled word order — and none of that should fail the join.
    """
    table = record.table
    if not isinstance(table, dict):
        return {}
    columns = [fold_for_matching(c) for c in (table.get("columns") or [])]
    out: dict[Any, Any] = {}
    for row in table.get("rows") or []:
        if not row:
            continue
        label = str(row[0]).strip()
        first = True
        for i, cell in enumerate(row[1:], start=1):
            # `parse_numeric`, NOT `to_float`. A table cell is a chart VALUE, and
            # `eval.metrics.to_float` is the official parser for gold ANSWERS: it divides a
            # trailing `%` by 100 and cannot read a spaced thousand. Using it here made every
            # percentage chart's evidence 100x too small, so a plan verified against the
            # annotation's own values then failed its round-trip against the table's —
            # `sum` over two 43.6% bars executed to 0.821 against an answer of 82.1. Third
            # occurrence of one root cause; `DECISIONS.md` 0082 is the first two.
            number = parse_numeric(cell)
            if number is None:
                continue
            if i < len(columns):
                out.setdefault((columns[i], label), number)
            if first:
                out.setdefault(label, number)
                first = False
    return out


#: How far the gold table's value for a label may sit from the value of the element whose
#: box we are about to emit, before we call them different marks. 2% is loose enough for the
#: two sources rounding differently and tight enough to catch the swapped pairs below.
VALUE_AGREEMENT_TOLERANCE = 0.02


def values_agree(table_value: Any, element_value: Any) -> bool:
    """Do the gold table and the chart annotation describe the *same mark*?

    An evidence entry takes its **value** from the table and its **box** from the
    annotation, joined only by a label string. On 110 of 1,893 measured entries the two
    sources disagree, and they disagree in *swapped pairs* — the table says Finland is 9.4
    and Hungary 9.9 while the annotation has them the other way round. Emitting such an
    entry teaches the model to box one mark and state another's number, which is exactly
    the association this project exists to teach (`DECISIONS.md` 0075).

    **The 100x case is not a disagreement.** `to_float` divides a "%" cell by 100 because
    the *official metric* does: `relaxed_correctness(gold="81.9%", pred="0.819")` is True
    and `pred="81.9"` is False. So a table value of 0.819 beside an annotation value of
    81.9 is the convention the evaluator requires, not an error. Measured: all 29 records
    in that state ship scale-invariant plans (27 `ratio`, 2 `count`) and every one of them
    round-trips.
    """
    # Both sides are chart VALUES, so both use `parse_numeric`. Using the official answer
    # parser here compared a table's 43.6 against an annotation's '43.6%' -- which it reads
    # as 0.436 -- and refused the record for a disagreement that did not exist.
    a, b = parse_numeric(table_value), parse_numeric(element_value)
    if a is None or b is None:
        return True                      # nothing to compare; other guards handle it
    if abs(a - b) <= VALUE_AGREEMENT_TOLERANCE * max(abs(a), abs(b), 1e-9):
        return True
    return abs(a * 100.0 - b) <= VALUE_AGREEMENT_TOLERANCE * max(abs(b), 1e-9)


def _refuse_on_value_disagreement(record: ChartRecord, label: str,
                                  table_value: Any, element: dict[str, Any]) -> None:
    if not values_agree(table_value, element.get("value")):
        raise TargetError(
            f"{record.record_id}: the gold table says {label!r} is {table_value!r} but the "
            f"annotated element we would box has value {element.get('value')!r}. The two "
            f"sources disagree about which mark this label names, so the target would "
            f"point at one mark and state another's number.")


def _evidence_from(record: ChartRecord) -> list[dict[str, Any]]:
    """Evidence entries — **the ones the plan needs**, not the first `MAX_EVIDENCE` boxes.

    This distinction is not cosmetic. Selecting the first eight boxes produced targets
    where the mined plan referenced a label that was not among them, and the executor
    refused with *"lookup of unknown evidence label: 'Indonesia'"*: **1 of 636 ChartQA
    records yielded a usable target**. Selecting by label instead fixes the join, and it
    also teaches the behaviour `DECISIONS.md` 0014 wants — point at what the answer needs
    and nothing else — rather than "point at the first eight things on the chart".

    An aggregate with empty args folds over whatever evidence is present
    (`DECISIONS.md` 0041), so those keep the leading elements up to the cap.
    """
    elements = [e for e in (record.meta.get(ELEMENTS_KEY) or [])
                if isinstance(e, dict)]
    # One name per element, unique within the chart. On a grouped chart "2019" names one bar
    # per series, and the two sides of this contract resolved it differently -- this module
    # kept the FIRST match and `plans.executor` kept the LAST -- so a plan pointed at one bar
    # and stated another's number (`AUDIT.md` H3). Only colliding labels are qualified, so
    # 77.4% of charts keep their labels exactly as the chart draws them.
    names = qualified_labels(elements)
    if len(set(names)) != len(names) and elements:
        repeated = next(n for n in names if names.count(n) > 1)
        raise TargetError(
            f"{record.record_id}: {repeated!r} still names more than one mark after "
            f"qualifying by series, so no plan can say which one it means. Picking one "
            f"would point at a mark we did not choose.")
    by_name = dict(zip(names, elements))
    series_of = {n: e.get("series") for n, e in zip(names, elements)}
    bare_of = {n: str(e.get("label")) for n, e in zip(names, elements)}

    def value_for(name: str, element: dict[str, Any]) -> Any:
        """The gold table's value for this element, by series where the table has one.

        The bare-label fallback is used **only for an element whose label does not collide**.
        On a grouped chart the bare key returns the row's FIRST numeric cell, which belongs
        to whichever series the table happens to list first — so falling back to it for a
        qualified element silently hands one series another series' number. Where the
        series-to-column join fails on such an element, the annotation's own value is the
        only source that is certainly about this mark.
        """
        keyed = table_values.get((fold_for_matching(series_of.get(name)), bare_of[name]))
        if keyed is not None:
            return keyed
        if name == bare_of[name]:                 # label did not collide; the row is unique
            return table_values.get(bare_of[name], element.get("value"))
        return element.get("value")
    boxes = record.boxes or []
    wanted = plan_labels(record.plan)
    # The plan was mined and verified against the gold TABLE, so the table is the
    # authority on values. The annotation is the authority on boxes. Reading values from
    # the annotation instead made 35 of 105 planned records disagree with their own
    # answer, because the two sources round and format numbers differently.
    table_values = _table_values(record)

    def entry(label: Any, value: Any, unit: Any, box: Any) -> dict[str, Any]:
        # Every box in a plan target passes through here, which is why the check lives
        # here. `ChartRecord.from_dict` does not validate `boxes` — it takes whatever the
        # cache holds — so a malformed one used to reach `tuple(box)` and raise
        # `TypeError`. The feed catches `TargetError`, not `TypeError`, so that killed the
        # run rather than costing one record. `build_grounding_only_target` already
        # refused these; now both paths refuse them the same way (`DECISIONS.md` 0116).
        if not (isinstance(box, (list, tuple)) and len(box) == 4
                and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                        for v in box)):
            raise TargetError(
                f"{record.record_id}: evidence {str(label)!r} has no usable box "
                f"({box!r}). Four numbers are required; a target cannot point at this.")
        return {"label": str(label), "value": value, "unit": unit,
                "bbox": clamp_for_official_evaluator(tuple(box))}

    # A plan that folds over the chart needs the *whole* chart in its evidence, even when
    # it also names labels. `difference("Alpha", mean-of-everything)` names one label; give
    # it one evidence item and the mean is that item, so the difference is exactly zero.
    # That is what happened to **all 6,000 L4 records** — the compositional level, and the
    # scarcest supervision in the mixture (`DECISIONS.md` 0071).
    # `wanted` is deliberately NOT required here. A plan that folds and also names a
    # label -- `difference("Alpha", mean-of-everything)` -- was caught; a plan that
    # only folds -- a bare `argmax()`, the common case -- has no labels at all and fell
    # through to the truncating branch below, silently keeping the first eight elements
    # of a chart with more. The round-trip then refused the record with *"own plan does
    # not reproduce its own answer"*, blaming the plan for evidence we had cut. The
    # median ChartQA chart has 10 elements and 64.4% have more than eight.
    if elements and folds_over_evidence(record.plan):
        if len(elements) > MAX_EVIDENCE:
            raise TargetError(
                f"{record.record_id}: the plan folds over all {len(elements)} elements, "
                f"more than the schema's {MAX_EVIDENCE}. Truncating would change the "
                f"aggregate and the target would no longer reproduce its answer.")
        have = {n for n, e in by_name.items() if e.get("bbox") is not None}
        missing = [label for label in wanted if label not in have]
        if missing:
            raise TargetError(
                f"{record.record_id}: the plan references {missing[0]!r}, which has no "
                f"element box.")
        folded = []
        for name, e in by_name.items():
            if e.get("bbox") is None:
                continue
            value = value_for(name, e)
            _refuse_on_value_disagreement(record, name, value, e)
            folded.append(entry(name, value, e.get("unit"), e["bbox"]))
        return folded

    if elements and wanted:
        picked: list[dict[str, Any]] = []
        for label in dict.fromkeys(wanted):          # de-duplicated, order preserved
            element = by_name.get(label)
            if element is None or element.get("bbox") is None:
                raise TargetError(
                    f"{record.record_id}: the plan references {label!r}, which has no "
                    f"element box. Emitting the record without it would train a plan "
                    f"that cannot execute.")
            value = value_for(label, element)
            _refuse_on_value_disagreement(record, label, value, element)
            picked.append(entry(label, value, element.get("unit"), element["bbox"]))
        if len(picked) > MAX_EVIDENCE:
            raise TargetError(f"{record.record_id}: the plan needs {len(picked)} evidence "
                              f"items, more than the schema's {MAX_EVIDENCE}")
        return picked

    if elements:
        # An aggregate over everything, or no plan yet: keep the leading elements.
        return [entry(n, value_for(n, e), e.get("unit"), e["bbox"])
                for n, e in list(by_name.items())[:MAX_EVIDENCE]
                if e.get("bbox") is not None]

    # Boxes with no element metadata — RefChartQA. Labels are placeholders; whether such a
    # record can be supervised at all is decided in `build_record`.
    return [entry(f"item{i + 1}", None, None, box)
            for i, box in enumerate(boxes[:MAX_EVIDENCE])]


def build_record(record: ChartRecord) -> dict[str, Any]:
    """The JSON object the model should emit for this training example.

    **The plan must be executable against the evidence in the same record.** RefChartQA
    supplies boxes but no per-element values, and the first version of this function filled
    those with `null` and a `lookup` plan — producing targets where **100% of 800 sampled
    records failed the round-trip**. Training on them would have taught the model to emit
    non-executable plans, on the very metric the project exists to move.

    Where the value is genuinely recoverable it is recovered: a record with **one** box and
    a numeric answer is a lookup whose result is that answer, which is both true and
    executable (52% of RefChartQA). Where it is not recoverable, the record is refused
    rather than filled in — `PLAN.md` 3.6's "never given an invented plan", applied to
    values as well as operations.
    """
    if record.answer is None:
        raise TargetError(f"{record.record_id} has no answer; nothing to supervise")

    evidence = _evidence_from(record)
    if not evidence:
        raise TargetError(f"{record.record_id} has no evidence boxes")

    plan = record.plan if isinstance(record.plan, dict) and record.plan.get("op") else None

    if plan is None:
        # One box and a numeric answer: the box IS the answer, so `lookup` over it is the
        # correct plan and the value is known. This is the only case where a plan can be
        # supplied without inventing anything.
        answer_value = to_float(record.answer)
        if len(evidence) == 1 and answer_value is not None:
            if evidence[0]["value"] is None:
                evidence[0]["value"] = answer_value
            plan = {"op": "lookup", "args": [evidence[0]["label"]]}
        else:
            raise NoPlanAvailable(
                f"{record.record_id}: no mined plan, and one cannot be derived — "
                f"{len(evidence)} evidence item(s), answer {record.answer!r}. Supplying "
                f"one would train a plan that does not execute.")

    return {
        "answerable": True,
        "evidence": evidence,
        "plan": plan,
        "model_answer": str(record.answer),
    }


def build_target(record: ChartRecord, *, verify: bool = True) -> str:
    """The exact target string, verified to be something our own parser accepts.

    `verify` exists only for speed on a hot path that has already been checked; leaving it
    on is the default because a silently invalid target is the failure this module is for.
    """
    obj = build_record(record)
    text = json.dumps(obj, separators=COMPACT, ensure_ascii=False)
    if not verify:
        return text

    parsed = parse_record(text)
    if not parsed.ok:
        raise TargetError(f"{record.record_id}: own target does not parse — {parsed.reason}")
    result = validate_record(parsed.record)
    if not result.ok:
        raise TargetError(f"{record.record_id}: own target fails the schema — "
                          f"{'; '.join(result.errors[:2])}")

    # The invariant that matters most: the target's own plan must reproduce the target's
    # own answer. A target that fails this teaches the model to fail the round-trip.
    trip = check_record(parsed.record)
    if not trip.ok:
        raise TargetError(f"{record.record_id}: own plan does not reproduce its own "
                          f"answer ({trip.outcome}: executed {trip.executed!r} vs stated "
                          f"{trip.stated!r}{'; ' + trip.error if trip.error else ''})")
    return text


def build_answer_only_target(record: ChartRecord) -> str:
    """The direct-answer control of `PLAN.md` 6.4 — the answer and nothing else.

    Required, not optional: it is what separates ordinary domain adaptation from the
    contribution of grounding, plans and execution. Trained on the *same* records, so the
    only difference between the arms is what the model is asked to emit.
    """
    if record.answer is None:
        raise TargetError(f"{record.record_id} has no answer")
    return str(record.answer)


def has_question_specific_boxes(record: ChartRecord) -> bool:
    """Do this record's boxes mark the evidence for **this question**?

    The distinction a grounding-only target lives or dies on, and it is not obvious from
    the record: both kinds have `boxes`, and both look like grounding.

    * **RefChartQA** annotates, per question, which regions a person used to answer it.
      Those boxes *are* the answer to "where should the model point", so a target built
      from them teaches exactly that.
    * **ChartQA** annotates the *chart* — every element of it, regardless of the question.
      Its `boxes` are its `elements`: 12 boxes for a 12-element chart, the same 12 for
      every question asked about it.

    Building a grounding-only target from the second kind teaches *"point at the whole
    chart"*. Measured on real records, that is not a theoretical worry — the target for
    *"Which year has the most crime?"* (answer: 2014) came out pointing at all six years,
    which is the behaviour `DECISIONS.md` 0014 exists to prevent and which AP@0.5 scores
    close to zero. It is the same mistake `_evidence_from` was written to avoid, arriving
    through a different door (`DECISIONS.md` 0116).

    **Every source now declares this at ingestion** — `chartqa.py` sets it False, and
    `refchartqa.py` and the synthetic reader set it True — so the semantics travel with the
    record instead of being re-derived by each consumer. That is the narrow version of the
    change `Prompt.md` Ideas 1 and 2 ask about: 0108 enumerated nine consumers of `boxes`
    and recorded two assumptions as *"safe by circumstance rather than by contract"*, and
    this is the one that then fired (`DECISIONS.md` 0119).

    The `refchartqa_id` fallback remains for records cached before the flag existed. A new
    source that declares nothing is treated as whole-chart, which is the safe direction:
    it loses grounding-only targets rather than emitting wrong ones.
    """
    flag = record.meta.get("question_specific_boxes")
    if flag is not None:
        return bool(flag)
    return "refchartqa_id" in record.meta


def build_grounding_only_target(record: ChartRecord, *, verify: bool = True) -> str:
    """Boxes and the answer, for a record whose plan we do not have.

    **What this recovers.** `build_record` requires a plan and refuses without one, so a
    record with gold grounding and no derivable plan is dropped from every mixture. Measured
    over the RefChartQA cache: **31.2% of records have boxes and no plan**, projecting to
    about **17,381** across the full train split — more than the entire stage-1 cap.

    That is a strange thing to discard, because `PLAN.md` 6.1 makes **stage 1 grounding
    only**: it teaches the model where to point *before* teaching it to reason. Refusing a
    record with gold boxes for want of a plan that stage does not yet use throws away exactly
    the supervision that stage exists for.

    **What is emitted, and what is not.** These records carry gold boxes *and* a gold answer;
    only the plan is missing. So both are supervised and the plan is **omitted** — not filled
    with `unanswerable`, which would be false, and not derived, which `PLAN.md` 3.6 forbids.
    The result is a strict subset of the full record's fields, so stage 1 teaches a prefix
    that stage 2 completes rather than a different format it must unlearn.

    It is deliberately **not** schema-valid: `OUTPUT_SCHEMA` requires a plan, and it should,
    because a *generation* without one is incomplete. This is a training target for one stage,
    the same exception `build_answer_only_target` already takes for the control arm.
    """
    if record.answer is None:
        raise TargetError(f"{record.record_id} has no answer; nothing to supervise")
    if not has_question_specific_boxes(record):
        raise TargetError(
            f"{record.record_id}: its boxes describe the whole chart rather than marking "
            f"the evidence for this question, so a grounding-only target would teach "
            f"'point at everything'. Only a plan can select from an annotation like this.")
    evidence = _evidence_from(record)
    if not evidence:
        raise TargetError(f"{record.record_id} has no evidence boxes")

    text = json.dumps({"answerable": True, "evidence": evidence,
                       "model_answer": str(record.answer)},
                      separators=COMPACT, ensure_ascii=False)
    if verify:
        # NOT `parse_record`: it requires a `plan`, correctly, because a *generation*
        # without one is incomplete. This is a stage-1 training target, so the checks are
        # the ones that still apply — it is valid JSON, and every box it claims is usable.
        try:
            back = json.loads(text)
        except json.JSONDecodeError as exc:                       # pragma: no cover
            raise TargetError(f"{record.record_id}: own target is not JSON — {exc}") from exc
        items = back.get("evidence") or []
        if not items:
            raise TargetError(f"{record.record_id}: emitted no evidence")
        for entry in items:
            box = entry.get("bbox")
            if not (isinstance(box, list) and len(box) == 4
                    and all(isinstance(v, (int, float)) for v in box)
                    and box[2] > box[0] and box[3] > box[1]):
                raise TargetError(
                    f"{record.record_id}: evidence entry {entry.get('label')!r} has no "
                    f"usable box ({box!r}). A grounding-only target is nothing BUT its "
                    f"boxes, so an unusable one makes the record worthless rather than "
                    f"merely incomplete.")
    return text


__all__ = [
    "COMPACT",
    "NoPlanAvailable",
    "TargetError",
    "build_answer_only_target",
    "build_grounding_only_target",
    "build_record",
    "build_target",
    "has_question_specific_boxes",
]
