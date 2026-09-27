from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from chartqa_dt.eval.generate import Generation, generate_one, interpret, read_generations
from chartqa_dt.eval.metrics import bootstrap_ci, relaxed_correctness
from chartqa_dt.eval.official import score_chartqa, score_refchartqa
from chartqa_dt.model.coords import clamp_for_official_evaluator
from chartqa_dt.model.parsing import evaluation_schema_ok, parse_record
from chartqa_dt.plans.roundtrip import check_record
from chartqa_dt.plans.schema import EVALUATION_MAX_FACTS


def generate_all(loaded: Any, rows: Sequence[dict[str, Any]], *, mode: str, path: Path,
                 progress_every: int = 25) -> list[Generation]:
    done = read_generations(path) if path.exists() else []
    if done:
        print(f"  resuming after {len(done)} existing generations")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows[len(done):]:
            raw, seconds, n_tokens, capped = generate_one(loaded, row["question"], row["image"], mode=mode)
            generation = interpret(Generation(record_id=row["record_id"], raw=raw, seconds=seconds,
                                              prompt_mode=mode, new_tokens=n_tokens, hit_token_cap=capped))
            done.append(generation)
            fh.write(json.dumps(asdict(generation), allow_nan=False) + "\n")
            fh.flush()
            if progress_every and len(done) % progress_every == 0:
                print(f"    {len(done)}/{len(rows)} items", flush=True)
    return done


def _as_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def plan_statistics(rows: Sequence[dict[str, Any]], generations: Sequence[Generation]) -> dict[str, Any]:
    with_plan = executes = agrees = executed_correct = verified_correct = 0
    for row, generation in zip(rows, generations):
        result = parse_record(generation.raw)
        if not result.ok or result.record is None or not evaluation_schema_ok(result.record)[0]:
            continue
        if result.record.get("plan") is None:
            continue
        with_plan += 1
        trip = check_record(result.record, max_facts=EVALUATION_MAX_FACTS)
        if trip.outcome in ("agrees", "disagrees"):
            executes += 1
            executed_correct += relaxed_correctness(row["answer"], _as_answer(trip.executed))
        if trip.outcome == "agrees":
            agrees += 1
            verified_correct += relaxed_correctness(row["answer"], generation.answer.strip())
    n = len(rows) or 1
    return {
        "answers_with_plan": with_plan,
        "plans_that_execute": executes,
        "plans_reproducing_stated_answer": agrees,
        "executed_plan_accuracy": executed_correct / n,
        "plan_verified_accuracy": verified_correct / n,
    }


def summarise_chartqa(rows: Sequence[dict[str, Any]], generations: Sequence[Generation], *,
                      structured: bool) -> dict[str, Any]:
    pairs = [(row["answer"], g.answer) for row, g in zip(rows, generations)]
    correct = [float(relaxed_correctness(gold, pred.strip())) for gold, pred in pairs]
    ci = bootstrap_ci(correct, n_resamples=10_000, seed=0)
    summary: dict[str, Any] = {
        "n": len(rows),
        "relaxed_accuracy": score_chartqa(pairs),
        "by_subset": {
            kind: score_chartqa([p for p, row in zip(pairs, rows) if row["question_kind"] == kind])
            for kind in ("human", "machine")
        },
        "ci95": [ci.lo, ci.hi],
    }
    if structured:
        summary["valid_outputs"] = sum(g.parsed_ok and not g.reason for g in generations)
        summary["plans"] = plan_statistics(rows, generations)
    return summary


def summarise_refchartqa(rows: Sequence[dict[str, Any]], generations: Sequence[Generation]) -> dict[str, Any]:
    items = [{
        "pred_boxes": [clamp_for_official_evaluator(tuple(box)) for box in g.boxes],
        "answer": g.answer,
        "label": row["answer"],
        "image_size": row["image_size"],
        "grounding_bboxes": row["raw_boxes"],
        "question_kind": row["question_kind"],
    } for row, g in zip(rows, generations)]
    summary: dict[str, Any] = {"n": len(rows), "official": score_refchartqa(items), "by_subset": {}}
    for kind in ("human", "machine", "pot"):
        subset = [item for item in items if item["question_kind"] == kind]
        if subset:
            summary["by_subset"][kind] = score_refchartqa(subset)
    return summary
