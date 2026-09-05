#!/usr/bin/env python3
"""A stratified audit set for the things executor agreement cannot measure.

`Prompt.md`'s **MANUAL SEMANTIC AUDIT SET** asks for a small, carefully selected, manually
inspected set of *difficult* cases, and is explicit about why: it exists to measure
**semantic correctness**, which round-trip agreement cannot. A plan that executes to the
gold answer can still be the wrong plan — `distinguish.coincidences` measures how often
another operand pair reaches the same number, and that is a lower bound on the problem, not
the whole of it.

The brief is equally explicit about the trap: *"Document sampling methodology so the audit
is reproducible and not cherry-picked."* So:

**Sampling methodology.**

1. Every candidate record is passed through the detectors in `STRATA`, each of which
   answers one question: *is this record a hard case of this kind?* A record may match
   several strata and is then eligible for each.
2. Within a stratum, candidates are sorted by `record_id` — a sha256-derived string, so the
   order is deterministic, machine-independent, and unrelated to anything that would
   correlate with difficulty or correctness.
3. `random.Random(seed)` samples `--per-stratum` of them. The seed is written into the
   output, so the exact set is reproducible.
4. **Strata that come up short are reported, not padded.** A stratum with four eligible
   records yields four rows and says so. Topping it up from an easier stratum is how an
   audit set stops being an audit of what it claims.

Nothing here judges anything. The output carries `verdict: null` per row for a person to
fill in, plus the fields needed to judge without re-deriving them.

Rule 7: the output holds labels and values, which are dataset content, so it goes to the
**data cache** and never to git.
"""
from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from chartqa_dt.data.records import ChartRecord
from chartqa_dt.env import get_env
from chartqa_dt.plans.executor import parse_numeric, plan_depth, plan_labels

#: A stratum is a name and a predicate. The predicate says *hard case of this kind*, not
#: *record of this source* — a stratum that just means "RefChartQA" would audit the source
#: rather than the difficulty.
Detector = Callable[[ChartRecord], bool]


def _values(record: ChartRecord) -> list[float]:
    out = []
    for element in record.elements or []:
        value = parse_numeric(element.get("value"))
        if value is not None:
            out.append(value)
    return out


def _labels(record: ChartRecord) -> list[str]:
    return [str(e.get("label")) for e in (record.elements or [])
            if e.get("label") is not None]


def has_mined_plan(r: ChartRecord) -> bool:
    return bool(r.plan)


def has_duplicate_labels(r: ChartRecord) -> bool:
    labels = _labels(r)
    return len(set(labels)) < len(labels)


def is_multi_series(r: ChartRecord) -> bool:
    return len({e.get("series") for e in (r.elements or []) if e.get("series")}) > 1


def has_year_labels(r: ChartRecord) -> bool:
    labels = _labels(r)
    return bool(labels) and all(
        x.isdigit() and 1800 <= int(x) <= 2100 for x in labels)


def has_numeric_labels_that_are_not_years(r: ChartRecord) -> bool:
    labels = _labels(r)
    if not labels or has_year_labels(r):
        return False
    return all(parse_numeric(x) is not None for x in labels)


def is_percentage_chart(r: ChartRecord) -> bool:
    values = _values(r)
    if len(values) < 3:
        return False
    return 97.0 <= sum(values) <= 103.0 or any(
        (e.get("unit") or "") == "%" for e in (r.elements or []))


def is_count_question(r: ChartRecord) -> bool:
    q = r.question.lower()
    return q.startswith("how many") or "number of" in q


def is_extremum_question(r: ChartRecord) -> bool:
    op = (r.plan or {}).get("op")
    if op in {"argmax", "argmin", "max", "min"}:
        return True
    q = r.question.lower()
    return any(w in q for w in ("highest", "lowest", "largest", "smallest", "most",
                                "least", "peak", "maximum", "minimum"))


def is_nested_program(r: ChartRecord) -> bool:
    return bool(r.plan) and plan_depth(r.plan) >= 2


def is_aggregate_plan(r: ChartRecord) -> bool:
    return bool(r.plan) and not (r.plan.get("args") or [])


def has_tied_extremum(r: ChartRecord) -> bool:
    """The ambiguity 0127 refuses to *generate* and cannot stop existing in real data."""
    values = _values(r)
    if len(values) < 2:
        return False
    return values.count(max(values)) > 1 or values.count(min(values)) > 1


def has_a_small_element(r: ChartRecord, *, area: float = 400.0) -> bool:
    """A box under `area` in 0-1000 space — 2% of the image on a side. These are where
    grounding actually fails, and 41.3% of targets are under one visual token (0095)."""
    for element in r.elements or []:
        box = element.get("bbox")
        if (isinstance(box, list) and len(box) == 4
                and (box[2] - box[0]) * (box[3] - box[1]) < area):
            return True
    return False


def is_weak_match(r: ChartRecord, *, margin: float = 0.9) -> bool:
    """An aligned box whose runner-up was close. The alignment refuses below `MIN_MARGIN`,
    so these are the ones it accepted least confidently (0077)."""
    return any(e.get("match_margin") is not None and e["match_margin"] < margin
               for e in (r.elements or []))


def references_missing_evidence(r: ChartRecord) -> bool:
    if not r.plan:
        return False
    return bool(set(map(str, plan_labels(r.plan) or [])) - set(_labels(r)))


STRATA: dict[str, Detector] = {
    "mined_plan_correctness": has_mined_plan,
    "evidence_correctness": lambda r: bool(r.evidence),
    "chartqa_label_value_bbox": lambda r: r.source == "chartqa" and bool(r.elements),
    "refchartqa_weak_match": is_weak_match,
    "duplicate_labels": has_duplicate_labels,
    "multi_series": is_multi_series,
    "years": has_year_labels,
    "numeric_looking_categories": has_numeric_labels_that_are_not_years,
    "percentages": is_percentage_chart,
    "count_questions": is_count_question,
    "extrema": is_extremum_question,
    "nested_programs": is_nested_program,
    "aggregate_plans": is_aggregate_plan,
    "ambiguous_tied_extremum": has_tied_extremum,
    "small_visual_elements": has_a_small_element,
    "plan_references_missing_evidence": references_missing_evidence,
}


def sample(records: list[ChartRecord], *, per_stratum: int, seed: int
           ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Draw up to `per_stratum` from each stratum. Short strata stay short."""
    rows: list[dict[str, Any]] = []
    sizes: dict[str, int] = {}
    for name, detector in STRATA.items():
        eligible = sorted((r for r in records if detector(r)),
                          key=lambda r: r.record_id)
        sizes[name] = len(eligible)
        rng = random.Random(f"{seed}:{name}")
        for record in rng.sample(eligible, min(per_stratum, len(eligible))):
            rows.append({
                "stratum": name,
                "record_id": record.record_id,
                "source": record.source,
                "question": record.question,
                "answer": record.answer,
                "plan": record.plan,
                "plan_depth": plan_depth(record.plan) if record.plan else None,
                "n_elements": len(record.elements or []),
                "n_evidence": len(record.evidence or []) if record.evidence else 0,
                "labels": _labels(record)[:12],
                "image_path": record.image_path,
                # For a person to fill in. Never written by this script.
                "verdict": None,
                "reason": None,
            })
    return rows, sizes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-stratum", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chartqa-limit", type=int, default=4000)
    ap.add_argument("--refchartqa-cap", type=int, default=8000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from chartqa_dt.data.chartqa import ArchiveReader
    from scripts.build_mixtures import (
        archive_path,
        chartqa_records,
        refchartqa_records,
        synthetic_records,
    )

    root = Path(get_env().data_root)
    records: list[ChartRecord] = []
    records += chartqa_records(ArchiveReader(archive_path()),
                               limit=args.chartqa_limit, seed=args.seed)
    records += refchartqa_records(cap=args.refchartqa_cap,
                                  cache=root / "refchartqa_train.jsonl")
    manifest = root / "synthetic/train/manifest.json"
    if manifest.exists():
        records += synthetic_records(manifest)
    print(f"\ncandidate pool: {len(records):,} records")

    rows, sizes = sample(records, per_stratum=args.per_stratum, seed=args.seed)
    out = args.out or (root / "semantic_audit.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")

    print(f"\n{'stratum':34s} {'eligible':>9s} {'sampled':>8s}")
    short = []
    for name in STRATA:
        drawn = sum(1 for r in rows if r["stratum"] == name)
        flag = ""
        if drawn < args.per_stratum:
            flag = "  <- short, NOT padded"
            short.append(name)
        print(f"{name:34s} {sizes[name]:>9,} {drawn:>8,}{flag}")
    print(f"\n{len(rows):,} rows -> {out}")
    print("Every row has verdict=null. Judge them, then measure semantic correctness "
          "per stratum — that is the number executor agreement cannot give you.")
    if short:
        print(f"\n{len(short)} strata came up short and were left short: "
              f"{', '.join(short)}")


if __name__ == "__main__":
    main()
