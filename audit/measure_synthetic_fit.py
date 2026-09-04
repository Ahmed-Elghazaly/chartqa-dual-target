#!/usr/bin/env python3
"""MEASUREMENT · does the synthetic corpus resemble the corpus it prepares the model for?

`synth/generator.py` states its own purpose: *"this is the primary source of plan
supervision, given that the uniqueness rule admits only ~5.7% of real ChartQA questions."*

**That premise no longer holds.** The uniqueness rule is off the supervision path — a reader
mines plans for real ChartQA questions directly (`DECISIONS.md` 0085, 0088) — so synthetic
data is no longer the only way to teach a plan. Its distribution was chosen for a job it no
longer has, and this measures how far that distribution sits from the real one.

Two comparisons, both against real ChartQA train:

  * **chart type**, counted from the annotations
  * **plan operation**, against the 60-question hand judgement in 0081. That sample is small
    (n=60) and its confidence intervals are wide; it is quoted as the best available estimate
    of what real questions ask for, not as a precise figure.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

#: What Claude judged 60 random ChartQA training questions to need (`DECISIONS.md` 0081).
#: Percentages of the 56 that were expressible at all.
REAL_OPERATIONS = {"lookup": 36, "argmax/argmin": 12, "sum": 2, "count": 2,
                   "ratio": 1, "difference": 1, "mean": 1, "max": 1}

#: The synthetic generator's own chart-type names, mapped onto ChartQA's vocabulary.
FAMILY = {"vbar": "bar", "hbar": "bar", "grouped_bar": "bar", "v_bar": "bar",
          "h_bar": "bar", "line": "line", "multi_line": "line", "pie": "pie",
          "scatter": "scatter", "area": "area"}


def skew(synthetic: float, real: float) -> str:
    if real == 0:
        return "NOT IN CHARTQA" if synthetic else "—"
    if synthetic == 0:
        return "absent"
    return f"{synthetic / real:.1f}x over" if synthetic > real else f"{real / synthetic:.1f}x under"


def table(title: str, synth: collections.Counter, real: collections.Counter) -> None:
    ns, nr = sum(synth.values()) or 1, sum(real.values()) or 1
    print(f"\n{title}\n")
    print(f"  {'':<16}{'synthetic':>11}{'real':>10}   ")
    for key in sorted(set(synth) | set(real), key=lambda k: -real.get(k, 0)):
        a, b = 100 * synth.get(key, 0) / ns, 100 * real.get(key, 0) / nr
        print(f"  {key:<16}{a:>10.1f}%{b:>9.1f}%   {skew(a, b)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = Path.home() / ".cache/chartqa_dt/data/synthetic/train/manifest.json"
    if not manifest.exists():
        print(f"no synthetic manifest at {manifest}")
        return 1
    examples = json.loads(manifest.read_text(encoding="utf-8"))["examples"]

    from scripts.build_mixtures import archive_path

    from chartqa_dt.data.chartqa import ArchiveReader, annotation_path

    real_types: collections.Counter[str] = collections.Counter()
    with ArchiveReader(archive_path()) as reader:
        pool = [r for k in ("human", "machine") for r in reader.qa_rows("train", k)]
        random.Random(args.seed).shuffle(pool)
        seen: set[str] = set()
        for row in pool:
            if len(seen) >= args.limit:
                break
            name = row["imgname"]
            if name in seen or not reader.exists(annotation_path("train", name)):
                continue
            seen.add(name)
            kind = str(reader.read_json(annotation_path("train", name)).get("type"))
            real_types[FAMILY.get(kind, kind)] += 1

    synth_types = collections.Counter(
        FAMILY.get(str(e.get("chart_type")), str(e.get("chart_type"))) for e in examples)
    table(f"CHART TYPE — {len(examples):,} synthetic vs {len(seen):,} real charts",
          synth_types, real_types)

    def top(plan):
        op = plan.get("op") if isinstance(plan, dict) else None
        return "argmax/argmin" if op in ("argmax", "argmin") else op

    synth_ops = collections.Counter(top(e.get("plan")) for e in examples)
    table("PLAN OPERATION — synthetic vs Claude's judgement of 60 random real questions",
          synth_ops, collections.Counter(REAL_OPERATIONS))

    absent = sum(v for k, v in synth_types.items() if not real_types.get(k))
    print(f"\n  synthetic examples on chart types ChartQA does not contain: "
          f"{absent:,} of {len(examples):,} ({100 * absent / len(examples):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
