#!/usr/bin/env python3
"""End-to-end smoke over the real data: build targets, then **print some and read them**.

This exists because of one defect. Grounding-only targets were extended to ChartQA, and the
composition report showed ChartQA going from 5 records to 4,944. It looked like a large
win. One actual target read:

    "Which year has the most crime?"  answer 2014  evidence: all six years.

4,939 records that teach *"point at everything"*, and every aggregate number said the change
was working (`DECISIONS.md` 0116).

So this script deliberately does the thing counting cannot: it renders whole targets and
puts them in front of you. **A total that moves the way you hoped is the least trustworthy
evidence available.** Run it after any change to data generation, record construction, or
target building — before committing, not after.

It also asserts the invariants that are cheap to check and expensive to lose:

* every source sets `elements`, and says whether it knows its `evidence` (0124);
* no evidence item is emitted without a usable box;
* a grounding-only target never comes from whole-chart boxes (0116);
* the per-source usable rate has not moved by more than `--tolerance` against
  `data/composition_snapshot.json`, which is the check that turns "5 -> 4,944" from a
  pleasant surprise into a failure.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "composition_snapshot.json"


def build(limit_chartqa: int, cap_refchartqa: int, seed: int):
    from chartqa_dt.data.chartqa import ArchiveReader
    from chartqa_dt.env import get_env
    from scripts.build_mixtures import (
        archive_path,
        chartqa_records,
        refchartqa_records,
        split_by_usability,
        synthetic_records,
    )

    root = Path(get_env().data_root)
    pools = {}
    if Path(archive_path()).exists():
        pools["chartqa"] = chartqa_records(ArchiveReader(archive_path()),
                                           limit=limit_chartqa, seed=seed)
    cache = root / "refchartqa_train.jsonl"
    if cache.exists():
        pools["refchartqa"] = refchartqa_records(cap=cap_refchartqa, cache=cache)
    manifest = root / "synthetic/train/manifest.json"
    if manifest.exists():
        pools["synthetic"] = synthetic_records(manifest)[:cap_refchartqa]
    if not pools:
        print("no data available locally; nothing to smoke")
        return None
    return {name: split_by_usability(records, name) for name, records in pools.items()}


def show(pools, *, n: int, seed: int) -> None:
    """Print whole targets. The point of the script."""
    from chartqa_dt.train.targets import (
        NoPlanAvailable,
        TargetError,
        build_grounding_only_target,
        build_target,
    )

    rng = random.Random(seed)
    print("\n" + "=" * 78)
    print("READ THESE. Counting cannot see what is wrong with a target; you can.")
    print("=" * 78)
    for name, (plans, grounding) in pools.items():
        for kind, records in (("plan", plans), ("grounding-only", grounding)):
            if not records:
                continue
            for record in rng.sample(records, min(n, len(records))):
                try:
                    text = (build_target(record) if kind == "plan"
                            else build_grounding_only_target(record))
                except (TargetError, NoPlanAvailable) as exc:
                    print(f"\n[{name}/{kind}] {record.record_id} REFUSED: {exc}")
                    continue
                target = json.loads(text)
                print(f"\n[{name}/{kind}] {record.record_id}")
                print(f"  Q: {record.question}")
                print(f"  gold answer : {record.answer!r}")
                print(f"  plan        : {json.dumps(target.get('plan'))}")
                print(f"  evidence    : {len(target['evidence'])} of "
                      f"{len(record.elements or [])} chart elements")
                for item in target["evidence"][:6]:
                    print(f"      {item['label']!r} = {item['value']!r} "
                          f"bbox={item['bbox']}")
                if len(target["evidence"]) > 6:
                    print(f"      ... and {len(target['evidence']) - 6} more")


def check_invariants(pools) -> list[str]:
    problems = []
    for name, (plans, grounding) in pools.items():
        for record in (plans + grounding)[:2000]:
            if record.elements is None:
                problems.append(f"{name}: {record.record_id} has no elements")
                break
        for record in grounding[:2000]:
            if not record.has_question_evidence:
                problems.append(
                    f"{name}: {record.record_id} became a grounding-only target from "
                    f"whole-chart boxes — this is DECISIONS.md 0116 happening again")
                break
    return problems


def compare_to_snapshot(pools, tolerance: float) -> list[str]:
    current = {name: {"plan": len(p), "grounding": len(g)}
               for name, (p, g) in pools.items()}
    if not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"\nno snapshot yet — wrote one to {SNAPSHOT.relative_to(Path.cwd())}")
        return []
    before = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    drifted = []
    print(f"\n{'source/kind':28s} {'snapshot':>10s} {'now':>10s} {'change':>9s}")
    for name, kinds in current.items():
        for kind, now in kinds.items():
            was = (before.get(name) or {}).get(kind)
            if was is None:
                print(f"{name + '/' + kind:28s} {'(new)':>10s} {now:>10,}")
                continue
            delta = (now - was) / was if was else (1.0 if now else 0.0)
            flag = ""
            if abs(delta) > tolerance:
                flag = "  <-- DRIFT"
                drifted.append(f"{name}/{kind}: {was:,} -> {now:,} ({delta:+.0%})")
            print(f"{name + '/' + kind:28s} {was:>10,} {now:>10,} {delta:>8.0%}{flag}")
    return drifted


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chartqa-limit", type=int, default=300)
    ap.add_argument("--refchartqa-cap", type=int, default=1500)
    ap.add_argument("--show", type=int, default=2, help="targets to print per source/kind")
    ap.add_argument("--tolerance", type=float, default=0.05,
                    help="fractional change in a usable count that counts as drift")
    ap.add_argument("--update-snapshot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pools = build(args.chartqa_limit, args.refchartqa_cap, args.seed)
    if pools is None:
        return
    show(pools, n=args.show, seed=args.seed)

    problems = check_invariants(pools)
    if args.update_snapshot and SNAPSHOT.exists():
        SNAPSHOT.unlink()
    drifted = compare_to_snapshot(pools, args.tolerance)

    print()
    if problems:
        print("INVARIANTS BROKEN:")
        for p in problems:
            print(f"  {p}")
    if drifted:
        print("COMPOSITION DRIFTED beyond tolerance:")
        for d in drifted:
            print(f"  {d}")
        print("  If this is intended, rerun with --update-snapshot and say why in "
              "DECISIONS.md. If it is a pleasant surprise, read the targets above first.")
    if problems or drifted:
        raise SystemExit(1)
    print("end-to-end smoke: OK")


if __name__ == "__main__":
    main()
