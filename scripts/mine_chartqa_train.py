"""Run Appendix E plan mining over ChartQA training questions and report the yield.

`PLAN.md` 3.6. `IDEA.md` §5.1 predicts the uniqueness rule will admit roughly 5.7% of
real ChartQA questions — that estimate is the reason synthetic charts became the primary
source of plan supervision. This measures it on the actual training split, broken down by
chart source, so the mixture in 3.7 is built on a number rather than an expectation.

Reads members straight out of the pinned archive; nothing is extracted.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from chartqa_dt.data.chartqa import (
    ArchiveReader,
    annotation_path,
    parse_table,
    table_path,
)
from chartqa_dt.data.download import load_manifest
from chartqa_dt.env import get_env
from chartqa_dt.plans.mining import mine_plan


def archive_path() -> Path:
    manifest = load_manifest()
    entry = manifest["archives"]["chartqa"]
    return (Path(get_env().data_root) / "hf" / "datasets--ahmed-masry--ChartQA"
            / "snapshots" / entry["revision"] / entry["filename"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=4000, help="questions per kind")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("verification/mining_yield.json"))
    args = ap.parse_args()

    if args.split != "train":
        raise SystemExit(
            f"refusing to mine the {args.split!r} split. Rule 1: design parameters are "
            f"never chosen by looking at held-out data."
        )

    reader = ArchiveReader(archive_path())
    rng = random.Random(args.seed)
    stats: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    ops = collections.Counter()
    by_charttype = collections.defaultdict(collections.Counter)
    examples: list[dict] = []

    for kind in ("human", "machine"):
        rows = reader.qa_rows(args.split, kind)
        picked = rng.sample(rows, min(args.limit, len(rows)))
        for row in picked:
            stats[kind]["questions"] += 1
            tbl_name = table_path(args.split, row["imgname"])
            if not reader.exists(tbl_name):
                stats[kind]["no_table"] += 1
                continue
            try:
                table = parse_table(reader.read_text(tbl_name))
            except ValueError:
                stats[kind]["bad_table"] += 1
                continue

            chart_type = "unknown"
            ann_name = annotation_path(args.split, row["imgname"])
            if reader.exists(ann_name):
                chart_type = str(reader.read_json(ann_name).get("type", "unknown"))

            # `candidate_sets` documents that `rows` INCLUDES the header;
            # `parse_table` splits it off, so it has to be put back or the
            # first data row of every table is silently dropped.
            mined = mine_plan([table["columns"], *table["rows"]], row.get("label"))
            by_charttype[chart_type]["questions"] += 1
            if mined.plan is None:
                stats[kind][f"rejected:{mined.status}"] += 1
                continue
            stats[kind]["mined"] += 1
            by_charttype[chart_type]["mined"] += 1
            ops[mined.plan["op"]] += 1
            if len(examples) < 40:
                examples.append({"imgname": row["imgname"], "kind": kind,
                                 "chart_type": chart_type, "question": row["query"],
                                 "answer": row.get("label"), "plan": mined.plan})

    total_q = sum(s["questions"] for s in stats.values())
    total_m = sum(s["mined"] for s in stats.values())
    print(f"\nPlan mining on ChartQA {args.split} — {total_q:,} questions sampled\n")
    print(f"{'kind':<10}{'questions':>11}{'mined':>9}{'yield':>9}")
    for kind, s in stats.items():
        print(f"  {kind:<8}{s['questions']:>11,}{s['mined']:>9,}"
              f"{100 * s['mined'] / max(1, s['questions']):>8.2f}%")
    print(f"  {'ALL':<8}{total_q:>11,}{total_m:>9,}{100 * total_m / max(1, total_q):>8.2f}%")

    print("\nby chart type:")
    print(f"{'type':<12}{'questions':>11}{'mined':>9}{'yield':>9}")
    for t, c in sorted(by_charttype.items(), key=lambda kv: -kv[1]["questions"]):
        print(f"  {t:<10}{c['questions']:>11,}{c['mined']:>9,}"
              f"{100 * c['mined'] / max(1, c['questions']):>8.2f}%")

    print("\nmined operations:")
    for op, n in ops.most_common():
        print(f"  {op:<16}{n:>7,}  {100 * n / max(1, total_m):>5.1f}%")

    print("\nwhy questions were rejected (top reasons):")
    merged: collections.Counter = collections.Counter()
    for s in stats.values():
        for k, v in s.items():
            if k.startswith("rejected:") or k in ("no_table", "bad_table"):
                merged[k] += v
    for k, v in merged.most_common(8):
        print(f"  {k:<28}{v:>7,}  {100 * v / max(1, total_q):>5.1f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "split": args.split, "sampled_per_kind": args.limit, "seed": args.seed,
        "questions": total_q, "mined": total_m,
        "yield_pct": round(100 * total_m / max(1, total_q), 2),
        "by_kind": {k: dict(v) for k, v in stats.items()},
        "by_chart_type": {k: dict(v) for k, v in by_charttype.items()},
        "operations": dict(ops), "examples": examples,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
