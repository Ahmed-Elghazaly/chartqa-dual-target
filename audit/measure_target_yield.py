#!/usr/bin/env python3
"""MEASUREMENT · how many real ChartQA records yield a usable training target.

This is the number that matters: not how many plans are mined, but how many survive all the
way to a target whose own plan reproduces its own answer. It is measured twice over the same
records -- once with the parser as it now is, once with the pre-fix semantics restored -- so
the value of the fix is a delta on identical inputs rather than an assertion.

The two pre-fix behaviours being priced:
  * a trailing `%` divided the value by 100, while the miner's parser did not
  * a space-separated thousand (`'3 071'`) raised instead of parsing
"""
from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord  # noqa: E402
from chartqa_dt.plans import executor  # noqa: E402
from chartqa_dt.plans.mining import mine_plan  # noqa: E402
from chartqa_dt.train.targets import TargetError, build_target  # noqa: E402

_REAL = executor.parse_numeric


def _pre_fix(x):
    """`executor.to_number` as it was: divides percents, cannot read a spaced thousand."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return _REAL(x)
    if not isinstance(x, str):
        return None
    s = x.strip().replace(",", "")
    pct = s.endswith("%")
    if pct:
        s = s[:-1]
    try:
        v = float(s)
    except ValueError:
        return None
    return v / 100.0 if pct else v


def run(records: list[ChartRecord], *, label: str) -> collections.Counter:
    out: collections.Counter[str] = collections.Counter()
    for r in records:
        try:
            build_target(r)
            out["built"] += 1
        except TargetError as exc:
            msg = str(exc)
            if "still names more than one mark" in msg:
                out["label repeats within one series"] += 1
            elif "no element box" in msg:
                out["plan names a label with no box"] += 1
            elif "the two sources disagree" in msg or "annotated element" in msg:
                out["table and annotation disagree on the value"] += 1
            elif "not numeric" in msg:
                out["value could not be parsed"] += 1
            elif "does not reproduce" in msg:
                out["plan disagrees with the answer"] += 1
            elif "more than the schema" in msg or "evidence items" in msg:
                out["needs more evidence than the cap"] += 1
            elif "no mined plan" in msg:
                out["no plan"] += 1
            else:
                out["other"] += 1
    print(f"  {label}")
    total = sum(out.values())
    for k, v in out.most_common():
        print(f"    {k:<36}{v:>6,}  ({100 * v / total:>5.1f}%)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scripts.build_mixtures import archive_path

    from chartqa_dt.data.chartqa import (
        ArchiveReader,
        annotation_boxes,
        annotation_path,
        image_path,
        parse_table,
        table_path,
    )

    records: list[ChartRecord] = []
    with ArchiveReader(archive_path()) as reader:
        pool = [r for k in ("human", "machine") for r in reader.qa_rows("train", k)]
        random.Random(args.seed).shuffle(pool)
        for row in pool:
            if len(records) >= args.limit:
                break
            ann, img, tbl = (annotation_path("train", row["imgname"]),
                             image_path("train", row["imgname"]),
                             table_path("train", row["imgname"]))
            if not (reader.exists(ann) and reader.exists(img) and reader.exists(tbl)):
                continue
            try:
                table = parse_table(reader.read_text(tbl))
            except ValueError:
                continue
            w, h = reader.image_size(img)
            elements = annotation_boxes(reader.read_json(ann), w, h)
            if not elements:
                continue
            mined = mine_plan([table["columns"], *table["rows"]], row.get("label"))
            records.append(ChartRecord(
                record_id=f"r{len(records)}", source="chartqa", split="train",
                image_path=str(img), image_sha256="x", question=str(row["query"]),
                answer=str(row.get("label", "")), question_kind="human",
                table=table, plan=mined.plan,
                boxes=[e["bbox"] for e in elements if e.get("bbox")],
                meta={ELEMENTS_KEY: elements}))

    planned = sum(1 for r in records if r.plan)
    print(f"{len(records):,} real ChartQA records, {planned:,} with a mined plan\n")
    executor.parse_numeric = _pre_fix
    before = run(records, label="BEFORE — percents divided, spaced thousands refused")
    executor.parse_numeric = _REAL
    print()
    after = run(records, label="AFTER — one shared parser")
    gain = after["built"] - before["built"]
    print(f"\n  targets built: {before['built']:,} -> {after['built']:,}   "
          f"{gain:+,} ({100 * gain / max(before['built'], 1):+.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
