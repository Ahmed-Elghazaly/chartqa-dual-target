#!/usr/bin/env python3
"""AUDIT · an UNBIASED sample for judging DSL expressiveness.

`audit/llm_mining_sample.jsonl` was drawn from records the deterministic miner could not
settle. That pool is enriched by construction for the hardest question types, so the 45%
"blocked by a missing operator" rate measured on it describes miner failures, not the
corpus. `audit/measure_dsl_coverage.py` puts the corpus rate at >= 7.5%, but 76% of
questions fall into a `numeric_other` catch-all that the regexes cannot judge.

This draws a plain random sample of ChartQA training questions -- no filtering on whether
the miner succeeded -- with the gold table attached, so each one can be judged on whether
the DSL can express it at all. The estimate that comes out is unbiased for the corpus.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.data.chartqa import ArchiveReader, parse_table, table_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="audit/dsl_expressiveness_sample.jsonl")
    args = ap.parse_args()

    from scripts.build_mixtures import archive_path

    rng = random.Random(args.seed)
    with ArchiveReader(archive_path()) as reader:
        rows = [(kind, r) for kind in ("human", "machine")
                for r in reader.qa_rows("train", kind)]
        print(f"population: {len(rows):,} ChartQA train questions "
              f"(no filtering on miner outcome)")
        picked, seen = [], 0
        for kind, row in rng.sample(rows, len(rows)):
            seen += 1
            tbl = table_path("train", row["imgname"])
            if not reader.exists(tbl):
                continue
            try:
                table = parse_table(reader.read_text(tbl))
            except ValueError:
                continue
            picked.append({"kind": kind, "imgname": row["imgname"],
                           "question": str(row["query"]), "answer": str(row.get("label", "")),
                           "columns": table["columns"], "rows": table["rows"][:14]})
            if len(picked) >= args.n:
                break

    out = Path(args.out)
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in picked),
                   encoding="utf-8")
    print(f"drew {len(picked)} after scanning {seen:,} (seed {args.seed}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
