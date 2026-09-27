"""
    python scripts/prepare_refchartqa.py
    python scripts/prepare_refchartqa.py --skip-cache
"""

from __future__ import annotations

import argparse
import json

from chartqa_dt.config import build_config
from chartqa_dt.data.chartqa import ArchiveReader, chartqa_training_records, heldout_image_hashes
from chartqa_dt.data.records import ChartRecord
from chartqa_dt.data.refchartqa import (
    align_training_cache,
    cache_training_split,
    read_jsonl,
    write_jsonl,
)
from chartqa_dt.data.sources import chartqa_archive_path, load_refchartqa
from chartqa_dt.paths import cache_root, data_root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--skip-cache", action="store_true", help="reuse an existing refchartqa_train.jsonl")
    args = ap.parse_args()
    cfg = build_config(args.config)
    root = data_root().resolve()
    cache = root / "refchartqa_train.jsonl"

    if not args.skip_cache:
        records, outcomes = cache_training_split(
            load_refchartqa("train", cache_root()),
            data_root=root, image_dir=root / "refchartqa" / "train", heldout=heldout_image_hashes(),
        )
        write_jsonl(cache, records)
        print(f"cached {len(records):,} records -> {cache}; outcomes {outcomes}")

    records_ = [ChartRecord.from_dict(raw) for raw in read_jsonl(cache)]
    with ArchiveReader(chartqa_archive_path(root)) as reader:
        chartqa = chartqa_training_records(reader, limit=cfg.data.chartqa_limit_per_kind,
                                           seed=cfg.data.mixture_seed)
    aligned, stats = align_training_cache(records_, chartqa)
    write_jsonl(root / "refchartqa_aligned.jsonl", aligned)
    stats_out = {"min_iou": 0.9, "min_margin": 0.5, "cached_records": len(records_), **stats,
                 "aligned_pct": round(100 * stats["aligned"] / max(len(records_), 1), 2)}
    (root / "refchartqa_alignment.json").write_text(json.dumps(stats_out, indent=1) + "\n")
    print(f"aligned {stats['aligned']:,} of {len(records_):,} records")


if __name__ == "__main__":
    main()
