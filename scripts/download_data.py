"""
    python scripts/download_data.py
    python scripts/download_data.py --splits train,validation,test
    python scripts/download_data.py --heldout-hashes
"""

from __future__ import annotations

import argparse
import json

from chartqa_dt.data.chartqa import HELDOUT_IMAGES, ArchiveReader, build_heldout_image_hashes
from chartqa_dt.data.sources import download_chartqa, download_refchartqa
from chartqa_dt.paths import cache_root, data_root, load_dotenv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="train,validation", help="RefChartQA splits to download")
    ap.add_argument("--heldout-hashes", action="store_true", help="rebuild data/heldout_images.json")
    args = ap.parse_args()
    load_dotenv()
    archive = download_chartqa(data_root())
    print(f"ChartQA archive: {archive}")
    for split in (s.strip() for s in args.splits.split(",") if s.strip()):
        shards = download_refchartqa(split, cache_root())
        print(f"RefChartQA {split}: {len(shards)} parquet shard(s)")
    if args.heldout_hashes:
        with ArchiveReader(archive) as reader:
            hashes = build_heldout_image_hashes(reader)
        HELDOUT_IMAGES.write_text(json.dumps(hashes, indent=0) + "\n", encoding="utf-8")
        print(f"held-out image hashes: {len(hashes['val'])} validation, {len(hashes['test'])} test")


if __name__ == "__main__":
    main()
