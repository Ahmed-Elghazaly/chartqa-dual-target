"""Stream RefChartQA training rows into a local record cache.

The 3.4 audit passed at 100% (`DECISIONS.md` 0047), so RefChartQA training rows go into
the mixtures, starting at the plan's 4,000 single-box cap. Streaming rather than
downloading the full 2.88 GB: the images are written once, at a size the machine can
hold, and the cache is a JSONL of `ChartRecord`s — ids and hashes, no dataset content
in the repo (rule 7).

Resumable. A stream that dies at row 3,000 should not cost the first 3,000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from chartqa_dt.data.records import ChartRecord
from chartqa_dt.data.refchartqa import row_to_record
from chartqa_dt.data.sources import REFCHARTQA_PARQUET as SPEC
from chartqa_dt.env import get_env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path.home() / ".cache/chartqa_dt/data/refchartqa_train.jsonl")
    ap.add_argument("--image-dir", type=Path, default=None)
    args = ap.parse_args()

    from datasets import load_dataset

    image_dir = args.image_dir or (Path(get_env().data_root) / "refchartqa" / "train")
    image_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line:
                done.add(json.loads(line)["meta"]["refchartqa_id"])
        print(f"resuming: {len(done):,} rows already cached")
    if len(done) >= args.cap:
        print(f"cache already holds {len(done):,} >= cap {args.cap:,}; nothing to do")
        return

    stream = load_dataset(SPEC.repo_id, split="train", streaming=True,
                          revision=SPEC.revision).shuffle(seed=args.seed, buffer_size=5000)

    written = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for row in stream:
            rid = row.get("id")
            if rid in done:
                continue
            if len(done) + written >= args.cap:
                break
            image = row["image"].convert("RGB")
            raw_path = image_dir / f"{rid}.png"
            image.save(raw_path)
            digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            try:
                record: ChartRecord = row_to_record(
                    row, split="train", image_path=raw_path, image_sha256=digest,
                    image_size=image.size)
            except ValueError as exc:
                print(f"  skipped {rid}: {exc}")
                raw_path.unlink(missing_ok=True)
                continue
            if not record.boxes:
                raw_path.unlink(missing_ok=True)
                continue      # no usable box: nothing to ground on
            fh.write(json.dumps(record.to_dict()) + "\n")
            fh.flush()
            written += 1
            if written % 250 == 0:
                print(f"  {len(done) + written:,}/{args.cap:,}")

    print(f"\n{len(done) + written:,} records in {args.out}")
    print(f"images in {image_dir}")


if __name__ == "__main__":
    main()
