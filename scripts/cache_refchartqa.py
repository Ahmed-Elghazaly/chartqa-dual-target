"""Stream RefChartQA training rows into a local record cache.

The 3.4 audit passed at 100% (`DECISIONS.md` 0047), so RefChartQA training rows go into
the mixtures. Streaming rather than downloading the full 2.88 GB: the images are written
once, at a size the machine can hold, and the cache is a JSONL of `ChartRecord`s — ids
and hashes, no dataset content in the repo (rule 7).

**The default caches the whole split, and that is not a detail.** It used to be 4,000,
matching `PLAN.md` 3.4's *starting* rung, and it silently became the ceiling: the cache
held 3,996 of 55,789 rows, so the ladder's 10,000 and 25,000 rungs had no data to run on
and the project trained on 7.2% of the dataset for a month (`DECISIONS.md` 0112, 0115).
A cap that exists to be raised should not be the thing that stops you raising it.

Resumable, and keyed on `refchartqa_id`, so raising the cap is strictly additive: a run
that dies at row 3,000 costs nothing, and re-running with a larger cap keeps what is
already there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chartqa_dt.data.records import ChartRecord, image_content_sha256
from chartqa_dt.data.refchartqa import row_to_record
from chartqa_dt.data.sources import REFCHARTQA_PARQUET as SPEC
from chartqa_dt.env import get_env
from chartqa_dt.splits import sealed_image_hashes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    #: The whole split is 55,789 rows; 60,000 caches all of it with headroom. See the
    #: module docstring for why this is not 4,000 any more.
    ap.add_argument("--cap", type=int, default=60_000)
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

    sealed = sealed_image_hashes()
    if not sealed:
        raise SystemExit(
            "data/sealed_images.json is missing — run scripts/build_sealed_images.py "
            "first. Without it this cache cannot tell a training chart from a held-out "
            "one, and 4 rows in 4,000 are held-out charts labelled 'train'."
        )
    written = contaminated = 0
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
            digest = image_content_sha256(image)
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
            if record.image_sha256 in sealed:
                # RefChartQA labels this row "train", but the CHART is one ChartQA holds
                # out. Measured at 4 in 4,000 (`DECISIONS.md` 0049). Dropped here, at the
                # point it enters the project, and counted — the mixture builder asserts
                # the same thing as a last line of defence and should never fire.
                raw_path.unlink(missing_ok=True)
                contaminated += 1
                continue
            fh.write(json.dumps(record.to_dict()) + "\n")
            fh.flush()
            written += 1
            if written % 250 == 0:
                print(f"  {len(done) + written:,}/{args.cap:,}")

    print(f"\n{len(done) + written:,} records in {args.out}")
    print(f"{contaminated} rows dropped: labelled 'train' but using a held-out ChartQA "
          f"chart")
    print(f"images in {image_dir}")


if __name__ == "__main__":
    main()
