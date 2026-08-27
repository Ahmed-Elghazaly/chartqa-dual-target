"""Record the pixel hash of every ChartQA validation and test image.

Rule 1 forbids training on held-out data. The obvious reading is "do not put val/test
*records* in the mixture", and `tests/test_mixture.py` enforces that. It is not enough.

**Measured:** of 4,000 cached RefChartQA *training* rows, 3,996 use a ChartQA *training*
image — and **2 use a ChartQA test image, 2 a ChartQA validation image**. RefChartQA
labels those rows "train", so every split check in this project would pass them, and the
model would still have seen ChartQA test charts during training. The contamination is
image-level, and no amount of checking record splits finds it.

The output is a list of hashes — derived data, no dataset content — so it is committed and
the guard works on any machine without the archive present (rule 7).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chartqa_dt.data.chartqa import ArchiveReader, image_path
from chartqa_dt.data.download import load_manifest
from chartqa_dt.data.records import image_content_sha256
from chartqa_dt.env import get_env

SEALED_PATH = Path("data/sealed_images.json")
SEALED_SPLITS = ("val", "test")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=SEALED_PATH)
    args = ap.parse_args()

    entry = load_manifest()["archives"]["chartqa"]
    archive = (Path(get_env().data_root) / "hf" / "datasets--ahmed-masry--ChartQA"
               / "snapshots" / entry["revision"] / entry["filename"])
    reader = ArchiveReader(archive)

    hashes: dict[str, list[str]] = {}
    for split in SEALED_SPLITS:
        prefix = image_path(split, "")
        names = [n for n in reader._names if n.startswith(prefix) and n.endswith(".png")]
        hashes[split] = sorted({image_content_sha256(reader.read(n)) for n in names})
        print(f"  {split:<6} {len(names):>6,} images -> {len(hashes[split]):,} distinct hashes")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "_purpose": "Pixel hashes of every ChartQA validation and test image. Any training "
                    "record whose image hashes to one of these is contaminated, whatever "
                    "split its own dataset assigns it. Hashes only — no dataset content.",
        "_source": {"repo_id": entry["repo_id"], "revision": entry["revision"],
                    "archive_sha256": entry["sha256"]},
        "hashes": hashes,
    }, indent=2) + "\n", encoding="utf-8")
    total = sum(len(v) for v in hashes.values())
    print(f"\n{total:,} sealed image hashes written to {args.out}")


if __name__ == "__main__":
    main()
