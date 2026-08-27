"""Record the expected SHA-256 of every parquet file, without downloading them.

`PLAN.md` 3.1 requires hash-verified downloads. RefChartQA is 2.88 GB and is *streamed*
here rather than downloaded — the audit needs 200 rows and the development machine has no
room for the rest — so there is no local file to hash.

The Hugging Face API publishes the SHA-256 of every LFS object at a given revision, so the
expected hashes can be recorded now, from the pinned revision, for a few kilobytes. The
verification then happens wherever the download actually happens (Kaggle, for training),
against a hash committed before anyone had the file.

That ordering is the point. A hash recorded *after* downloading only proves the file has
not changed since you fetched it; one recorded from the pinned revision beforehand proves
you got the file the project was designed against.
"""

from __future__ import annotations

import argparse
import json

from chartqa_dt.data.download import MANIFEST_PATH, load_manifest, save_manifest
from chartqa_dt.data.sources import SOURCES, ParquetSpec
from chartqa_dt.net import get_json


def parquet_hashes(spec: ParquetSpec) -> dict[str, dict[str, object]]:
    url = (f"https://huggingface.co/api/datasets/{spec.repo_id}/revision/"
           f"{spec.revision}?blobs=true")
    data = get_json(url, timeout=90)
    out: dict[str, dict[str, object]] = {}
    for sibling in data.get("siblings", []):
        name = sibling["rfilename"]
        if not name.endswith(".parquet"):
            continue
        lfs = sibling.get("lfs") or {}
        digest = lfs.get("sha256") or lfs.get("oid")
        if not digest:
            continue
        out[name] = {"sha256": digest, "size_bytes": lfs.get("size") or sibling.get("size")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="refchartqa")
    args = ap.parse_args()

    manifest = load_manifest()
    manifest.setdefault("parquet", {})
    for key in (k.strip() for k in args.datasets.split(",") if k.strip()):
        spec = SOURCES[key]
        if not isinstance(spec, ParquetSpec):
            raise SystemExit(f"{key} is an archive, not a parquet dataset")
        files = parquet_hashes(spec)
        if not files:
            raise SystemExit(f"{key}: the API returned no parquet hashes")
        total = sum(int(f["size_bytes"] or 0) for f in files.values())
        manifest["parquet"][key] = {
            "repo_id": spec.repo_id, "revision": spec.revision,
            "files": files, "total_bytes": total,
            "_note": "Expected hashes read from the Hugging Face API at the pinned "
                     "revision. Nothing was downloaded; verification happens wherever "
                     "the download does, against a hash committed beforehand.",
        }
        print(f"{key}: {len(files)} parquet files, {total:,} bytes total")
        for name, entry in sorted(files.items()):
            print(f"  {name:<34} {int(entry['size_bytes']):>13,}  "
                  f"{str(entry['sha256'])[:16]}…")

    save_manifest(manifest, MANIFEST_PATH)
    print(f"\nrecorded in {MANIFEST_PATH}")
    print(json.dumps({"parquet_datasets": list(manifest["parquet"])}, indent=2))


if __name__ == "__main__":
    main()
