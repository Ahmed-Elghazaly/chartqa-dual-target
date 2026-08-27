"""Freeze the validation slices Phase 5 iterates on — `PLAN.md` 5.1, 5.2, 5.3, 5.4.

Three slices, built once, recorded by id and hash:

* `chartqa_variant_200` — the **frozen 200-question ChartQA validation slice, balanced
  between human and machine**, on which 5.2 chooses Instruct or Thinking.
* `chartqa_val` — the ChartQA validation set for the 5.3 zero-shot number.
* `refchartqa_val` — RefChartQA validation for the 5.4 grounding number.

**Why freeze them, and why now.** 5.1 says prompt design iterates *on validation data
only*, and 5.2 says the comparison runs on a *frozen* slice. A slice re-sampled between
runs would let a prompt that got lucky on one draw look better than one that did not, and
nobody would see it happen. Sampling once, before any prompt exists, removes the
opportunity — the same reason evaluation is built before training.

Test splits are untouched: `chartqa_dt.splits` refuses them, and nothing here asks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from chartqa_dt.data.chartqa import ArchiveReader, annotation_path, image_path, qa_path
from chartqa_dt.data.download import load_manifest
from chartqa_dt.data.records import image_content_sha256, make_record_id
from chartqa_dt.env import get_env

SLICE_DIR = Path("data/slices")
VARIANT_SLICE_SIZE = 200


def archive() -> ArchiveReader:
    entry = load_manifest()["archives"]["chartqa"]
    return ArchiveReader(Path(get_env().data_root) / "hf"
                         / "datasets--ahmed-masry--ChartQA" / "snapshots"
                         / entry["revision"] / entry["filename"])


def chartqa_val_rows(reader: ArchiveReader) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind in ("human", "machine"):
        for row in reader.read_json(qa_path("val", kind)):
            rows.append({**row, "question_kind": kind})
    return rows


def build_chartqa_slice(reader: ArchiveReader, rows: list[dict[str, Any]], *,
                        size: int, seed: int) -> list[dict[str, Any]]:
    """A balanced sample: `PLAN.md` 5.2 asks for human and machine in equal measure."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    per_kind = size // 2
    for kind in ("human", "machine"):
        pool = [r for r in rows if r["question_kind"] == kind]
        for row in rng.sample(pool, min(per_kind, len(pool))):
            name = image_path("val", row["imgname"])
            if not reader.exists(name):
                continue
            digest = image_content_sha256(reader.read(name))
            out.append({
                "record_id": make_record_id("chartqa", "val", digest, row["query"]),
                "image_member": name,
                "image_sha256": digest,
                "imgname": row["imgname"],
                "question": row["query"],
                "answer": str(row["label"]),
                "question_kind": kind,
                "chart_type": (reader.read_json(annotation_path("val", row["imgname"]))
                               .get("type") if reader.exists(
                                   annotation_path("val", row["imgname"])) else None),
            })
    rng.shuffle(out)
    return out


def write_slice(name: str, rows: list[dict[str, Any]], meta: dict[str, Any]) -> Path:
    """Ids and hashes only — rule 7 forbids dataset content in the repository."""
    SLICE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"name": name, "n": len(rows), **meta,
               "record_ids": [r["record_id"] for r in rows],
               "image_sha256": [r["image_sha256"] for r in rows]}
    digest = hashlib.sha256(
        json.dumps(payload["record_ids"], sort_keys=True).encode()).hexdigest()
    payload["slice_sha256"] = digest
    path = SLICE_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # The working copy, with questions and answers, stays out of the repository.
    local = Path(get_env().data_root) / "slices"
    local.mkdir(parents=True, exist_ok=True)
    (local / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"  {name:<22} n={len(rows):<6} sha256 {digest[:16]}…  "
          f"-> {path} (+ working copy in {local})")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--variant-size", type=int, default=VARIANT_SLICE_SIZE)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even though a slice already exists — a frozen slice "
                         "that quietly changes defeats the reason it is frozen")
    args = ap.parse_args()

    existing = sorted(SLICE_DIR.glob("*.json")) if SLICE_DIR.exists() else []
    if existing and not args.force:
        print("slices already frozen; refusing to rebuild without --force:")
        for p in existing:
            d = json.loads(p.read_text())
            print(f"  {d['name']:<22} n={d['n']:<6} sha256 {d['slice_sha256'][:16]}…")
        return

    reader = archive()
    rows = chartqa_val_rows(reader)
    print(f"\nChartQA validation: {len(rows):,} questions "
          f"({sum(r['question_kind'] == 'human' for r in rows):,} human)\n")

    variant = build_chartqa_slice(reader, rows, size=args.variant_size, seed=args.seed)
    write_slice("chartqa_variant_200", variant,
                {"purpose": "PLAN 5.2 model-variant selection; balanced human/machine",
                 "seed": args.seed, "split": "val", "source": "chartqa"})

    full = build_chartqa_slice(reader, rows, size=len(rows), seed=args.seed)
    write_slice("chartqa_val", full,
                {"purpose": "PLAN 5.3 zero-shot ChartQA", "seed": args.seed,
                 "split": "val", "source": "chartqa"})
    print("\nRefChartQA validation is streamed at run time; see scripts/cache_refchartqa.py")


if __name__ == "__main__":
    main()
