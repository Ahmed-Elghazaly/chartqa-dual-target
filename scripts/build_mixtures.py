"""Build `data/mixture_stage1.json` and `data/mixture_stage2.json` — `PLAN.md` 3.7.

Sources, in the order the plan specifies:

* **Synthetic** — exact boxes, exact answers, exact plans, curriculum levels L1–L4.
* **ChartQA training annotations** — real charts with gold element boxes
  (`DECISIONS.md` 0042), plus mined typed plans where the uniqueness rule admits one
  (`DECISIONS.md` 0045/0046). Questions without a unique plan are kept as answer and
  evidence supervision, never given an invented plan.
* **RefChartQA training rows** — included because the 3.4 audit passed at 100%
  (`DECISIONS.md` 0047), starting at the plan's 4,000 single-box cap.

Training split only, and the leak check runs on the inputs rather than the survivors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from chartqa_dt.data.chartqa import (
    ArchiveReader,
    annotation_boxes,
    annotation_path,
    image_path,
    parse_table,
    table_path,
)
from chartqa_dt.data.download import load_manifest
from chartqa_dt.data.mixture import (
    STAGE1_CAP,
    STAGE2_CAP,
    build_stage1,
    build_stage2,
    write_mixture,
)
from chartqa_dt.data.records import ChartRecord, make_record_id
from chartqa_dt.env import get_env
from chartqa_dt.plans.mining import mine_plan

REFCHARTQA_CAP = 4_000       # `PLAN.md` 3.4: start at the single-box cap


def archive_path() -> Path:
    entry = load_manifest()["archives"]["chartqa"]
    return (Path(get_env().data_root) / "hf" / "datasets--ahmed-masry--ChartQA"
            / "snapshots" / entry["revision"] / entry["filename"])


def synthetic_records(manifest: Path) -> list[ChartRecord]:
    """Generated examples, read back from the generator's manifest."""
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    out = []
    for e in data["examples"]:
        if e["holdout"]:
            continue          # sealed for Phase 9.5
        out.append(ChartRecord(
            record_id=e["example_id"], source="synthetic", split="train",
            image_path=e["image_path"], image_sha256=e["image_sha256"],
            question=e["question"], answer=e["answer"], question_kind="synthetic",
            table=e["table"], boxes=[ev["bbox"] for ev in e["evidence"]],
            plan=e["plan"],
            meta={"level": e["level"], "chart_type": e["chart_type"],
                  "evidence": e["evidence"], "style_seed": e["style_seed"],
                  "data_seed": e["data_seed"]}))
    return out


def chartqa_records(reader: ArchiveReader, *, limit: int, seed: int) -> list[ChartRecord]:
    """Real ChartQA training rows with gold element boxes and, where unique, a plan."""
    rng = random.Random(seed)
    out: list[ChartRecord] = []
    for kind in ("human", "machine"):
        rows = reader.qa_rows("train", kind)
        for row in rng.sample(rows, min(limit, len(rows))):
            img_name = image_path("train", row["imgname"])
            ann_name = annotation_path("train", row["imgname"])
            if not (reader.exists(img_name) and reader.exists(ann_name)):
                continue
            raw = reader.read(img_name)
            width, height = reader.image_size(img_name)
            elements = annotation_boxes(reader.read_json(ann_name), width, height)
            if not elements:
                continue      # no gold boxes -> nothing to ground on

            plan = None
            table = None
            tbl_name = table_path("train", row["imgname"])
            if reader.exists(tbl_name):
                try:
                    table = parse_table(reader.read_text(tbl_name))
                except ValueError:
                    table = None
            if table is not None:
                mined = mine_plan([table["columns"], *table["rows"]], row.get("label"))
                plan = mined.plan     # None unless the uniqueness rule admitted one

            question = str(row["query"])
            digest = hashlib.sha256(raw).hexdigest()
            out.append(ChartRecord(
                record_id=make_record_id("chartqa", "train", digest, question),
                source="chartqa", split="train", image_path=img_name,
                image_sha256=digest, question=question,
                answer=None if row.get("label") is None else str(row["label"]),
                question_kind=kind, table=table,
                boxes=[e["bbox"] for e in elements], plan=plan,
                meta={"chart_type": reader.read_json(ann_name).get("type"),
                      "imgname": row["imgname"], "image_size": [width, height],
                      "n_elements": len(elements)}))
    return out


def refchartqa_records(*, cap: int, seed: int, cache: Path) -> list[ChartRecord]:
    """Streamed RefChartQA training rows. Cached, because streaming them is the slow part."""
    if cache.exists():
        return [ChartRecord.from_dict(json.loads(line))
                for line in cache.read_text(encoding="utf-8").splitlines() if line]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic-manifest", type=Path,
                    default=Path.home() / ".cache/chartqa_dt/data/synthetic/train/manifest.json")
    ap.add_argument("--chartqa-limit", type=int, default=6000, help="questions per kind")
    ap.add_argument("--refchartqa-cache", type=Path,
                    default=Path.home() / ".cache/chartqa_dt/data/refchartqa_train.jsonl")
    ap.add_argument("--refchartqa-cap", type=int, default=REFCHARTQA_CAP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage1-cap", type=int, default=STAGE1_CAP)
    ap.add_argument("--stage2-cap", type=int, default=STAGE2_CAP)
    args = ap.parse_args()

    synth = synthetic_records(args.synthetic_manifest)
    reader = ArchiveReader(archive_path())
    real = chartqa_records(reader, limit=args.chartqa_limit, seed=args.seed)
    ref = refchartqa_records(cap=args.refchartqa_cap, seed=args.seed,
                             cache=args.refchartqa_cache)

    print(f"\nsources: synthetic={len(synth):,}  chartqa={len(real):,}  "
          f"refchartqa={len(ref):,}")
    if not ref:
        print("  (RefChartQA cache absent — the audit passed, so its rows belong in the\n"
              "   mixture; run scripts/cache_refchartqa.py to add them.)")

    s1, c1 = build_stage1(synth, [*real, *ref], cap=args.stage1_cap)
    write_mixture(s1, c1, "data/mixture_stage1.json")

    plan_bearing = [r for r in [*synth, *real, *ref] if r.plan or r.boxes]
    s2, c2 = build_stage2(plan_bearing, synth, cap=args.stage2_cap, seed=args.seed)
    write_mixture(s2, c2, "data/mixture_stage2.json")

    for comp in (c1, c2):
        print(f"\n{comp.stage}: {comp.total:,} records")
        print(f"  by source        : {dict(comp.by_source)}")
        print(f"  by question kind : {dict(comp.by_question_kind)}")
        print(f"  by level         : {dict(comp.by_level)}")
        print(f"  with boxes       : {comp.with_boxes:,}")
        print(f"  with plan        : {comp.with_plan:,} "
              f"(compositional: {comp.with_compositional_plan:,})")
        print(f"  dedup            : {comp.dedup_summary}")


if __name__ == "__main__":
    main()
