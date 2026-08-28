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
    SYNTHETIC_REPLAY,
    build_stage1,
    build_stage2,
    write_mixture,
)
from chartqa_dt.data.records import ChartRecord, image_content_sha256, make_record_id
from chartqa_dt.env import get_env
from chartqa_dt.plans.mining import mine_plan
from chartqa_dt.splits import sealed_image_hashes

REFCHARTQA_CAP = 4_000       # `PLAN.md` 3.4: start at the single-box cap


def archive_path() -> Path:
    entry = load_manifest()["archives"]["chartqa"]
    return (Path(get_env().data_root) / "hf" / "datasets--ahmed-masry--ChartQA"
            / "snapshots" / entry["revision"] / entry["filename"])


def balance_by_level(records: list[ChartRecord], total: int, *, seed: int
                     ) -> list[ChartRecord]:
    """An equal share of each curriculum level, sampled — not the first `total`.

    `build_stage1` orders L1→L4 and then takes the cap, so handing it every synthetic
    example fills the whole mixture with L1 and L2 and excludes L3, L4 **and all real
    grounding data**. Sampling per level keeps the curriculum and leaves room for real
    charts, which is where the domain actually matters (`DECISIONS.md` 0066).
    """
    if not total or total >= len(records):
        return records
    rng = random.Random(seed)
    by_level: dict[str, list[ChartRecord]] = {}
    for record in records:
        by_level.setdefault(str(record.meta.get("level")), []).append(record)
    per_level = max(1, total // max(1, len(by_level)))
    out: list[ChartRecord] = []
    for level in sorted(by_level):
        pool = by_level[level]
        out.extend(rng.sample(pool, min(per_level, len(pool))))
    return out


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
    sealed = sealed_image_hashes()
    dropped = 0
    out: list[ChartRecord] = []
    for kind in ("human", "machine"):
        rows = reader.qa_rows("train", kind)
        for row in rng.sample(rows, min(limit, len(rows))):
            img_name = image_path("train", row["imgname"])
            ann_name = annotation_path("train", row["imgname"])
            if not (reader.exists(img_name) and reader.exists(ann_name)):
                continue
            raw = reader.read(img_name)
            digest = image_content_sha256(raw)
            if digest in sealed:
                # ChartQA's OWN train split contains 15 images that are pixel-identical
                # to val/test charts (`DECISIONS.md` 0049). Not our bug, still our
                # contamination.
                dropped += 1
                continue
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
            out.append(ChartRecord(
                record_id=make_record_id("chartqa", "train", digest, question),
                source="chartqa", split="train", image_path=img_name,
                image_sha256=digest, question=question,
                answer=None if row.get("label") is None else str(row["label"]),
                question_kind=kind, table=table,
                boxes=[e["bbox"] for e in elements], plan=plan,
                meta={"chart_type": reader.read_json(ann_name).get("type"),
                      "imgname": row["imgname"], "image_size": [width, height],
                      "n_elements": len(elements),
                      # The per-element label, value and unit — not just the count.
                      # Dropping them left `meta["elements"]` empty while `boxes` was
                      # full, so every training target fell back to "item1" placeholders
                      # and the mined plan's labels matched nothing: 1 of 636 records
                      # produced an executable target.
                      "elements": elements}))
    if dropped:
        print(f"  chartqa: {dropped} rows dropped — a train image identical to a "
              f"held-out chart")
    return out


def refchartqa_records(*, cap: int, cache: Path) -> list[ChartRecord]:
    """Streamed RefChartQA training rows. Cached, because streaming them is the slow part."""
    if not cache.exists():
        return []
    records = [ChartRecord.from_dict(json.loads(line))
               for line in cache.read_text(encoding="utf-8").splitlines() if line]
    return records[:cap]


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
    ap.add_argument("--replay", type=int, default=None,
                    help="synthetic examples in stage 2. `PLAN.md` 3.7 says ~2,000; the "
                         "plan-rich arm raises it because only 15.7%% of stage 2 teaches "
                         "a compositional plan (DECISIONS.md 0066)")
    ap.add_argument("--synthetic-stage1", type=int, default=6000,
                    help="synthetic examples offered to stage 1, balanced across L1-L4. "
                         "The rest of the cap is real grounding data.")
    ap.add_argument("--suffix", type=str, default="",
                    help="written as data/mixture_stageN<suffix>.json, so the "
                         "pre-registered mixture and the plan-rich arm coexist")
    args = ap.parse_args()

    if args.replay is None:
        args.replay = SYNTHETIC_REPLAY
    synth_all = synthetic_records(args.synthetic_manifest)
    synth = balance_by_level(synth_all, args.synthetic_stage1, seed=args.seed)
    reader = ArchiveReader(archive_path())
    real = chartqa_records(reader, limit=args.chartqa_limit, seed=args.seed)
    ref = refchartqa_records(cap=args.refchartqa_cap, cache=args.refchartqa_cache)

    print(f"\nsources: synthetic={len(synth):,}  chartqa={len(real):,}  "
          f"refchartqa={len(ref):,}")
    if not ref:
        print("  (RefChartQA cache absent — the audit passed, so its rows belong in the\n"
              "   mixture; run scripts/cache_refchartqa.py to add them.)")

    s1, c1 = build_stage1(synth, [*real, *ref], cap=args.stage1_cap)
    write_mixture(s1, c1, f"data/mixture_stage1{args.suffix}.json")

    # Real records only here; the synthetic replay is the second argument. Passing synth
    # in both would just merge it with itself and the replay size would control nothing.
    plan_bearing = [r for r in [*real, *ref] if r.plan or r.boxes]
    # Stage 2 draws replay from the FULL synthetic pool, not the stage-1 subsample, so
    # raising `--replay` actually adds plan supervision.
    s2, c2 = build_stage2(plan_bearing, synth_all, cap=args.stage2_cap,
                          replay=args.replay, seed=args.seed)
    write_mixture(s2, c2, f"data/mixture_stage2{args.suffix}.json")

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
