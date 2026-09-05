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
from dataclasses import replace
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
    ABSENT_FROM_EVALUATION,
    CHARTQA_DRAW,
    REFCHARTQA_CAP,
    STAGE1_CAP,
    STAGE2_CAP,
    SYNTHETIC_REPLAY,
    build_stage1,
    build_stage2,
    drop_absent_chart_types,
    write_mixture,
)
from chartqa_dt.data.records import (
    ELEMENTS_KEY,
    ChartRecord,
    image_content_sha256,
    make_record_id,
)
from chartqa_dt.env import get_env
from chartqa_dt.plans.teacher import PROVENANCE_KEY
from chartqa_dt.splits import sealed_image_hashes
from chartqa_dt.train.targets import (
    NoPlanAvailable,
    TargetError,
    build_grounding_only_target,
    build_target,
)


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
            # `elements`, not `evidence`. `build_target` reads `meta["elements"]` —
            # ChartQA's reader sets that key, and this one did not, so every synthetic
            # record fell through to the placeholder branch, got evidence labelled
            # `item1, item2, ...`, and was then refused because its plan referenced the
            # real labels. All 12,000 stage-1 records, silently (`DECISIONS.md` 0071).
            # The generator now emits every mark on the chart and says which of them the
            # question needs, so the record carries both (`DECISIONS.md` 0124). Older
            # manifests have no `elements`, and fall back to the operands.
            elements=[{**el, "grounding_provenance": "synthetic_exact",
                       "value_provenance": "synthetic_generated"}
                      for el in (e.get("elements") or e["evidence"])],
            evidence=(e.get("evidence_index")
                      if e.get("elements") else list(range(len(e["evidence"])))),
            meta={"level": e["level"], "chart_type": e["chart_type"],
                  "style_seed": e["style_seed"], "data_seed": e["data_seed"]}))
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

            table = None
            tbl_name = table_path("train", row["imgname"])
            if reader.exists(tbl_name):
                try:
                    table = parse_table(reader.read_text(tbl_name))
                except ValueError:
                    table = None

            question = str(row["query"])
            # No plan is mined here. Records are built COMPLETE — boxes, labels, values,
            # series, colour — and plans are attached afterwards from `chartqa_plans.jsonl`,
            # which a language model produces by reading the finished records
            # (`DECISIONS.md` 0088). The deterministic miner used to run at this point; it
            # searched backwards from the gold answer and had to refuse whenever more than
            # one operation reproduced it, which is 53.9% of rows and is what working
            # backwards means rather than a defect to patch (0085).
            plan = None
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
                      # Dropping them left the elements empty while `boxes` was full, so
                      # every training target fell back to "item1" placeholders and the
                      # mined plan's labels matched nothing: 1 of 636 records produced an
                      # executable target (`DECISIONS.md` 0067). The same defect survived
                      # in `synthetic_records` under a different spelling until 0071,
                      # which is why the key is now a shared constant.
                      ELEMENTS_KEY: elements},
                elements=[{**e, "grounding_provenance": "chartqa_annotation",
                           "value_provenance": "chartqa_annotation"} for e in elements],
                # **`None`, and that is the whole point.** ChartQA annotates the CHART:
                # these elements are every mark on the image, identical for every question
                # asked about it, so nothing here knows which subset answers this one.
                # Only a plan can select. `None` means unknown, and a consumer that needs
                # question-level grounding must refuse rather than assume the whole chart
                # (`DECISIONS.md` 0124).
                evidence=None))
    if dropped:
        print(f"  chartqa: {dropped} rows dropped — a train image identical to a "
              f"held-out chart")
    return attach_mined_plans(out, cache=chartqa_plans_path())


def chartqa_plans_path() -> Path:
    """Where the reader's verified plans are cached. Never in git — rule 7."""
    return Path.home() / ".cache/chartqa_dt/data/chartqa_plans.jsonl"


def attach_mined_plans(records: list[ChartRecord], *, cache: Path) -> list[ChartRecord]:
    """Join verified plans onto finished records, by record id.

    **The attachment must happen in the reader, not downstream.** A mixture stores record
    ids and training rehydrates from these readers, so anything added after this point is
    discarded before training ever sees it — which is how the dedup merge was silently lost
    (`AUDIT.md` H2).

    A record with no plan keeps `plan=None` and is refused later by `build_target` with a
    reason, rather than being given an invented one.
    """
    if not cache.exists():
        return records
    by_id: dict[str, dict] = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line:
            entry = json.loads(line)
            if entry.get("plan"):
                by_id[entry["record_id"]] = entry
    if not by_id:
        return records
    out, attached = [], 0
    for r in records:
        entry = by_id.get(r.record_id)
        if entry is None:
            out.append(r)
            continue
        meta = {**r.meta, PROVENANCE_KEY: entry.get(PROVENANCE_KEY)
                or entry.get("provenance") or {"method": "unrecorded"}}
        out.append(replace(r, plan=entry["plan"], meta=meta))
        attached += 1
    print(f"  chartqa: {attached:,} of {len(records):,} records carry a mined plan")
    return out


def _with_evidence(record: ChartRecord) -> ChartRecord:
    """Give a legacy cached RefChartQA record the `elements`/`evidence` it predates.

    The cache is a JSONL written before those were fields, so its rows carry `boxes` and
    nothing else. Read back as-is they would have `evidence is None`, which now means
    *"unknown"* — and a RefChartQA record whose evidence is unknown is indistinguishable
    from ChartQA's whole-chart annotation, so it would lose its grounding-only target
    (`DECISIONS.md` 0124).

    Every box RefChartQA gives is evidence for that question, which is what the dataset
    annotates, so this reconstructs exactly what `row_to_record` now writes. Rebuilding
    the 55,486-row cache would do the same thing and take twenty-five minutes.
    """
    if record.evidence is not None or not record.boxes:
        return record
    elements = record.elements or [
        {"label": None, "value": None, "unit": None, "bbox": b,
         # Same provenance `row_to_record` now writes: the box is RefChartQA's own
         # per-question annotation, and the value does not exist until alignment (0126).
         "grounding_provenance": "refchartqa_gold", "value_provenance": "unknown"}
        for b in record.boxes]
    return replace(record, elements=elements, evidence=list(range(len(elements))))


def attach_pot_plans(records: list[ChartRecord]) -> list[ChartRecord]:
    """Give a record the plan its own gold derivation implies — if it verifies.

    RefChartQA ships a program of thought on 63.6% of rows and nothing read it
    (`DECISIONS.md` 0133). `plans/pot.py` converts the ones whose shape is unambiguous;
    this attaches the result **only when it executes against the record's own evidence and
    reproduces the gold answer**, which is the same gate every other plan passes.

    Records that already carry a plan are left alone: a mined plan has been through the
    full five gates, and this one has been through one of them.
    """
    from chartqa_dt.eval.metrics import relaxed_correctness
    from chartqa_dt.plans.executor import (
        FOLD_OPS,
        EvidenceItem,
        ExecutorError,
        execute,
        parse_numeric,
    )
    from chartqa_dt.plans.pot import to_plan

    def rendered(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, float):
            return (str(int(value)) if value == int(value)
                    else f"{value:.6f}".rstrip("0").rstrip("."))
        return str(value)

    out: list[ChartRecord] = []
    attached = refused = 0
    for record in records:
        response = (record.meta or {}).get("response")
        if record.plan or not response or record.answer is None:
            out.append(record)
            continue
        plan = to_plan(response)
        if plan is None:
            out.append(record)
            continue
        evidence = [EvidenceItem(label=str(e.get("label")),
                                 value=parse_numeric(e.get("value")),
                                 unit=e.get("unit"))
                    for e in (record.elements or []) if e.get("label") is not None]
        try:
            got = rendered(execute(plan, evidence)) if evidence else None
        except ExecutorError:
            got = None
        # **A fold over one item is not a verified fold.** `max([x])` is `x`, so
        # "it reproduces the gold answer" proves nothing — and the target would teach the
        # model to emit one box and trivially fold it, which is the circular supervision
        # `align_refchartqa.py` already warns about: it "makes the round-trip pass by
        # construction and teaches nothing about reading the chart".
        #
        # Measured: 97.6% of the fold plans this converter produced folded over a single
        # marked element, because RefChartQA marks only the region a person used — for
        # "what is the maximum", that is the maximum bar alone (`DECISIONS.md` 0133).
        folds = plan["op"] in FOLD_OPS and not plan.get("args")
        if folds and len(evidence) < 2:
            refused += 1
            out.append(record)
            continue
        if got is not None and relaxed_correctness(got, str(record.answer)):
            out.append(replace(record, plan=plan,
                               meta={**record.meta, "plan_provenance": "refchartqa_pot"}))
            attached += 1
        else:
            refused += 1
            out.append(record)
    if attached or refused:
        print(f"  refchartqa: {attached:,} plans recovered from gold derivations, "
              f"{refused:,} refused by the executor")
    return out


def refchartqa_records(*, cap: int, cache: Path) -> list[ChartRecord]:
    """Streamed RefChartQA training rows, enriched with semantic identity where available.

    RefChartQA marks *which* regions answer a question but not what they are, so an
    unenriched record produces evidence named `item1, item2, …` with no value, and
    `build_record` can only derive a plan for the single-box case — by setting the evidence
    value **to the answer**, which makes the round-trip pass by construction.

    `scripts/align_refchartqa.py` matches each grounding box to a ChartQA element on the
    same image (measured: 98.9% at IoU >= 0.9, median 1.000 — they are the same boxes) and
    caches the result. When that cache exists, the matched elements and the chart's gold
    table are attached, so the record carries real labels, real values and a table to mine
    a plan against (`AUDIT.md` H1, `DECISIONS.md` 0077).

    **The enrichment must be attached here, not merged later.** A mixture stores record ids
    and training rehydrates from these readers, so anything added downstream of this point
    is discarded before training ever sees it (`AUDIT.md` H2).
    """
    if not cache.exists():
        return []
    records = [_with_evidence(ChartRecord.from_dict(json.loads(line)))
               for line in cache.read_text(encoding="utf-8").splitlines() if line]

    aligned_path = cache.with_name("refchartqa_aligned.jsonl")
    if aligned_path.exists():
        by_id = {}
        for line in aligned_path.read_text(encoding="utf-8").splitlines():
            if line:
                a = json.loads(line)
                by_id[a["record_id"]] = a
        enriched = 0
        out: list[ChartRecord] = []
        for r in records:
            a = by_id.get(r.record_id)
            if a is None:
                out.append(r)
                continue
            aligned = [{**e,
                        # Matched to a ChartQA element, so the box is RefChartQA's and the
                        # value is ChartQA's table. `match_iou` / `match_margin` already on
                        # the element say how good the match was (0077, 0126).
                        "grounding_provenance": "refchartqa_aligned",
                        "value_provenance": "chartqa_table"}
                       for e in a[ELEMENTS_KEY]]
            meta = {**r.meta, ELEMENTS_KEY: aligned, "aligned_to_chartqa": True}
            # RefChartQA marks, per question, the regions a person used, and alignment
            # gives those regions their identity. Every one of them is evidence — that is
            # what the dataset annotates — so `evidence` lists them all (0124).
            out.append(replace(r, meta=meta, table=r.table or a.get("table"),
                               plan=r.plan or a.get("plan"),
                               elements=aligned, evidence=list(range(len(aligned)))))
            enriched += 1
        records = out
        print(f"  refchartqa: {enriched:,} of {len(records):,} records enriched with "
              f"ChartQA element identity")
    return attach_pot_plans(records[:cap])


def usable_only(records: list[ChartRecord], label: str) -> list[ChartRecord]:
    """Drop records that cannot become a training target.

    A mixture slot filled by a record `build_target` refuses is a slot that trains
    nothing: the feed catches the refusal, counts it and moves on. Measured before this
    filter existed, stage 2 held **3,265 usable records of 12,000** — so the effective
    training set was a quarter of the pre-registered one, and nothing said so
    (`DECISIONS.md` 0072).

    This does not change *what* the model learns, only how many of the 12,000 slots teach
    it something. The supply of real records that yield targets is itself the binding
    constraint: 2,420 of 22,947 ChartQA rows and 2,063 of 3,996 cached RefChartQA rows.
    """
    keep, _, refused = _partition(records)
    if refused:
        print(f"  {label:<12}dropped {refused:,} of {len(records):,} — no training target "
              f"({100 * len(keep) / max(len(records), 1):.1f}% usable)")
    return keep


def _partition(records: list[ChartRecord]
               ) -> tuple[list[ChartRecord], list[ChartRecord], int]:
    """Split into plan-usable, grounding-only-usable, and the count of the rest."""
    plans: list[ChartRecord] = []
    grounding: list[ChartRecord] = []
    refused = 0
    for record in records:
        try:
            build_target(record)
        except NoPlanAvailable:
            try:
                build_grounding_only_target(record)
            except TargetError:
                refused += 1
                continue
            grounding.append(record)
            continue
        except TargetError:
            refused += 1
            continue
        plans.append(record)
    return plans, grounding, refused


def split_by_usability(records: list[ChartRecord], label: str
                       ) -> tuple[list[ChartRecord], list[ChartRecord]]:
    """The same drop as `usable_only`, but keeping the grounding-only half separately.

    `usable_only` calls `build_target` and drops whatever it refuses, and that filter runs
    **before** the feed — so a record it drops never reaches the feed's grounding-only
    fallback at all. Wiring the fallback into the feed and not here would have made
    `DECISIONS.md` 0116 dead code in every built mixture, which is what the first version
    of it was: the mixture still reported *"refchartqa dropped 1,735 of 4,000"*.

    The two halves are kept apart rather than merged because they are not
    interchangeable. Stage 1 is grounding only (`PLAN.md` 6.1) and can use both; stage 2
    trains the plan, and a grounding-only record there is supervision with the answer
    taken out.
    """
    plans, grounding, refused = _partition(records)
    if refused or grounding:
        print(f"  {label:<12}{len(plans):,} plan targets, {len(grounding):,} "
              f"grounding-only (stage 1 only), {refused:,} dropped of {len(records):,} "
              f"({100 * (len(plans) + len(grounding)) / max(len(records), 1):.1f}% usable)")
    return plans, grounding


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic-manifest", type=Path,
                    default=Path.home() / ".cache/chartqa_dt/data/synthetic/train/manifest.json")
    ap.add_argument("--chartqa-limit", type=int, default=CHARTQA_DRAW,
                    help="questions per kind")
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
    ap.add_argument("--keep-unusable", action="store_true",
                   help="keep records that do not yield a training target. The default is "
                        "to drop them: the feed skips them anyway, so they only consume "
                        "mixture slots (DECISIONS.md 0072)")
    ap.add_argument("--suffix", type=str, default="",
                    help="written as data/mixture_stageN<suffix>.json, so the "
                         "pre-registered mixture and the plan-rich arm coexist")
    args = ap.parse_args()

    if args.replay is None:
        args.replay = SYNTHETIC_REPLAY
    synth_all = synthetic_records(args.synthetic_manifest)
    reader = ArchiveReader(archive_path())
    real = chartqa_records(reader, limit=args.chartqa_limit, seed=args.seed)
    ref = refchartqa_records(cap=args.refchartqa_cap, cache=args.refchartqa_cache)

    real_grounding: list[ChartRecord] = []
    ref_grounding: list[ChartRecord] = []
    if not args.keep_unusable:
        synth_all = usable_only(synth_all, "synthetic")
        real, real_grounding = split_by_usability(real, "chartqa")
        ref, ref_grounding = split_by_usability(ref, "refchartqa")
    # Drop the chart families ChartQA does not contain BEFORE balancing, so the per-level
    # sample is drawn from a pool that is already the right shape (`DECISIONS.md` 0091).
    synth_all, absent = drop_absent_chart_types(synth_all)
    if absent:
        print(f"  synthetic: {absent:,} records dropped — chart types the evaluation "
              f"corpus does not contain ({', '.join(sorted(ABSENT_FROM_EVALUATION))})")
    synth = balance_by_level(synth_all, args.synthetic_stage1, seed=args.seed)

    print(f"\nsources: synthetic={len(synth):,}  chartqa={len(real):,}  "
          f"refchartqa={len(ref):,}  "
          f"(+{len(real_grounding) + len(ref_grounding):,} grounding-only for stage 1)")
    if not ref:
        print("  (RefChartQA cache absent — the audit passed, so its rows belong in the\n"
              "   mixture; run scripts/cache_refchartqa.py to add them.)")

    # Plan-bearing records first: if `--stage1-cap` ever binds, the richer supervision
    # should be what survives it.
    s1, c1 = build_stage1(synth, [*real, *ref, *real_grounding, *ref_grounding],
                          cap=args.stage1_cap)
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
        print(f"  synthetic share  : {comp.synthetic_share:.1%}"
              + ("   <- the realised stage-2 replay ratio; SYNTHETIC_REPLAY sets a "
                 "count, not this (DECISIONS.md 0117)" if comp.stage == "stage2" else ""))
        print(f"  with boxes       : {comp.with_boxes:,}")
        print(f"  with plan        : {comp.with_plan:,} "
              f"(compositional: {comp.with_compositional_plan:,})")
        print(f"  dedup            : {comp.dedup_summary}")


if __name__ == "__main__":
    main()
