#!/usr/bin/env python3
"""Mine plans with a teacher model, verify every one, and keep only what survives.

    # see exactly what would be sent, without a network call
    python scripts/mine_with_llm.py --source chartqa --limit 5 --dry-run

    # score proposals produced elsewhere (no API key needed)
    python scripts/mine_with_llm.py --source chartqa --proposals audit/proposals.jsonl

    # the real thing (needs ANTHROPIC_API_KEY)
    python scripts/mine_with_llm.py --source chartqa --limit 2000 --model claude-opus-5

Three separable parts, so the expensive one can be skipped or replaced:

  build a prompt   `chartqa_dt.plans.teacher.build_prompt` -- deterministic and hashed
  get a proposal   a model call, a cache hit, or a line from --proposals
  decide           `chartqa_dt.plans.llm_mining.verify` -- five gates, no repairs

Only the middle part costs anything, and only it is cached. Rejections are counted by gate
and reported, because a teacher's failure profile is the thing that says whether to trust it.
Nothing is written for a record whose proposal fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.data.records import ELEMENTS_KEY, qualified_labels  # noqa: E402
from chartqa_dt.plans import llm_mining, teacher  # noqa: E402

CACHE = Path.home() / ".cache/chartqa_dt/llm_mining"


def stable_id(source: str, image: str, question: str) -> str:
    """A record id that is the same on every run and every machine.

    Python's built-in `hash()` of a string is salted per process unless PYTHONHASHSEED is
    pinned, so using it here would give each run fresh ids, every cache lookup would miss,
    and a resumed run would pay for every record again.
    """
    digest = hashlib.sha256(f"{image}\x00{question}".encode()).hexdigest()
    return f"{source}:{digest[:24]}"


def cache_path(record_id: str, model: str, prompt_sha: str) -> Path:
    """One file per (record, model, prompt). Changing the prompt cannot reuse old replies."""
    return CACHE / model.replace("/", "_") / f"{record_id}.{prompt_sha[:16]}.json"


def load_requests(source: str, *, limit: int, seed: int) -> list[teacher.TeacherRequest]:
    """Build one request per record, from whichever source was asked for."""
    from chartqa_dt.data.chartqa import ArchiveReader, parse_table, table_path
    from scripts.build_mixtures import archive_path

    out: list[teacher.TeacherRequest] = []
    rng = random.Random(seed)

    if source == "chartqa":
        from chartqa_dt.data.chartqa import annotation_boxes, annotation_path, image_path
        with ArchiveReader(archive_path()) as reader:
            pool = [r for k in ("human", "machine") for r in reader.qa_rows("train", k)]
            rng.shuffle(pool)
            for row in pool:
                if len(out) >= limit:
                    break
                tbl = table_path("train", row["imgname"])
                ann = annotation_path("train", row["imgname"])
                img = image_path("train", row["imgname"])
                if not (reader.exists(tbl) and reader.exists(ann) and reader.exists(img)):
                    continue
                try:
                    table = parse_table(reader.read_text(tbl))
                except ValueError:
                    continue
                width, height = reader.image_size(img)
                elements = annotation_boxes(reader.read_json(ann), width, height)
                if not elements:
                    continue
                # The teacher must be shown the labels the EXECUTOR will resolve against,
                # which are the chart's annotated elements -- not the table's row headers.
                # The two differ: element labels are truncated relative to table labels on
                # 3.1% of charts (`DECISIONS.md` 0078). A plan naming a table label that no
                # element carries passes every gate here and then builds no target at all,
                # because `build_target` joins against `meta[ELEMENTS_KEY]`. The table is
                # still shown, as context the teacher can read the question against.
                # The same names `train.targets` will use. A grouped chart draws "2019"
                # once per series; showing the teacher three identical labels asks it to
                # choose between marks it cannot tell apart (`AUDIT.md` H3).
                evidence = [{"label": n, "value": e.get("value"), "unit": e.get("unit")}
                            for n, e in zip(qualified_labels(elements), elements)]
                out.append(teacher.TeacherRequest(
                    record_id=stable_id("chartqa", row["imgname"], str(row["query"])),
                    prompt=teacher.build_prompt(
                        question=str(row["query"]), answer=row.get("label"),
                        table=table, evidence=evidence),
                    evidence=evidence, answer=row.get("label")))
        return out

    cache = Path.home() / ".cache/chartqa_dt/data/refchartqa_aligned.jsonl"
    if not cache.exists():
        raise SystemExit(f"no aligned RefChartQA cache at {cache}; "
                         "run scripts/align_refchartqa.py first")
    rows = [json.loads(x) for x in cache.read_text(encoding="utf-8").splitlines() if x]
    rng.shuffle(rows)
    for r in rows[:limit]:
        elements = r.get(ELEMENTS_KEY) or []
        marked = {str(e["label"]) for e in elements}
        table = r.get("table")
        evidence = ([{"label": str(x[0]), "value": x[1]}
                     for x in (table or {}).get("rows", []) if len(x) >= 2]
                    or [{"label": str(e["label"]), "value": e.get("value")}
                        for e in elements])
        out.append(teacher.TeacherRequest(
            record_id=r["record_id"],
            prompt=teacher.build_prompt(question=r.get("question", ""),
                                        answer=r.get("answer"), table=table,
                                        evidence=evidence, marked_labels=marked),
            evidence=evidence, answer=r.get("answer"), marked_labels=marked))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("chartqa", "refchartqa"), default="chartqa")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--proposals", help="JSONL of {record_id, plan|refused} produced "
                                        "elsewhere; skips every model call")
    ap.add_argument("--dry-run", action="store_true", help="print prompts and stop")
    ap.add_argument("--out", default="audit/llm_mined_plans.jsonl")
    args = ap.parse_args()

    requests = load_requests(args.source, limit=args.limit, seed=args.seed)
    print(f"{len(requests):,} records from {args.source} (seed {args.seed})")

    if args.dry_run:
        for r in requests[:3]:
            print(f"\n{'=' * 78}\n{r.record_id}  prompt sha {r.sha256[:16]}\n"
                  f"{'=' * 78}\n{r.prompt}")
        print(f"\n[dry run] {len(requests):,} prompts built, nothing sent.")
        return 0

    offline: dict[str, dict] = {}
    if args.proposals:
        for line in Path(args.proposals).read_text(encoding="utf-8").splitlines():
            if line:
                p = json.loads(line)
                offline[p["record_id"]] = p

    accepted, refusals, errors, cached = [], 0, 0, 0
    proposals: list[dict] = []
    for i, req in enumerate(requests, 1):
        if args.proposals:
            got = offline.get(req.record_id)
            if got is None:
                continue
            plan, refused, note = got.get("plan"), bool(got.get("refused")), ""
        else:
            path = cache_path(req.record_id, args.model, req.sha256)
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))["raw"]
                cached += 1
            else:
                try:
                    raw = teacher.call_anthropic(req.prompt, model=args.model)
                except teacher.TeacherError as exc:
                    print(f"\n{exc}")
                    return 2
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"raw": raw, "model": args.model,
                                            "prompt_sha256": req.sha256,
                                            "at": time.time()}), encoding="utf-8")
                time.sleep(0.05)
            plan, refused, note = teacher.parse_proposal(raw)
        if refused:
            refusals += 1
            continue
        if plan is None:
            errors += 1
            continue
        proposals.append({"record_id": req.record_id, "plan": plan, "answer": req.answer,
                          "evidence": req.evidence, "marked_labels": req.marked_labels,
                          "note": note})
        if i % 200 == 0:
            print(f"  … {i:,}/{len(requests):,}")

    verdicts, stats = llm_mining.verify_many(proposals)
    for p, v in zip(proposals, verdicts):
        if v.accepted:
            accepted.append({
                "record_id": p["record_id"], "plan": p["plan"],
                teacher.PROVENANCE_KEY: teacher.provenance(
                    model=args.model,
                    prompt_sha256=next(r.sha256 for r in requests
                                       if r.record_id == p["record_id"]),
                    verifier_gates=["shape", "grounded", "executes", "reproduces_answer",
                                    *(["uses_marked_regions"] if p["marked_labels"] else [])]),
            })

    n = len(requests)
    print(f"\n  teacher refused          : {refusals:,}  ({100 * refusals / max(n, 1):.1f}%)")
    print(f"  unusable replies         : {errors:,}")
    if cached:
        print(f"  served from cache        : {cached:,}")
    print(f"\n{stats.describe()}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in accepted),
                   encoding="utf-8")
    print(f"\n  kept {len(accepted):,} verified plans -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
