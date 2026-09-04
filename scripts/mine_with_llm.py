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
import collections
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
    ap.add_argument("--model", default="claude-opus-5",
                    choices=sorted(teacher.PRICING))
    ap.add_argument("--effort", default=teacher.DEFAULT_EFFORT,
                    choices=("low", "medium", "high", "xhigh", "max"),
                    help="thinking depth per record; sweep it on a small sample before a "
                         "full run rather than assuming this default is right")
    ap.add_argument("--batch", action="store_true",
                    help="submit through the Message Batches API at HALF price. Mining is "
                         "not latency-sensitive, so this is the right default for a large "
                         "run; results come back asynchronously.")
    ap.add_argument("--estimate", action="store_true",
                    help="price the run and stop, without sending anything")
    ap.add_argument("--proposals", help="JSONL of {record_id, plan|refused} produced "
                                        "elsewhere; skips every model call")
    ap.add_argument("--dry-run", action="store_true", help="print prompts and stop")
    ap.add_argument("--out", default="audit/llm_mined_plans.jsonl")
    args = ap.parse_args()

    requests = load_requests(args.source, limit=args.limit, seed=args.seed)
    print(f"{len(requests):,} records from {args.source} (seed {args.seed})")

    if args.estimate:
        system = teacher.build_system()
        avg = sum(len(r.prompt) for r in requests) / max(len(requests), 1)
        for batch in (False, True):
            c = teacher.estimate_cost(n_records=len(requests), system_chars=len(system),
                                      record_chars=int(avg), model=args.model, batch=batch)
            how = "Message Batches API (half price)" if batch else "one request per record"
            print(f"\n  {how}, {args.model}")
            print(f"    instructions, first call      ${c['input_system_first']:>9.2f}")
            print(f"    instructions, cache reads     ${c['input_system_cached']:>9.2f}")
            print(f"    per-record input              ${c['input_records']:>9.2f}")
            print(f"    output                        ${c['output']:>9.2f}")
            print(f"    {'TOTAL':<30}${c['total']:>9.2f}")
        print("\n  Rough: tokens estimated at ~3.6 chars each and output at 700 tokens per\n"
              "  record, which errs high. Nothing was sent.")
        return 0

    if args.dry_run:
        for r in requests[:3]:
            print(f"\n{'=' * 78}\n{r.record_id}  prompt sha {r.sha256[:16]}\n"
                  f"{'=' * 78}\n{r.prompt}")
        print(f"\n[dry run] {len(requests):,} prompts built, nothing sent.")
        return 0

    # Batch mode fetches every reply up front at half price, then falls through to the same
    # per-record loop, which finds each one already in the cache.
    if args.batch and not args.proposals:
        todo = [(r.record_id, r.prompt) for r in requests
                if not cache_path(r.record_id, args.model, r.sha256).exists()]
        print(f"  {len(requests) - len(todo):,} already cached, {len(todo):,} to fetch")
        if todo:
            try:
                replies = teacher.run_batch(todo, model=args.model, effort=args.effort,
                                            on_progress=lambda m: print(f"  {m}"))
            except teacher.TeacherError as exc:
                print(f"\n{exc}")
                return 2
            by_id = {r.record_id: r for r in requests}
            for rid, raw in replies.items():
                path = cache_path(rid, args.model, by_id[rid].sha256)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"raw": raw, "model": args.model,
                                            "prompt_sha256": by_id[rid].sha256,
                                            "at": time.time(), "via": "batch"}),
                                encoding="utf-8")
            missing = len(todo) - len(replies)
            if missing:
                print(f"  {missing:,} requests did not come back and are left unmined "
                      f"— rerun to retry only those")

    offline: dict[str, dict] = {}
    if args.proposals:
        for line in Path(args.proposals).read_text(encoding="utf-8").splitlines():
            if line:
                p = json.loads(line)
                offline[p["record_id"]] = p

    accepted, refusals, errors, cached = [], 0, 0, 0
    proposals: list[dict] = []
    #: Operations the teacher asked for and we do not have. Ranked and reported, because a
    #: question that is answerable but inexpressible is the only signal that says what to
    #: build next -- and folding it into "refused" would hide it entirely.
    wanted_ops: collections.Counter[str] = collections.Counter()
    wanted_detail: dict[str, dict] = {}
    for i, req in enumerate(requests, 1):
        if args.proposals:
            got = offline.get(req.record_id)
            if got is None:
                continue
            reply = teacher.Reply(plan=got.get("plan"), refused=bool(got.get("refused")),
                                  needs_operator=got.get("needs_operator"),
                                  note=str(got.get("refused") or ""))
        else:
            path = cache_path(req.record_id, args.model, req.sha256)
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))["raw"]
                cached += 1
            else:
                try:
                    raw = teacher.call_anthropic(req.prompt, system=teacher.build_system(),
                                                 model=args.model, effort=args.effort)
                except teacher.TeacherError as exc:
                    print(f"\n{exc}")
                    return 2
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"raw": raw, "model": args.model,
                                            "prompt_sha256": req.sha256,
                                            "at": time.time()}), encoding="utf-8")
                time.sleep(0.05)
            reply = teacher.parse_proposal(raw)
        if reply.needs_operator:
            name = reply.needs_operator["name"]
            wanted_ops[name] += 1
            wanted_detail.setdefault(name, reply.needs_operator)
            continue
        if reply.refused:
            refusals += 1
            continue
        if reply.plan is None:
            errors += 1
            continue
        proposals.append({"record_id": req.record_id, "plan": reply.plan,
                          "answer": req.answer, "evidence": req.evidence,
                          "marked_labels": req.marked_labels, "note": reply.note})
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
    if wanted_ops:
        asked = sum(wanted_ops.values())
        print(f"  asked for a new operator : {asked:,}  ({100 * asked / max(n, 1):.1f}%)")
    print(f"  unusable replies         : {errors:,}")
    if cached:
        print(f"  served from cache        : {cached:,}")
    print(f"\n{stats.describe()}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in accepted),
                   encoding="utf-8")
    print(f"\n  kept {len(accepted):,} verified plans -> {out}")

    if wanted_ops:
        print("\n  operations the teacher asked for and we do not have, most-wanted first:")
        for name, k in wanted_ops.most_common(12):
            d = wanted_detail[name]
            print(f"    {k:>5}x  {name}")
            if d.get("signature"):
                print(f"           {d['signature']}")
            if d.get("why"):
                print(f"           e.g. {d['why']}")
        ops_out = out.with_name(out.stem + "_wanted_operators.json")
        ops_out.write_text(json.dumps(
            [{"name": nm, "requests": k, **wanted_detail[nm]}
             for nm, k in wanted_ops.most_common()], indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"\n  -> {ops_out}   (each is a proposal to weigh, not a change to make)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
