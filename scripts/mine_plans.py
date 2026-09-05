#!/usr/bin/env python3
"""Mine a plan for every finished ChartRecord, with a language model, verifying each one.

    python scripts/mine_plans.py --limit 500 --write-batches   # prompts for a reader
    python scripts/mine_plans.py --score audit/plans/answers.json
    python scripts/mine_plans.py --limit 20000 --api           # needs ANTHROPIC_API_KEY

Two stages, in this order (`DECISIONS.md` 0088):

  1. `scripts/build_mixtures.py` assembles **complete** records — image, boxes, labels,
     values, series, colour, table — and mines nothing.
  2. This reads those finished records, asks a model what reasoning answers each question,
     and writes the plans that survive verification to
     `~/.cache/chartqa_dt/data/chartqa_plans.jsonl`, which the reader then joins back on.

**Every proposal passes the same five gates** (`plans.llm_mining`): shape, operands that
exist in the evidence, executes, reproduces the gold answer at the answer's own precision,
and stays inside the marked regions where grounding exists. A proposal failing any gate is
**discarded, never repaired** — repairing it would make this pipeline the author of its own
supervision.

Batches are plain text so they can be answered by whatever is available: a Claude Code
session, Codex, or a person. `--api` exists for when a console key does, and uses the Message
Batches API at half price.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord  # noqa: E402
from chartqa_dt.plans import llm_mining, teacher  # noqa: E402

OUT = Path("audit/plans")
#: Items shown per record. A reader cannot use eighty labels and the batch stops being dense.
#: The verifier still checks against the FULL evidence, so trimming can only cause a refusal,
#: never a wrong acceptance.
SHOW_ITEMS = 24


def report_conflicts(proposals: list[dict], verdicts: list) -> None:
    """Where the model and the record's existing plan disagree, and how.

    **Nothing here accepts or rejects.** The verifier has already run; this only reports.
    A record that already carries a plan is one where an independent method committed to an
    answer, so a disagreement says something about the *teacher* — and agreement on a
    trivial `lookup` says the teacher is at least not inventing operations (0143).

    Three outcomes worth separating:

    * **agree** — same operation, same operands. The cheapest possible confirmation.
    * **same operation, different operands** — the dangerous one. Both may reproduce the
      gold answer while pointing at different marks, which is exactly what
      `distinguish.coincidences` exists to detect and what numeric agreement cannot.
    * **different operation** — the model saw reasoning the derived plan could not. Usually
      the model is right, since a derived plan is only ever `lookup`; that is the whole
      reason for running this comparison.
    """
    import collections

    outcome = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)
    for proposal, verdict in zip(proposals, verdicts):
        prior = proposal.get("prior_plan")
        if not prior or not verdict.accepted:
            continue
        new = proposal["plan"]
        if not isinstance(new, dict):
            continue
        if prior.get("op") != new.get("op"):
            key = "different operation"
        elif (prior.get("args") or []) != (new.get("args") or []):
            key = "same operation, DIFFERENT operands"
        else:
            key = "agree"
        outcome[key] += 1
        if len(examples[key]) < 3:
            examples[key].append((proposal["record_id"], prior, new))

    total = sum(outcome.values())
    if not total:
        return
    print(f"\n  comparison against the {total:,} records that already had a plan:")
    for key in ("agree", "different operation", "same operation, DIFFERENT operands"):
        n = outcome.get(key, 0)
        print(f"    {key:38s} {n:6,} ({n / total:5.1%})")
        for rid, prior, new in examples[key][:2]:
            print(f"        [{rid[:20]}] was {json.dumps(prior)[:44]} "
                  f"-> {json.dumps(new)[:44]}")
    if outcome.get("same operation, DIFFERENT operands"):
        print("    ^ these reproduce the same answer from different marks. Read them: "
              "numeric agreement is not semantic agreement.")


def finished_records(*, limit: int, seed: int, kind: str,
                     source: str = "chartqa") -> list[ChartRecord]:
    """Complete records, straight from the readers the mixture builder uses.

    **RefChartQA is included on purpose**, and that is new. Its records already carry a
    trivial `lookup` plan wherever a single marked box and a numeric answer make one
    unambiguous — not mined, just the only plan such a record can have. Ahmed asked for the
    model to be run over them anyway *"and compare them and if there r conflicts we ll see
    them"*.

    That is a free validation set and the only one available: 22,780 records where an
    independent method already committed to an answer, so a disagreement is a signal about
    the *teacher* rather than about the chart (`DECISIONS.md` 0143).
    """
    from pathlib import Path as _Path

    from chartqa_dt.data.chartqa import ArchiveReader
    from chartqa_dt.env import get_env
    from scripts.build_mixtures import archive_path, chartqa_records, refchartqa_records

    records: list[ChartRecord] = []
    if source in ("chartqa", "all"):
        with ArchiveReader(archive_path()) as reader:
            records += chartqa_records(reader, limit=limit, seed=seed)
    if source in ("refchartqa", "all"):
        cache = _Path(get_env().data_root) / "refchartqa_train.jsonl"
        records += refchartqa_records(cap=limit, cache=cache)
    if kind != "all":
        records = [r for r in records if r.question_kind == kind]
    records = [r for r in records if r.answer is not None and r.meta.get(ELEMENTS_KEY)]
    return deduplicate_for_mining(records)


def deduplicate_for_mining(records: list[ChartRecord]) -> list[ChartRecord]:
    """One prompt per (image, question), never two.

    **Ahmed:** *"mining ll be done after merging or aligning chartqa with refchartqa not
    before because we know they are duplicates."* He is right, and the overlap is much
    larger than the sources suggest — **17,920 records share an (image, question) key**,
    which is **78.8% of all ChartQA**. Mining the two pools separately would send 77,737
    prompts where the union needs 59,817, paying for 17,920 twice.

    Cost is the smaller half. The real problem is that the same question mined twice can
    come back with **two different plans**, and nothing downstream would know which record
    it was looking at — a reconciliation problem created for no reason
    (`DECISIONS.md` 0145).

    Which copy survives is decided by what a *teacher* needs, not by source: the prompt
    shows the question, the chart's data and the gold answer, so the record with the richer
    table and more elements is the better prompt. `ChartRecord.key` is the dedup key
    already used at mixture time (`data/dedup.py`), so mining and training agree about what
    a duplicate is.
    """
    def richness(record: ChartRecord) -> tuple[int, int]:
        table = record.table or {}
        cells = sum(len(row) for row in (table.get("rows") or [])
                    if isinstance(row, list))
        return (cells, len(record.meta.get(ELEMENTS_KEY) or []))

    best: dict[str, ChartRecord] = {}
    for record in records:
        current = best.get(record.key)
        if current is None or richness(record) > richness(current):
            best[record.key] = record
    return list(best.values())


def evidence_of(record: ChartRecord) -> list[dict]:
    """What the executor will resolve labels against, named as the target builder names it."""
    from chartqa_dt.data.records import qualified_labels

    elements = record.meta.get(ELEMENTS_KEY) or []
    return [{"label": n, "value": e.get("value"), "unit": e.get("unit"),
             "colour": e.get("colour")}
            for n, e in zip(qualified_labels(elements), elements)]


def render(batch: list[tuple[ChartRecord, list[dict]]]) -> str:
    blocks = []
    for i, (record, evidence) in enumerate(batch):
        shown = ", ".join(
            f"{e['label']!r}={e['value']!r}"
            + (f" [{teacher.describe_colour(e['colour'])}]" if e.get("colour") else "")
            for e in evidence[:SHOW_ITEMS])
        more = (f"   … and {len(evidence) - SHOW_ITEMS} more items"
                if len(evidence) > SHOW_ITEMS else "")
        blocks.append(f"{i}. Q: {record.question}\n   ANSWER: {record.answer!r}\n"
                      f"   items: {shown}{more}")
    return (f"{teacher.build_system()}\n\n"
            "--- Answer every numbered item below. Reply with ONE ```json block mapping each\n"
            "--- item number to a plan, to {\"refused\": \"...\"}, or to "
            "{\"needs_operator\": {...}}.\n\n"
            + "\n\n".join(blocks)
            + '\n\n```json\n{"0": {"op": "...", "args": [...]}, "1": {"refused": "..."}}\n```')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", choices=("all", "human", "machine"), default="all")
    ap.add_argument("--source", choices=("chartqa", "refchartqa", "all"),
                    default="chartqa",
                    help="which pool to mine. `refchartqa` records already carry a trivial "
                         "derived plan, so mining them yields a comparison (0143)")
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--write-batches", action="store_true")
    ap.add_argument("--score", help="JSON of {batch: {item: plan|refusal}} to verify")
    ap.add_argument("--api", action="store_true", help="call the model directly")
    ap.add_argument("--model", default="claude-opus-5", choices=sorted(teacher.PRICING))
    args = ap.parse_args()

    records = finished_records(limit=args.limit, seed=args.seed, kind=args.kind,
                               source=args.source)
    pairs = [(r, evidence_of(r)) for r in records]
    batches = [pairs[i:i + args.batch_size] for i in range(0, len(pairs), args.batch_size)]
    print(f"{len(records):,} finished records ({args.kind}), "
          f"{len(batches)} batches of up to {args.batch_size}")
    OUT.mkdir(parents=True, exist_ok=True)

    if args.write_batches:
        for i, batch in enumerate(batches):
            (OUT / f"batch_{i:03d}.txt").write_text(render(batch), encoding="utf-8")
        (OUT / "index.json").write_text(json.dumps(
            [[r.record_id for r, _ in b] for b in batches]), encoding="utf-8")
        print(f"  -> {OUT}/batch_NNN.txt")
        print("  answer them, save as {\"0\": {\"0\": <plan>, ...}, ...}, rerun with --score")
        return 0

    if args.api:
        try:
            replies = teacher.run_batch(
                [(r.record_id, render([(r, e)])) for r, e in pairs],
                model=args.model, on_progress=lambda m: print(f"  {m}"))
        except teacher.TeacherError as exc:
            print(f"\n{exc}")
            return 2
        answers = {"0": {}}
        for i, (record, _) in enumerate(pairs):
            reply = teacher.parse_proposal(replies.get(record.record_id, ""))
            if reply.plan:
                answers["0"][str(i)] = reply.plan
        batches = [pairs]
    elif args.score:
        answers = json.loads(Path(args.score).read_text(encoding="utf-8"))
    else:
        print("\nnothing to do — pass --write-batches, --score <file>, or --api")
        return 0

    proposals, refused, asked = [], 0, {}
    for bi, batch in enumerate(batches):
        given = answers.get(str(bi)) or {}
        for qi, (record, evidence) in enumerate(batch):
            reply = given.get(str(qi))
            if reply is None:
                continue
            if isinstance(reply, dict) and "needs_operator" in reply:
                asked.setdefault(reply["needs_operator"].get("name", "?"),
                                 reply["needs_operator"])
                continue
            if isinstance(reply, dict) and "refused" in reply:
                refused += 1
                continue
            proposals.append({"record_id": record.record_id, "plan": reply,
                              "answer": record.answer, "evidence": evidence,
                              "marked_labels": None,
                              # What this record already believed, if anything. Compared
                              # after verification, never used to accept or reject (0143).
                              "prior_plan": record.plan})

    verdicts, stats = llm_mining.verify_many(proposals)
    kept = [{"record_id": p["record_id"], "plan": p["plan"],
             teacher.PROVENANCE_KEY: teacher.provenance(
                 model=args.model, prompt_sha256="batch",
                 verifier_gates=["shape", "grounded", "executes", "reproduces_answer"])}
            for p, v in zip(proposals, verdicts) if v.accepted]

    print(f"\n  reader refused : {refused:,}")
    print(f"\n{stats.describe()}")
    for p, v in zip(proposals, verdicts):
        if not v.accepted:
            print(f"    [{p['record_id'][:22]}] {v.status}: {v.detail[:64]}")

    from scripts.build_mixtures import chartqa_plans_path
    cache = chartqa_plans_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line:
                entry = json.loads(line)
                existing[entry["record_id"]] = entry
    for entry in kept:
        existing[entry["record_id"]] = entry      # a rerun replaces, never duplicates
    cache.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n"
                             for v in existing.values()), encoding="utf-8")
    report_conflicts(proposals, verdicts)
    print(f"\n  kept {len(kept):,} verified plans; cache now holds {len(existing):,}")
    print(f"  -> {cache}")
    if asked:
        print("\n  operators the reader asked for:")
        for name, d in asked.items():
            print(f"    {name}: {d.get('signature', '')} — {d.get('why', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
