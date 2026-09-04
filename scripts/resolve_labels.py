#!/usr/bin/env python3
"""Resolve which label a question means, for the lookups `plans.forward` could not identify.

    # write batches for a reader to answer (a Claude Code session, Codex, or a person)
    python scripts/resolve_labels.py --limit 3000 --batch-size 60

    # score the answers back into verified plans
    python scripts/resolve_labels.py --answers audit/resolve/answers.json

The design is in `chartqa_dt.plans.resolve`: the candidate label is chosen by arithmetic —
it is the one element whose value equals the gold answer — and the reader only judges whether
the question is asking about it. A binary judgement per record, so batches pack tightly and
this runs on a subscription rather than an API budget.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.data.records import qualified_labels  # noqa: E402
from chartqa_dt.plans import forward, resolve  # noqa: E402

OUT = Path("audit/resolve")


def collect(*, limit: int, seed: int) -> list[resolve.Question]:
    """Records where the arithmetic is settled and only the reading is open."""
    from chartqa_dt.data.chartqa import (
        ArchiveReader,
        annotation_boxes,
        annotation_path,
        image_path,
    )
    from scripts.build_mixtures import archive_path
    from scripts.mine_with_llm import stable_id

    out: list[resolve.Question] = []
    seen = 0
    with ArchiveReader(archive_path()) as reader:
        pool = [r for k in ("human", "machine") for r in reader.qa_rows("train", k)]
        random.Random(seed).shuffle(pool)
        for row in pool:
            if seen >= limit:
                break
            name = row["imgname"]
            ann, img = annotation_path("train", name), image_path("train", name)
            if not (reader.exists(ann) and reader.exists(img)):
                continue
            w, h = reader.image_size(img)
            elements = annotation_boxes(reader.read_json(ann), w, h)
            if not elements:
                continue
            seen += 1
            question, answer = str(row["query"]), str(row.get("label", ""))
            evidence = [{"label": n, "value": e.get("value"), "unit": e.get("unit")}
                        for n, e in zip(qualified_labels(elements), elements)]
            if forward.build(question, answer=answer, evidence=evidence).ok:
                continue          # already settled without asking anyone
            candidate = resolve.candidates(question=question, answer=answer,
                                           evidence=evidence)
            if candidate is None:
                continue          # no unique holder; a yes/no could not settle it
            out.append(resolve.Question(
                record_id=stable_id("chartqa", name, question), question=question,
                answer=answer, candidate=candidate,
                labels=[str(e["label"]) for e in evidence]))
    print(f"scanned {seen:,} records; {len(out):,} need only a reading "
          f"({100 * len(out) / max(seen, 1):.1f}%)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=60)
    ap.add_argument("--answers", help="a JSON file of {batch_index: {item: bool}} to score")
    args = ap.parse_args()

    questions = collect(limit=args.limit, seed=args.seed)
    batches = [questions[i:i + args.batch_size]
               for i in range(0, len(questions), args.batch_size)]
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.answers:
        index = []
        for i, batch in enumerate(batches):
            (OUT / f"batch_{i:03d}.txt").write_text(resolve.build_batch(batch),
                                                    encoding="utf-8")
            index.append([{"record_id": q.record_id, "candidate": q.candidate,
                           "question": q.question, "answer": q.answer} for q in batch])
        (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
        print(f"wrote {len(batches)} batches of up to {args.batch_size} -> {OUT}/")
        print("answer each batch_NNN.txt, save the verdicts as "
              "{\"0\": {\"0\": true, ...}, ...}, then rerun with --answers")
        return 0

    index = json.loads((OUT / "index.json").read_text(encoding="utf-8"))
    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    kept, said_no, unjudged = [], 0, 0
    for bi, batch in enumerate(index):
        verdicts = {int(k): v for k, v in (answers.get(str(bi)) or {}).items()}
        for qi, item in enumerate(batch):
            verdict = verdicts.get(qi)
            if verdict is None:
                unjudged += 1
            elif verdict:
                kept.append({"record_id": item["record_id"],
                             "plan": resolve.plan_for(item["candidate"]),
                             "resolved_label": item["candidate"],
                             "provenance": {"method": "llm_label_resolution",
                                            "arithmetic": "value equals the gold answer",
                                            "fidelity": "reader agreed the question "
                                                        "refers to this label"}})
            else:
                said_no += 1
    total = len(kept) + said_no + unjudged
    print(f"\n  reader said yes : {len(kept):,}  ({100 * len(kept) / max(total, 1):.1f}%)"
          f"   <-- verified plans")
    print(f"  reader said no  : {said_no:,}")
    print(f"  not judged      : {unjudged:,}   (left for a later pass, never counted as no)")
    path = OUT / "resolved_plans.jsonl"
    path.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept),
                    encoding="utf-8")
    print(f"\n  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
