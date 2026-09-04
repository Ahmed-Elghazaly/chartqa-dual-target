#!/usr/bin/env python3
"""Mine the human-written ChartQA questions, which are half the score and where patterns fail.

    python scripts/mine_human_questions.py --limit 400 --batch-size 40   # write batches
    python scripts/mine_human_questions.py --score audit/human/answers.json

`DECISIONS.md` 0086: `plans.forward` builds a plan for 53.5% of machine-generated questions
and only **14.8%** of human-written ones, because it recognises templates rather than reading
language. ChartQA's test split is 50/50 and its headline metric averages the halves, so
supervision mined by pattern alone is ~92% machine-generated while half the score is decided
by human questions.

This writes those questions out in dense batches for a reader — a Claude Code session, Codex,
or a person — and scores the replies through the same five gates every other proposal faces
(`plans.llm_mining`): shape, grounded operands, executes, reproduces the gold answer, and
uses the marked regions where grounding exists. **A proposal that fails any gate is discarded,
never repaired.**

Records `plans.forward` already settles are skipped: they are free, verified, and asking a
reader to redo them wastes the only budget that is actually scarce here.
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
from chartqa_dt.plans import forward, llm_mining, teacher  # noqa: E402

OUT = Path("audit/human")
#: Charts with more items than this have their item list trimmed in the prompt. A reader
#: cannot use eighty labels and the batch stops being dense; the full list is still what the
#: verifier checks against, so trimming can only cause a refusal, never a wrong accept.
SHOW_ITEMS = 24


def collect(*, limit: int, seed: int) -> list[dict]:
    from chartqa_dt.data.chartqa import (
        ArchiveReader,
        annotation_boxes,
        annotation_path,
        image_path,
    )
    from scripts.build_mixtures import archive_path
    from scripts.mine_with_llm import stable_id

    out, seen, already = [], 0, 0
    with ArchiveReader(archive_path()) as reader:
        rows = list(reader.qa_rows("train", "human"))
        random.Random(seed).shuffle(rows)
        for row in rows:
            if len(out) >= limit:
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
                already += 1
                continue
            out.append({"record_id": stable_id("chartqa", name, question),
                        "question": question, "answer": answer, "evidence": evidence})
    print(f"scanned {seen:,} human questions; {already:,} already settled by plans.forward; "
          f"{len(out):,} need a reader")
    return out


def render(batch: list[dict]) -> str:
    lines = []
    for i, r in enumerate(batch):
        items = r["evidence"][:SHOW_ITEMS]
        shown = ", ".join(f"{e['label']!r}={e['value']!r}" for e in items)
        more = (f"   … and {len(r['evidence']) - SHOW_ITEMS} more items"
                if len(r["evidence"]) > SHOW_ITEMS else "")
        lines.append(f"{i}. Q: {r['question']}\n   ANSWER: {r['answer']!r}\n"
                     f"   items: {shown}{more}")
    body = "\n\n".join(lines)
    return f"""{teacher.build_system()}

--- Answer every numbered item below. Reply with ONE ```json block mapping each item number
--- to a plan, or to {{"refused": "..."}}, or to {{"needs_operator": {{...}}}}.

{body}

```json
{{"0": {{"op": "...", "args": [...]}}, "1": {{"refused": "..."}}, ...}}
```"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--score", help="JSON of {batch: {item: plan|refusal}} to verify")
    args = ap.parse_args()

    records = collect(limit=args.limit, seed=args.seed)
    batches = [records[i:i + args.batch_size]
               for i in range(0, len(records), args.batch_size)]
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.score:
        for i, batch in enumerate(batches):
            (OUT / f"batch_{i:03d}.txt").write_text(render(batch), encoding="utf-8")
        (OUT / "index.json").write_text(json.dumps(records, ensure_ascii=False),
                                        encoding="utf-8")
        print(f"wrote {len(batches)} batches of up to {args.batch_size} -> {OUT}/")
        return 0

    records = json.loads((OUT / "index.json").read_text(encoding="utf-8"))
    answers = json.loads(Path(args.score).read_text(encoding="utf-8"))
    proposals, refused, asked = [], 0, {}
    for bi, batch in enumerate(batches):
        given = answers.get(str(bi)) or {}
        for qi, rec in enumerate(batch):
            reply = given.get(str(qi))
            if reply is None:
                continue
            if isinstance(reply, dict) and "needs_operator" in reply:
                asked[reply["needs_operator"].get("name", "?")] = reply["needs_operator"]
                continue
            if isinstance(reply, dict) and "refused" in reply:
                refused += 1
                continue
            proposals.append({"record_id": rec["record_id"], "plan": reply,
                              "answer": rec["answer"], "evidence": rec["evidence"],
                              "marked_labels": None})

    verdicts, stats = llm_mining.verify_many(proposals)
    kept = [{"record_id": p["record_id"], "plan": p["plan"],
             teacher.PROVENANCE_KEY: {"method": "llm_teacher_batch",
                                      "question_kind": "human"}}
            for p, v in zip(proposals, verdicts) if v.accepted]
    print(f"\n  reader refused : {refused:,}")
    print(f"\n{stats.describe()}")
    for p, v in zip(proposals, verdicts):
        if not v.accepted:
            print(f"    [{p['record_id'][:20]}] {v.status}: {v.detail[:70]}")
    path = OUT / "human_plans.jsonl"
    path.write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept),
                    encoding="utf-8")
    print(f"\n  kept {len(kept):,} verified plans -> {path}")
    if asked:
        print("\n  operators the reader asked for:")
        for name, d in asked.items():
            print(f"    {name}: {d.get('signature','')} — {d.get('why','')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
