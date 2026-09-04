#!/usr/bin/env python3
"""MEASUREMENT · what `MAX_EVIDENCE = 8` blocks, and what raising it would cost.

Two of the teacher's 27 proposals were rejected for needing more evidence than the schema
allows, both bare aggregates over charts with 17 and 25 elements. The cap was set on a
measured token budget (`DECISIONS.md` 0060: a target with 8 evidence items is 241 tokens
against `ModelConfig.max_seq_len = 1024`), so re-opening it means pricing sequence length
against step time — not asserting that bigger is better.

This measures the two halves separately:

  BLOCKED   a question whose answer needs an operation that folds over the whole chart
            (`argmax`, `max`, `median`, `count`, `trend`, …) on a chart with more elements
            than the cap. A plan naming specific labels is unaffected however large the
            chart, because `train.targets` selects the elements the plan names.

  COST      tokens per additional evidence item, with the real tokenizer where it is
            available, so the trade is in the same units as 0060.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from audit.measure_dsl_coverage import classify  # noqa: E402
from chartqa_dt.plans.schema import MAX_EVIDENCE  # noqa: E402

#: Categories whose plan folds over every element, so the whole chart must be in evidence.
FOLD_SHAPED = {"argmax_argmin", "rank_n", "argmax_over_computed"}


def token_cost() -> None:
    """Tokens per evidence item, measured rather than estimated where possible."""
    def record(n: int) -> str:
        return json.dumps({
            "answerable": True,
            "evidence": [{"label": f"Series {i} · 20{i:02d}", "value": 1234.5,
                          "unit": "%", "bbox": [100 + i, 200, 150 + i, 800]}
                         for i in range(n)],
            "plan": {"op": "argmax", "args": []},
            "model_answer": "2019"}, separators=(",", ":"), ensure_ascii=False)

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct",
                                            trust_remote_code=True)
        def count(s: str) -> int:
            return len(tok(s)["input_ids"])
        how = "real Qwen3-VL tokenizer"
    except Exception as exc:                     # noqa: BLE001 — offline is expected
        print(f"  (tokenizer unavailable: {type(exc).__name__}; "
              f"reporting characters instead of tokens)")
        def count(s: str) -> int:
            return len(s)
        how = "CHARACTERS, not tokens — treat as a shape, not a budget"

    print(f"\n  cost per evidence item ({how}):")
    base = count(record(1))
    for n in (2, 8, 12, 16, 20):
        c = count(record(n))
        print(f"    {n:>3} items {c:>6,}   ({(c - base) / (n - 1):>5.1f} per extra item)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scripts.build_mixtures import archive_path

    from chartqa_dt.data.chartqa import (
        ArchiveReader,
        annotation_boxes,
        annotation_path,
        image_path,
    )

    sizes: dict[str, list[int]] = collections.defaultdict(list)
    seen = 0
    with ArchiveReader(archive_path()) as reader:
        pool = [r for k in ("human", "machine") for r in reader.qa_rows("train", k)]
        random.Random(args.seed).shuffle(pool)
        cache: dict[str, int] = {}
        for row in pool:
            if seen >= args.limit:
                break
            name = row["imgname"]
            if name not in cache:
                ann, img = annotation_path("train", name), image_path("train", name)
                if not (reader.exists(ann) and reader.exists(img)):
                    continue
                w, h = reader.image_size(img)
                cache[name] = len(annotation_boxes(reader.read_json(ann), w, h))
            n = cache[name]
            if not n:
                continue
            seen += 1
            sizes[classify(str(row["query"]), str(row.get("label", "")))].append(n)

    fold = [n for k, v in sizes.items() if k in FOLD_SHAPED for n in v]
    named = [n for k, v in sizes.items() if k not in FOLD_SHAPED for n in v]
    print(f"{seen:,} real ChartQA train questions (seed {args.seed})\n")
    print(f"  plans that name their own labels : {len(named):,} "
          f"({100 * len(named) / seen:.1f}%) — the cap never applies")
    print(f"  plans that fold over the chart   : {len(fold):,} "
          f"({100 * len(fold) / seen:.1f}%) — the whole chart must fit\n")
    if fold:
        blocked = sum(1 for n in fold if n > MAX_EVIDENCE)
        print(f"  of those, over the cap of {MAX_EVIDENCE}: {blocked:,} "
              f"({100 * blocked / len(fold):.1f}% of fold-shaped, "
              f"{100 * blocked / seen:.1f}% of ALL questions)   <-- what raising it buys")
        print("\n  how much of the fold-shaped set each cap would admit:")
        for cap in (8, 10, 12, 16, 20, 24, 32):
            ok = sum(1 for n in fold if n <= cap)
            print(f"    cap {cap:>3}: {ok:>6,}/{len(fold):,} ({100 * ok / len(fold):>5.1f}%)"
                  f"   +{100 * (ok - sum(1 for n in fold if n <= MAX_EVIDENCE)) / seen:>4.1f}% "
                  f"of all questions")
    token_cost()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
