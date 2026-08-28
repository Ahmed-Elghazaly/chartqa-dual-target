#!/usr/bin/env python3
"""How many mixture records actually become training examples — on CPU, before any GPU.

`PLAN.md` 6.6 pre-registers a budget of 24,000 presentations. That budget assumes the
mixture yields examples, and it does not: `build_target` refuses a record whose plan does
not round-trip against its own evidence (`DECISIONS.md` 0067), and `build_batch` refuses one
that does not fit `max_seq_len` (0064). Both refusals are silent at training time — the feed
skips and moves on — so a mixture that yields half of what it claims looks like a run that
is simply slower than expected.

That is a ten-GPU-hour way to learn something measurable for free. This script rehydrates a
mixture, builds every target, and reports the yield and the reasons for refusal. With
`--tokens` it also runs the real processor over a sample and reports the sequence-length
distribution against the limit.

Nothing here needs model weights.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chartqa_dt.train.targets import TargetError, build_answer_only_target, build_target


def _reason(exc: Exception) -> str:
    """Collapse a refusal to its kind, so the histogram is readable."""
    text = str(exc)
    for needle, label in [
        ("has no answer", "no answer"),
        ("no element box", "plan references a label with no box"),
        ("more than the schema", "more evidence than the schema allows"),
        ("does not reproduce", "plan does not reproduce the answer"),
        ("round-trip", "plan does not reproduce the answer"),
        ("no mined plan", "no mined plan, and none derivable"),
        ("no plan", "no plan"),
        ("schema", "target fails the schema"),
    ]:
        if needle in text:
            return label
    return text.split(":")[-1].strip()[:60] or type(exc).__name__


def _source_pool(name: str, ctx, args) -> list:
    """One source's whole pool, at the caps `cli/train.py` uses to rebuild records."""
    from pathlib import Path as _Path

    from scripts.build_mixtures import (
        archive_path,
        chartqa_records,
        refchartqa_records,
        synthetic_records,
    )

    root = _Path(ctx.env.data_root)
    if name == "synth":
        return synthetic_records(root / "synthetic/train/manifest.json")
    if name == "refchartqa":
        return refchartqa_records(cap=args.refchartqa_cap,
                                  cache=root / "refchartqa_train.jsonl")
    from chartqa_dt.data.chartqa import ArchiveReader

    return chartqa_records(ArchiveReader(archive_path()), limit=args.chartqa_limit,
                           seed=ctx.cfg.seed)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mixture", default="data/mixture_stage1.json")
    p.add_argument("--source", default="", choices=["", "synth", "chartqa", "refchartqa"],
                   help="measure a whole source pool instead of a mixture — the ceiling on "
                        "how much usable supervision that source can contribute")
    p.add_argument("--chartqa-limit", type=int, default=8000,
                   help="rows sampled per question kind when --source chartqa")
    p.add_argument("--refchartqa-cap", type=int, default=4000)
    p.add_argument("--limit", type=int, default=0, help="0 = the whole mixture")
    p.add_argument("--answer-only", action="store_true",
                   help="measure the direct-answer control's target instead")
    p.add_argument("--tokens", type=int, default=0,
                   help="also tokenise this many examples with the real processor")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    from chartqa_dt.cli.train import _all_source_records
    from chartqa_dt.config import build_config
    from chartqa_dt.env import get_env

    ctx = argparse.Namespace(args=argparse.Namespace(mixture=args.mixture),
                             cfg=build_config(None), env=get_env())
    if args.source:
        records = _source_pool(args.source, ctx, args)
        by_id = {r.record_id: r for r in records}
        ids = [r.record_id for r in records]
        label = f"source {args.source}"
    else:
        by_id = {r.record_id: r for r in _all_source_records(ctx)}
        ids = json.loads(Path(args.mixture).read_text(encoding="utf-8"))["record_ids"]
        label = args.mixture
    if args.limit:
        ids = ids[:args.limit]

    build = build_answer_only_target if args.answer_only else build_target
    missing, ok = 0, []
    reasons: collections.Counter[str] = collections.Counter()
    reasons_by_source: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter)
    by_source: collections.Counter[str] = collections.Counter()
    ok_by_source: collections.Counter[str] = collections.Counter()

    for record_id in ids:
        record = by_id.get(record_id)
        if record is None:
            missing += 1
            continue
        source = record_id.split("_", 1)[0] if "_" in record_id else record.source
        by_source[source] += 1
        try:
            target = build(record)
        except TargetError as exc:
            reasons[_reason(exc)] += 1
            reasons_by_source[source][_reason(exc)] += 1
            continue
        ok.append((record, target))
        ok_by_source[source] += 1

    n = len(ids)
    print(f"\npool      : {label}  ({n:,} ids"
          f"{', answer-only target' if args.answer_only else ''})")
    if missing:
        print(f"  MISSING  : {missing:,} ids are not in any source — the mixture is stale")
    print(f"  usable   : {len(ok):,} / {n - missing:,} "
          f"({100 * len(ok) / max(n - missing, 1):.1f}%)")
    for reason, count in reasons.most_common():
        print(f"    refused: {count:>6,}  {reason}")
    print("\n  by source:")
    for source in sorted(by_source):
        got, tot = ok_by_source[source], by_source[source]
        print(f"    {source:<14}{got:>7,} / {tot:>7,}  ({100 * got / tot:5.1f}%)")
        for reason, count in reasons_by_source[source].most_common(4):
            print(f"        {count:>6,}  {reason}")

    lengths: list[int] = []
    over = 0
    if args.tokens and ok:
        from chartqa_dt.model.loader import ModelConfig
        from transformers import AutoProcessor

        from chartqa_dt.train.collate import Example, build_batch

        cfg = ModelConfig(image_max_pixels=512 * 512)
        processor = AutoProcessor.from_pretrained(cfg.hf_id)
        rng = random.Random(args.seed)
        sample = rng.sample(ok, min(args.tokens, len(ok)))
        print(f"\n  tokenising {len(sample)} examples with the real processor "
              f"(max_seq_len {cfg.max_seq_len})...")
        from chartqa_dt.train.feed import MixtureFeed

        feed = MixtureFeed([r for r, _ in sample], answer_only=args.answer_only)
        for i, (record, target) in enumerate(sample):
            try:
                image = feed._image(record)
                batch = build_batch(processor, [Example(image=image,
                                                        question=record.question,
                                                        target=target)],
                                    cfg.max_seq_len, strict=False)
                lengths.append(int(batch["input_ids"].shape[1]))
            except (OSError, ValueError) as exc:
                over += 1
                if over <= 3:
                    print(f"    refused: {exc}")
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(sample)}", flush=True)
        if lengths:
            lengths.sort()
            def pct(q: float) -> int:
                return lengths[min(int(q * len(lengths)), len(lengths) - 1)]
            print(f"    tokens   : median {pct(0.5):,}  p90 {pct(0.9):,}  "
                  f"p99 {pct(0.99):,}  max {lengths[-1]:,}  (limit {cfg.max_seq_len:,})")
            print(f"    over limit: {over} of {len(sample)} "
                  f"({100 * over / len(sample):.1f}%)")

    presentations = len(ok) * 2
    print(f"\n  two epochs over the usable records = {presentations:,} presentations "
          f"(budget 24,000)")
    if presentations < 24000:
        print(f"  SHORT by {24000 - presentations:,}. The pre-registered budget needs "
              f"{24000 / 2 / max(len(ok), 1):.2f} epochs, or a larger mixture.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "pool": label, "answer_only": args.answer_only,
            "ids": n, "missing": missing, "usable": len(ok),
            "usable_pct": round(100 * len(ok) / max(n - missing, 1), 2),
            "refusals": dict(reasons),
            "by_source": {s: {"total": by_source[s], "usable": ok_by_source[s],
                              "refusals": dict(reasons_by_source[s])}
                          for s in sorted(by_source)},
            "token_lengths": {"n": len(lengths), "over_limit": over,
                              "median": lengths[len(lengths) // 2] if lengths else None,
                              "max": lengths[-1] if lengths else None},
        }, indent=1) + "\n", encoding="utf-8")
        print(f"\n  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
