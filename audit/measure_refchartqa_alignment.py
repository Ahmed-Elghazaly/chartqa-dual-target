#!/usr/bin/env python3
"""AUDIT · Idea 3 — can RefChartQA's gold grounding be aligned to ChartQA's semantic elements?

RefChartQA gives, per question, the regions a correct answer should point at — but no
labels and no values. Our target builder therefore names them `item1, item2, ...`
(`targets.py`, the no-elements branch), which teaches the model to emit placeholder labels
and gives the plan nothing to reason over.

ChartQA annotates the same images with *semantic* elements: label, value, series, bbox.

If a RefChartQA box can be matched to a ChartQA element with high confidence, that
grounding becomes semantically identified, and a RefChartQA question becomes eligible for
a real plan instead of no plan at all.

This measures whether the match is clean enough to trust. It does NOT force matches:
the output is a distribution, so the decision about thresholds is made on evidence.

Matching is deliberately conservative:
  * IoU as the primary score, in the shared 0-1000 space;
  * the MARGIN between the best and second-best element is recorded, because a box that
    matches two elements almost equally is ambiguous, not matched;
  * containment is recorded separately, since a grounding box may enclose a mark plus its
    label text, which lowers IoU without being wrong.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.config import build_config  # noqa: E402
from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord  # noqa: E402
from chartqa_dt.env import get_env  # noqa: E402


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix, iy = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def containment(inner, outer) -> float:
    """Fraction of `inner`'s area inside `outer` — a grounding box may enclose extra text."""
    ax1, ay1, ax2, ay2 = inner
    bx1, by1, bx2, by2 = outer
    ix, iy = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    return (ix * iy) / area if area > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="audit/refchartqa_alignment.json")
    args = ap.parse_args()

    from scripts.build_mixtures import archive_path, chartqa_records

    from chartqa_dt.data.chartqa import ArchiveReader
    from chartqa_dt.data.mixture import CHARTQA_DRAW

    cfg = build_config(None)
    root = Path(get_env().data_root)

    # ChartQA elements, keyed by decoded-pixel image hash.
    elements_by_image: dict[str, list] = {}
    for r in chartqa_records(ArchiveReader(archive_path()), limit=CHARTQA_DRAW, seed=cfg.seed):
        els = [e for e in (r.meta.get(ELEMENTS_KEY) or []) if isinstance(e, dict)]
        if els:
            elements_by_image.setdefault(r.image_sha256, els)
    print(f"ChartQA images carrying elements: {len(elements_by_image):,}")

    cache = root / "refchartqa_train.jsonl"
    ref = [ChartRecord.from_dict(json.loads(line))
           for line in cache.read_text(encoding="utf-8").splitlines() if line]
    print(f"RefChartQA cached records: {len(ref):,}")

    stats = collections.Counter()
    best_ious, margins, best_containments = [], [], []
    per_box = []
    for r in ref:
        els = elements_by_image.get(r.image_sha256)
        if not els:
            stats["no_chartqa_elements_for_this_image"] += 1
            continue
        stats["records_with_candidate_elements"] += 1
        cand = [(e, e["bbox"]) for e in els if e.get("bbox")]
        for gb in (r.boxes or []):
            scored = sorted(((iou(gb, eb), e, eb) for e, eb in cand),
                            key=lambda t: -t[0])
            if not scored:
                continue
            best, second = scored[0][0], (scored[1][0] if len(scored) > 1 else 0.0)
            stats["grounding_boxes"] += 1
            best_ious.append(best)
            margins.append(best - second)
            best_containments.append(containment(scored[0][2], gb))
            per_box.append({"record": r.record_id, "best_iou": round(best, 4),
                            "margin": round(best - second, 4),
                            "label": scored[0][1].get("label"),
                            "value": scored[0][1].get("value"),
                            "series": scored[0][1].get("series")})

    n = stats["grounding_boxes"]
    print(f"\nrecords whose image has ChartQA elements: "
          f"{stats['records_with_candidate_elements']:,} of {len(ref):,}")
    print(f"grounding boxes scored: {n:,}\n")
    if n:
        def share(pred):
            return 100 * sum(1 for v in best_ious if pred(v)) / n
        print("best-match IoU between a RefChartQA grounding box and a ChartQA element:")
        for t in (0.9, 0.75, 0.5, 0.3, 0.1):
            print(f"   IoU >= {t:<5} {share(lambda v, t=t: v >= t):5.1f}%")
        print(f"   median {statistics.median(best_ious):.3f}")
        clean = [i for i, (b, m) in enumerate(zip(best_ious, margins))
                 if b >= 0.5 and m >= 0.2]
        print(f"\nCONFIDENTLY matched (IoU >= 0.5 AND margin over runner-up >= 0.2): "
              f"{len(clean):,} of {n:,}  ({100 * len(clean) / n:.1f}%)")
        print(f"median containment of the matched element inside the grounding box: "
              f"{statistics.median(best_containments):.3f}")

    Path(args.out).write_text(json.dumps(
        {"stats": dict(stats),
         "best_iou_median": statistics.median(best_ious) if best_ious else None,
         "n_boxes": n, "per_box_sample": per_box[:400]}, indent=1) + "\n", encoding="utf-8")
    print(f"\n  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
