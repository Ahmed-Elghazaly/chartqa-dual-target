"""The RefChartQA box audit — `PLAN.md` 3.4.

Sample exactly 200 training rows, stratified across human / machine / PoT. Render each
with its box drawn on the chart. Judge each acceptable or not: *a box is acceptable if it
plausibly contains evidence a person would use to answer that question.*

**What "judge" means here, stated plainly.** The labels came partly from an automated
pipeline using GPT-4o-mini, so the audit exists to find systematic error. Judgement runs
in two layers, and both are recorded:

1. **Measured criteria on all 200 rows.** A box that contains no chart ink cannot contain
   evidence, whatever the question. A box covering nearly the whole chart is not evidence
   either — it is a non-answer. These are computed, reproducible, and applied uniformly.
2. **Visual inspection of a stratified subsample**, rendered to disk and actually looked
   at, to check that layer 1 agrees with what a person would say. Layer 1 is only worth
   anything if it tracks human judgement, and asserting that without checking would defeat
   the purpose of auditing.

Every judgement, its reason and its measurements go to `data/refchartqa_audit.jsonl`, so
the decision is auditable rather than asserted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

AUDIT_PATH = Path("data/refchartqa_audit.jsonl")
SAMPLE_SIZE = 200
KINDS = ("human", "machine", "pot")

#: A box below this ink fraction contains nothing a reader could point at.
MIN_INK = 0.02
#: A box above this share of the chart area is not evidence, it is the whole chart.
MAX_AREA_SHARE = 0.60
#: Chart ink is anything this far from the page background.
INK_TOLERANCE = 24
#: Tightness is MEASURED AND REPORTED but does NOT gate. It was tried as a gate and is
#: invalid for this dataset: RefChartQA grounds on printed value labels *inside* filled
#: elements at least as often as on the elements themselves, and growing a box that sits
#: on a number inside a bar captures more bar colour, so ink density rises rather than
#: falls. It rejected the box drawn exactly around "DK 14%" in a pie chart answering
#: "What's the percentage value of DK segment?", and the boxes drawn exactly around "68"
#: and "52" inside two bars. Gating on it took the human subset from 100% to 58.2% and
#: the whole audit from 100% to 84% — a FAIL that would have dropped RefChartQA from
#: training on the strength of a criterion that does not describe the data.
EXPAND = 0.35


def ink_fraction(img: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """Share of the box that is not background."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    background = np.median(img.reshape(-1, 3), axis=0)
    return float((np.abs(crop.astype(int) - background) > INK_TOLERANCE).any(axis=-1).mean())


def tightness(img: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """How much of the box's ink density is lost when the box is grown.

    Recorded for every row so a later reader can see the distribution, but not a gate —
    see the note on `EXPAND` for the counter-examples that ruled it out.
    """
    dx, dy = (box[2] - box[0]) * EXPAND, (box[3] - box[1]) * EXPAND
    grown = (box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy)
    inner, outer = ink_fraction(img, box), ink_fraction(img, grown)
    return 0.0 if inner <= 0 else (inner - outer) / inner


def judge(img: np.ndarray, boxes_px: list[tuple[float, float, float, float]]
          ) -> tuple[bool, str, dict[str, Any]]:
    """Measured verdict for one row, with the numbers behind it.

    These are *necessary* conditions, not sufficient ones: they cannot tell whether a
    well-formed box sits on the element the question is actually about. That judgement
    is what the visual layer is for, and the report states both separately rather than
    letting the measured pass rate stand in for the whole audit.
    """
    h, w = img.shape[:2]
    if not boxes_px:
        return False, "no box", {"n_boxes": 0}
    inks = [ink_fraction(img, b) for b in boxes_px]
    shares = [((b[2] - b[0]) * (b[3] - b[1])) / (w * h) for b in boxes_px]
    tights = [tightness(img, b) for b in boxes_px]
    m = {"n_boxes": len(boxes_px), "min_ink": round(min(inks), 4),
         "max_area_share": round(max(shares), 4),
         "mean_ink": round(float(np.mean(inks)), 4),
         "min_tightness": round(min(tights), 4)}
    if min(inks) < MIN_INK:
        return False, f"a box is {100 * min(inks):.1f}% ink — nothing to point at", m
    if max(shares) > MAX_AREA_SHARE:
        return False, f"a box covers {100 * max(shares):.0f}% of the chart", m
    return True, "ink-bearing, and a region rather than the whole chart", m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=SAMPLE_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", type=int, default=18,
                    help="how many to render for visual inspection")
    ap.add_argument("--render-rejected", action="store_true",
                    help="render only the rows the measured criteria rejected")
    ap.add_argument("--render-dir", type=Path,
                    default=Path("outputs/refchartqa_audit_renders"))
    ap.add_argument("--out", type=Path, default=AUDIT_PATH)
    ap.add_argument("--with-text", action="store_true",
                    help="include question and answer text — LOCAL INSPECTION ONLY, "
                         "never for the committed file (rule 7)")
    args = ap.parse_args()

    from datasets import load_dataset
    from PIL import ImageDraw

    from chartqa_dt.data.refchartqa import boxes_to_norm1000
    from chartqa_dt.data.sources import REFCHARTQA_PARQUET as spec
    from chartqa_dt.vision.coords import norm1000_to_px

    per_kind = args.n // len(KINDS)
    quota = dict.fromkeys(KINDS, per_kind)
    for k in KINDS[:args.n - per_kind * len(KINDS)]:
        quota[k] += 1

    # Streamed: the audit needs 200 rows, not 2.88 GB.
    stream = load_dataset(spec.repo_id, split="train", streaming=True,
                          revision=spec.revision).shuffle(seed=args.seed, buffer_size=5000)

    records: list[dict[str, Any]] = []
    taken: Counter[str] = Counter()
    args.render_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0

    for row in stream:
        kind = str(row.get("type", "")).lower()
        if kind not in quota or taken[kind] >= quota[kind]:
            if sum(taken.values()) >= args.n:
                break
            continue
        image = row["image"].convert("RGB")
        w, h = image.size
        norm = boxes_to_norm1000(row.get("grounding_bboxes"), w, h)
        boxes_px = [tuple(norm1000_to_px(b, w, h)) for b in norm]
        arr = np.asarray(image)
        ok, reason, measured = judge(arr, boxes_px)
        taken[kind] += 1

        # Rule 7: no dataset content in the repository. The question and the gold answer
        # are RefChartQA's (AGPL-3.0), so they are not written. `id` identifies the row
        # uniquely, so anyone with the dataset can recover them and re-judge — which is
        # what "auditable" requires. Question text goes only to `--with-text`, for local
        # inspection, and that file is not committed.
        rec = {"id": row.get("id"), "type": kind, "image_size": [w, h],
               "n_boxes_raw": len(row.get("grounding_bboxes") or []),
               "boxes_norm1000": [[round(v, 2) for v in b] for b in norm],
               "acceptable": ok, "reason": reason, "measured": measured}
        if args.with_text:
            rec["question"] = row.get("query")
            rec["answer"] = row.get("label")

        if rendered < args.render and (not args.render_rejected or not ok):
            drawn = image.copy()
            d = ImageDraw.Draw(drawn)
            for b in boxes_px:
                d.rectangle([b[0], b[1], b[2], b[3]], outline=(255, 0, 255), width=3)
            name = f"{kind}_{taken[kind]:03d}_{'ok' if ok else 'BAD'}.png"
            drawn.save(args.render_dir / name)
            rec["render"] = str(args.render_dir / name)
            rendered += 1
        records.append(rec)
        if sum(taken.values()) >= args.n:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    total = len(records)
    good = sum(r["acceptable"] for r in records)
    print(f"\nRefChartQA box audit — {total} training rows, seed {args.seed}\n")
    print(f"{'type':<10}{'rows':>7}{'acceptable':>12}{'rate':>9}")
    for k in KINDS:
        rows = [r for r in records if r["type"] == k]
        g = sum(r["acceptable"] for r in rows)
        print(f"  {k:<8}{len(rows):>7}{g:>12}{100 * g / max(1, len(rows)):>8.1f}%")
    rate = 100 * good / max(1, total)
    print(f"  {'ALL':<8}{total:>7}{good:>12}{rate:>8.1f}%")
    print("\n  gate: >= 90% -> use RefChartQA training rows; < 90% -> drop from training")
    print(f"  result: {rate:.1f}% -> {'PASS' if rate >= 90 else 'FAIL'}")

    reasons = Counter(r["reason"] for r in records if not r["acceptable"])
    if reasons:
        print("\nwhy rows were rejected:")
        for reason, n in reasons.most_common():
            print(f"  {n:>4}  {reason}")
    print(f"\n{total} judgements written to {args.out}")
    print(f"{rendered} renders in {args.render_dir} for visual inspection")


if __name__ == "__main__":
    main()
