#!/usr/bin/env python3
"""Give RefChartQA's grounding boxes their semantic identity, from ChartQA's elements.

`AUDIT.md` H1. RefChartQA marks *which* regions answer a question but says nothing about
what they are. `targets._evidence_from` therefore names them `item1, item2, …` with no
value, and `build_record` can only derive a plan for the single-box case — by setting the
evidence value **to the answer**, which makes the round-trip pass by construction and
teaches nothing about reading the chart.

ChartQA annotates the same images with semantic elements. Measured over 6,340 grounding
boxes (`audit/measure_refchartqa_alignment.py`): **98.9% match a ChartQA element at
IoU ≥ 0.9, median IoU 1.000, median margin over the runner-up 1.000.** The boxes are not
similar, they are the same boxes — RefChartQA is ChartQA's element geometry plus a
per-question selection.

So this is a lookup, not an inference problem. It is still written to **refuse**:

* a box must match at `MIN_IOU` **and** beat the runner-up by `MIN_MARGIN`, so a box that
  fits two elements almost equally is ambiguous rather than matched;
* **all** of a record's boxes must match, or the record stays unaligned — a half-aligned
  record would mix real labels with `item1` placeholders in one evidence list;
* the match score and margin are recorded per element, so a later filter can be stricter
  without re-running this.

Output goes to the **data cache**, never to git: it contains labels and values, which are
dataset content under rule 7.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from chartqa_dt.config import build_config  # noqa: E402
from chartqa_dt.data.records import ELEMENTS_KEY, ChartRecord  # noqa: E402
from chartqa_dt.env import get_env  # noqa: E402
from chartqa_dt.plans.mining import mine_plan  # noqa: E402
from chartqa_dt.train.targets import plan_labels  # noqa: E402

#: Deliberately strict. 98.9% of boxes already clear IoU 0.9, so precision costs ~nothing.
MIN_IOU = 0.90
#: The best match must beat the second best by this much, or the box is ambiguous.
MIN_MARGIN = 0.50


def normalise_value(raw: Any) -> tuple[float | None, str | None]:
    """A chart annotation's value as a number, plus its unit — or (None, None).

    ChartQA annotations store values as *written on the chart*: ``'460 000'`` with a space
    thousands separator, ``'9,891'`` with a comma, ``'64%'`` with a percent sign. Passed
    through unchanged they are unusable — `to_float('460 000')` is None and the executor
    *raises* on it, while `'9,891'` parses in one and not the other.

    **The percent magnitude is kept, not divided.** RefChartQA's gold answers are written in
    percentage points (`'64'` for a bar reading `64%`), and the official metric would score
    a predicted `0.64` against a gold `64` as wrong. This differs from the ChartQA path,
    where the table cell and the gold answer *both* carry `%` and both parse to a fraction
    (`DECISIONS.md` 0075). The two never meet in one record, because a record is either
    ChartQA-sourced or RefChartQA-sourced.
    """
    if raw is None:
        return None, None
    if isinstance(raw, (int, float)):
        return float(raw), None
    text = str(raw).strip().replace("\u00a0", " ")
    unit = None
    if text.endswith("%"):
        unit, text = "%", text[:-1].strip()
    # Space and comma are thousands separators on these charts, never decimal points.
    text = text.replace(" ", "").replace(",", "")
    try:
        return float(text), unit
    except ValueError:
        return None, unit


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ub = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def align_record(record: ChartRecord, elements: list[dict]) -> dict | None:
    """The gold-selected elements for this record, or None if any box is not confident.

    Assignment is greedy and one-to-one: an element already claimed by one grounding box
    cannot be claimed by another. Two boxes resolving to the same element would emit
    duplicate evidence labels, which `executor.by_label` and `targets.by_label` resolve
    *differently* (`AUDIT.md` H3) — so it is refused here rather than left to them.
    """
    boxes = record.boxes or []
    if not boxes:
        return None
    candidates = [e for e in elements if e.get("bbox")]
    if not candidates:
        return None

    taken: set[int] = set()
    matched: list[dict] = []
    for gb in boxes:
        scored = sorted(
            ((iou(gb, e["bbox"]), i, e) for i, e in enumerate(candidates) if i not in taken),
            key=lambda t: -t[0])
        if not scored:
            return None
        best_iou, index, element = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0
        if best_iou < MIN_IOU or (best_iou - runner_up) < MIN_MARGIN:
            return None
        taken.add(index)
        value, unit = normalise_value(element.get("value"))
        matched.append({**element, "value": value, "unit": unit or element.get("unit"),
                        "value_raw": element.get("value"),
                        "match_iou": round(best_iou, 4),
                        "match_margin": round(best_iou - runner_up, 4)})
    return {"record_id": record.record_id, ELEMENTS_KEY: matched}


def labels_cover(used: set[str], marked: set[str]) -> bool:
    """Is every operand the miner read one of the regions the annotation marks?

    Compared with truncation tolerance. ChartQA's *element* labels are stored as drawn and
    are clipped: the table says `'MSCI Global, excluding U.S.'` where the annotation says
    `'MSCI Global, excluding'`. Measured over 63,069 elements — 93.5% match the table
    exactly, **3.1% are a prefix of it**, 0.5% the reverse. Comparing on equality alone
    counted those as the miner reading the wrong rows, which they are not.

    A prefix is accepted only when it is **unambiguous**: if the truncated label prefixes two
    different marked regions, the pairing is unknown and the record is not certified.
    """
    for operand in used:
        exact = operand in marked
        if exact:
            continue
        prefixes = [m for m in marked
                    if operand.startswith(m) or m.startswith(operand)]
        if len(prefixes) != 1:
            return False
    return True


def mine_grounded_plan(record: ChartRecord, table: dict | None,
                       marked: list[dict]) -> tuple[dict | None, str]:
    """Mine a plan, and keep it only if it uses **the regions RefChartQA marked**.

    This is the check the deterministic miner cannot make on its own. `mine_plan` accepts an
    operation when exactly one reproduces the gold answer — but *numerical agreement is not
    semantic correctness*, and a plan can hit the right number through the wrong rows
    (`Prompt.md` Idea 7; `DECISIONS.md` 0045 recorded `difference -> 2096` explaining a gold
    answer of `2019`).

    RefChartQA independently states which regions a correct answer uses. So when the miner's
    operands are exactly those regions, two independent sources agree on both the *value*
    and the *operands*, which is far stronger than either alone. When they disagree, the
    mined plan reached the right number from the wrong marks and is **rejected** — the
    grounding is trusted over the arithmetic, because the grounding is gold and the plan is
    inferred.

    Returns the plan and a status, so the rejection rate is measurable rather than silent.
    """
    if not table or record.answer is None:
        return None, "no_table"
    rows = [table.get("columns") or [], *(table.get("rows") or [])]
    if len(rows) < 2:
        return None, "no_table"

    mined = mine_plan(rows, record.answer)
    if mined.plan is None:
        return None, f"miner:{mined.status}"

    wanted = {str(label) for label in plan_labels(mined.plan)}
    if not wanted:
        return None, "plan_uses_no_named_operand"
    marked_labels = {str(e.get("label")) for e in marked}
    if not labels_cover(wanted, marked_labels):
        return None, "operands_outside_the_marked_regions"
    if not labels_cover(marked_labels, wanted):
        # The question marks regions the plan never reads. That is not necessarily wrong —
        # a grounding annotation may include context — but it is not the clean agreement
        # this path exists to certify, so it is recorded rather than accepted.
        return None, "marked_regions_unused_by_the_plan"
    return mined.plan, "agreed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <data_root>/refchartqa_aligned.jsonl")
    ap.add_argument("--stats", default="verification/refchartqa_alignment.json")
    args = ap.parse_args()

    from chartqa_dt.data.chartqa import ArchiveReader
    from chartqa_dt.data.mixture import CHARTQA_DRAW
    from scripts.build_mixtures import archive_path, chartqa_records

    cfg = build_config(None)
    root = Path(get_env().data_root)
    out = args.out or (root / "refchartqa_aligned.jsonl")

    elements_by_image: dict[str, list] = {}
    table_by_image: dict[str, dict] = {}
    for r in chartqa_records(ArchiveReader(archive_path()), limit=CHARTQA_DRAW, seed=cfg.seed):
        els = [e for e in (r.meta.get(ELEMENTS_KEY) or []) if isinstance(e, dict)]
        if els:
            elements_by_image.setdefault(r.image_sha256, els)
            if r.table:
                table_by_image.setdefault(r.image_sha256, r.table)
    print(f"ChartQA images with elements: {len(elements_by_image):,}")

    cache = root / "refchartqa_train.jsonl"
    ref = [ChartRecord.from_dict(json.loads(line))
           for line in cache.read_text(encoding="utf-8").splitlines() if line]

    stats = collections.Counter()
    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for r in ref:
            els = elements_by_image.get(r.image_sha256)
            if not els:
                stats["no_chartqa_elements"] += 1
                continue
            aligned = align_record(r, els)
            if aligned is None:
                stats["refused_low_confidence_or_ambiguous"] += 1
                continue
            table = table_by_image.get(r.image_sha256)
            aligned["table"] = table
            aligned["n_boxes"] = len(r.boxes or [])
            plan, plan_status = mine_grounded_plan(r, table, aligned[ELEMENTS_KEY])
            if plan is not None:
                aligned["plan"] = plan
            stats[f"plan:{plan_status}"] += 1
            fh.write(json.dumps(aligned) + "\n")
            written += 1
            stats["aligned"] += 1

    total = len(ref)
    print(f"\nRefChartQA cached records : {total:,}")
    for k, v in sorted(stats.items()):
        print(f"  {k:<38}{v:>7,}  ({100 * v / total:5.1f}%)")
    print(f"\n  aligned records -> {out}")

    Path(args.stats).write_text(json.dumps({
        "min_iou": MIN_IOU, "min_margin": MIN_MARGIN,
        "cached_records": total, **dict(stats),
        "aligned_pct": round(100 * stats["aligned"] / max(total, 1), 2),
        "_note": ("Counts only. The aligned records themselves hold labels and values, "
                  "which are dataset content, so they live in the data cache (rule 7)."),
    }, indent=1) + "\n", encoding="utf-8")
    print(f"  statistics      -> {args.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
