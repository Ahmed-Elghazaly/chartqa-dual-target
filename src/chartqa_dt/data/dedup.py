"""Merge records that are the same (chart, question) seen twice.

`PLAN.md` 3.3. RefChartQA is derived partly from ChartQA, so mixing them naively
double-counts the same question and inflates the apparent size of the training set — and
a training set that is 15% smaller than believed is a silent confound on every result
that follows.

The plan is explicit that duplicates are **merged**, not dropped and not counted twice:
keep the answer, union the boxes, keep any exact plan. Dropping one would throw away
whichever half had the boxes; keeping both would train twice on one example.

Two properties are worth stating because they are easy to get wrong:

* **Merging is order-independent.** Records arrive from different loaders in whatever
  order a mixture happens to iterate. If merge order changed the result, two runs of the
  same pipeline could differ, so the merge is deliberately commutative and it is tested
  by shuffling.
* **Splits are never merged across.** A train record and a test record that share a key
  are a *leak*, not a duplicate. Merging them would hide exactly the thing rule 1 exists
  to prevent, so a cross-split collision is reported, never silently resolved.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from chartqa_dt.data.records import ChartRecord

#: Which source's answer wins when two disagree. ChartQA's gold labels are the ones the
#: official metric scores against, so they take precedence over derived datasets.
SOURCE_PRIORITY = {"chartqa": 0, "chartqapro": 1, "refchartqa": 2, "synthetic": 3}

#: Boxes closer than this in every coordinate (0-1000 scale) are the same box.
BOX_EPSILON = 1.0


@dataclass
class DedupReport:
    """What the merge did, in enough detail to put in a phase report."""

    input_records: int = 0
    output_records: int = 0
    merges: int = 0
    merged_pairs: Counter[str] = field(default_factory=Counter)
    answer_conflicts: int = 0
    boxes_gained: int = 0
    plans_gained: int = 0
    cross_split_collisions: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def duplicates_removed(self) -> int:
        return self.input_records - self.output_records

    def summary(self) -> str:
        pairs = ", ".join(f"{k}={v}" for k, v in sorted(self.merged_pairs.items())) or "none"
        return (f"{self.input_records:,} in -> {self.output_records:,} out "
                f"({self.duplicates_removed:,} duplicates merged across {self.merges:,} "
                f"merge operations; {pairs}); {self.boxes_gained:,} records gained boxes, "
                f"{self.plans_gained:,} gained a plan, {self.answer_conflicts:,} answer "
                f"conflicts, {len(self.cross_split_collisions):,} cross-split collisions")


def _same_box(a: list[float], b: list[float]) -> bool:
    return all(abs(x - y) <= BOX_EPSILON for x, y in zip(a, b))


def union_boxes(*groups: list[list[float]] | None) -> list[list[float]] | None:
    """Every distinct box across the inputs, in first-seen order.

    Order is preserved rather than sorted because `DECISIONS.md` 0014 has the model emit
    boxes best-first; sorting would destroy that ordering for no gain.
    """
    out: list[list[float]] = []
    for group in groups:
        for box in group or ():
            if not any(_same_box(box, kept) for kept in out):
                out.append(list(box))
    return out or None


def merge_pair(a: ChartRecord, b: ChartRecord, report: DedupReport | None = None
               ) -> ChartRecord:
    """One merged record from two that share a `dedup_key`.

    Commutative by construction: the winner of every field is chosen by a rule that does
    not depend on argument order.
    """
    if a.split != b.split:
        raise ValueError(
            f"refusing to merge across splits: {a.record_id} is {a.split!r} and "
            f"{b.record_id} is {b.split!r}. A shared key across splits is a leak, not a "
            f"duplicate (rule 1)."
        )
    primary, other = (a, b) if SOURCE_PRIORITY.get(a.source, 9) <= SOURCE_PRIORITY.get(b.source, 9) else (b, a)

    answer = primary.answer if primary.answer is not None else other.answer
    if (a.answer is not None and b.answer is not None
            and a.answer.strip().lower() != b.answer.strip().lower() and report):
        report.answer_conflicts += 1

    boxes = union_boxes(primary.boxes, other.boxes)
    plan = primary.plan or other.plan
    table = primary.table or other.table

    if report is not None:
        if boxes and not primary.boxes:
            report.boxes_gained += 1
        if plan and not primary.plan:
            report.plans_gained += 1

    meta = dict(other.meta)
    meta.update(primary.meta)
    meta["merged_from"] = sorted({*_sources(a), *_sources(b)})
    meta["merged_record_ids"] = sorted({a.record_id, b.record_id,
                                        *a.meta.get("merged_record_ids", []),
                                        *b.meta.get("merged_record_ids", [])})

    return replace(primary, answer=answer, boxes=boxes, plan=plan, table=table, meta=meta)


def _sources(r: ChartRecord) -> list[str]:
    return list(r.meta.get("merged_from") or [r.source])


def deduplicate(records: Iterable[ChartRecord]) -> tuple[list[ChartRecord], DedupReport]:
    """Merge every group that shares a `dedup_key`, preserving first-seen order."""
    report = DedupReport()
    order: list[tuple[str, str]] = []
    by_key: dict[tuple[str, str], ChartRecord] = {}
    splits_seen: dict[str, str] = {}

    for record in records:
        report.input_records += 1
        key = record.key
        # Deduplicate WITHIN a split and only report across it. A key shared by two
        # splits is a leak, and dropping one side would resolve it silently — the exact
        # failure rule 1 exists to make impossible.
        prior_split = splits_seen.setdefault(key, record.split)
        if prior_split != record.split:
            report.cross_split_collisions.append((key, prior_split, record.split))

        slot = (record.split, key)
        existing = by_key.get(slot)
        if existing is None:
            by_key[slot] = record
            order.append(slot)
            continue
        pair = " + ".join(sorted({existing.source, record.source}))
        by_key[slot] = merge_pair(existing, record, report)
        report.merges += 1
        report.merged_pairs[pair] += 1

    out = [by_key[k] for k in order]
    report.output_records = len(out)
    return out, report


def find_cross_split_leaks(records: Iterable[ChartRecord]) -> dict[str, dict[str, Any]]:
    """Keys that appear in more than one split — the rule 1 check, done on keys."""
    seen: dict[str, dict[str, Any]] = {}
    for r in records:
        entry = seen.setdefault(r.key, {"splits": set(), "record_ids": []})
        entry["splits"].add(r.split)
        entry["record_ids"].append(r.record_id)
    return {k: {"splits": sorted(v["splits"]), "record_ids": v["record_ids"]}
            for k, v in seen.items() if len(v["splits"]) > 1}


__all__ = ["BOX_EPSILON", "SOURCE_PRIORITY", "DedupReport", "deduplicate",
           "find_cross_split_leaks", "merge_pair", "union_boxes"]
