"""Stage-1 and stage-2 training mixtures, with their composition recorded.

`PLAN.md` 3.7.

* **Stage 1 — grounding only.** Ordered `L1 -> L4` synthetic, then audited real boxes.
  Cap 12,000. The order is the point: difficulty is a curriculum, so stage 1 is *not*
  shuffled.
* **Stage 2 — joint box + plan + answer.** Shuffled, including ~2,000 exact synthetic
  replay examples. Cap 12,000.

Both are deduplicated, and both are checked for validation/test records before they are
written — `PLAN.md` 3.7 requires zero, and `build_mixture` raises rather than filtering
silently. A mixture that quietly dropped a leaked record would hide the fact that
something upstream produced one.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chartqa_dt.data.dedup import deduplicate
from chartqa_dt.data.records import ChartRecord
from chartqa_dt.splits import assert_no_held_out_images

STAGE1_CAP = 12_000
STAGE2_CAP = 12_000
SYNTHETIC_REPLAY = 2_000
TRAIN_ONLY = "train"

#: The curriculum order for stage 1. Not shuffled — that is what makes it a curriculum.
LEVEL_ORDER = ("L1", "L2", "L3", "L4")


class LeakageError(RuntimeError):
    """A mixture contained a record from a split it must never see."""


@dataclass
class MixtureComposition:
    """Exact counts by source, question kind and difficulty level."""

    stage: str
    total: int = 0
    by_source: Counter[str] = field(default_factory=Counter)
    by_question_kind: Counter[str] = field(default_factory=Counter)
    by_level: Counter[str] = field(default_factory=Counter)
    by_chart_type: Counter[str] = field(default_factory=Counter)
    with_boxes: int = 0
    with_plan: int = 0
    with_compositional_plan: int = 0
    dedup_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "total": self.total,
                "by_source": dict(self.by_source),
                "by_question_kind": dict(self.by_question_kind),
                "by_level": dict(self.by_level),
                "by_chart_type": dict(self.by_chart_type),
                "with_boxes": self.with_boxes, "with_plan": self.with_plan,
                "with_compositional_plan": self.with_compositional_plan,
                "dedup": self.dedup_summary}


def is_compositional(plan: dict | None) -> bool:
    """A plan that teaches more than "read this cell".

    73.6% of plans mined from real ChartQA are bare `lookup` (`DECISIONS.md` 0046), and
    a lookup teaches the output format but nothing about typed expression trees. The
    distinction is tracked so a mixture cannot look plan-rich while being lookup-only.
    """
    if not plan:
        return False
    if plan.get("op") != "lookup":
        return True
    return any(isinstance(a, dict) for a in plan.get("args") or ())


def assert_train_only(records: list[ChartRecord], stage: str) -> None:
    """`PLAN.md` 3.7: zero validation or test records in either mixture."""
    offenders = [r for r in records if r.split != TRAIN_ONLY]
    if offenders:
        splits = Counter(r.split for r in offenders)
        raise LeakageError(
            f"{stage}: {len(offenders)} records are not from the training split "
            f"({dict(splits)}). The first is {offenders[0].record_id}. This is not "
            f"filtered automatically — something upstream produced it and that needs "
            f"fixing, not hiding."
        )


def _describe(records: list[ChartRecord], stage: str, dedup_summary: str
              ) -> MixtureComposition:
    comp = MixtureComposition(stage=stage, total=len(records), dedup_summary=dedup_summary)
    for r in records:
        comp.by_source[r.source] += 1
        comp.by_question_kind[r.question_kind] += 1
        comp.by_level[str(r.meta.get("level", "n/a"))] += 1
        comp.by_chart_type[str(r.meta.get("chart_type", "unknown"))] += 1
        comp.with_boxes += bool(r.boxes)
        comp.with_plan += bool(r.plan)
        comp.with_compositional_plan += is_compositional(r.plan)
    return comp


def build_stage1(synthetic: list[ChartRecord], real_boxes: list[ChartRecord], *,
                 cap: int = STAGE1_CAP) -> tuple[list[ChartRecord], MixtureComposition]:
    """Grounding only: synthetic ordered L1->L4, then audited real boxes.

    Deduplication runs *before* the cap, so the cap counts distinct examples rather than
    whatever survived a merge.
    """
    # Check the INPUTS, before any filtering. Checking the survivors would let a leaked
    # record slip through simply by being dropped for some other reason first.
    assert_train_only([*synthetic, *real_boxes], "stage1")
    # And check the IMAGES, not just the split labels: RefChartQA ships rows labelled
    # "train" that use ChartQA test charts (`DECISIONS.md` 0049).
    assert_no_held_out_images([*synthetic, *real_boxes], "stage1")

    unlevelled = [r for r in synthetic if r.meta.get("level") not in LEVEL_ORDER]
    if unlevelled:
        raise ValueError(
            f"stage1: {len(unlevelled)} synthetic records have no curriculum level "
            f"(first: {unlevelled[0].record_id}, level={unlevelled[0].meta.get('level')!r}). "
            f"Stage 1 is ordered L1->L4, so an unlevelled record has no position in it — "
            f"dropping it silently would shrink the mixture without saying so."
        )

    ordered: list[ChartRecord] = []
    for lvl in LEVEL_ORDER:
        ordered.extend(r for r in synthetic if r.meta.get("level") == lvl)
    ordered.extend(r for r in real_boxes if r.boxes)

    merged, report = deduplicate(ordered)
    out = merged[:cap]
    return out, _describe(out, "stage1", report.summary())


def build_stage2(records: list[ChartRecord], synthetic_replay: list[ChartRecord], *,
                 cap: int = STAGE2_CAP, replay: int = SYNTHETIC_REPLAY,
                 seed: int = 0) -> tuple[list[ChartRecord], MixtureComposition]:
    """Joint box + plan + answer, shuffled, with exact synthetic replay mixed in.

    Replay examples are the ones whose plan AND boxes are both exact by construction;
    they are what stops the model drifting away from emitting plans it can execute.
    """
    rng = random.Random(seed)
    pool = [*records, *synthetic_replay[:replay]]
    assert_train_only(pool, "stage2")     # the inputs, before anything is dropped
    assert_no_held_out_images(pool, "stage2")
    merged, report = deduplicate(pool)
    rng.shuffle(merged)
    out = merged[:cap]
    return out, _describe(out, "stage2", report.summary())


def write_mixture(records: list[ChartRecord], composition: MixtureComposition,
                  path: str | Path) -> MixtureComposition:
    """Write a mixture file: composition first, then the record ids it contains.

    Record **ids**, not images or questions — rule 7 forbids committing dataset content,
    and a mixture must stay reproducible without carrying any.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "composition": composition.to_dict(),
        "record_ids": [r.record_id for r in records],
        "keys": [r.key for r in records],
    }, indent=2) + "\n", encoding="utf-8")
    return composition


__all__ = ["LEVEL_ORDER", "STAGE1_CAP", "STAGE2_CAP", "SYNTHETIC_REPLAY", "LeakageError",
           "MixtureComposition", "assert_no_held_out_images", "assert_train_only",
           "build_stage1", "build_stage2",
           "is_compositional", "write_mixture"]
