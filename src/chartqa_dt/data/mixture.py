from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chartqa_dt.data.chartqa import canonical_chart_type
from chartqa_dt.data.dedup import deduplicate
from chartqa_dt.data.records import ChartRecord
from chartqa_dt.synth.curriculum import LEVELS
from chartqa_dt.train.targets import TargetError, build_stage1_target, build_target

ABSENT_FROM_EVALUATION = frozenset({"area", "scatter"})


@dataclass
class MixtureComposition:
    stage: str
    total: int = 0
    by_source: Counter[str] = field(default_factory=Counter)
    by_question_kind: Counter[str] = field(default_factory=Counter)
    by_level: Counter[str] = field(default_factory=Counter)
    by_chart_type: Counter[str] = field(default_factory=Counter)
    by_source_chart_type: Counter[str] = field(default_factory=Counter)
    with_boxes: int = 0
    with_plan: int = 0
    with_compositional_plan: int = 0
    dedup_summary: str = ""

    @property
    def synthetic_share(self) -> float:
        return self.by_source.get("synthetic", 0) / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "total": self.total,
            "by_source": dict(self.by_source),
            "by_question_kind": dict(self.by_question_kind),
            "by_level": dict(self.by_level),
            "by_chart_type": dict(self.by_chart_type),
            "by_source_chart_type": dict(self.by_source_chart_type),
            "with_boxes": self.with_boxes,
            "with_plan": self.with_plan,
            "with_compositional_plan": self.with_compositional_plan,
            "synthetic_share": round(self.synthetic_share, 4),
            "dedup": self.dedup_summary,
        }


def is_compositional(plan: dict[str, Any] | None) -> bool:
    if not plan:
        return False
    if plan.get("op") != "lookup":
        return True
    return any(isinstance(arg, dict) and "op" in arg for arg in plan.get("args") or ())


def describe(records: list[ChartRecord], stage: str, dedup_summary: str) -> MixtureComposition:
    comp = MixtureComposition(stage=stage, total=len(records), dedup_summary=dedup_summary)
    for r in records:
        comp.by_source[r.source] += 1
        comp.by_question_kind[r.question_kind] += 1
        comp.by_level[str(r.meta.get("level", "n/a"))] += 1
        chart_type = canonical_chart_type(r.meta.get("chart_type"))
        comp.by_chart_type[chart_type] += 1
        comp.by_source_chart_type[f"{r.source}:{chart_type}"] += 1
        comp.with_boxes += bool(r.boxes)
        if stage != "stage1":
            comp.with_plan += bool(r.plan)
            comp.with_compositional_plan += is_compositional(r.plan)
    return comp


def drop_absent_chart_types(records: list[ChartRecord]) -> list[ChartRecord]:
    return [r for r in records
            if canonical_chart_type(r.meta.get("chart_type")) not in ABSENT_FROM_EVALUATION]


def split_by_stage_usability(records: list[ChartRecord], label: str
                             ) -> tuple[list[ChartRecord], list[ChartRecord]]:
    joint: list[ChartRecord] = []
    grounding: list[ChartRecord] = []
    for record in records:
        try:
            build_target(record)
        except TargetError:
            pass
        else:
            joint.append(record)
        try:
            build_stage1_target(record)
        except TargetError:
            pass
        else:
            grounding.append(record)
    print(f"  {label:<12}joint {len(joint):,}/{len(records):,}; "
          f"stage1 grounding {len(grounding):,}/{len(records):,}")
    return joint, grounding


def balance_by_level(records: list[ChartRecord], total: int, *, seed: int) -> list[ChartRecord]:
    if total == 0:
        return []
    if total > len(records):
        raise ValueError(f"synthetic sample requests {total} rows but only {len(records)} are available")
    rng = random.Random(seed)
    by_level: dict[str, list[ChartRecord]] = {}
    for record in records:
        by_level.setdefault(str(record.meta.get("level")), []).append(record)
    levels = sorted(by_level)
    for level in levels:
        by_level[level].sort(key=lambda record: (record.record_id, record.key))
        rng.shuffle(by_level[level])
    base, remainder = divmod(total, len(levels))
    allocation = {
        level: min(len(by_level[level]), base + (index < remainder)) for index, level in enumerate(levels)
    }
    left = total - sum(allocation.values())
    while left:
        for level in levels:
            if allocation[level] < len(by_level[level]):
                allocation[level] += 1
                left -= 1
                if not left:
                    break
    out: list[ChartRecord] = []
    for level in levels:
        out.extend(by_level[level][: allocation[level]])
    return out


def build_stage1(synthetic: list[ChartRecord], real: list[ChartRecord], *, cap: int
                 ) -> tuple[list[ChartRecord], MixtureComposition]:
    ordered: list[ChartRecord] = []
    for level in LEVELS:
        ordered.extend(r for r in synthetic if r.meta.get("level") == level)
    ordered.extend(r for r in real if r.boxes)
    merged, report = deduplicate(ordered)
    out = merged[:cap]
    return out, describe(out, "stage1", report.summary())


def build_stage2(real: list[ChartRecord], stage1_synthetic: list[ChartRecord], *, cap: int,
                 replay: int, seed: int) -> tuple[list[ChartRecord], MixtureComposition]:
    replay_candidates = sorted(stage1_synthetic, key=lambda record: (record.record_id, record.key))
    rng = random.Random(seed)
    chosen_replay = rng.sample(replay_candidates, replay)
    merged, report = deduplicate([*real, *chosen_replay])
    chosen_ids = {record.record_id for record in chosen_replay}
    kept_replay = [record for record in merged if record.record_id in chosen_ids]
    real_rows = sorted(
        (record for record in merged if record.record_id not in chosen_ids),
        key=lambda record: (record.record_id, record.key),
    )
    rng.shuffle(real_rows)
    out = [*real_rows[: cap - replay], *kept_replay]
    rng.shuffle(out)
    return out, describe(out, "stage2", report.summary())


def write_mixture(path: Path, records: list[ChartRecord], composition: MixtureComposition) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "composition": composition.to_dict(),
        "record_ids": [record.record_id for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_mixture_ids(path: Path) -> list[str]:
    ids: list[str] = json.loads(path.read_text(encoding="utf-8"))["record_ids"]
    return ids
