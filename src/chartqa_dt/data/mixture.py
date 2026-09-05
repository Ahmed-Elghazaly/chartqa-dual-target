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

#: **Where 12,000 comes from.** It is not a round number chosen for tidiness, and it is not
#: a property of the data — it is the compute budget, arrived at backwards:
#:
#:     12,000 records x 1 epoch / effective batch 8 (batch 2 x grad_accum 4) = 1,500 steps
#:     x 2 stages                                                            = 3,000 steps
#:     x 11.903 s/step measured at 512px (`DECISIONS.md` 0060)               = 9.92 hours
#:     against a Kaggle session limit of                                       10 hours
#:
#: The reasoning was invisible until `DECISIONS.md` 0092 went looking: it lives across four
#: constants in three files and a measured step time in a decision record, and nothing said
#: they were connected. Changing any one of them silently changes what fits in a session.
#:
#: **The constraint that set it has since been lifted** — three Kaggle accounts give ~90 h a
#: week, and `train/checkpoint.py` resumes across sessions with the resume verified against
#: an uninterrupted run. So this is now a *choice* rather than a ceiling. It has not been
#: raised, because more supervision is gated on mining and whether more data helps at all is
#: what the deferred scaling ladder exists to answer (0092).
#: **No cap.** Ahmed: *"why r we even putting caps on training why not train on all the
#: data we have"* — and he is right that the reason had expired. 12,000 was a *compute*
#: budget from when a single Kaggle account's 30 GPU-hours was the binding constraint;
#: there are now three accounts, ~90 hours a week, against ~19 committed.
#:
#: `cli.train.steps_for` makes stage 1 **one pass** over its mixture (`PLAN.md` 6.1), so
#: removing the cap does not mean training longer on the same data — it means seeing all
#: of the data once instead of a third of it. That is the principled version of what the
#: cap was approximating (`DECISIONS.md` 0142).
#:
#: Kept as a number rather than `None` so `build_stage1`'s slice needs no special case, and
#: so a run can still be capped from the command line when someone wants a short one.
STAGE1_CAP = 1_000_000
#: The same number as stage 1, and for the same reason: two stages of 12,000 at effective
#: batch 8 is 3,000 optimizer steps, which is the compute budget (see `STAGE1_CAP`). It is
#: not independently motivated -- if the ladder raises one, it should raise both, or the two
#: stages stop being comparable on compute.
STAGE2_CAP = 1_000_000
#: How much synthetic data is replayed into stage 2, to stop the model forgetting the output
#: format while it learns the task.
#:
#: **This ratio is not measured**, and the sentence that used to stand here -- *"2,000 of a
#: 12,000 mixture is one sixth"* -- was also **wrong in practice**. It assumed stage 2 fills
#: `STAGE2_CAP`; it does not, because the real supply is smaller than the cap. A fixed count
#: against a variable pool is a fixed count, not a ratio, and the ratio is what matters:
#:
#: | real records in stage 2 | mixture | replay kept | share |
#: |---:|---:|---:|---:|
#: | 2,264 *(built today)* | 4,264 | 2,000 | **46.9%** |
#: | 10,000 *(the documented case)* | 12,000 | 2,000 | 16.7% |
#: | 48,000 *(if the ladder fills stage 2)* | 12,000 | 480 | 4.0% |
#:
#: So nearly half of stage 2 is synthetic today, in the stage whose job is to teach plans on
#: **real** charts, and the same constant gives 4% once the ladder runs. The share swings 12x
#: across plausible supply and nothing announced it (`DECISIONS.md` 0117).
#:
#: The value is unchanged because changing it needs the experiment nobody has run: if format
#: validity collapses in stage 2 it is too low; if stage-2 accuracy lags the control it may
#: be too high. Both are visible in the Phase 6 numbers. What *has* changed is that the
#: realised share is now printed with every mixture, so it cannot drift silently again.
SYNTHETIC_REPLAY = 2_000
TRAIN_ONLY = "train"

#: How much of each source is drawn when records are rebuilt. **These belong here, not at
#: each call site.** A mixture holds record ids only (rule 7), so training rehydrates every
#: record from the sources; if it rehydrates a *smaller* pool than the mixture was built
#: from, the ids at the tail resolve to nothing. `load_mixture_records` refuses on that
#: drift — loudly, but on the GPU, an hour into a run. `DECISIONS.md` 0072 raised the
#: ChartQA draw to the whole split because only 10.5% of it yields a training target.
CHARTQA_DRAW = 30_000        # per question kind; the split has 7,398 human + 20,901 machine
#: How many RefChartQA rows a mixture may draw. **This is a starting point and `PLAN.md` 3.4
#: says what ends it**: a scaling ladder at 4,000 / 10,000 / 25,000 rows, measuring validation
#: grounding at each and keeping the point where the curve flattens.
#:
#: It stayed at the start for a month while the ladder went unrun, and meanwhile the *cache*
#: — a separate `--cap` on `scripts/cache_refchartqa.py`, also 4,000 — held 3,996 of the
#: split's 55,789 rows, so the project trained on **7.2%** of the dataset (`DECISIONS.md`
#: 0112). The cache is being filled; this number moves only when the ladder says where to.
#:
#: 4,000 was also chosen when only *single-box* records were usable, which was 52% of
#: RefChartQA. `build_grounding_only_target` raised that to 98.5% (0104), so the same number
#: now discards a much larger share of a much larger pool.
#: **No cap**, for the same reason as the stage caps (0142). This began as rung 1 of
#: `PLAN.md` 3.4's scaling ladder — 4,000 / 10,000 / 25,000, measuring grounding at each and
#: keeping where the curve flattens — and 0115 found the ladder could not have been run at
#: all, because the *cache* also held 3,996 rows.
#:
#: The ladder is a **measurement**, not an improvement: it establishes how much data is
#: enough, which is a different goal from getting the best model. Ahmed's priority is
#: explicit — *"I mainly care for improvement over baselines and published numbers"* — and
#: he does not want runs that do not improve the result. So the whole cache goes in, and if
#: someone later wants the curve, the rungs are still `--refchartqa-cap 4000|10000|25000`.
REFCHARTQA_CAP = 1_000_000

#: Chart families the generator draws that **ChartQA does not contain**. Measured over 3,000
#: real train charts: bar 83.6%, line 12.8%, pie 3.6%, and area and scatter exactly 0.0%
#: (`DECISIONS.md` 0091). They are a quarter of the 24,000 synthetic examples, so stage 1 was
#: spending a quarter of its budget teaching the model to ground chart types it will never be
#: asked about.
#:
#: Excluded at composition time rather than removed from the generator: the examples cost
#: nothing where they sit, they are the honest thing to train on if the evaluation corpus ever
#: changes, and regenerating 24,000 charts to drop 6,000 would spend hours to achieve a
#: different `if`.
ABSENT_FROM_EVALUATION = frozenset({"area", "scatter"})


def drop_absent_chart_types(records: list[ChartRecord]) -> tuple[list[ChartRecord], int]:
    """Keep only chart families the evaluation corpus actually contains.

    Returns the survivors and how many went, because a filter that shrinks a mixture without
    saying so is how 12,000 stage-1 records were silently lost once already (0071).
    """
    kept = [r for r in records
            if str(r.meta.get("chart_type", "")).lower() not in ABSENT_FROM_EVALUATION]
    return kept, len(records) - len(kept)


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

    @property
    def synthetic_share(self) -> float:
        """What fraction of this mixture is synthetic.

        Derived, never stored, because the thing it guards against is precisely a stored
        number drifting from the mixture it describes. In stage 2 this is the realised
        replay ratio, which `SYNTHETIC_REPLAY` sets only indirectly: it is a count, and
        the share depends on how much real data there happened to be (0117).
        """
        return self.by_source.get("synthetic", 0) / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "total": self.total,
                "by_source": dict(self.by_source),
                "by_question_kind": dict(self.by_question_kind),
                "by_level": dict(self.by_level),
                "by_chart_type": dict(self.by_chart_type),
                "with_boxes": self.with_boxes, "with_plan": self.with_plan,
                "with_compositional_plan": self.with_compositional_plan,
                "synthetic_share": round(self.synthetic_share, 4),
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
