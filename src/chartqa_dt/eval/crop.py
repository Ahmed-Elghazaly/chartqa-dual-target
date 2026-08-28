"""The crop re-read — `PLAN.md` 8.2.

When the calibrator says a record is unreliable and its focus box looks valid, expand that
box by 15%, crop from the **original high-resolution** image, resize, and run the same model
once more. Boxes from the second pass are remapped to original coordinates by
`vision.coords.remap_crop_box_to_original`.

`PLAN.md` says outright that this is expected to fail: *"if the predicted box is slightly
wrong, cropping to it removes the answer from the image"*, and calls a clean negative result
a genuine contribution. So the accounting is built to make a negative visible rather than
recoverable:

* **help and harm are counted separately.** A net figure hides the case that matters — a
  technique that fixes twenty records and breaks twenty is not neutral, it is a coin flip
  applied to answers that were already right.
* **the acceptance rule is frozen before the numbers exist.** `AcceptanceRule` is
  constructed from the pre-registration and compared by value, so a rule tuned after seeing
  the help rate is a different object and says so.
* **never a third pass.** `CropBudget` allows one re-read per record and raises on a second,
  rather than trusting a loop not to iterate.

The crop is *offered* only when the reliability is low. Cropping a record the system already
got right can only cost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from chartqa_dt.vision.coords import norm1000_to_px

#: `PLAN.md` 8.2. A box that is exactly right still needs margin: the answer's text often
#: sits just outside the element it labels.
CROP_EXPANSION = 0.15
#: A crop smaller than this fraction of the original is almost certainly a bad box, and
#: cropping to it removes the rest of the chart — including the axis the value is read from.
MIN_CROP_FRACTION = 0.02


class ThirdPassRefused(RuntimeError):
    """`PLAN.md` 8.2: never a third pass."""


@dataclass(frozen=True)
class AcceptanceRule:
    """When a re-read may replace the first answer. Frozen on validation, before Phase 7.

    Compared by value, so a report can assert that the rule it used is the rule that was
    pre-registered rather than one tuned after the help rate was visible.
    """

    max_reliability: float = 0.5
    require_schema_valid: bool = True
    require_executor_agreement: bool = True
    require_boxes_inside_crop: bool = True

    def accepts(self, second: dict[str, Any]) -> bool:
        if self.require_schema_valid and not second.get("schema_valid"):
            return False
        if self.require_executor_agreement and not second.get("agrees"):
            return False
        return not (self.require_boxes_inside_crop and not second.get("boxes_in_crop", True))


@dataclass
class CropBudget:
    """One re-read per record. Structural, not a convention."""

    used: set[str] = field(default_factory=set)

    def take(self, record_id: str) -> None:
        if record_id in self.used:
            raise ThirdPassRefused(
                f"{record_id} has already been re-read once. `PLAN.md` 8.2 allows a second "
                f"pass and forbids a third; a loop that re-crops until it likes the answer "
                f"is search over the test set.")
        self.used.add(record_id)


def expand_box(box: Sequence[float], *, fraction: float = CROP_EXPANSION
               ) -> tuple[float, float, float, float]:
    """Grow a 0–1000 box by `fraction` on each side, clamped to the image."""
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    dx, dy = (x2 - x1) * fraction / 2, (y2 - y1) * fraction / 2
    return (max(0.0, x1 - dx), max(0.0, y1 - dy),
            min(1000.0, x2 + dx), min(1000.0, y2 + dy))


def crop_region(box: Sequence[float], width: int, height: int, *,
                fraction: float = CROP_EXPANSION) -> tuple[int, int, int, int] | None:
    """The pixel box to crop from the **original** image, or None if it is unusable."""
    expanded = expand_box(box, fraction=fraction)
    x1, y1, x2, y2 = norm1000_to_px(expanded, width, height)
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, round(x2)), min(height, round(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    if ((x2 - x1) * (y2 - y1)) / float(width * height) < MIN_CROP_FRACTION:
        return None
    return (x1, y1, x2, y2)


def should_crop(item: dict[str, Any], rule: AcceptanceRule) -> bool:
    """Whether to offer a re-read at all. Only unreliable records are worth the risk."""
    reliability = item.get("reliability")
    if reliability is None or reliability > rule.max_reliability:
        return False
    return bool(item.get("focus_box"))


@dataclass
class CropOutcome:
    requested: int = 0
    unusable_box: int = 0
    accepted: int = 0
    rejected: int = 0
    helped: int = 0
    harmed: int = 0
    unchanged: int = 0

    def _rate(self, n: int, d: int) -> float:
        return n / d if d else 0.0

    def to_dict(self, total: int) -> dict[str, Any]:
        return {
            "n": total,
            "request_rate": self._rate(self.requested, total),
            "unusable_box_rate": self._rate(self.unusable_box, self.requested),
            "accepted_rate": self._rate(self.accepted, self.requested),
            # Help and harm are never netted. A technique that fixes twenty records and
            # breaks twenty is a coin flip applied to answers that were already right.
            "help_rate": self._rate(self.helped, self.accepted),
            "harm_rate": self._rate(self.harmed, self.accepted),
            "net_records": self.helped - self.harmed,
            **asdict(self),
        }

    def describe(self, total: int) -> str:
        d = self.to_dict(total)
        return (f"  crop requested {self.requested}/{total} "
                f"({100 * d['request_rate']:.1f}%), accepted {self.accepted} "
                f"({100 * d['accepted_rate']:.1f}% of requests)\n"
                f"    helped {self.helped}   harmed {self.harmed}   "
                f"unchanged {self.unchanged}   net {d['net_records']:+d}")


def run_crop_pass(items: Sequence[dict[str, Any]], reread: Any, *,
                  rule: AcceptanceRule | None = None,
                  budget: CropBudget | None = None) -> tuple[list[dict[str, Any]], CropOutcome]:
    """Apply the crop policy over a scored prediction set.

    `reread(item, region)` performs the second pass and returns a dict with at least
    `correct`, `schema_valid`, `agrees`, and optionally `boxes_in_crop`. Injecting it keeps
    the policy testable without a GPU — and the policy, not the generation, is where this
    ablation goes wrong.
    """
    rule = rule or AcceptanceRule()
    budget = budget or CropBudget()
    outcome = CropOutcome()
    final: list[dict[str, Any]] = []

    for item in items:
        result = dict(item)
        if not should_crop(item, rule):
            final.append(result)
            continue
        outcome.requested += 1
        size = item.get("image_size") or (0, 0)
        region = crop_region(item["focus_box"], int(size[0]), int(size[1]))
        if region is None:
            outcome.unusable_box += 1
            final.append(result)
            continue

        budget.take(str(item.get("id", id(item))))
        second = reread(item, region) or {}
        if not rule.accepts(second):
            outcome.rejected += 1
            final.append(result)
            continue

        outcome.accepted += 1
        was, now = bool(item.get("correct")), bool(second.get("correct"))
        outcome.helped += (now and not was)
        outcome.harmed += (was and not now)
        outcome.unchanged += (was == now)
        result.update(second)
        result["crop_region"] = region
        final.append(result)

    return final, outcome


__all__ = ["CROP_EXPANSION", "MIN_CROP_FRACTION", "AcceptanceRule", "CropBudget",
           "CropOutcome", "ThirdPassRefused", "crop_region", "expand_box",
           "run_crop_pass", "should_crop"]
