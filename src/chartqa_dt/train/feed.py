"""Turning a mixture into training examples, in the order the stage requires.

`PLAN.md` 6.1 orders stage 1 easy→hard and 6.2 shuffles stage 2. That difference is the
curriculum, so it is a property of the feed rather than a flag someone remembers to set —
`shuffle=True` is the default in most dataloaders and would silently destroy stage 1.

The feed also carries its own **position**, because `PLAN.md` 6.3 requires the dataloader
position in every checkpoint. A resume that restarts the epoch trains on the first examples
twice and never reaches the last ones, and nothing about the loss curve would show it.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chartqa_dt.data.records import ChartRecord
from chartqa_dt.train.collate import Example
from chartqa_dt.train.targets import TargetError, build_answer_only_target, build_target


@dataclass
class FeedStats:
    """What the feed accepted and refused — a target that cannot be built is not silent."""

    offered: int = 0
    usable: int = 0
    refused: dict[str, int] = field(default_factory=dict)

    def note_refusal(self, error: Exception) -> None:
        text = str(error)
        key = ("no plan derivable" if "cannot be derived" in text
               else "plan does not round-trip" if "does not reproduce" in text
               else "references a missing box" if "references" in text
               else text.split(":")[-1].strip()[:48])
        self.refused[key] = self.refused.get(key, 0) + 1

    def describe(self) -> str:
        lines = [f"  usable examples : {self.usable}/{self.offered} "
                 f"({100 * self.usable / max(1, self.offered):.1f}%)"]
        for reason, n in sorted(self.refused.items(), key=lambda kv: -kv[1])[:6]:
            lines.append(f"    refused {n:>6}  {reason}")
        return "\n".join(lines)


class MixtureFeed:
    """An ordered, resumable stream of training examples over a list of records."""

    def __init__(self, records: Sequence[ChartRecord], *, shuffle: bool, seed: int = 0,
                 answer_only: bool = False, image_root: Path | None = None) -> None:
        self.records = list(records)
        self.shuffle = shuffle
        self.seed = seed
        self.answer_only = answer_only
        self.image_root = Path(image_root) if image_root else None
        self.stats = FeedStats()
        self.position = 0
        self.epoch = 0
        self._order = self._make_order()

    def _make_order(self) -> list[int]:
        order = list(range(len(self.records)))
        if self.shuffle:
            # Seeded per epoch, so a resume reproduces the same order it left.
            random.Random(self.seed + self.epoch).shuffle(order)
        return order

    def state_dict(self) -> dict[str, Any]:
        return {"position": self.position, "epoch": self.epoch, "seed": self.seed,
                "shuffle": self.shuffle, "n": len(self.records)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("n") != len(self.records):
            raise ValueError(
                f"checkpoint was taken over {state.get('n')} records, this feed has "
                f"{len(self.records)}. Resuming would train on a different mixture.")
        self.epoch = int(state.get("epoch", 0))
        self.position = int(state.get("position", 0))
        self.seed = int(state.get("seed", self.seed))
        self._order = self._make_order()

    def _image(self, record: ChartRecord) -> Any:
        from PIL import Image

        path = Path(record.image_path)
        if not path.is_absolute() and self.image_root is not None:
            path = self.image_root / path
        return Image.open(path).convert("RGB")

    def _example(self, record: ChartRecord) -> Example | None:
        try:
            target = (build_answer_only_target(record) if self.answer_only
                      else build_target(record))
        except TargetError as exc:
            self.stats.note_refusal(exc)
            return None
        try:
            image = self._image(record)
        except (OSError, ValueError) as exc:
            self.stats.note_refusal(exc)
            return None
        self.stats.usable += 1
        return Example(image=image, question=record.question, target=target)

    def batches(self, batch_size: int) -> Iterator[list[Example]]:
        """Yield batches forever, advancing `position` and rolling epochs."""
        pending: list[Example] = []
        while True:
            if self.position >= len(self._order):
                self.epoch += 1
                self.position = 0
                self._order = self._make_order()
            index = self._order[self.position]
            self.position += 1
            self.stats.offered += 1
            example = self._example(self.records[index])
            if example is None:
                continue
            pending.append(example)
            if len(pending) == batch_size:
                yield pending
                pending = []


def load_mixture_records(path: str | Path, records_by_id: dict[str, ChartRecord]
                         ) -> list[ChartRecord]:
    """Rehydrate a mixture file, which stores ids rather than content (rule 7)."""
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [i for i in data["record_ids"] if i not in records_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(data['record_ids'])} mixture ids are not in the "
            f"rebuilt record set (first: {missing[0]}). The mixture and the sources have "
            f"drifted; rebuild rather than training on a different set than was recorded.")
    return [records_by_id[i] for i in data["record_ids"]]


__all__ = ["FeedStats", "MixtureFeed", "load_mixture_records"]
