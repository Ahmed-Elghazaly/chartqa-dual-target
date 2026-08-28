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


class FeedRefusedTooMuch(RuntimeError):
    """The feed is discarding so much of its mixture that the run is not the planned one."""


#: Refusals are checked once this many records have been offered — enough that the rate is
#: not noise, and early enough that a broken run dies in its first minute rather than its
#: tenth hour.
REFUSAL_CHECK_AFTER = 200
#: The fraction of offered records that must become examples. Measured yields after the
#: fixes in `DECISIONS.md` 0071-0073 are 99.5%; the 0.5% loss is examples over
#: `max_seq_len`. 0.90 leaves generous room for that while catching the failures that
#: actually occurred, which cost 23%, 38%, 46% and 100%.
MIN_USABLE_FRACTION = 0.90


class MixtureFeed:
    """An ordered, resumable stream of training examples over a list of records."""

    def __init__(self, records: Sequence[ChartRecord], *, shuffle: bool, seed: int = 0,
                 answer_only: bool = False, image_root: Path | None = None,
                 archive: Any = None) -> None:
        self.records = list(records)
        self.shuffle = shuffle
        self.seed = seed
        self.answer_only = answer_only
        self.image_root = Path(image_root) if image_root else None
        #: An `ArchiveReader`, for the ChartQA images that live inside the zip rather than
        #: on disk. Without it every ChartQA record is refused with `No such file or
        #: directory` — 23% of stage 1 and 38% of stage 2, counted and skipped, on a host
        #: where the archive was never extracted (`DECISIONS.md` 0073).
        self.archive = archive
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
        """The chart image, from disk if it is there and from the archive if it is not.

        ChartQA ships as a single zip and this project never extracts it — `ArchiveReader`
        reads members in place, which is why the mixtures could be built at all. But a
        record's `image_path` is the member name, which looks exactly like a relative disk
        path, so opening it directly succeeds on a host that happens to have extracted the
        archive and fails everywhere else. The failure is an `OSError`, which `_example`
        catches and counts as a refusal, so it costs records rather than raising.
        """
        import io

        from PIL import Image

        path = Path(record.image_path)
        if not path.is_absolute() and self.image_root is not None:
            path = self.image_root / path
        if path.exists():
            return Image.open(path).convert("RGB")
        if self.archive is not None and self.archive.exists(record.image_path):
            return Image.open(io.BytesIO(self.archive.read(record.image_path))).convert("RGB")
        raise FileNotFoundError(
            f"{record.image_path} is neither on disk nor in the archive"
            f"{'' if self.archive is not None else ' (no archive was supplied)'}")

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

    def check_refusal_rate(self) -> None:
        """Stop a run that is training on a fraction of its mixture.

        Four defects in this project have had the same shape: something the feed cannot
        turn into an example, caught by an `except`, counted, and skipped. Each produced a
        smaller training set rather than an error, so the run finished on schedule and
        reported its step count truthfully — 100% of stage-1 targets lost to a misspelled
        metadata key, 100% of the compositional level to an evidence-selection interaction,
        46% of a mixture to unusable records, 38% to chart images that were in a zip rather
        than on disk (`DECISIONS.md` 0071, 0072, 0073).

        `FeedStats` recorded every one of them. Recording is not enough: from outside, an
        `except` that counts and continues is indistinguishable from there being no
        failures. So the rate is now a gate.
        """
        if self.stats.offered < REFUSAL_CHECK_AFTER:
            return
        fraction = self.stats.usable / self.stats.offered
        if fraction >= MIN_USABLE_FRACTION:
            return
        raise FeedRefusedTooMuch(
            f"the feed turned only {self.stats.usable}/{self.stats.offered} "
            f"({100 * fraction:.1f}%) of offered records into examples, below the "
            f"{100 * MIN_USABLE_FRACTION:.0f}% floor. Training would proceed on a "
            f"fraction of the mixture and report its step count truthfully.\n"
            f"{self.stats.describe()}\n"
            f"Run `scripts/measure_target_yield.py --mixture <file> --tokens 200` to see "
            f"the same refusals without booking a GPU.")

    def batches(self, batch_size: int) -> Iterator[list[Example]]:
        """Yield batches forever, advancing `position` and rolling epochs."""
        pending: list[Example] = []
        checked = False
        while True:
            if self.position >= len(self._order):
                self.epoch += 1
                self.position = 0
                self._order = self._make_order()
            index = self._order[self.position]
            self.position += 1
            self.stats.offered += 1
            example = self._example(self.records[index])
            if not checked and self.stats.offered >= REFUSAL_CHECK_AFTER:
                checked = True
                self.check_refusal_rate()
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


__all__ = [
    "MIN_USABLE_FRACTION",
    "REFUSAL_CHECK_AFTER",
    "FeedRefusedTooMuch",
    "FeedStats",
    "MixtureFeed",
    "load_mixture_records",
]
