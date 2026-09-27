from __future__ import annotations

import io
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from chartqa_dt.data.records import ChartRecord
from chartqa_dt.train.collate import Example
from chartqa_dt.train.targets import TargetError, build_stage1_target, build_target


class MixtureFeed:
    def __init__(self, records: Sequence[ChartRecord], *, shuffle: bool, seed: int,
                 grounding_only: bool, image_root: Path, archive: Any) -> None:
        self.records = list(records)
        self.shuffle = shuffle
        self.seed = seed
        self.grounding_only = grounding_only
        self.image_root = image_root
        self.archive = archive
        self.refused = 0
        self.position = 0
        self.epoch = 0
        self._order = self._make_order()

    def _make_order(self) -> list[int]:
        order = list(range(len(self.records)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        return order

    def state_dict(self) -> dict[str, Any]:
        return {"position": self.position, "epoch": self.epoch, "refused": self.refused}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.position = state["position"]
        self.epoch = state["epoch"]
        self.refused = state.get("refused", 0)
        self._order = self._make_order()

    def image(self, record: ChartRecord) -> Any:
        from PIL import Image

        path = Path(record.image_path)
        if not path.is_absolute():
            path = self.image_root / path
        if path.exists():
            with Image.open(path) as source:
                return source.convert("RGB")
        if self.archive is not None and self.archive.exists(record.image_path):
            with Image.open(io.BytesIO(self.archive.read(record.image_path))) as source:
                return source.convert("RGB")
        raise FileNotFoundError(f"{record.image_path} is neither on disk nor in the archive")

    def example(self, record: ChartRecord) -> Example | None:
        try:
            target = build_stage1_target(record) if self.grounding_only else build_target(record)
            image = self.image(record)
        except (TargetError, OSError, ValueError):
            return None
        return Example(image=image, question=record.question, target=target)

    def batches(self, batch_size: int) -> Iterator[list[Example]]:
        pending: list[Example] = []
        while True:
            if self.position >= len(self._order):
                self.epoch += 1
                self.position = 0
                self._order = self._make_order()
            index = self._order[self.position]
            self.position += 1
            example = self.example(self.records[index])
            if example is None:
                self.refused += 1
                continue
            pending.append(example)
            if len(pending) == batch_size:
                yield pending
                pending = []
