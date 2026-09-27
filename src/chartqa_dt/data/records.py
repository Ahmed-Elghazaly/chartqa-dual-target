from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from chartqa_dt.plans.executor import SERIES_SEPARATOR

Source = Literal["chartqa", "refchartqa", "synthetic"]
Split = Literal["train", "val", "test"]
QuestionKind = Literal["human", "machine", "pot", "synthetic"]

ELEMENTS_KEY = "elements"


def image_content_sha256(source: Any) -> str:
    import io

    import numpy as np
    from PIL import Image

    if isinstance(source, (bytes, bytearray)):
        handle: Any = io.BytesIO(source)
    elif hasattr(source, "convert"):
        handle = None
    else:
        handle = source

    image = source if handle is None else Image.open(handle)
    array = np.asarray(image.convert("RGB"))
    digest = hashlib.sha256()
    digest.update(f"{array.shape[1]}x{array.shape[0]}:".encode())
    digest.update(array.tobytes())
    if handle is not None:
        image.close()
    return digest.hexdigest()


def normalise_question(q: str) -> str:
    q = unicodedata.normalize("NFKC", q).strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q.rstrip(" ?.!").strip()


def dedup_key(image_sha256: str, question: str) -> str:
    qh = hashlib.sha256(normalise_question(question).encode("utf-8")).hexdigest()[:16]
    return f"{image_sha256[:16]}:{qh}"


def make_record_id(source: str, split: str, image_sha256: str, question: str,
                   source_row_index: int) -> str:
    h = hashlib.sha256(
        f"{source}|{split}|{source_row_index}|{image_sha256}|{normalise_question(question)}"
        .encode()
    ).hexdigest()[:16]
    return f"{source}_{split}_{h}"


@dataclass(frozen=True)
class ChartRecord:
    schema_version: Literal[2] = field(default=2, kw_only=True)
    source_row_index: int = field(default=0, kw_only=True)
    record_id: str
    source: Source
    split: Split
    image_path: str
    image_sha256: str
    question: str
    answer: str | None
    question_kind: QuestionKind
    table: dict[str, Any] | None = None
    boxes: list[list[float]] | None = None
    plan: dict[str, Any] | None = None
    elements: list[dict[str, Any]] | None = None
    evidence: list[int] | None = None
    source_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return dedup_key(self.image_sha256, self.question)

    @property
    def evidence_elements(self) -> list[dict[str, Any]] | None:
        if self.evidence is None or self.elements is None:
            return None
        return [self.elements[i] for i in self.evidence]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChartRecord:
        return cls(**d)


_CONFUSABLES = str.maketrans({"А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
                              "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
                              "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x"})


def fold_for_matching(text: Any) -> str:
    return " ".join(str(text).translate(_CONFUSABLES).lower().split())


def qualified_labels(elements: Sequence[Mapping[str, Any]]) -> list[str]:
    labels = [str(e.get("label")) for e in elements]
    collides = {label for label, n in Counter(labels).items() if n > 1}
    out = []
    for element, label in zip(elements, labels):
        series = element.get("series")
        out.append(f"{series}{SERIES_SEPARATOR}{label}"
                   if label in collides and series else label)
    return out
