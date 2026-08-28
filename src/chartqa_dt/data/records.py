"""The one record type every data source is normalised into.

`PLAN.md` 3.2 specifies this. Having a single shape means the mixture builder,
the deduplicator, the leakage test and the training collator all speak one
language, and a new source is a new loader rather than a new special case.

`record_id` is deterministic — derived from the source, split and content — so the
same input produces the same id on every machine and in every run. That is what
makes a mixture file comparable across sessions.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Source = Literal["chartqa", "refchartqa", "synthetic", "chartqapro"]
Split = Literal["train", "val", "test"]
QuestionKind = Literal["human", "machine", "pot", "synthetic"]


def image_content_sha256(source: Any) -> str:
    """SHA-256 of an image's DECODED PIXELS, not its file bytes.

    This distinction is what makes cross-dataset deduplication work at all. RefChartQA
    is derived partly from ChartQA, but its images travel through parquet and come back
    re-encoded: **0 of 4,000** cached RefChartQA training images match a ChartQA image by
    file-byte hash, while matches appear immediately once the comparison is on pixels.
    Keying `dedup_key` on file bytes would therefore have found zero duplicates and
    reported a clean merge — the exact silent double-counting `PLAN.md` 3.3 exists to
    prevent, and it would have looked like success.

    Accepts a path, raw bytes, or an already-open PIL image.
    """
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
    """Canonical question text, for deduplication and leakage checks.

    Verbatim from `PLAN.md` 3.3. NFKC folds unicode variants, case is dropped,
    runs of whitespace collapse, and trailing punctuation is stripped — so
    "What is the median value?" and "what is the median value" are one question.
    """
    q = unicodedata.normalize("NFKC", q).strip().lower()
    q = re.sub(r"\s+", " ", q)
    return q.rstrip(" ?.!").strip()


def dedup_key(image_sha256: str, question: str) -> str:
    """Identity of a (chart, question) pair.

    Verbatim from `PLAN.md` 3.3, and the image half is load-bearing. Measured in
    `DECISIONS.md` 0028: generic questions such as "what is the median value"
    appear on many different charts — one of them on three separate ChartQA test
    charts — so a key on question text alone produces false duplicates and would
    have flagged phantom leakage.
    """
    qh = hashlib.sha256(normalise_question(question).encode("utf-8")).hexdigest()[:16]
    return f"{image_sha256[:16]}:{qh}"


ELEMENTS_KEY = "elements"


@dataclass(frozen=True)
class ChartRecord:
    """One (chart, question) example, whatever it came from."""

    record_id: str
    source: Source
    split: Split
    image_path: str
    image_sha256: str        # of the DECODED PIXELS — see `image_content_sha256`
    question: str
    answer: str | None
    question_kind: QuestionKind
    table: dict | None = None
    boxes: list[list[float]] | None = None       # 0-1000 normalised [x1,y1,x2,y2]
    plan: dict | None = None                      # typed tree, only when known exactly
    meta: dict[str, Any] = field(default_factory=dict)
    #: The `meta` key holding per-element label/value/unit/bbox dictionaries. Named here
    #: because `build_target` joins the plan's labels against it, and a source that spells
    #: it differently produces records that look complete and refuse silently: the
    #: synthetic reader wrote `evidence` and all 12,000 stage-1 targets were lost
    #: (`DECISIONS.md` 0071). Both readers and the target builder use this constant.

    @property
    def key(self) -> str:
        return dedup_key(self.image_sha256, self.question)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChartRecord:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def make_record_id(source: str, split: str, image_sha256: str, question: str,
                   index: int | None = None) -> str:
    """Deterministic id: same input, same id, on any machine.

    ``index`` disambiguates the genuinely rare case of one chart carrying two
    questions that normalise identically — which does occur in ChartQA.
    """
    h = hashlib.sha256(
        f"{source}|{split}|{image_sha256}|{normalise_question(question)}|{index if index is not None else ''}"
        .encode()
    ).hexdigest()[:16]
    return f"{source}_{split}_{h}"


def write_jsonl(records: list[ChartRecord], path: str) -> int:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    return len(records)


def read_jsonl(path: str) -> list[ChartRecord]:
    out: list[ChartRecord] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(ChartRecord.from_dict(json.loads(line)))
    return out
