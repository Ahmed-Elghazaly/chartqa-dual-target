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
from collections import Counter
from collections.abc import Mapping, Sequence
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

#: Where an element's **box** came from, and therefore how much to trust it.
#: `Prompt.md` Idea 15 asks for exactly this, and asks that it be used for filtering,
#: weighting, curriculum, debugging, auditing, ablations and reproducibility — **not**
#: exposed to the model. It is carried per element rather than per record because a
#: RefChartQA record can hold aligned and unaligned boxes at once (`DECISIONS.md` 0126).
GROUNDING_PROVENANCE = {
    #: Drawn by our own generator and checked against the rendered pixels. Exact.
    "synthetic_exact",
    #: RefChartQA's own per-question grounding annotation, as published.
    "refchartqa_gold",
    #: A RefChartQA box matched to a ChartQA element, so it also carries a label and a
    #: value. `match_iou` and `match_margin` on the element say how good the match was
    #: (measured: 98.9% at IoU >= 0.9, median 1.000 — 0077).
    "refchartqa_aligned",
    #: ChartQA's chart annotation: every element of the image, not question-specific.
    "chartqa_annotation",
}

#: Where an element's **value** came from. Separate from the box, because the two can
#: disagree — and did: reading values from the annotation instead of the gold table made
#: 35 of 105 planned records contradict their own answer (`targets._evidence_from`).
VALUE_PROVENANCE = {
    "synthetic_generated",   # the generator chose it; exact by construction
    "chartqa_table",         # the gold data table for that chart
    "chartqa_annotation",    # the annotated element's own value
    "derived",               # inferred, e.g. a single-box value set from the answer
    "unknown",               # a box with no identity yet — an unaligned RefChartQA row
}



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
    #: **Every semantic object the chart draws** — one dict per mark, with `label`,
    #: `value`, `unit` and `bbox`. A first-class field rather than a `meta` key, because
    #: as a `meta` key it meant two different things on two sources and nothing said so
    #: (`DECISIONS.md` 0098).
    elements: list[dict[str, Any]] | None = None
    #: **Which of those elements answer *this* question**, as indices into `elements` —
    #: or `None` for *"unknown"*.
    #:
    #: The distinction this field exists to carry, and which four defects came from not
    #: carrying it (0067, 0071, 0098, 0116):
    #:
    #: * **ChartQA** annotates the *chart*, not the question. Its elements are the same
    #:   for every question asked about that image, so `evidence` is `None`: nothing in
    #:   the record knows which subset answers it, and only a plan can select one.
    #: * **RefChartQA** marks, per question, the regions a person used. Every element is
    #:   evidence, so `evidence` lists them all.
    #: * **synthetic** knows exactly which marks its plan needs, and now keeps the *rest
    #:   of the chart* as well — information the old representation discarded.
    #:
    #: `None` means unknown and is the safe default: a consumer that needs question-level
    #: grounding must refuse rather than assume the whole chart is the answer.
    evidence: list[int] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return dedup_key(self.image_sha256, self.question)

    @property
    def evidence_elements(self) -> list[dict[str, Any]] | None:
        """The elements that answer this question, or `None` when that is not known."""
        if self.evidence is None or self.elements is None:
            return None
        return [self.elements[i] for i in self.evidence
                if 0 <= i < len(self.elements)]

    @property
    def has_question_evidence(self) -> bool:
        """Whether this record knows which marks answer its own question.

        Replaces the `question_specific_boxes` meta flag from 0119: that put a fact about
        the boxes *beside* the boxes, where this makes the record say what it holds.
        """
        return self.evidence is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChartRecord:
        known = set(cls.__dataclass_fields__)
        kept = {k: v for k, v in d.items() if k in known}
        # Records cached before `elements` became a field carry it under `meta`. Lifting
        # it here means an old cache stays readable instead of silently producing records
        # with no elements at all — which is exactly the shape of defect 0071.
        if kept.get("elements") is None:
            legacy = (d.get("meta") or {}).get(ELEMENTS_KEY)
            if legacy:
                kept["elements"] = legacy
        return cls(**kept)


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


#: Separator between a series name and a label in a qualified element name. A middle dot
#: with spaces: measured over 800 colliding ChartQA charts, **no** existing label contains
#: it, so qualifying cannot collide with a real label. Defined in `plans.executor`, which
#: has to parse it back out for `within`, and imported here so the two cannot drift.
from chartqa_dt.plans.executor import SERIES_SEPARATOR  # noqa: E402

#: Cyrillic letters that ChartQA annotations use where Latin is meant. Real, and invisible:
#: `'Оррose'` (Cyrillic О, р, р) and `'Oppose'` render identically and compare unequal.
#: Only used to match a series name to a table column, never to rewrite a label.
_CONFUSABLES = str.maketrans({"А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
                              "О": "O", "Р": "P", "С": "C", "Т": "T", "Х": "X",
                              "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x"})


def fold_for_matching(text: Any) -> str:
    """A form in which two spellings of the same series name compare equal.

    Only for joining an annotation's series to a table's column. 14.4% of colliding charts
    spell them differently — Cyrillic homoglyphs, a stray leading letter, scrambled word
    order — and none of that is worth failing a join over.
    """
    return " ".join(str(text).translate(_CONFUSABLES).lower().split())


def qualified_labels(elements: Sequence[Mapping[str, Any]]) -> list[str]:
    """One name per element, unique within the chart wherever the annotation allows it.

    On a grouped chart `"2019"` names one bar per series, and the two sides of our own
    contract disagreed about which one it meant: `train.targets` kept the FIRST element with
    a label and `plans.executor` kept the LAST. A plan saying `lookup("2019")` therefore
    pointed at one bar and stated another's number.

    The annotation already carries the discriminator — `chartqa.py::_series_elements` writes
    `"series"` on every element — and nothing downstream read it. Measured over 3,000 charts
    sampled at random: a label collides on **22.6%**, every one of those has a series name,
    and **(series, label) is unique on 94.4%** of them (`AUDIT.md` H3).

    Only colliding labels are qualified, so 77.4% of charts are untouched and their labels
    stay exactly the chart's own text. The remaining 5.6% cannot be separated even with the
    series; this returns the duplicates as they are and the caller refuses, rather than
    silently picking one.
    """
    labels = [str(e.get("label")) for e in elements]
    collides = {label for label, n in Counter(labels).items() if n > 1}
    out = []
    for element, label in zip(elements, labels):
        series = element.get("series")
        out.append(f"{series}{SERIES_SEPARATOR}{label}"
                   if label in collides and series else label)
    return out
