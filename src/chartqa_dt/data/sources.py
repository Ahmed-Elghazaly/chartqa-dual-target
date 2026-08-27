"""What each dataset is, where it lives, and at which revision.

Every fact here was established in Phase 0 and is recorded in
`verification/measured_facts.json`; nothing is repeated from memory. Two of them shape
the whole download path:

* **ChartQA's gold tables are not in the parquet.** The HF viewer version exposes only
  ``imgname, query, label, type, image``. The data tables that Appendix E plan mining
  depends on live in ``ChartQA Dataset.zip`` in the same repo (`phase0.md` F6), so
  ChartQA is fetched as an archive, not through ``load_dataset``.
* **RefChartQA boxes are absolute pixels**, given as ``{x, y, w, h}`` — not normalised,
  not corner form (`phase0.md` F7). Converting them is the loader's job and the
  conversion is tested, because getting it wrong would silently poison every grounding
  number in the project.

Revisions are pinned. An unpinned dataset can change under a running experiment and make
two of our own numbers incomparable without any code changing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchiveSpec:
    """One downloadable file, pinned to a revision and checked against a size."""

    key: str
    repo_id: str
    filename: str
    revision: str
    expected_bytes: int | None = None
    note: str = ""


@dataclass(frozen=True)
class ParquetSpec:
    """A dataset read through ``datasets``, pinned to a revision."""

    key: str
    repo_id: str
    revision: str
    splits: dict[str, int]
    note: str = ""


#: `phase0.md` F6 — the zip, not the parquet: only the zip carries the gold tables.
CHARTQA_ARCHIVE = ArchiveSpec(
    key="chartqa",
    repo_id="ahmed-masry/ChartQA",
    filename="ChartQA Dataset.zip",
    revision="af8b6f5c08c95085271561c2a3f9d15f2b5a9031",
    expected_bytes=875_370_872,
    note="Gold data tables and chart-element annotations; needed for plan mining.",
)

#: `phase0.md` F7 — grounding_bboxes are absolute-pixel {x, y, w, h}.
REFCHARTQA_PARQUET = ParquetSpec(
    key="refchartqa",
    repo_id="omoured/RefChartQA",
    revision="c6b6504adb96cf72f0852a5f73ba4c62b718f843",
    splits={"train": 55_789, "validation": 6_223, "test": 11_690},
    note="Boxes are absolute pixels in {x, y, w, h} form.",
)

#: Only if the ChartQAPro extension is approved (`PLAN.md` 3.1). Not fetched by default.
CHARTQAPRO_PARQUET = ParquetSpec(
    key="chartqapro",
    repo_id="ahmed-masry/ChartQAPro",
    revision="e27c2874825874d6767d2bbc538ed4f0dc2c64c2",
    splits={"test": 1_948},
    note="Gated extension; test-only.",
)

CHARTQA_SPLIT_ROWS = {"train": 28_299, "val": 1_920, "test": 2_500}

SOURCES: dict[str, ArchiveSpec | ParquetSpec] = {
    CHARTQA_ARCHIVE.key: CHARTQA_ARCHIVE,
    REFCHARTQA_PARQUET.key: REFCHARTQA_PARQUET,
    CHARTQAPRO_PARQUET.key: CHARTQAPRO_PARQUET,
}

DEFAULT_SOURCES = ("chartqa", "refchartqa")

__all__ = ["CHARTQAPRO_PARQUET", "CHARTQA_ARCHIVE", "CHARTQA_SPLIT_ROWS", "DEFAULT_SOURCES",
           "REFCHARTQA_PARQUET", "SOURCES", "ArchiveSpec", "ParquetSpec"]
