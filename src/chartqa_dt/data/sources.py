from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArchiveSpec:
    repo_id: str
    filename: str
    revision: str


@dataclass(frozen=True)
class ParquetSpec:
    repo_id: str
    revision: str
    shards: dict[str, tuple[str, ...]]


CHARTQA_ARCHIVE = ArchiveSpec(
    repo_id="ahmed-masry/ChartQA",
    filename="ChartQA Dataset.zip",
    revision="af8b6f5c08c95085271561c2a3f9d15f2b5a9031",
)

REFCHARTQA_PARQUET = ParquetSpec(
    repo_id="omoured/RefChartQA",
    revision="c6b6504adb96cf72f0852a5f73ba4c62b718f843",
    shards={
        "train": tuple(f"data/train-0000{i}-of-00006.parquet" for i in range(6)),
        "validation": ("data/validation-00000-of-00001.parquet",),
        "test": tuple(f"data/test-0000{i}-of-00002.parquet" for i in range(2)),
    },
)


def _upstream_split(split: str) -> str:
    return "validation" if split in {"val", "valid", "dev"} else split


def download_chartqa(data_root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    cache = data_root / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(
        repo_id=CHARTQA_ARCHIVE.repo_id,
        filename=CHARTQA_ARCHIVE.filename,
        revision=CHARTQA_ARCHIVE.revision,
        repo_type="dataset",
        cache_dir=str(cache),
    ))


def chartqa_archive_path(data_root: Path) -> Path:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(repo_id=CHARTQA_ARCHIVE.repo_id,
                                 filename=CHARTQA_ARCHIVE.filename,
                                 revision=CHARTQA_ARCHIVE.revision,
                                 repo_type="dataset", cache_dir=str(data_root / "hf"))
    if not isinstance(hit, str):
        raise FileNotFoundError("ChartQA archive is not downloaded; run scripts/download_data.py")
    return Path(hit)


def download_refchartqa(split: str, cache_root: Path) -> list[Path]:
    from huggingface_hub import hf_hub_download

    hub = cache_root / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    return [
        Path(hf_hub_download(
            repo_id=REFCHARTQA_PARQUET.repo_id,
            filename=name,
            revision=REFCHARTQA_PARQUET.revision,
            repo_type="dataset",
            cache_dir=str(hub),
        ))
        for name in REFCHARTQA_PARQUET.shards[_upstream_split(split)]
    ]


def load_refchartqa(split: str, cache_root: Path) -> Any:
    from datasets import load_dataset
    from huggingface_hub import try_to_load_from_cache

    upstream = _upstream_split(split)
    paths = []
    for name in sorted(REFCHARTQA_PARQUET.shards[upstream]):
        hit = try_to_load_from_cache(repo_id=REFCHARTQA_PARQUET.repo_id, filename=name,
                                     revision=REFCHARTQA_PARQUET.revision,
                                     repo_type="dataset", cache_dir=str(cache_root / "hub"))
        if not isinstance(hit, str):
            raise FileNotFoundError(
                f"RefChartQA {upstream} shard {name} is not downloaded; run scripts/download_data.py"
            )
        paths.append(str(Path(hit).resolve()))
    return load_dataset(
        "parquet",
        data_files={upstream: paths},
        split=upstream,
        streaming=True,
        cache_dir=str(cache_root.resolve() / "datasets"),
    )
