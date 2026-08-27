"""Resumable, hash-verified dataset downloads and the manifest that records them.

`PLAN.md` 3.1. Three properties are load-bearing:

* **Resumable.** ``hf_hub_download`` resumes a partial transfer and caches by revision,
  so an interrupted 2.88 GB fetch does not start over. On a free Kaggle session that is
  the difference between finishing and not.
* **Hash-verified.** Every archive's SHA-256 goes into `data/MANIFEST.json`. The point is
  not corruption — HTTPS mostly handles that — it is that a *silently different file*
  under the same name would make two of our own numbers incomparable with no code change
  to explain it. The manifest is committed; the data it describes never is (rule 7).
* **Pinned.** Revisions come from `data/sources.py`, which was cross-checked against
  `verification/measured_facts.json`.

`--dev` materialises a ~200-example subset from the parquet copies so every downstream
component can be built and tested without the full download. Dev mode deliberately
cannot supply ChartQA's gold tables — those exist only in the zip (`phase0.md` F6) — and
says so rather than producing something that looks complete.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chartqa_dt.data.sources import SOURCES, ArchiveSpec, ParquetSpec

MANIFEST_PATH = Path("data/MANIFEST.json")
DEV_ROWS = 200
_CHUNK = 1 << 20


class DownloadError(RuntimeError):
    """A download completed but did not match what was expected of it."""


def sha256_file(path: str | Path, *, chunk: int = _CHUNK) -> str:
    """Streaming SHA-256. Archives are gigabytes; reading one into memory is not an option."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class ArchiveResult:
    key: str
    path: Path
    size_bytes: int
    sha256: str
    revision: str
    cached: bool


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"_purpose": "SHA-256 of every dataset archive this project downloaded. "
                            "Committed so a changed file is detectable; the data itself "
                            "is never committed (rule 7).",
                "archives": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def save_manifest(manifest: dict[str, Any], path: str | Path = MANIFEST_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def record_archive(result: ArchiveResult, *, path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    """Add or confirm an archive in the manifest.

    A recorded hash is never silently overwritten: if the same key and revision now
    hashes differently, that is exactly the event the manifest exists to catch.
    """
    manifest = load_manifest(path)
    prior = manifest["archives"].get(result.key)
    if prior and prior["revision"] == result.revision and prior["sha256"] != result.sha256:
        raise DownloadError(
            f"{result.key}: SHA-256 changed at the same pinned revision "
            f"{result.revision[:12]}. Recorded {prior['sha256'][:16]}, now "
            f"{result.sha256[:16]}. Do not proceed — any number measured before and "
            f"after this change is not comparable."
        )
    manifest["archives"][result.key] = {
        "repo_id": SOURCES[result.key].repo_id,
        "filename": getattr(SOURCES[result.key], "filename", None),
        "revision": result.revision,
        "size_bytes": result.size_bytes,
        "sha256": result.sha256,
        "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    save_manifest(manifest, path)
    return manifest


def fetch_archive(spec: ArchiveSpec, *, data_root: Path,
                  manifest_path: str | Path = MANIFEST_PATH) -> ArchiveResult:
    """Download one archive, verify its size, hash it and record it."""
    from huggingface_hub import hf_hub_download

    from chartqa_dt.hub import get_token

    cache = Path(data_root) / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    before = _cached_path(spec, cache)

    local = hf_hub_download(
        repo_id=spec.repo_id,
        filename=spec.filename,
        revision=spec.revision,
        repo_type="dataset",
        cache_dir=str(cache),
        token=get_token(),
    )
    size = os.path.getsize(local)
    if spec.expected_bytes is not None and size != spec.expected_bytes:
        raise DownloadError(
            f"{spec.key}: expected {spec.expected_bytes:,} bytes, got {size:,}. "
            f"The pinned revision {spec.revision[:12]} should be byte-identical."
        )
    result = ArchiveResult(spec.key, Path(local), size, sha256_file(local),
                           spec.revision, cached=before is not None)
    record_archive(result, path=manifest_path)
    return result


def _cached_path(spec: ArchiveSpec, cache: Path) -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover - huggingface_hub is a hard dependency
        return None
    hit = try_to_load_from_cache(repo_id=spec.repo_id, filename=spec.filename,
                                 revision=spec.revision, repo_type="dataset",
                                 cache_dir=str(cache))
    return Path(hit) if isinstance(hit, str) else None


def verify_manifest(*, data_root: Path, path: str | Path = MANIFEST_PATH) -> dict[str, str]:
    """Re-hash everything the manifest claims and report per-archive status."""
    manifest = load_manifest(path)
    out: dict[str, str] = {}
    for key, entry in manifest["archives"].items():
        spec = SOURCES.get(key)
        if not isinstance(spec, ArchiveSpec):
            out[key] = "not an archive"
            continue
        local = _cached_path(spec, Path(data_root) / "hf")
        if local is None or not local.exists():
            out[key] = "absent"
        elif sha256_file(local) != entry["sha256"]:
            out[key] = "MISMATCH"
        else:
            out[key] = "ok"
    return out


def materialise_dev_subset(key: str, *, data_root: Path, rows: int = DEV_ROWS) -> Path:
    """Write a ~`rows`-example subset so downstream work needs no full download.

    Streamed, so this costs a few megabytes rather than the full archive. Images are
    written as files and referenced by path, matching what the full loaders produce, so
    nothing downstream has to branch on dev versus full.
    """
    from datasets import load_dataset

    spec = SOURCES[key]
    if not isinstance(spec, ParquetSpec):
        raise DownloadError(
            f"{key} has no parquet copy to stream. Its gold tables exist only in "
            f"{getattr(spec, 'filename', '?')} (phase0.md F6), so --dev cannot supply "
            f"them; run a full download for anything that needs tables."
        )
    out = Path(data_root) / "dev" / key
    (out / "images").mkdir(parents=True, exist_ok=True)

    written: list[dict[str, Any]] = []
    per_split = max(1, rows // len(spec.splits))
    for split in spec.splits:
        stream = load_dataset(spec.repo_id, split=split, streaming=True,
                              revision=spec.revision)
        for i, row in enumerate(stream):
            if i >= per_split:
                break
            image = row.pop("image", None)
            name = f"{split}_{i:05d}.png"
            if image is not None:
                image.save(out / "images" / name)
            row["_image_file"] = name
            row["_split"] = split
            written.append(_jsonable(row))

    (out / "rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in written) + "\n", encoding="utf-8")
    (out / "README.txt").write_text(
        f"Dev subset of {spec.repo_id} @ {spec.revision[:12]}: {len(written)} rows "
        f"across {sorted(spec.splits)}.\nStreamed, not the full dataset. "
        f"Not a substitute for it in any measurement.\n", encoding="utf-8")
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


__all__ = ["DEV_ROWS", "MANIFEST_PATH", "ArchiveResult", "DownloadError", "fetch_archive",
           "load_manifest", "materialise_dev_subset", "record_archive", "save_manifest",
           "sha256_file", "verify_manifest"]
