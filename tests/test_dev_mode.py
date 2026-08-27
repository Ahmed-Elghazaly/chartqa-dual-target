"""`--dev` mode and the download layer — `PLAN.md` 3.1.

Nothing here touches the network. The acceptance criterion is that every downstream
component can be built and tested without the full download, and a test that needed the
full download to prove that would defeat itself.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from chartqa_dt.data.download import (
    ArchiveResult,
    DownloadError,
    load_manifest,
    record_archive,
    save_manifest,
    sha256_file,
)
from chartqa_dt.data.sources import CHARTQA_ARCHIVE, SOURCES, ArchiveSpec, ParquetSpec


def test_streaming_sha256_matches_hashlib(tmp_path):
    blob = b"chart" * 100_000
    p = tmp_path / "a.bin"
    p.write_bytes(blob)
    assert sha256_file(p, chunk=1024) == hashlib.sha256(blob).hexdigest()


def test_a_manifest_round_trips(tmp_path):
    path = tmp_path / "MANIFEST.json"
    result = ArchiveResult("chartqa", path, 875_370_872, "ab" * 32,
                           CHARTQA_ARCHIVE.revision, cached=False)
    record_archive(result, path=path)
    manifest = load_manifest(path)
    entry = manifest["archives"]["chartqa"]
    assert entry["size_bytes"] == 875_370_872
    assert entry["revision"] == CHARTQA_ARCHIVE.revision
    assert entry["repo_id"] == "ahmed-masry/ChartQA"


def test_a_changed_hash_at_the_same_revision_is_refused(tmp_path):
    """The event the manifest exists to catch.

    A file that differs under the same pinned revision makes every number measured
    before and after it incomparable. Overwriting the record would erase the evidence.
    """
    path = tmp_path / "MANIFEST.json"
    rev = CHARTQA_ARCHIVE.revision
    record_archive(ArchiveResult("chartqa", path, 1, "aa" * 32, rev, False), path=path)
    with pytest.raises(DownloadError, match="SHA-256 changed"):
        record_archive(ArchiveResult("chartqa", path, 1, "bb" * 32, rev, False), path=path)


def test_recording_the_same_hash_twice_is_fine(tmp_path):
    path = tmp_path / "MANIFEST.json"
    rev = CHARTQA_ARCHIVE.revision
    for _ in range(2):
        record_archive(ArchiveResult("chartqa", path, 1, "aa" * 32, rev, False), path=path)
    assert load_manifest(path)["archives"]["chartqa"]["sha256"] == "aa" * 32


def test_an_absent_manifest_reads_as_empty(tmp_path):
    manifest = load_manifest(tmp_path / "nope.json")
    assert manifest["archives"] == {}
    save_manifest(manifest, tmp_path / "written.json")
    assert json.loads((tmp_path / "written.json").read_text())["archives"] == {}


def test_dev_mode_refuses_where_it_cannot_deliver():
    """ChartQA's gold tables exist only in the zip, so `--dev` must say so, not fake it."""
    from chartqa_dt.data.download import materialise_dev_subset

    with pytest.raises(DownloadError, match=r"phase0\.md F6"):
        materialise_dev_subset("chartqa", data_root=".")


def test_every_source_is_pinned_to_a_full_revision():
    for key, spec in SOURCES.items():
        assert len(spec.revision) == 40, f"{key} is not pinned to a full commit sha"
        assert spec.repo_id.count("/") == 1


def test_sources_agree_with_the_measured_facts_file():
    """One source of truth. A revision that drifts here is a silent confound."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    facts = json.loads((root / "verification/measured_facts.json").read_text())
    for spec in SOURCES.values():
        assert spec.revision == facts["pinned_revisions"][spec.repo_id]
    assert CHARTQA_ARCHIVE.expected_bytes == facts["datasets"]["chartqa_archive_bytes"]
    assert SOURCES["refchartqa"].splits == facts["datasets"]["refchartqa_splits"]


def test_the_two_source_kinds_are_distinguishable():
    assert isinstance(SOURCES["chartqa"], ArchiveSpec)
    assert isinstance(SOURCES["refchartqa"], ParquetSpec)
    assert SOURCES["chartqa"].filename == "ChartQA Dataset.zip"
