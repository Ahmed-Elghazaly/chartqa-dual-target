"""Read individual members of a remote zip without downloading it.

A zip's central directory sits at the end of the file, and every member records its own
offset, so an HTTP Range request can fetch one file out of an 875 MB archive for the cost
of that file. Two uses:

* **Learning the layout before committing to a download.** The ChartQA archive's
  structure and the exact schema of its annotations were established this way, at a cost
  of a few megabytes, before Phase 3.1 fetched anything.
* **`--dev` subsets.** A few hundred members instead of the whole archive.

This is not a substitute for the real download: anything measured for the record uses the
hash-verified archive from `data/download.py`.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from urllib.parse import quote

from chartqa_dt.net import get_bytes, get_range

_EOCD_SIG = b"PK\x05\x06"
_CD_SIG = b"PK\x01\x02"
_EOCD_SEARCH = 1 << 16
_STORED, _DEFLATED = 0, 8


class RemoteZipError(RuntimeError):
    """The archive could not be read remotely."""


@dataclass(frozen=True)
class ZipEntry:
    name: str
    local_header_offset: int
    compressed_size: int
    uncompressed_size: int
    method: int


class RemoteZip:
    """Random access to a remote zip over HTTP Range requests."""

    def __init__(self, url: str, size: int) -> None:
        self.url = url
        self.size = size
        self._entries: dict[str, ZipEntry] | None = None

    @classmethod
    def for_hf_dataset(cls, repo_id: str, filename: str, revision: str,
                       size: int) -> RemoteZip:
        url = (f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/"
               f"{quote(filename)}")
        return cls(url, size)

    @property
    def entries(self) -> dict[str, ZipEntry]:
        if self._entries is None:
            self._entries = self._read_central_directory()
        return self._entries

    def _read_central_directory(self) -> dict[str, ZipEntry]:
        tail_start = max(0, self.size - _EOCD_SEARCH)
        tail = get_range(self.url, tail_start, self.size - 1)
        i = tail.rfind(_EOCD_SIG)
        if i < 0:
            raise RemoteZipError(
                "no end-of-central-directory record in the last 64 KiB; the file is not "
                "a zip, or it uses zip64 (not supported here)"
            )
        n_entries, cd_size, cd_offset = struct.unpack("<HII", tail[i + 10:i + 20])
        if cd_offset == 0xFFFFFFFF or n_entries == 0xFFFF:
            raise RemoteZipError("zip64 central directory is not supported")

        cd = get_range(self.url, cd_offset, cd_offset + cd_size - 1)
        out: dict[str, ZipEntry] = {}
        p = 0
        while p < len(cd) and cd[p:p + 4] == _CD_SIG:
            method, = struct.unpack("<H", cd[p + 10:p + 12])
            csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
            nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
            lho, = struct.unpack("<I", cd[p + 42:p + 46])
            name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
            if not name.endswith("/"):          # skip directory entries
                out[name] = ZipEntry(name, lho, csize, usize, method)
            p += 46 + nlen + elen + clen
        return out

    def read(self, name: str) -> bytes:
        """One member's bytes. Two range requests: local header, then the data."""
        try:
            e = self.entries[name]
        except KeyError:
            raise RemoteZipError(f"{name!r} is not in the archive") from None
        if e.compressed_size == 0:
            return b""
        header = get_range(self.url, e.local_header_offset, e.local_header_offset + 29)
        nlen, elen = struct.unpack("<HH", header[26:30])
        start = e.local_header_offset + 30 + nlen + elen
        raw = get_range(self.url, start, start + e.compressed_size - 1)
        if e.method == _DEFLATED:
            return zlib.decompressobj(-zlib.MAX_WBITS).decompress(raw)
        if e.method == _STORED:
            return raw
        raise RemoteZipError(f"{name!r} uses compression method {e.method}, "
                             f"expected stored or deflate")

    def read_json(self, name: str):
        return json.loads(self.read(name))

    def read_text(self, name: str, encoding: str = "utf-8") -> str:
        return self.read(name).decode(encoding, "replace")

    def names_under(self, prefix: str) -> list[str]:
        return sorted(n for n in self.entries if n.startswith(prefix))


def chartqa_remote() -> RemoteZip:
    """The pinned ChartQA archive, ready for random access."""
    from chartqa_dt.data.sources import CHARTQA_ARCHIVE as spec

    if spec.expected_bytes is None:  # pragma: no cover - the spec pins a size
        raise RemoteZipError("ChartQA archive size is not pinned")
    return RemoteZip.for_hf_dataset(spec.repo_id, spec.filename, spec.revision,
                                    spec.expected_bytes)


__all__ = ["RemoteZip", "RemoteZipError", "ZipEntry", "chartqa_remote", "get_bytes"]
