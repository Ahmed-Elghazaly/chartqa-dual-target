from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from chartqa_dt.data.records import ChartRecord


def read_plans(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            entries[entry["record_id"]] = entry
    return entries


def write_plans(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in sorted(entries, key=lambda e: e["record_id"]):
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":")) + "\n")


def attach_plans(records: list[ChartRecord], path: Path) -> list[ChartRecord]:
    if not path.exists():
        return records
    entries = read_plans(path)
    out = [replace(record, plan=entries[record.record_id]["plan"])
           if record.record_id in entries else record
           for record in records]
    attached = sum(record.record_id in entries for record in records)
    print(f"  canonical real pool: {attached:,} of {len(records):,} records carry a mined plan")
    return out
