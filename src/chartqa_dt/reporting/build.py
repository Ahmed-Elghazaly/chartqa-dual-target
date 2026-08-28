"""Assemble the report's generated fragments — `PLAN.md` 10.1.

`load_results` gathers every recorded JSON this project produces into one dictionary, so a
builder asks for `results["mining_yield"]` and never for a path. Anything absent is simply
absent, and the builder that wanted it emits `\\TODO{}`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chartqa_dt.reporting.latex import has_todo
from chartqa_dt.reporting.tables import BUILDERS

#: Recorded file → the key builders look it up under. Missing files are not an error;
#: they are the normal state of a report being filled continuously from Phase 1.
SOURCES = {
    "verification/measured_facts.json": "measured_facts",
    "verification/mining_yield.json": "mining_yield",
    "verification/level_b_reproduction.json": "level_b",
    "verification/evaluator_crosscheck.json": "crosscheck",
    "results/headline.json": "headline",
    "results/oracle.json": "oracle",
    "results/stratified.json": "stratified",
    "results/structured_cost.json": "structured_cost",
    "results/compute.json": "compute",
}


def load_results(root: Path, extra: Path | None = None) -> dict[str, Any]:
    """Every recorded result, keyed for the builders. Absent files are left out."""
    out: dict[str, Any] = {}
    for relative, key in SOURCES.items():
        path = root / relative
        if path.is_file():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    if extra is not None and extra.is_dir():
        for path in sorted(extra.glob("*.json")):
            out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return out


def build_tables(results: dict[str, Any], out_dir: Path,
                 what: str = "all") -> dict[str, dict[str, Any]]:
    """Write the requested fragments. Returns what was written and whether it is complete."""
    names = sorted(BUILDERS) if what == "all" else [what]
    unknown = [n for n in names if n not in BUILDERS]
    if unknown:
        raise ValueError(f"unknown table {unknown[0]!r}; expected one of {sorted(BUILDERS)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, Any]] = {}
    for name in names:
        text = BUILDERS[name](results)
        path = out_dir / f"tab_{name}.tex"
        path.write_text(text, encoding="utf-8")
        written[name] = {"path": str(path), "complete": not has_todo(text),
                         "todos": text.count("\\TODO")}
    return written


def summarise(written: dict[str, dict[str, Any]]) -> str:
    complete = [n for n, d in written.items() if d["complete"]]
    lines = [f"  {len(complete)}/{len(written)} tables complete"]
    for name, d in sorted(written.items()):
        mark = "ok  " if d["complete"] else f"{d['todos']:>3} TODO"
        lines.append(f"    {mark}  tab_{name}.tex")
    if len(complete) != len(written):
        lines.append("  A build with TODOs left in is not final (report/main.tex).")
    return "\n".join(lines)


__all__ = ["SOURCES", "build_tables", "load_results", "summarise"]
