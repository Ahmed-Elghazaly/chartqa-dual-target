"""``cdt-data`` — download, verify, load and mix the datasets (PLAN Phase 3)."""

from __future__ import annotations

from chartqa_dt.cli._common import NotYetBuilt, base_parser, setup


def main() -> None:
    p = base_parser("cdt-data", "Download, hash-verify, load, deduplicate and mix the datasets.")
    p.add_argument("command", nargs="?", default="status",
                   choices=["status", "download", "verify", "audit", "dedup", "mixture"],
                   help="status: show what is present; download: fetch archives; "
                        "verify: check SHA-256 against MANIFEST.json; audit: the 200-row "
                        "RefChartQA box audit; dedup: merge duplicate records; "
                        "mixture: build stage-1 and stage-2 mixtures")
    p.add_argument("--datasets", type=str, default="chartqa,refchartqa",
                   help="comma-separated: chartqa, refchartqa, chartqapro")
    ctx = setup(p)
    if ctx.args.command == "status":
        print("\nPhase 3 not started. Nothing downloaded.")
        print(f"data_root would be: {ctx.env.data_root}")
        return
    raise NotYetBuilt(f"cdt-data {ctx.args.command}", "Phase 3 — Data")
