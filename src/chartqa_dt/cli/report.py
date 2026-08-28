"""``cdt-report`` — build result tables and figures for the report (PLAN Phase 10).

Every number the report prints comes through here from a recorded JSON file, so a figure in
the PDF cannot disagree with `verification/measured_facts.json`. A table whose results do
not exist yet is still written, with `\\TODO{}` in the cells — the skeleton is meant to
compile from Phase 1 onwards with the gaps visible in red.
"""

from __future__ import annotations

from pathlib import Path

from chartqa_dt.cli._common import base_parser, setup
from chartqa_dt.reporting.build import build_tables, load_results, summarise
from chartqa_dt.reporting.tables import BUILDERS


def main() -> None:
    p = base_parser("cdt-report", "Assemble LaTeX tables and figures from recorded results.")
    p.add_argument("--what", type=str, default="all",
                   help=", ".join(sorted(BUILDERS)) + ", or all")
    p.add_argument("--results-dir", type=str, default=None,
                   help="extra directory of *.json results, keyed by filename stem")
    p.add_argument("--report-dir", type=str, default="report",
                   help="the LaTeX skeleton to write into")
    ctx = setup(p)  # validates config, dumps provenance
    args = ctx.args

    root = Path(__file__).resolve().parents[3]
    extra = Path(args.results_dir) if args.results_dir else None
    results = load_results(root, extra)
    print(f"  results loaded : {', '.join(sorted(results)) or 'none'}")

    written = build_tables(results, Path(args.report_dir) / "tables", args.what)
    print(summarise(written))


if __name__ == "__main__":
    main()
