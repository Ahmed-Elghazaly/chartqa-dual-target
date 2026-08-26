"""``cdt-report`` — build result tables and figures for the report (PLAN Phase 10)."""

from __future__ import annotations

from chartqa_dt.cli._common import NotYetBuilt, base_parser, setup


def main() -> None:
    p = base_parser("cdt-report", "Assemble LaTeX tables and figures from recorded results.")
    p.add_argument("--what", type=str, default="all",
                   help="headline, oracle, stratified, structured_cost, crop, resolution, "
                        "variant_selection, plan_yield, compute, or all")
    p.add_argument("--results-dir", type=str, default=None)
    setup(p)  # validates config, dumps provenance
    raise NotYetBuilt("cdt-report", "Phase 10 — Deliverables")
