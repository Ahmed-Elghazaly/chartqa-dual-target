"""``cdt-mine`` — mine typed plans from real ChartQA gold tables (PLAN Phase 3.6, Appendix E)."""

from __future__ import annotations

from chartqa_dt.cli._common import NotYetBuilt, base_parser, setup


def main() -> None:
    p = base_parser("cdt-mine", "Mine typed plans from ChartQA training tables by the uniqueness rule.")
    p.add_argument("--split", type=str, default="train", choices=["train"],
                   help="training only — rule 1 seals val/test for this")
    p.add_argument("--sample", type=int, default=None, help="mine only N questions (for a quick yield estimate)")
    p.add_argument("--report-by-source", action="store_true", default=True,
                   help="report yield separately for human and machine questions (mandatory)")
    setup(p)  # validates config, dumps provenance
    raise NotYetBuilt("cdt-mine", "Phase 3.6 — Plan mining on real ChartQA")
