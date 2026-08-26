"""``cdt-gen`` — the synthetic chart generator (PLAN Phase 3.5)."""

from __future__ import annotations

from chartqa_dt.cli._common import NotYetBuilt, base_parser, setup


def main() -> None:
    p = base_parser("cdt-gen", "Generate synthetic charts with exact boxes, answers and typed plans.")
    p.add_argument("-n", "--num", type=int, default=100, help="number of examples to generate")
    p.add_argument("--levels", type=str, default="L1,L2,L3,L4", help="curriculum levels to emit")
    p.add_argument("--chart-types", type=str, default="bar,line,pie,scatter")
    p.add_argument("--holdout", action="store_true", help="emit from the sealed holdout style/seed ranges")
    p.add_argument("--verify", action="store_true", help="run the mandatory box-correctness self-test")
    setup(p)  # validates config, dumps provenance
    raise NotYetBuilt("cdt-gen", "Phase 3.5 — The synthetic generator")


