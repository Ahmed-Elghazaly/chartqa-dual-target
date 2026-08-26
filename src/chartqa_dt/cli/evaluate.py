"""``cdt-eval`` — evaluation against the official protocols (PLAN Phase 4, 5, 7)."""

from __future__ import annotations

from chartqa_dt.cli._common import NotYetBuilt, base_parser, setup


def main() -> None:
    p = base_parser("cdt-eval", "Evaluate on ChartQA and RefChartQA using the official evaluators.")
    p.add_argument("--dataset", type=str, default=None, choices=["chartqa", "refchartqa", "chartqapro"])
    p.add_argument("--split", type=str, default=None, choices=["val", "test"],
                   help="test splits stay sealed until Phase 7 and require --i-have-preregistered")
    p.add_argument("--adapter", type=str, default=None, help="adapter path or hub id; omit for zero-shot")
    p.add_argument("--predictions", type=str, default=None, help="score an existing predictions file instead")
    p.add_argument("--i-have-preregistered", action="store_true",
                   help="required to open a test split; asserts PREREGISTRATION.md is committed (rule 1)")
    setup(p)  # validates config, dumps provenance
    raise NotYetBuilt("cdt-eval", "Phase 4 — Evaluation, built before training")
