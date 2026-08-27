"""``cdt-eval`` — evaluation against the official protocols (PLAN Phase 4, 5, 7)."""

from __future__ import annotations

from chartqa_dt.cli._common import NotYetBuilt, base_parser, setup
from chartqa_dt.splits import assert_split_allowed


def main() -> None:
    p = base_parser("cdt-eval", "Evaluate on ChartQA and RefChartQA using the official evaluators.")
    p.add_argument("--dataset", type=str, default=None, choices=["chartqa", "refchartqa", "chartqapro"])
    p.add_argument("--split", type=str, default=None, choices=["val", "test"],
                   help="test splits stay sealed until Phase 7 and require --i-have-preregistered")
    p.add_argument("--adapter", type=str, default=None, help="adapter path or hub id; omit for zero-shot")
    p.add_argument("--predictions", type=str, default=None, help="score an existing predictions file instead")
    p.add_argument("--i-have-preregistered", action="store_true",
                   help="required to open a sealed split; the run additionally verifies that "
                        "PREREGISTRATION.md is committed and clean (rule 1, PLAN.md 5.5)")
    p.add_argument("--seal-reason", type=str, default="",
                   help="why a sealed split is being opened; recorded in the run log")
    ctx = setup(p)

    # Rule 1, enforced before anything else happens. Refusal is the default, and
    # authorisation additionally requires a committed, clean PREREGISTRATION.md
    # (PLAN.md 5.5). See DECISIONS.md 0031 for why this is a control and not a
    # sentence.
    assert_split_allowed(
        ctx.args.dataset or ctx.cfg.eval.dataset,
        ctx.args.split or ctx.cfg.eval.split,
        authorised=ctx.args.i_have_preregistered,
        reason=ctx.args.seal_reason,
    )

    raise NotYetBuilt("cdt-eval", "Phase 4 — Evaluation, built before training")
