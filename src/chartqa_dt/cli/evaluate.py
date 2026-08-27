"""``cdt-eval`` — evaluation against the official protocols (PLAN Phase 4, 5, 7)."""

from __future__ import annotations

import json
from pathlib import Path

from chartqa_dt.cli._common import NotYetBuilt, base_parser, setup
from chartqa_dt.splits import assert_split_allowed


def main() -> None:
    p = base_parser("cdt-eval",
                    "Evaluate on ChartQA and RefChartQA using the official evaluators.")
    p.add_argument("--dataset", type=str, default=None,
                   choices=["chartqa", "refchartqa", "chartqapro"])
    p.add_argument("--split", type=str, default=None, choices=["val", "test"],
                   help="test splits stay sealed until Phase 7 and require "
                        "--i-have-preregistered")
    p.add_argument("--adapter", type=str, default=None,
                   help="adapter path or hub id; omit for zero-shot")
    p.add_argument("--predictions", type=str, default=None,
                   help="score an existing predictions JSONL instead of generating")
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--results", type=Path, default=None,
                   help="where to write the structured results JSON")
    p.add_argument("--i-have-preregistered", action="store_true",
                   help="required to open a sealed split; the run additionally verifies "
                        "that PREREGISTRATION.md is committed and clean (rule 1, "
                        "PLAN.md 5.5)")
    p.add_argument("--seal-reason", type=str, default="",
                   help="why a sealed split is being opened; recorded in the run log")
    ctx = setup(p)

    # Rule 1, enforced before anything else happens. Refusal is the default, and
    # authorisation additionally requires a committed, clean PREREGISTRATION.md
    # (PLAN.md 5.5). See DECISIONS.md 0031 for why this is a control and not a
    # sentence. --dev uses a synthetic fixture and touches no real split.
    if not ctx.args.dev:
        assert_split_allowed(
            ctx.args.dataset or ctx.cfg.eval.dataset,
            ctx.args.split or ctx.cfg.eval.split,
            authorised=ctx.args.i_have_preregistered,
            reason=ctx.args.seal_reason,
        )

    from chartqa_dt.eval.runner import evaluate_predictions, score_item, write_results
    from chartqa_dt.eval.stratified import stratify

    seeds = [int(s) for s in ctx.args.seeds.split(",") if s.strip()]

    # `--dev` comes from the shared base parser, so every command means the same
    # thing by it.
    if ctx.args.dev:
        rows = _dev_rows()
        source = "dev fixture"
    elif ctx.args.predictions:
        rows = _read_predictions(Path(ctx.args.predictions))
        source = str(ctx.args.predictions)
    else:
        raise NotYetBuilt("cdt-eval with a model", "Phase 5 — Zero-shot baselines")

    items = [score_item(r["id"], r["gold"], r["prediction"],
                        pred_boxes=r.get("pred_boxes"), gt_boxes=r.get("gt_boxes"),
                        subset=r.get("subset", "")) for r in rows]
    result = evaluate_predictions(items, seeds=seeds)

    report = stratify(
        [{"pred_boxes": i.pred_boxes, "gt_boxes": i.gt_boxes,
          "resized_size": r.get("resized_size", (512, 512))}
         for i, r in zip(items, rows)])

    print(f"\nscored {len(items):,} predictions from {source}\n")
    print(result.describe())
    print()
    print(report.describe())

    out = ctx.args.results or (ctx.out_dir / "results.json")
    write_results(result, out,
                  meta={"source": source, "seeds": seeds,
                        "dataset": ctx.args.dataset, "split": ctx.args.split,
                        "adapter": ctx.args.adapter},
                  stratified=report.to_dict())
    print(f"\nwritten to {out}")


def _read_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _dev_rows() -> list[dict]:
    """A small, self-contained fixture so `--dev` needs no model and no download.

    The Phase 4 acceptance criterion asks that `cdt-eval` run end to end on dev data. A
    fixture that reached for a dataset would make that criterion depend on the network,
    which is exactly what `--dev` exists to avoid. The rows exercise every branch that
    matters: a correct numeric answer within tolerance, one outside it, a non-numeric
    answer, a missing box, a spurious box, and a sub-token target.
    """
    full = [100.0, 100.0, 300.0, 300.0]
    small = [500.0, 500.0, 520.0, 520.0]          # sub-token at 512 px
    other = [600.0, 600.0, 800.0, 800.0]
    return [
        {"id": "dev-1", "gold": "10", "prediction": "10.4", "subset": "human",
         "pred_boxes": [full], "gt_boxes": [full]},
        {"id": "dev-2", "gold": "10", "prediction": "10.6", "subset": "human",
         "pred_boxes": [full], "gt_boxes": [full]},
        {"id": "dev-3", "gold": "Yes", "prediction": "yes", "subset": "human",
         "pred_boxes": [], "gt_boxes": [full]},
        {"id": "dev-4", "gold": "0", "prediction": "0", "subset": "machine",
         "pred_boxes": [full, other], "gt_boxes": [full]},
        {"id": "dev-5", "gold": "42", "prediction": "42", "subset": "machine",
         "pred_boxes": [small], "gt_boxes": [small]},
        {"id": "dev-6", "gold": "7", "prediction": "9", "subset": "pot",
         "pred_boxes": [other], "gt_boxes": [full]},
        {"id": "dev-7", "gold": "3.5", "prediction": "3.6", "subset": "pot",
         "pred_boxes": [full, small], "gt_boxes": [full, small]},
        {"id": "dev-8", "gold": "", "prediction": "", "subset": "pot",
         "pred_boxes": [], "gt_boxes": []},
    ]


if __name__ == "__main__":
    main()
