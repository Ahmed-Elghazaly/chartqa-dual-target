"""``cdt-mine`` — Appendix E plan mining over ChartQA training questions (PLAN 3.6)."""

from __future__ import annotations

from chartqa_dt.cli._common import base_parser, setup


def main() -> None:
    p = base_parser("cdt-mine",
                    "Mine typed plans from ChartQA gold tables, training split only.")
    p.add_argument("--limit", type=int, default=40_000, help="questions per kind")
    p.add_argument("--split", default="train", choices=["train"],
                   help="training only — rule 1 forbids the rest")
    ctx = setup(p)

    import sys

    from scripts.mine_chartqa_train import main as run

    sys.argv = ["cdt-mine", "--limit", str(ctx.args.limit), "--seed", str(ctx.cfg.seed)]
    run()


if __name__ == "__main__":
    main()
