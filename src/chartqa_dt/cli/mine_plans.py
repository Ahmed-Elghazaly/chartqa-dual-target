"""``cdt-mine`` — mine typed plans for ChartQA training questions.

Drives `scripts/mine_plans.py`, which reads FINISHED records and has a language
model propose a plan for each, verifying every one through the five gates in
`plans.llm_mining` (`DECISIONS.md` 0088). The deterministic miner it used to call
is off the supervision path: it searched backwards from the gold answer and had to
refuse whenever more than one operation reproduced it (0085).
"""

from __future__ import annotations

from chartqa_dt.cli._common import base_parser, setup


def main() -> None:
    p = base_parser("cdt-mine",
                    "Mine typed plans from ChartQA gold tables, training split only.")
    p.add_argument("--limit", type=int, default=20_000, help="records to mine")
    p.add_argument("--split", default="train", choices=["train"],
                   help="training only — rule 1 forbids the rest")
    ctx = setup(p)

    import sys

    from scripts.mine_plans import main as run

    sys.argv = ["cdt-mine", "--limit", str(ctx.args.limit),
                "--seed", str(ctx.cfg.seed), "--write-batches"]
    run()


if __name__ == "__main__":
    main()
