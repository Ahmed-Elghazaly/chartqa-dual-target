"""``cdt-gen`` — the synthetic chart generator (PLAN Phase 3.5)."""

from __future__ import annotations

from pathlib import Path

from chartqa_dt.cli._common import base_parser, setup


def main() -> None:
    p = base_parser("cdt-gen",
                    "Generate synthetic charts with exact boxes, answers and typed plans.")
    p.add_argument("-n", "--num", type=int, default=100, help="number of examples to generate")
    p.add_argument("--levels", type=str, default="L1,L2,L3,L4",
                   help="curriculum levels to emit")
    p.add_argument("--chart-types", type=str, default="", help="default: all eight")
    p.add_argument("--holdout", action="store_true",
                   help="emit from the sealed holdout style/seed ranges (Phase 9.5 only)")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the box-correctness self-test (not for training data)")
    p.add_argument("--out", type=Path, default=None,
                   help="default: <output_root>/synthetic")
    ctx = setup(p)

    from chartqa_dt.synth.generator import CHART_TYPES, generate_batch, write_manifest

    args = ctx.args
    out = args.out or (ctx.env.output_root / "synthetic" /
                       ("holdout" if args.holdout else "train"))
    types = tuple(t.strip() for t in args.chart_types.split(",") if t.strip()) or CHART_TYPES
    unknown = set(types) - set(CHART_TYPES)
    if unknown:
        raise SystemExit(f"unknown chart types: {sorted(unknown)}; expected {list(CHART_TYPES)}")
    levels = tuple(x.strip() for x in args.levels.split(",") if x.strip())

    # ctx.cfg.seed, not args.seed: the latter is None unless passed, and
    # random.Random(None) seeds from OS entropy — silently unreproducible.
    examples = generate_batch(args.num, out, seed=ctx.cfg.seed, chart_types=types,
                              levels=levels, holdout=args.holdout,
                              verify=not args.no_verify)
    summary = write_manifest(examples, out / "manifest.json")

    print(f"\n{len(examples)}/{args.num} examples written to {out}")
    if len(examples) < args.num:
        print(f"  {args.num - len(examples)} attempts produced nothing — a question that "
              f"could not be built, a degenerate box, or a failed verification. "
              f"Rejections are the verifier working.")
    print(f"  by chart type: {summary['by_chart_type']}")
    print(f"  by level     : {summary['by_level']}")
    print(f"  holdout      : {summary['holdout']}")
    print(f"  manifest     : {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
