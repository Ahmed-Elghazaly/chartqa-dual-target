"""
    python scripts/generate_synthetic.py -n 24000 --seed 20250824
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chartqa_dt.paths import data_root
from chartqa_dt.synth.generator import generate_batch, write_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--num", type=int, default=24_000)
    ap.add_argument("--seed", type=int, default=20250824)
    ap.add_argument("--out", type=Path, default=None, help="default: <data root>/synthetic/train")
    args = ap.parse_args()
    out = args.out or data_root() / "synthetic" / "train"
    outcomes: dict[str, int] = {}
    examples = generate_batch(args.num, out, seed=args.seed, outcomes=outcomes)
    summary = write_manifest(examples, out / "manifest.json", seed=args.seed, outcomes=outcomes)
    print(f"{len(examples)} examples written to {out}")
    print(f"  by chart type: {summary['by_chart_type']}")
    print(f"  by level     : {summary['by_level']}")
    print(f"  outcomes     : {summary['generation_outcomes']}")


if __name__ == "__main__":
    main()
