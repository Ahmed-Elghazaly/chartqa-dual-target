"""
    python scripts/build_mixtures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chartqa_dt.config import build_config
from chartqa_dt.data.mixture import (
    balance_by_level,
    build_stage1,
    build_stage2,
    drop_absent_chart_types,
    split_by_stage_usability,
    write_mixture,
)
from chartqa_dt.data.pool import load_real_pool, synthetic_records
from chartqa_dt.data.sources import chartqa_archive_path
from chartqa_dt.paths import data_root
from chartqa_dt.plans.cache import attach_plans


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--output-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    cfg = build_config(args.config).data
    root = data_root()

    synthetic = synthetic_records(root / "synthetic" / "train" / "manifest.json")
    real = load_real_pool(chartqa_archive_path(root), chartqa_limit=cfg.chartqa_limit_per_kind,
                          seed=cfg.mixture_seed, refchartqa_cache=root / "refchartqa_train.jsonl",
                          aligned_cache=root / "refchartqa_aligned.jsonl")
    real = attach_plans(real, root / "real_plans.jsonl")

    _, synth_grounding = split_by_stage_usability(synthetic, "synthetic")
    real_joint, real_grounding = split_by_stage_usability(real, "real")
    synth_grounding = drop_absent_chart_types(synth_grounding)
    stage1_synthetic = balance_by_level(synth_grounding, cfg.synthetic_stage1, seed=cfg.mixture_seed)

    s1, c1 = build_stage1(stage1_synthetic, real_grounding, cap=cfg.stage1_cap)
    write_mixture(args.output_dir / "mixture_stage1.json", s1, c1)
    s2, c2 = build_stage2(real_joint, stage1_synthetic, cap=cfg.stage2_cap, replay=cfg.synthetic_replay,
                          seed=cfg.mixture_seed)
    write_mixture(args.output_dir / "mixture_stage2.json", s2, c2)
    for comp in (c1, c2):
        print(f"\n{comp.stage}: {comp.total:,} records")
        print(f"  by source        : {dict(comp.by_source)}")
        print(f"  by question kind : {dict(comp.by_question_kind)}")
        print(f"  with plan        : {comp.with_plan:,} (compositional: {comp.with_compositional_plan:,})")


if __name__ == "__main__":
    main()
