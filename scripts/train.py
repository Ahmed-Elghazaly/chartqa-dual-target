"""
    python scripts/train.py --config configs/stage1.yaml
    python scripts/train.py --config configs/stage2.yaml \\
        --init-adapter outputs/stage1/checkpoints/stage1-best-step6349
    python scripts/train.py --config configs/stage2.yaml --resume <checkpoint directory>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chartqa_dt.config import build_config
from chartqa_dt.paths import cache_root, data_root, load_dotenv, output_root
from chartqa_dt.train.run import run_stage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mixture", type=Path, default=None, help="default: data/mixture_<stage>.json")
    ap.add_argument("--init-adapter", default=None, help="Stage 2: the best Stage-1 checkpoint")
    ap.add_argument("--resume", type=Path, default=None, help="checkpoint directory to resume from")
    ap.add_argument("--output-dir", type=Path, default=None)
    args, overrides = ap.parse_known_args()
    load_dotenv()
    cache_root()
    cfg = build_config(args.config, overrides)
    if cfg.train.stage == "stage2" and args.init_adapter is None and args.resume is None:
        ap.error("stage 2 starts from the best Stage-1 adapter: pass --init-adapter")
    run_stage(
        cfg,
        mixture=args.mixture or Path("data") / f"mixture_{cfg.train.stage}.json",
        data_root=data_root(),
        out_dir=args.output_dir or output_root() / cfg.run_name,
        init_adapter=args.init_adapter,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
