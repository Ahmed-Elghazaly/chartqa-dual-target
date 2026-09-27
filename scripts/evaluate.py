"""
    python scripts/evaluate.py chartqa --slice data/chartqa_val_200.json --tag base
    python scripts/evaluate.py chartqa --slice data/chartqa_val_200.json --tag stage2 \\
        --adapter weights/stage2 --adapter-stage stage2
    python scripts/evaluate.py refchartqa --n 200 --tag stage2 \\
        --adapter weights/stage2 --adapter-stage stage2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chartqa_dt.config import build_config
from chartqa_dt.eval.datasets import chartqa_rows, refchartqa_sample, select_slice
from chartqa_dt.eval.evaluate import generate_all, summarise_chartqa, summarise_refchartqa
from chartqa_dt.model.loading import load_adapter, load_model
from chartqa_dt.paths import cache_root, data_root, load_dotenv, output_root
from chartqa_dt.seeding import set_seed

STRUCTURED_MODE = {None: "structured", "stage1": "grounding", "stage2": "training"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["chartqa", "refchartqa"])
    ap.add_argument("--split", default="val", help="chartqa: val or test; refchartqa: validation or test")
    ap.add_argument("--slice", type=Path, default=None, help="chartqa: a file of record ids to evaluate")
    ap.add_argument("--n", type=int, default=200, help="refchartqa: rows to sample")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--adapter-stage", choices=["stage1", "stage2"], default=None)
    ap.add_argument("--tag", required=True, help="name for the output files, e.g. base or stage2")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--config", default="configs/base.yaml", help="model settings; fields can be overridden")
    args, overrides = ap.parse_known_args()
    if bool(args.adapter) != bool(args.adapter_stage):
        ap.error("--adapter and --adapter-stage go together")
    load_dotenv()
    out = args.output_dir or output_root() / "eval"
    out.mkdir(parents=True, exist_ok=True)

    loaded = load_model(build_config(args.config, overrides).model)
    if args.adapter:
        loaded = load_adapter(loaded, args.adapter, trainable=False)
    loaded.model.eval()
    structured_mode = STRUCTURED_MODE[args.adapter_stage]

    if args.dataset == "chartqa":
        rows = chartqa_rows(data_root(), args.split)
        if args.slice:
            rows = select_slice(rows, args.slice)
        results = {}
        for arm, mode in (("structured", structured_mode), ("plain", "plain")):
            set_seed(0)
            gens = generate_all(loaded, rows, mode=mode, path=out / f"chartqa_{args.split}_{arm}_{args.tag}.jsonl")
            results[arm] = summarise_chartqa(rows, gens, structured=mode != "plain")
            print(f"{arm}: relaxed accuracy {100 * results[arm]['relaxed_accuracy']:.2f}%  "
                  f"by subset {results[arm]['by_subset']}")
    else:
        rows = refchartqa_sample(cache_root(), "validation" if args.split == "val" else args.split, args.n)
        set_seed(0)
        gens = generate_all(loaded, rows, mode=structured_mode, path=out / f"refchartqa_{args.split}_{args.tag}.jsonl")
        results = summarise_refchartqa(rows, gens)
        official = results["official"]
        print(f"accuracy {100 * official['accuracy']:.2f}%  AP@0.5 {100 * official['AP_50']:.2f}%  "
              f"P@F1 {100 * official['P_at_FI']:.2f}%")
    (out / f"{args.dataset}_{args.split}_{args.tag}.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
