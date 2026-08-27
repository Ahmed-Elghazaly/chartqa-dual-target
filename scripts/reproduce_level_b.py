"""Level-B reproduction — `PLAN.md` 4.4.

    Download RefChartQA's released per-model prediction files and re-score them with
    their own evaluator. Confirm **32.83 AP@0.5** reproduces.

    Reproduces → the target is Level B-grade and independently confirmed. Record it.
    Does not reproduce → **stop and investigate.** Do not proceed with a target you
    cannot reproduce.

The whole project is measured against that number, so it is worth knowing that it is real
before spending compute trying to beat it. `DECISIONS.md` 0002 already established that
32.83 is the **human subset**, not the whole test set.

This runs the vendored `evaluate.py` functions verbatim on the vendored
`filtered_results.jsonl` — no reimplementation — and then re-scores the same predictions
with our own metrics, which is the shared-prediction-set half of `PLAN.md` 4.2.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
from pathlib import Path
from typing import Any

VENDOR = Path("verification/refchartqa_eval")
PUBLISHED = {"human": 32.83, "machine": 59.28, "pot": 39.32}


def load_official():
    spec = importlib.util.spec_from_file_location("official_evaluate", VENDOR / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["official_evaluate"] = module
    spec.loader.exec_module(module)
    return module


def load_test_split(revision: str) -> Any:
    """The official evaluator's own data path, with the revision pinned."""
    import pandas as pd
    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset("omoured/RefChartQA", split="test", revision=revision).to_pandas()

    def size(cell):
        img = Image.open(io.BytesIO(cell["bytes"]))
        return pd.Series({"width": img.width, "height": img.height})

    ds[["width", "height"]] = ds["image"].apply(size)
    return ds


def ours_on(rows: list[dict], official) -> dict[str, float]:
    """Re-score the same predictions with our metrics (`PLAN.md` 4.2, shared set)."""
    from chartqa_dt.eval.metrics import (
        average_precision_coco,
        p_at_f1,
        relaxed_correctness,
    )

    sep = official.GROUNDING_SEPERATOR_TOKEN
    preds: list[tuple[str, float, list[float]]] = []
    gts: dict[str, list[list[float]]] = {}
    pairs: list[tuple[list, list]] = []
    correct = 0

    for i, item in enumerate(rows):
        key = f"r{i}"
        answer = str(item.get("model_answer", ""))
        parts = answer.split(sep)
        if len(parts) == 2:
            boxes = official.extract_bounding_boxes(parts[0], bins=1000)
            correct += bool(relaxed_correctness(str(item["label"]), parts[1]))
        else:
            boxes = []
        gt = [official.transform_bbox_to_quantized(b, item["width"], item["height"], 1000)
              for b in item["grounding_bboxes"]]
        gts[key] = gt
        preds.extend((key, 1.0, b) for b in boxes)
        pairs.append((boxes, gt))

    return {
        "accuracy": correct / max(1, len(rows)),
        "AP_50": average_precision_coco(preds, gts, 0.5),
        "P_at_FI": p_at_f1(pairs, 0.5),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", default="c6b6504adb96cf72f0852a5f73ba4c62b718f843")
    ap.add_argument("--subsets", default="human,machine,pot")
    ap.add_argument("--limit", type=int, default=0, help="rows per subset, 0 = all")
    ap.add_argument("--out", type=Path,
                    default=Path("verification/level_b_reproduction.json"))
    args = ap.parse_args()

    official = load_official()
    import pandas as pd

    test = load_test_split(args.revision)
    results = pd.read_json(VENDOR / "filtered_results.jsonl", lines=True)
    # The official evaluator's own join: left on the test set, so surplus prediction ids
    # are dropped and missing ones become NaN (`phase0.md` F8 — the file has 8 extra
    # human rows).
    combined = pd.merge(test, results, on="id", how="left")

    report: dict[str, Any] = {"revision": args.revision, "published": PUBLISHED,
                              "subsets": {}}
    print(f"\nLevel-B reproduction — RefChartQA test, revision {args.revision[:12]}\n")
    print(f"{'subset':<10}{'rows':>7}{'published':>11}{'official':>11}{'delta':>9}"
          f"{'ours':>11}{'|diff|':>9}")

    for subset in (s.strip() for s in args.subsets.split(",") if s.strip()):
        frame = combined[combined["type"] == subset]
        if args.limit:
            frame = frame.head(args.limit)
        rows = frame.to_dict("records")
        for row in rows:
            if not isinstance(row.get("model_answer"), str):
                row["model_answer"] = ""      # a missing prediction is a wrong one

        theirs = official.analyse_dataset(rows, 1000)
        mine = ours_on(rows, official)
        published = PUBLISHED.get(subset)
        ap50 = 100 * theirs["AP_50"]
        delta = ap50 - published if published is not None else float("nan")
        print(f"  {subset:<8}{len(rows):>7,}{published:>11.2f}{ap50:>11.2f}"
              f"{delta:>+9.2f}{100 * mine['AP_50']:>11.2f}"
              f"{abs(100 * mine['AP_50'] - ap50):>9.3f}")
        report["subsets"][subset] = {
            "rows": len(rows), "published_ap50": published,
            "official": {k: float(v) for k, v in theirs.items()},
            "ours": {k: float(v) for k, v in mine.items()},
            "ap50_delta_vs_published": float(delta),
            "ap50_abs_diff_ours_vs_official": abs(float(mine["AP_50"] - theirs["AP_50"])),
        }

    human = report["subsets"].get("human")
    if human and human["published_ap50"] is not None:
        ok = abs(human["ap50_delta_vs_published"]) <= 0.05
        report["reproduces_32_83"] = bool(ok)
        print(f"\n  32.83 reproduces (within 0.05): {'YES' if ok else 'NO'}")
        if not ok:
            print("  PLAN.md 4.4: does not reproduce -> stop and investigate. "
                  "Do not proceed with a target you cannot reproduce.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
