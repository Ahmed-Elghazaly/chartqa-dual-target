"""Cross-check our metrics against the official evaluators — `PLAN.md` 4.2.

    Run both the official ChartQA and official RefChartQA evaluators on the same
    predictions as your implementation and assert agreement. If they disagree, **the
    official one wins** and you fix yours. Record the comparison.

Three comparisons, each on the same inputs:

1. **Relaxed accuracy** — ours against the `relaxed_accuracy` inside the vendored
   RefChartQA evaluator, over generated edge cases and over the released prediction file.
2. **AP@0.5** — ours against `torchmetrics.MeanAveragePrecision`, which is what the
   official evaluator calls, over randomised detection scenarios.
3. **P@F1** — ours against the official `is_image_grounding_correct`.

Randomised rather than hand-picked: hand-picked cases test what the author already
suspects. The seeds are fixed so a disagreement is reproducible.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from chartqa_dt.eval.metrics import (
    average_precision_at_iou,
    average_precision_coco,
    grounding_is_perfect,
    relaxed_correctness,
)

VENDOR = Path("verification/refchartqa_eval")


def load_official():
    """Import the vendored official evaluator, verbatim."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("official_evaluate", VENDOR / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["official_evaluate"] = module
    spec.loader.exec_module(module)
    return module


def official_ap(per_image_preds, per_image_gts, iou_threshold: float = 0.5) -> float:
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    metric = MeanAveragePrecision(iou_thresholds=[iou_threshold], class_metrics=False)
    for preds, gts in zip(per_image_preds, per_image_gts):
        metric.update(
            [{"boxes": torch.tensor(preds, dtype=torch.float).reshape(-1, 4),
              "scores": torch.ones(len(preds)),
              "labels": torch.ones(len(preds), dtype=torch.int64)}],
            [{"boxes": torch.tensor(gts, dtype=torch.float).reshape(-1, 4),
              "labels": torch.ones(len(gts), dtype=torch.int64)}],
        )
    return float(metric.compute()["map"])


def ours_ap(per_image_preds, per_image_gts, coco: bool = True,
            iou_threshold: float = 0.5) -> float:
    preds = [(f"i{k}", 1.0, b) for k, boxes in enumerate(per_image_preds) for b in boxes]
    gts = {f"i{k}": list(boxes) for k, boxes in enumerate(per_image_gts)}
    fn = average_precision_coco if coco else average_precision_at_iou
    return fn(preds, gts, iou_threshold)


def random_scene(rng: random.Random, n_images: int) -> tuple[list, list]:
    """Boxes on a 1000-unit canvas, as the official evaluator sees them."""
    preds, gts = [], []
    for _ in range(n_images):
        n_gt = rng.randint(1, 4)
        image_gts = []
        for _ in range(n_gt):
            x, y = rng.uniform(0, 800), rng.uniform(0, 800)
            w, h = rng.uniform(20, 200), rng.uniform(20, 200)
            image_gts.append([x, y, x + w, y + h])
        image_preds = []
        for g in image_gts:
            r = rng.random()
            if r < 0.15:
                continue                       # missed
            jitter = rng.uniform(0, 60) if r < 0.6 else rng.uniform(0, 200)
            image_preds.append([g[0] + jitter, g[1] + jitter,
                                g[2] + jitter, g[3] + jitter])
        if rng.random() < 0.4:                 # a spurious extra box
            x, y = rng.uniform(0, 800), rng.uniform(0, 800)
            image_preds.append([x, y, x + rng.uniform(20, 200), y + rng.uniform(20, 200)])
        preds.append(image_preds)
        gts.append(image_gts)
    return preds, gts


def check_relaxed_accuracy(official, cases: list[tuple[str, str]]) -> dict[str, Any]:
    disagreements = []
    for target, pred in cases:
        theirs = bool(official.relaxed_accuracy(pred, target))
        mine = relaxed_correctness(target, pred)
        if theirs != mine:
            disagreements.append({"target": target, "prediction": pred,
                                  "official": theirs, "ours": mine})
    return {"cases": len(cases), "disagreements": disagreements}


def check_ap(seeds: range, n_images: int) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        rng = random.Random(seed)
        preds, gts = random_scene(rng, n_images)
        rows.append({
            "seed": seed,
            "official": official_ap(preds, gts),
            "ours_coco": ours_ap(preds, gts, coco=True),
            "ours_appendix_d": ours_ap(preds, gts, coco=False),
        })
    coco_err = [abs(r["ours_coco"] - r["official"]) for r in rows]
    apd_err = [abs(r["ours_appendix_d"] - r["official"]) for r in rows]
    return {
        "scenarios": len(rows),
        "coco_max_abs_err": max(coco_err), "coco_mean_abs_err": float(np.mean(coco_err)),
        "appendix_d_max_abs_err": max(apd_err),
        "appendix_d_mean_abs_err": float(np.mean(apd_err)),
        "worst_coco": max(rows, key=lambda r: abs(r["ours_coco"] - r["official"])),
        "rows": rows,
    }


def check_p_at_f1(official, seeds: range) -> dict[str, Any]:
    disagreements = []
    for seed in seeds:
        rng = random.Random(10_000 + seed)
        preds, gts = random_scene(rng, 1)
        p, g = preds[0], gts[0]
        theirs = bool(official.is_image_grounding_correct(p, g)) if p and g else False
        mine = grounding_is_perfect(p, g)
        if theirs != mine:
            disagreements.append({"seed": seed, "official": theirs, "ours": mine,
                                  "n_pred": len(p), "n_gt": len(g)})
    return {"scenarios": len(list(seeds)), "disagreements": disagreements}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ap-scenarios", type=int, default=60)
    ap.add_argument("--images-per-scenario", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("verification/evaluator_crosscheck.json"))
    args = ap.parse_args()

    official = load_official()

    cases = [
        ("10", "10.4"), ("10", "10.6"), ("10", "10.5"), ("0", "0"), ("0", "0.1"),
        ("0", "0.0"), ("50%", "0.5"), ("0.5", "50%"), ("50%", "50"), ("Yes", "yes"),
        ("Yes", "Yes."), ("Yes, No", "yes, no"), ("", ""), ("42", ""), ("", "42"),
        ("1,234", "1234"), ("1234", "1,234"), ("1,234", "1,234"), ("-5", "-5.1"),
        ("100", "99"), ("3.14159", "3.14"), ("abc", "ABC"), ("2020", "2020.0"),
    ]
    rng = random.Random(0)
    for _ in range(400):                       # randomised numeric pairs
        t = rng.choice([0, 1, 5, 42, 100, 1000, 0.5, -7.25])
        noise = rng.uniform(-0.12, 0.12)
        cases.append((str(t), str(round(t * (1 + noise), 4))))

    acc = check_relaxed_accuracy(official, cases)
    ap_res = check_ap(range(args.ap_scenarios), args.images_per_scenario)
    pf1 = check_p_at_f1(official, range(40))

    print("\n=== 4.2 cross-check against the official evaluators ===\n")
    print(f"relaxed accuracy : {acc['cases']} cases, "
          f"{len(acc['disagreements'])} disagreement(s)")
    for d in acc["disagreements"][:10]:
        print(f"    target={d['target']!r:<12} pred={d['prediction']!r:<12} "
              f"official={d['official']}  ours={d['ours']}")

    print(f"\nAP@0.5           : {ap_res['scenarios']} randomised scenarios "
          f"x {args.images_per_scenario} images")
    print(f"    ours (COCO)        max abs err {ap_res['coco_max_abs_err']:.6f}   "
          f"mean {ap_res['coco_mean_abs_err']:.6f}")
    print(f"    Appendix D verbatim max abs err {ap_res['appendix_d_max_abs_err']:.6f}   "
          f"mean {ap_res['appendix_d_mean_abs_err']:.6f}")

    print(f"\nP@F1             : {pf1['scenarios']} scenarios, "
          f"{len(pf1['disagreements'])} disagreement(s)")
    for d in pf1["disagreements"][:5]:
        print(f"    seed={d['seed']} official={d['official']} ours={d['ours']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "relaxed_accuracy": acc,
        "ap": {k: v for k, v in ap_res.items() if k != "rows"},
        "p_at_f1": pf1,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
