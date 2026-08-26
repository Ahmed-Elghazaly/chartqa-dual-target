"""Characterise the official RefChartQA evaluator by executing it.

Phase 0.4 asked whether the evaluator is "present and runnable". Confirming a file
exists is not confirming it runs, and neither is reading it. This runs its metric
functions against synthetic predictions whose correct answers are known by
construction, and prints what the evaluator actually rewards.

It is the evidence behind `DECISIONS.md` 0003, 0004 and 0014.

Run:  python scripts/characterise_official_evaluator.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import warnings

warnings.filterwarnings("ignore")

EVALUATOR = pathlib.Path(__file__).resolve().parents[1] / "verification/refchartqa_eval/evaluate.py"

W, H = 800, 386
GT_A = {"x": 276.0, "y": 277.0, "w": 60.0, "h": 23.0}
GT_B = {"x": 500.0, "y": 100.0, "w": 60.0, "h": 23.0}
BAD = [[10, 10, 60, 40], [100, 100, 150, 140], [200, 200, 250, 240]]


def load_official():
    spec = importlib.util.spec_from_file_location("refeval", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_item(ev, boxes, gts, answer="47"):
    body = "".join(f"<box>{','.join(map(str, b))}</box>" for b in boxes)
    return {
        "model_answer": f"{body}<grounding-sep>{answer}",
        "label": answer,
        "width": W,
        "height": H,
        "grounding_bboxes": [dict(g) for g in gts],
    }


def main() -> None:
    ev = load_official()
    g_a = ev.transform_bbox_to_quantized(dict(GT_A), W, H, 1000)
    g_b = ev.transform_bbox_to_quantized(dict(GT_B), W, H, 1000)

    print("=" * 72)
    print("1. relaxed_accuracy — where it differs from PLAN.md Appendix D")
    print("=" * 72)
    for target, pred, note in [
        ("10", "10.4", "within 5%"),
        ("10", "10.6", "outside 5%"),
        ("0", "0", "zero-division guard"),
        ("0", "0.0", "DIFFERS: Appendix D says correct, official says not"),
        ("50%", "0.5", "percent divided by 100 on both sides"),
        ("Yes", "Yes.", "trailing period fails"),
        ("1,234", "1234", "DIFFERS: official does not strip commas"),
    ]:
        print(f"  target={target!r:<8} pred={pred!r:<8} -> {ev.relaxed_accuracy(pred, target)!s:<6} {note}")

    print("\n" + "=" * 72)
    print("2. extract_bounding_boxes — the silent 0..999 discard (decision 0004)")
    print("=" * 72)
    for text in ["<box>0,0,999,999</box>", "<box>0,0,1000,1000</box>", "<box>1,2,3</box>"]:
        print(f"  {text:<32} -> {ev.extract_bounding_boxes(text, bins=1000)}")
    print("  a single coordinate of 1000 discards the WHOLE box, with no error")

    print("\n" + "=" * 72)
    print("3. One image, one GT box — AP = 1 / (rank of the first correct box)")
    print("=" * 72)
    print(f"  {'prediction':<34}{'AP@0.5':>9}{'P@F1':>8}")
    for name, boxes in [
        ("[correct]", [g_a]),
        ("[correct, bad, bad, bad]", [g_a, *BAD]),
        ("[bad, correct]", [BAD[0], g_a]),
        ("[bad, bad, correct]", [*BAD[:2], g_a]),
        ("[bad, bad, bad, correct]", [*BAD, g_a]),
    ]:
        item = make_item(ev, boxes, [GT_A])
        print(f"  {name:<34}{ev.compute_AP_50([item]):>9.4f}{ev.compute_P_at_FI([item]):>8.4f}")

    print("\n" + "=" * 72)
    print("4. Twenty images — the same extras that looked free are devastating")
    print("=" * 72)
    n = 20
    print(f"  {'strategy':<48}{'AP@0.5':>9}{'P@F1':>8}")
    for name, make in [
        ("all correct, one box each", lambda i: [g_a]),
        ("all correct + 3 extra wrong each", lambda i: [g_a, *BAD]),
        ("all correct, one wrong box FIRST", lambda i: [BAD[0], g_a]),
        ("60% correct only, 40% nothing", lambda i: ([g_a] if i < 12 else [])),
        ("60% correct + extras, 40% nothing", lambda i: ([g_a, *BAD] if i < 12 else [])),
    ]:
        data = [make_item(ev, make(i), [GT_A]) for i in range(n)]
        print(f"  {name:<48}{ev.compute_AP_50(data):>9.4f}{ev.compute_P_at_FI(data):>8.4f}")

    print("\n" + "=" * 72)
    print("5. Two GT boxes")
    print("=" * 72)
    for name, boxes in [
        ("both correct", [g_a, g_b]),
        ("only one of two", [g_a]),
        ("one correct, one wrong (after)", [g_a, BAD[0]]),
        ("wrong first, then both correct", [BAD[0], g_a, g_b]),
    ]:
        item = make_item(ev, boxes, [GT_A, GT_B])
        print(f"  {name:<48}{ev.compute_AP_50([item]):>9.4f}{ev.compute_P_at_FI([item]):>8.4f}")

    print("\nConclusion (DECISIONS.md 0014): emit FEW boxes, BEST FIRST.")
    print("Every extra box is a global false positive; dataset AP pools all predictions")
    print("into one PR curve and every score is tied at 1.0, so extras cannot be ranked away.")


if __name__ == "__main__":
    main()
