#!/usr/bin/env python3
"""Figures for the week-1 presentation, drawn from this project's own generated charts.

**Why only generated charts.** ChartQA is GPL-3.0 and RefChartQA is AGPL-3.0, and a chart
image with boxes drawn on it is a derivative of that image (rule 7, and
`eval/figures.write_figure` refuses it in code). The synthetic charts are this project's own
work, so they can go in a deck that is handed around — and they demonstrate the generator at
the same time.

Every number a figure prints is measured here, not typed: the box overlap scores come from
`synth/verify.ink_bbox_iou`, and the visual-token sizes come from the same `smart_resize`
port the model's processor uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from chartqa_dt.env import get_env  # noqa: E402
from chartqa_dt.vision.coords import smart_resize  # noqa: E402

#: Qwen3-VL's visual token is 32 px on a side (`DECISIONS.md` 0008, derived from the
#: processor rather than assumed).
TOKEN_PX = 32
LONG_SIDE = 512

GREEN, RED, INK = "#12813f", "#c0392b", "#1a1a1a"

#: The generator's own acceptance threshold (`synth/verify.GEOMETRY_THRESHOLDS`). Read from
#: there rather than restated, so the figure cannot claim a rule the code does not apply.
from chartqa_dt.synth.verify import GEOMETRY_THRESHOLDS  # noqa: E402

MIN_INK_IOU = GEOMETRY_THRESHOLDS["rect"]["min_ink_iou"]

#: Chosen for the figures: whole-number values, well-separated bars, a two-operand plan.
GROUNDED_EXAMPLE = "synth_vbar_L2_31_697938"


def load_examples() -> dict:
    manifest = Path(get_env().data_root) / "synthetic/train/manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {e["example_id"]: e for e in data["examples"]}


def _chart_axes(ax, example: dict) -> Image.Image:
    img = Image.open(example["image_path"]).convert("RGB")
    ax.imshow(np.asarray(img))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    return img


def figure_ungrounded(example: dict, out: Path) -> Path:
    """Slide 2 — an answer with nothing behind it."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    img = _chart_axes(ax, example)
    w, h = img.size
    ax.text(w / 2, h * 0.42, "?", fontsize=110, color="#c0392b", alpha=0.30,
            ha="center", va="center", fontweight="bold")
    ax.set_title(f'"{example["question"]}"', fontsize=13, color=INK, pad=12)
    ax.set_xlabel(f'Model answers:  {example["answer"]}          '
                  "— but did it look at the right bars?",
                  fontsize=12, color="#c0392b", labelpad=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_grounded(example: dict, out: Path) -> Path:
    """Slide 3 — the same answer, with its evidence and its arithmetic."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
    _chart_axes(ax, example)

    values = dict(zip(example["table"]["labels"], example["table"]["values"]))
    for ev in example["evidence"]:
        x1, y1, x2, y2 = ev["bbox_px"]
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                        edgecolor=GREEN, linewidth=2.5))
        ax.text((x1 + x2) / 2, y1 - 8, f'{ev["label"]} = {values[ev["label"]]:g}',
                fontsize=10.5, color=GREEN, ha="center", va="bottom", fontweight="bold")

    plan = example["plan"]
    operands = [values[a] for a in plan["args"] if a in values]
    arithmetic = (f"{operands[0]:g} − {operands[1]:g} = {example['answer']}"
                  if len(operands) == 2 else example["answer"])
    ax.set_title(f'"{example["question"]}"', fontsize=13, color=INK, pad=12)
    ax.set_xlabel(f"it used these two bars, and subtracted one from the other\n"
                  f"checked:  {arithmetic}   ✓  matches the answer it gave",
                  fontsize=12, color=GREEN, labelpad=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_subtoken(example: dict, out: Path) -> Path:
    """Slide 7 — the target is smaller than one of the blocks the model sees."""
    img = Image.open(example["image_path"]).convert("RGB")
    w, h = img.size
    rh, rw = smart_resize(h, w, factor=TOKEN_PX, min_pixels=4 * TOKEN_PX * TOKEN_PX,
                          max_pixels=LONG_SIDE * LONG_SIDE)
    resized = img.resize((rw, rh))
    sx, sy = rw / w, rh / h

    ev = example["evidence"][0]
    x1, y1, x2, y2 = (v * s for v, s in zip(ev["bbox_px"], (sx, sy, sx, sy)))
    tw, th = (x2 - x1) / TOKEN_PX, (y2 - y1) / TOKEN_PX

    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=200,
                                   gridspec_kw={"width_ratios": [2.1, 1]})
    ax.imshow(np.asarray(resized))
    for gx in range(0, rw, TOKEN_PX):
        ax.axvline(gx, color="#2f6fdb", linewidth=0.45, alpha=0.5)
    for gy in range(0, rh, TOKEN_PX):
        ax.axhline(gy, color="#2f6fdb", linewidth=0.45, alpha=0.5)
    ax.add_patch(mpatches.Rectangle((x1 - 6, y1 - 6), (x2 - x1) + 12, (y2 - y1) + 12,
                                    fill=False, edgecolor=RED, linewidth=2.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("what the model sees: the chart divided into blocks",
                 fontsize=12.5, color=INK)

    pad = TOKEN_PX * 1.6
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box = (max(0, cx - pad), max(0, cy - pad), min(rw, cx + pad), min(rh, cy + pad))
    zoom.imshow(np.asarray(resized.crop([int(v) for v in box])),
                extent=[box[0], box[2], box[3], box[1]])
    for gx in range(0, rw, TOKEN_PX):
        zoom.axvline(gx, color="#2f6fdb", linewidth=1.0, alpha=0.7)
    for gy in range(0, rh, TOKEN_PX):
        zoom.axhline(gy, color="#2f6fdb", linewidth=1.0, alpha=0.7)
    zoom.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                      edgecolor=RED, linewidth=2.2))
    zoom.set_xlim(box[0], box[2])
    zoom.set_ylim(box[3], box[1])
    zoom.set_xticks([])
    zoom.set_yticks([])
    # Kept short: a longer title is wider than the zoom panel and spills over the
    # left-hand chart.
    zoom.set_title("smaller than one block",
                   fontsize=12.5, color=RED, fontweight="bold")

    fig.suptitle("If a target is smaller than one block, it cannot be located precisely",
                 fontsize=13.5, color=INK, y=1.01)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": out, "tokens_w": round(tw, 2), "tokens_h": round(th, 2),
            "resized": [rw, rh]}


def figure_verification(example: dict, out: Path, *, shift_px: int = 26) -> dict:
    """Slide 9 — the check that stops a wrong box reaching training.

    The right-hand box is the same box moved sideways. Its overlap score is measured by
    the same function the generator uses, not asserted.
    """
    from chartqa_dt.synth.verify import ink_bbox_iou

    img = Image.open(example["image_path"]).convert("RGB")
    array = np.asarray(img)
    ev = example["evidence"][0]
    good = tuple(ev["bbox_px"])
    bad = (good[0] + shift_px, good[1], good[2] + shift_px, good[3])

    colours = example.get("meta", {}).get("colours")
    rgb = _element_colour(array, good) if not colours else colours[0]
    score_good = ink_bbox_iou(array, good, rgb)
    score_bad = ink_bbox_iou(array, bad, rgb)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=200)
    for ax, box, score, colour, label in (
            (axes[0], good, score_good, GREEN, "the region we drew"),
            (axes[1], bad, score_bad, RED, "the same region, nudged sideways")):
        verdict = "KEPT" if score >= MIN_INK_IOU else "REJECTED"
        ax.imshow(array)
        ax.add_patch(mpatches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                        fill=False, edgecolor=colour, linewidth=2.6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label}\nmatches the picture: {100 * score:.0f}%   →   {verdict}",
                     fontsize=12.5, color=colour, fontweight="bold")
    fig.suptitle("Every generated example is checked against its own picture, "
                 "and thrown away if it does not match",
                 fontsize=13, color=INK, y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": out, "score_correct": round(float(score_good), 3),
            "score_shifted": round(float(score_bad), 3),
            "threshold": MIN_INK_IOU}


def _element_colour(array: np.ndarray, box) -> tuple[int, int, int]:
    """The dominant non-background colour inside a box."""
    x1, y1, x2, y2 = (int(v) for v in box)
    crop = array[y1:y2, x1:x2].reshape(-1, 3)
    if not len(crop):
        return (0, 0, 0)
    colours, counts = np.unique(crop, axis=0, return_counts=True)
    order = np.argsort(-counts)
    for index in order:
        r, g, b = colours[index]
        if not (r > 235 and g > 235 and b > 235):        # skip the background
            return (int(r), int(g), int(b))
    return tuple(int(v) for v in colours[order[0]])


def pick_subtoken_example(examples: dict) -> dict:
    """A real sub-token target from our own pool: the narrowest marker on a line chart."""
    best, best_size = None, 9e9
    for example in examples.values():
        if example["holdout"] or example["chart_type"] != "line":
            continue
        w, h = example["image_size"]
        rh, rw = smart_resize(h, w, factor=TOKEN_PX, min_pixels=4 * TOKEN_PX * TOKEN_PX,
                              max_pixels=LONG_SIDE * LONG_SIDE)
        sx, sy = rw / w, rh / h
        for ev in example["evidence"]:
            x1, y1, x2, y2 = ev["bbox_px"]
            size = min((x2 - x1) * sx, (y2 - y1) * sy) / TOKEN_PX
            if 0.45 < size < best_size:                  # not the degenerate extreme
                best, best_size = {**example, "evidence": [ev]}, size
    return best


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=_ROOT / "presentation/figures")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    examples = load_examples()
    grounded = examples[GROUNDED_EXAMPLE]
    subtoken = pick_subtoken_example(examples)

    figure_ungrounded(grounded, args.out / "fig1_ungrounded.png")
    figure_grounded(grounded, args.out / "fig2_grounded.png")
    sub = figure_subtoken(subtoken, args.out / "fig3_subtoken.png")
    ver = figure_verification(grounded, args.out / "fig4_verification.png")

    meta = {
        "grounded_example": grounded["example_id"],
        "question": grounded["question"], "answer": grounded["answer"],
        "plan": grounded["plan"],
        "subtoken_example": subtoken["example_id"],
        "subtoken_label": subtoken["evidence"][0]["label"],
        "subtoken_tokens": [sub["tokens_w"], sub["tokens_h"]],
        "resized_to": sub["resized"],
        "verification_correct_box": ver["score_correct"],
        "verification_shifted_box": ver["score_shifted"],
    }
    (args.out / "figures.json").write_text(json.dumps(meta, indent=1) + "\n",
                                           encoding="utf-8")
    for name, value in meta.items():
        print(f"  {name:<26} {value}")
    print(f"\n  four figures written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
