from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Optional

from chartqa_dt.model.coords import clamp_for_official_evaluator

THIRD_PARTY = Path(__file__).resolve().parents[3] / "third_party"
GROUNDING_SEPARATOR = "<grounding-sep>"


def score_chartqa(pairs: Sequence[tuple[str, str]]) -> float:
    source = (THIRD_PARTY / "chartqa_eval" / "metrics.py").read_text(encoding="utf-8")
    start = source.index("def relaxed_correctness")
    end = source.index("\ndef ", start + 1)
    namespace: dict[str, Any] = {"Optional": Optional}
    exec(compile(source[start:end], "pix2struct/metrics.py", "exec"), namespace)
    scorer = namespace["relaxed_correctness"]
    if not pairs:
        return 0.0
    return sum(bool(scorer(str(gold), str(prediction))) for gold, prediction in pairs) / len(pairs)


def format_prediction(boxes: Iterable[Sequence[float]], answer: str) -> str:
    parts = []
    for box in boxes:
        x1, y1, x2, y2 = clamp_for_official_evaluator((float(box[0]), float(box[1]), float(box[2]), float(box[3])))
        parts.append(f"<box>{x1},{y1},{x2},{y2}</box>")
    return "".join(parts) + GROUNDING_SEPARATOR + str(answer).strip()


def score_refchartqa(items: Sequence[dict[str, Any]], *, bins: int = 1000) -> dict[str, float]:
    path = THIRD_PARTY / "refchartqa_eval" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("official_evaluate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["official_evaluate"] = module
    spec.loader.exec_module(module)
    rows = []
    for item in items:
        width, height = item["image_size"]
        rows.append({
            "model_answer": format_prediction(item.get("pred_boxes") or [], item.get("answer", "")),
            "label": item["label"],
            "width": width,
            "height": height,
            "grounding_bboxes": item["grounding_bboxes"],
            "type": item.get("question_kind", ""),
        })
    return {k: float(v) for k, v in module.analyse_dataset(rows, bins).items()}
