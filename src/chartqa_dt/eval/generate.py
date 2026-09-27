from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chartqa_dt.model.parsing import answer_of, coerce_boxes, evaluation_schema_ok, parse_record
from chartqa_dt.model.prompts import (
    build_grounding_prompt,
    build_plain_prompt,
    build_structured_prompt,
    build_training_prompt,
)

MAX_NEW_TOKENS_STRUCTURED = 900
MAX_NEW_TOKENS_PLAIN = 32
MAX_NEW_TOKENS_TRAINING = 1536

MODES: dict[str, tuple[Any, int]] = {
    "structured": (build_structured_prompt, MAX_NEW_TOKENS_STRUCTURED),
    "grounding": (build_grounding_prompt, MAX_NEW_TOKENS_TRAINING),
    "training": (build_training_prompt, MAX_NEW_TOKENS_TRAINING),
    "plain": (build_plain_prompt, MAX_NEW_TOKENS_PLAIN),
}


@dataclass
class Generation:
    record_id: str
    raw: str
    seconds: float
    prompt_mode: str
    parsed_ok: bool = False
    repairs: list[str] = field(default_factory=list)
    reason: str = ""
    answer: str = ""
    boxes: list[list[float]] = field(default_factory=list)
    plan: dict[str, Any] | None = None
    new_tokens: int = 0
    hit_token_cap: bool = False


def interpret(generation: Generation) -> Generation:
    if generation.prompt_mode == "plain":
        generation.parsed_ok = True
        generation.answer = generation.raw.strip()
        return generation
    result = parse_record(generation.raw)
    generation.parsed_ok = result.ok
    generation.repairs = result.repairs
    generation.reason = result.reason
    if result.ok and result.record is not None:
        valid, why = evaluation_schema_ok(result.record)
        if valid:
            generation.answer = answer_of(result.record)
            generation.boxes = coerce_boxes(result.record)
            generation.plan = result.record.get("plan")
        else:
            generation.reason = f"schema-invalid: {why}"
    return generation


def build_messages(question: str, image: Any, mode: str) -> list[dict[str, Any]]:
    text = MODES[mode][0](question)
    return [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": text}]}]


def _pad_id(tokenizer: Any) -> int:
    pad_id = tokenizer.pad_token_id
    return tokenizer.eos_token_id if pad_id is None else pad_id


def generate_one(loaded: Any, question: str, image: Any, *, mode: str,
                 max_new_tokens: int | None = None) -> tuple[str, float, int, bool]:
    import torch

    processor, model = loaded.processor, loaded.model
    text = processor.apply_chat_template(build_messages(question, image, mode), tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    budget = max_new_tokens or MODES[mode][1]
    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=budget, do_sample=False, num_beams=1,
                             pad_token_id=_pad_id(processor.tokenizer))
    elapsed = time.perf_counter() - start
    produced = out[0][inputs["input_ids"].shape[1]:]
    decoded = processor.tokenizer.decode(produced, skip_special_tokens=True)
    n_produced = int(produced.shape[0])
    eos = processor.tokenizer.eos_token_id
    eos_ids = {int(v) for v in (eos if isinstance(eos, (list, tuple)) else [eos]) if v is not None}
    ended = bool(n_produced and int(produced[-1]) in eos_ids)
    return decoded, elapsed, n_produced, n_produced >= budget and not ended


def generate_batch(loaded: Any, questions: Sequence[str], images: Sequence[Any], *, mode: str,
                   max_new_tokens: int | None = None) -> list[tuple[str, int, bool]]:
    import torch

    processor, model = loaded.processor, loaded.model
    texts = [processor.apply_chat_template(build_messages(q, img, mode), tokenize=False,
                                           add_generation_prompt=True)
             for q, img in zip(questions, images)]
    tokenizer = processor.tokenizer
    previous_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        inputs = processor(text=texts, images=list(images), return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = previous_side
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    budget = max_new_tokens or MODES[mode][1]
    pad_id = _pad_id(tokenizer)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=budget, do_sample=False, num_beams=1,
                             pad_token_id=pad_id)
    stop_ids = {int(pad_id)}
    candidates: list[Any] = [tokenizer.eos_token_id,
                             getattr(getattr(model, "generation_config", None), "eos_token_id", None)]
    for value in candidates:
        values = value if isinstance(value, (list, tuple)) else [value]
        stop_ids.update(int(item) for item in values if item is not None)
    results: list[tuple[str, int, bool]] = []
    for row in out[:, int(inputs["input_ids"].shape[1]):].tolist():
        kept: list[int] = []
        finished = False
        for token in row:
            kept.append(int(token))
            if int(token) in stop_ids:
                finished = True
                break
        results.append((tokenizer.decode(kept, skip_special_tokens=True), len(kept),
                        not finished and len(kept) >= budget))
    return results


def read_generations(path: Path) -> list[Generation]:
    return [Generation(**json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
