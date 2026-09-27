from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

IGNORE_INDEX = -100
NON_TARGET_TOKENS = ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")


class CollateError(ValueError):
    pass


@dataclass
class Example:
    image: Any
    question: str
    target: str


def end_of_turn(processor: Any) -> str:
    return str(processor.tokenizer.eos_token)


def _texts(processor: Any, examples: Sequence[Example],
           prompt_builder: Callable[[str], str]) -> tuple[list[str], list[str], list[Any]]:
    stop = end_of_turn(processor)
    prompt_texts, texts, images = [], [], []
    for ex in examples:
        user_turn = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt_builder(ex.question)},
        ]}]
        prompt_text = processor.apply_chat_template(user_turn, tokenize=False, add_generation_prompt=True)
        prompt_texts.append(prompt_text)
        texts.append(prompt_text + ex.target + stop)
        images.append(ex.image)
    return prompt_texts, texts, images


def _lengths(batch: Any) -> list[int]:
    return [int(value) for value in batch["attention_mask"].sum(dim=1).tolist()]


def encoded_lengths(processor: Any, examples: Sequence[Example], *,
                    prompt_builder: Callable[[str], str]) -> list[int]:
    if not examples:
        return []
    _, texts, images = _texts(processor, examples, prompt_builder)
    return _lengths(processor(text=texts, images=images, return_tensors="pt", padding=True))


def build_batch(processor: Any, examples: Sequence[Example], max_len: int, *,
                prompt_builder: Callable[[str], str]) -> tuple[dict[str, Any], int]:
    import torch

    tok = processor.tokenizer
    if tok.pad_token_id is not None and tok.pad_token_id == tok.eos_token_id:
        raise CollateError("padding and end-of-turn share a token; masking padding would hide the stop")
    previous_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        prompt_texts, texts, images = _texts(processor, examples, prompt_builder)
        prompt_lens = _lengths(processor(text=prompt_texts, images=images, return_tensors="pt", padding=True))
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        full_lens = _lengths(batch)
        if max(full_lens) > max_len:
            raise CollateError(f"an example exceeds max_len={max_len} ({max(full_lens)} tokens)")
    finally:
        tok.padding_side = previous_side

    labels = batch["input_ids"].clone()
    for row, n_prompt in enumerate(prompt_lens):
        labels[row, :n_prompt] = IGNORE_INDEX
    if tok.pad_token_id is not None:
        labels[labels == tok.pad_token_id] = IGNORE_INDEX
    for token in NON_TARGET_TOKENS:
        tid = tok.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0:
            labels[labels == tid] = IGNORE_INDEX
    supervised = int((labels != IGNORE_INDEX).sum())
    if supervised == 0:
        raise CollateError("no supervised positions in this batch")
    batch["labels"] = labels
    inputs = {k: (v.to(torch.long) if k in ("input_ids", "labels") else v) for k, v in batch.items()}
    return inputs, supervised
