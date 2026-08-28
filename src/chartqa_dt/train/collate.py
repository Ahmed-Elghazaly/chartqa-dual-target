"""Batching and loss masking for Phase 6 — built on the path that already trained.

`train/smoke.py` ran successfully on Kaggle in Phase 2, so its `build_batch` is the
authority here rather than a fresh implementation. Two things it does are subtle and are
preserved exactly:

* **The prompt boundary is measured by running the processor on the prompt with the same
  image**, never by counting text tokens. An image expands into a variable number of visual
  tokens, so a text-only count puts the mask boundary in the wrong place. The first draft of
  that function supervised 23 of 45 positions instead of 8, and the loss curve looked fine.
* **Right padding**, because masking leading positions requires it.

Three things are added, each for a defect found by design pass before Phase 6 ran:

* **The end-of-turn token is appended and supervised.** The Qwen3-VL template closes an
  assistant turn with `<|im_end|>`; `smoke.py` concatenated `prompt + answer` and never
  supplied it. A model trained that way is never taught to stop, and every generation runs
  to `max_new_tokens`.
* **Truncation is detected and refused**, not silently accepted. A target cut off at the
  sequence limit teaches incomplete records while the loss curve stays plausible
  (`DECISIONS.md` 0064).
* **Pad and end-of-turn must be different tokens**, asserted. They are here — 151643 and
  151645 — but if they ever coincided, masking padding would also mask the stop token and
  the model would silently never learn to terminate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

IGNORE_INDEX = -100
#: Masked out of the loss: they are inputs, not things to predict.
NON_TARGET_TOKENS = ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>")


class CollateError(ValueError):
    """An example could not be batched without corrupting its supervision."""


@dataclass
class Example:
    """One training example: an image, the prompt text, and the exact target string."""

    image: Any
    question: str
    target: str


def end_of_turn(processor: Any) -> str:
    """The token that closes an assistant turn, taken from the tokenizer, not assumed."""
    tok = processor.tokenizer
    text = getattr(tok, "eos_token", None)
    if not text:
        raise CollateError("the tokenizer has no eos_token; cannot teach the model to stop")
    return text


def assert_stop_token_is_supervisable(processor: Any) -> None:
    """Padding and stopping must be different tokens, or the stop is masked away."""
    tok = processor.tokenizer
    pad, eos = getattr(tok, "pad_token_id", None), getattr(tok, "eos_token_id", None)
    if eos is None:
        raise CollateError("no eos_token_id; the model cannot be taught to stop")
    if pad is not None and pad == eos:
        raise CollateError(
            f"pad_token_id == eos_token_id ({pad}). Masking padding would also mask every "
            f"stop token, and the model would silently never learn to terminate."
        )


def build_batch(processor: Any, examples: Sequence[Example], max_len: int,
                *, prompt_builder=None, strict: bool = True) -> dict:
    """Tokenise a batch and mask everything except the target out of the loss."""
    import torch

    assert_stop_token_is_supervisable(processor)
    stop = end_of_turn(processor)
    tok = processor.tokenizer
    previous_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        texts, images, prompt_lens, full_lens = [], [], [], []
        for ex in examples:
            user_turn = [{"role": "user",
                          "content": [{"type": "image"},
                                      {"type": "text", "text": _prompt(ex, prompt_builder)}]}]
            prompt_text = processor.apply_chat_template(
                user_turn, tokenize=False, add_generation_prompt=True)
            # The stop token is part of the target, so the model is supervised to emit it.
            texts.append(prompt_text + ex.target + stop)
            images.append(ex.image)

            prompt_only = processor(text=[prompt_text], images=[ex.image],
                                    return_tensors="pt")
            prompt_lens.append(int(prompt_only["input_ids"].shape[1]))
            full_only = processor(text=[texts[-1]], images=[ex.image], return_tensors="pt")
            full_lens.append(int(full_only["input_ids"].shape[1]))

        over = [(i, n) for i, n in enumerate(full_lens) if n > max_len]
        if over and strict:
            raise CollateError(
                f"{len(over)} example(s) exceed max_len={max_len} — longest {max(n for _, n in over)}. "
                f"Truncating would teach incomplete records while the loss curve stays "
                f"plausible (DECISIONS.md 0064). Shorten the prompt or drop the example.")

        batch = processor(text=texts, images=images, return_tensors="pt",
                          padding=True, truncation=True, max_length=max_len)
    finally:
        tok.padding_side = previous_side

    labels = batch["input_ids"].clone()
    for row, n_prompt in enumerate(prompt_lens):
        labels[row, :n_prompt] = IGNORE_INDEX
    pad_id = getattr(tok, "pad_token_id", None)
    if pad_id is not None:
        labels[labels == pad_id] = IGNORE_INDEX
    for token in NON_TARGET_TOKENS:
        tid = tok.convert_tokens_to_ids(token)
        if isinstance(tid, int) and tid >= 0:
            labels[labels == tid] = IGNORE_INDEX

    supervised = int((labels != IGNORE_INDEX).sum())
    if supervised == 0:
        raise CollateError("no supervised positions in this batch; the loss would be empty")

    batch["labels"] = labels
    out = {k: (v.to(torch.long) if k in ("input_ids", "labels") else v)
           for k, v in batch.items()}
    out["_supervised_positions"] = supervised
    return out


def _prompt(example: Example, prompt_builder) -> str:
    if prompt_builder is not None:
        return prompt_builder(example.question)
    from chartqa_dt.prompting.prompts import build_training_prompt

    return build_training_prompt(example.question)


__all__ = ["IGNORE_INDEX", "NON_TARGET_TOKENS", "CollateError", "Example",
           "assert_stop_token_is_supervisable", "build_batch", "end_of_turn"]
