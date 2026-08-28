"""Batching and loss masking — `PLAN.md` 6.1, built on the Phase 2 path that trained.

Tested against a stub processor rather than the real one, deliberately: the properties
that matter are about *which positions carry loss* and *what the target ends with*, and
those are decidable without four gigabytes of weights. The real processor is exercised on
Kaggle, where a GPU exists.
"""

from __future__ import annotations

import importlib.util

import pytest

from chartqa_dt.train.collate import (
    IGNORE_INDEX,
    CollateError,
    Example,
    assert_stop_token_is_supervisable,
    build_batch,
    end_of_turn,
)

# The stub processor is torch-free, but `build_batch` builds real tensors. CI's fast job
# installs core+dev only, so these skip there and run in the job that has torch.
pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                                reason="build_batch needs torch")


class StubTokenizer:
    """Character-level, so token counts are predictable and assertions are exact."""

    def __init__(self, pad_id=1, eos_id=2, eos="<eos>"):
        self.pad_token_id = pad_id
        self.eos_token_id = eos_id
        self.eos_token = eos
        self.padding_side = "right"

    def convert_tokens_to_ids(self, token):
        return -1


class StubProcessor:
    def __init__(self, **kw):
        self.tokenizer = StubTokenizer(**kw)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = messages[0]["content"][-1]["text"]
        return f"<P>{text}<A>"

    def __call__(self, text, images, return_tensors="pt", padding=False,
                 truncation=False, max_length=None):
        import torch

        rows = [[ord(c) % 50 + 10 for c in t] for t in text]
        width = max(len(r) for r in rows)
        if truncation and max_length:
            width = min(width, max_length)
        ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
        for i, r in enumerate(rows):
            r = r[:width]
            ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": (ids != 1).long()}


def ex(target="TGT"):
    return Example(image=object(), question="Q", target=target)


def test_the_target_is_supervised_and_the_prompt_is_not():
    batch = build_batch(StubProcessor(), [ex()], max_len=512)
    labels = batch["labels"][0]
    supervised = (labels != IGNORE_INDEX).sum().item()
    # "<P>Q<A>" is the prompt; the target is "TGT" plus the stop token.
    assert supervised == len("TGT" + "<eos>")
    assert batch["_supervised_positions"] == supervised


def test_the_stop_token_is_appended_and_carries_loss():
    """`smoke.py` concatenated prompt + answer with no end-of-turn token.

    A model trained that way is never taught to stop, and every generation runs to
    `max_new_tokens`. The token comes from the tokenizer, not from a hard-coded string.
    """
    processor = StubProcessor()
    assert end_of_turn(processor) == "<eos>"
    batch = build_batch(processor, [ex("AB")], max_len=512)
    labels = batch["labels"][0]
    assert (labels != IGNORE_INDEX).sum().item() == len("AB<eos>")


def test_padding_is_masked_but_the_stop_token_is_not():
    batch = build_batch(StubProcessor(), [ex("A"), ex("LONGER-TARGET")], max_len=512)
    labels = batch["labels"]
    assert (labels[0] != IGNORE_INDEX).sum().item() == len("A<eos>")
    assert (labels[1] != IGNORE_INDEX).sum().item() == len("LONGER-TARGET<eos>")


def test_pad_and_stop_must_be_different_tokens():
    """If they coincided, masking padding would mask every stop token — silently."""
    with pytest.raises(CollateError, match="never learn to terminate"):
        assert_stop_token_is_supervisable(StubProcessor(pad_id=7, eos_id=7))


def test_a_tokenizer_with_no_stop_token_is_refused():
    processor = StubProcessor()
    processor.tokenizer.eos_token = None
    with pytest.raises(CollateError, match="cannot teach the model to stop"):
        end_of_turn(processor)
    processor.tokenizer.eos_token_id = None
    with pytest.raises(CollateError, match="cannot be taught to stop"):
        assert_stop_token_is_supervisable(processor)


def test_an_over_long_example_is_refused_not_truncated():
    """`DECISIONS.md` 0064: a truncated target teaches incomplete records while the loss
    curve stays entirely plausible."""
    long_target = "X" * 400
    with pytest.raises(CollateError, match="exceed max_len"):
        build_batch(StubProcessor(), [ex(long_target)], max_len=64)


def test_truncation_can_be_allowed_explicitly_but_never_by_default():
    """`strict=False` is an opt-in for callers that have already checked lengths."""
    batch = build_batch(StubProcessor(), [ex("X" * 400)], max_len=64, strict=False,
                        prompt_builder=lambda q: q)
    assert batch["input_ids"].shape[1] == 64, "explicit opt-in still truncates"
    # Part of the target survives inside the window, so there is still supervision —
    # which is exactly why silent truncation is dangerous rather than obviously broken.
    assert batch["_supervised_positions"] > 0


def test_an_empty_target_still_supervises_the_stop_token():
    """Appending the stop token means there is always something to learn — deliberately.

    A model that emits nothing but the stop token is wrong, and that is a *content*
    failure the metrics catch. An empty loss would be a *plumbing* failure that looks
    like slow convergence.
    """
    batch = build_batch(StubProcessor(), [Example(object(), "Q", "")], max_len=512)
    assert batch["_supervised_positions"] == len("<eos>")


def test_a_batch_with_nothing_to_supervise_is_refused():
    """Fires when the window is so small that only prompt survives."""
    with pytest.raises(CollateError, match="no supervised positions"):
        build_batch(StubProcessor(), [ex("TGT")], max_len=4, strict=False,
                    prompt_builder=lambda q: q)


def test_right_padding_is_restored_afterwards():
    processor = StubProcessor()
    processor.tokenizer.padding_side = "left"
    build_batch(processor, [ex()], max_len=512)
    assert processor.tokenizer.padding_side == "left", "the caller's setting is restored"


def test_the_training_prompt_is_used_by_default():
    """Not the 980-token zero-shot prompt — that overflows the budget (0064)."""
    seen = {}

    class Recording(StubProcessor):
        def apply_chat_template(self, messages, **kw):
            seen["text"] = messages[0]["content"][-1]["text"]
            return super().apply_chat_template(messages, **kw)

    build_batch(Recording(), [ex()], max_len=4096)
    assert "compact JSON object" in seen["text"]
    assert "Question: Q" in seen["text"]
    assert len(seen["text"]) < 700, "the short training prompt, not the zero-shot one"
