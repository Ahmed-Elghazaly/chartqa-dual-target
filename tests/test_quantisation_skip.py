"""The vision tower must stay out of 4-bit quantisation — verified, not assumed.

Quantising the visual encoder degrades exactly the capability this project
measures, so `BitsAndBytesConfig(llm_int8_skip_modules=...)` is used to keep it in
higher precision. (Despite the `int8` name, `quantizer_bnb_4bit.py` reads the same
field for the 4-bit path.)

The first version passed `["visual", "vision_tower", "lm_head"]` and did nothing.
`transformers.quantizers.quantizers_utils.should_convert_module` matches with::

    re.match(f"{key}\\.", full_name) or re.match(key, full_name) or full_name.endswith(key)

and `re.match` is anchored at the start of the string, so a bare ``"visual"`` never
matches ``"model.visual.blocks.0.attn.qkv"``. The whole vision tower was being
quantised while a code comment claimed the opposite — no error, no warning, just a
quietly worse result on the headline metric.
"""

from __future__ import annotations

import pytest

from chartqa_dt.modeling.backends.hf_peft_backend import VISION_SKIP_PATTERNS

pytestmark = pytest.mark.official

# Real Qwen3-VL module paths, read from named_modules() of the actual architecture.
VISION_MODULES = [
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.7.mlp.linear_fc1",
    "model.visual.blocks.23.mlp.linear_fc2",
    "model.visual.patch_embed.proj",
]
LANGUAGE_MODULES = [
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.layers.27.mlp.down_proj",
]


def _should_convert():
    return pytest.importorskip(
        "transformers.quantizers.quantizers_utils"
    ).should_convert_module


@pytest.mark.parametrize("name", VISION_MODULES)
def test_every_vision_module_is_kept_out_of_4bit(name):
    should_convert = _should_convert()
    assert not should_convert(name, list(VISION_SKIP_PATTERNS)), (
        f"{name} would be quantised. Skip patterns must be full module paths: "
        "re.match is anchored at the start of the string."
    )


@pytest.mark.parametrize("name", LANGUAGE_MODULES)
def test_language_modules_are_still_quantised(name):
    """The point of 4-bit is the language model; skipping it too would blow the memory gate."""
    should_convert = _should_convert()
    assert should_convert(name, list(VISION_SKIP_PATTERNS))


def test_the_bare_name_that_silently_failed_is_still_broken():
    """Pins the actual cause, so nobody 'simplifies' the patterns back."""
    should_convert = _should_convert()
    assert should_convert("model.visual.blocks.0.attn.qkv", ["visual"]), (
        "if this ever fails, transformers changed its matching rule and "
        "VISION_SKIP_PATTERNS should be re-derived"
    )


def test_lm_head_is_skipped_by_suffix_match():
    should_convert = _should_convert()
    assert not should_convert("lm_head", list(VISION_SKIP_PATTERNS))


def test_4bit_quantizer_reads_the_int8_named_field():
    """The field is called llm_int8_skip_modules but governs the 4-bit path too."""
    mod = pytest.importorskip("transformers.quantizers.quantizer_bnb_4bit")
    import inspect

    src = inspect.getsource(mod)
    assert "llm_int8_skip_modules" in src
