"""Backend: stock Hugging Face ``transformers`` + ``bitsandbytes`` + ``peft``.

This is the conservative path. It has no special support for any particular
model, which is exactly why it is the one most likely to work on Qwen3-VL-2B —
the size for which no Unsloth vision notebook exists (`IDEA.md` 7, re-verified in
Phase 0).

Two things here are deliberate and non-obvious.

**The vision tower's modules are named differently from the language model's.**
Qwen's vision blocks use a fused ``qkv`` projection plus ``proj``, while the
language model uses ``q_proj``/``k_proj``/``v_proj``/``o_proj``. Targeting only
the language names attaches LoRA to the language model alone — and does so
silently, producing a run that looks completely normal. That is precisely the
failure mode of QwenLM/Qwen3-VL issues #2016 and #2079, and it is why
:func:`~chartqa_dt.modeling.lora_assert.assert_lora_on_both_sides` runs after
every ``apply_lora`` rather than the flags being trusted.

**Quantisation must skip the vision tower.** Quantising the visual encoder to
4 bits alongside the language model degrades exactly the capability this project
measures. ``llm_int8_skip_modules`` keeps it in higher precision; it costs a few
hundred MB and protects the headline result.
"""

from __future__ import annotations

import time
from typing import Any

from chartqa_dt.config import ModelConfig
from chartqa_dt.modeling.backends.base import (
    Backend,
    LoadedModel,
    peak_reserved_gb,
    register_backend,
    reset_peak_memory,
    resolve_attn_implementation,
    resolve_dtype,
    resolve_target_modules,
)
from chartqa_dt.vision.coords import VisualGeometry

# Module paths kept out of 4-bit quantisation. Full paths, anchored at the start:
# see the note in `load()` for why bare names silently fail to match.
VISION_SKIP_PATTERNS: tuple[str, ...] = (
    "model.visual",
    "model.vision_tower",
    "model.vision_model",
    "visual",          # for models where the tower is top-level
    "lm_head",
)


@register_backend
class HFPeftBackend(Backend):
    name = "hf_peft"

    @staticmethod
    def available() -> tuple[bool, str]:
        try:
            import peft  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            return False, f"missing dependency: {exc.name}"
        return True, ""

    # ------------------------------------------------------------------ load

    def load(self, cfg: ModelConfig) -> LoadedModel:
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        reset_peak_memory()
        t0 = time.time()

        model_id = cfg.hf_id
        dtype, dtype_note = resolve_dtype(cfg.dtype)
        attn_impl, attn_note = resolve_attn_implementation(cfg.attn_implementation)
        if dtype_note:
            print(f"  dtype: {dtype_note}")
        if attn_note:
            print(f"  attn: {attn_note}")

        quant_config = None
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                # Keep the visual encoder out of 4-bit: quantising it damages the
                # exact capability this project measures.
                #
                # These must be FULL module paths, not bare names. transformers'
                # `should_convert_module` matches with
                #     re.match(f"{key}\\.", full_name) or re.match(key, full_name)
                #     or full_name.endswith(key)
                # and `re.match` is anchored at the START of the string. A bare
                # "visual" therefore does NOT match "model.visual.blocks.0.attn.qkv",
                # and the vision tower gets quantised anyway with no warning.
                # Verified in tests/test_quantisation_skip.py.
                llm_int8_skip_modules=list(VISION_SKIP_PATTERNS),
            )

        processor = AutoProcessor.from_pretrained(model_id)
        geometry = VisualGeometry.from_processor(processor)
        if cfg.image_max_pixels is not None:
            geometry = geometry.with_max_pixels(cfg.image_max_pixels)
            _set_processor_pixel_budget(processor, geometry)

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            quantization_config=quant_config,
            attn_implementation=attn_impl,
            # A SINGLE device, deliberately. Kaggle's "NvidiaTeslaT4" shape
            # provides TWO T4s, and device_map="auto" shards the model across
            # cuda:0 and cuda:1. That silently produced three separate problems:
            # the training loop sends every batch to the first parameter's device
            # and crashes when a later layer lives on the other one; each forward
            # pass pays inter-GPU transfers (measured +52% per step); and
            # torch.cuda.max_memory_reserved() reads device 0 only, so a sharded
            # run's real footprint was never being measured at all.
            #
            # IDEA.md 14's compute budget is for ONE card, so one card is what we
            # measure. Multi-GPU would be a different experiment.
            device_map={"": 0} if torch.cuda.is_available() else None,
        )

        arch = AutoConfig.from_pretrained(model_id)
        return LoadedModel(
            model=model,
            processor=processor,
            geometry=geometry,
            backend=self.name,
            model_id=model_id,
            load_seconds=time.time() - t0,
            peak_reserved_gb_after_load=peak_reserved_gb(),
            dtype=str(dtype).replace("torch.", ""),
            quantized_4bit=bool(cfg.load_in_4bit),
            notes={
                "dtype_note": dtype_note,
                "attn_note": attn_note,
                "model_type": getattr(arch, "model_type", "?"),
                "architectures": getattr(arch, "architectures", []),
                "image_max_pixels": geometry.max_pixels,
            },
        )

    # ------------------------------------------------------------- apply_lora

    def apply_lora(self, loaded: LoadedModel, cfg: ModelConfig) -> LoadedModel:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model = loaded.model
        if cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=cfg.gradient_checkpointing
            )

        targets = resolve_target_modules(cfg)
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
            # Without this, PEFT can decline to wrap modules it considers part of
            # the vision tower, which is the silent half of the #2016/#2079 bug.
            modules_to_save=None,
        )
        model = get_peft_model(model, lora_cfg)

        loaded.model = model
        loaded.notes["lora_target_modules"] = targets
        loaded.notes["lora_r"] = cfg.lora_r
        return loaded

    # ------------------------------------------------------- prepare_training

    def prepare_for_training(self, loaded: LoadedModel, cfg: ModelConfig) -> LoadedModel:
        model = loaded.model
        if cfg.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                # Needed with gradient checkpointing on a frozen embedding layer,
                # otherwise no gradient reaches the adapters at all.
                model.enable_input_require_grads()
        if hasattr(model, "config"):
            model.config.use_cache = False
        return loaded


def _set_processor_pixel_budget(processor: Any, geometry: VisualGeometry) -> None:
    """Apply a pixel budget to a loaded processor, whichever API it exposes.

    Different ``transformers`` versions accept ``min_pixels``/``max_pixels``
    attributes or a ``size`` dict with ``shortest_edge``/``longest_edge``. We set
    whatever is present rather than assuming a version.
    """
    ip = getattr(processor, "image_processor", processor)
    size = getattr(ip, "size", None)
    if isinstance(size, dict):
        size["shortest_edge"] = geometry.min_pixels
        size["longest_edge"] = geometry.max_pixels
    elif size is not None and hasattr(size, "shortest_edge"):
        try:
            size.shortest_edge = geometry.min_pixels
            size.longest_edge = geometry.max_pixels
        except (AttributeError, TypeError):
            pass
    if hasattr(ip, "min_pixels"):
        ip.min_pixels = geometry.min_pixels
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = geometry.max_pixels
