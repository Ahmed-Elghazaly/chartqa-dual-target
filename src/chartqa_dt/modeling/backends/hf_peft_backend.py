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
    resolve_target_modules,
)
from chartqa_dt.vision.coords import VisualGeometry


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
        dtype = getattr(torch, cfg.dtype, torch.bfloat16)

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
                llm_int8_skip_modules=["visual", "vision_tower", "lm_head"],
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
            attn_implementation=cfg.attn_implementation,
            device_map="auto" if torch.cuda.is_available() else None,
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
