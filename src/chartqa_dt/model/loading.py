from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from chartqa_dt.config import ModelConfig
from chartqa_dt.model.coords import VisualGeometry

QWEN_VISION_TARGETS: tuple[str, ...] = ("qkv", "attn.proj", "linear_fc1", "linear_fc2")


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    geometry: VisualGeometry
    dtype: str

    def describe(self) -> str:
        return f"dtype={self.dtype}  geometry: {self.geometry.describe()}"


def resolve_dtype(requested: str) -> Any:
    import torch

    want = getattr(torch, requested, torch.float16)
    if torch.cuda.is_available() and want is torch.bfloat16 and torch.cuda.get_device_capability(0)[0] < 8:
        print(f"  dtype: {torch.cuda.get_device_name(0)} has no native bfloat16; using float16")
        return torch.float16
    return want


def target_modules(cfg: ModelConfig) -> list[str]:
    mods: list[str] = list(cfg.lora_target_modules) if cfg.lora_on_language else []
    if cfg.lora_on_vision:
        mods += list(QWEN_VISION_TARGETS)
    return sorted(set(mods))


def _set_pixel_budget(processor: Any, geometry: VisualGeometry) -> None:
    ip = getattr(processor, "image_processor", processor)
    size = getattr(ip, "size", None)
    if isinstance(size, dict):
        size["shortest_edge"] = geometry.min_pixels
        size["longest_edge"] = geometry.max_pixels
    elif size is not None and hasattr(size, "shortest_edge"):
        size.shortest_edge = geometry.min_pixels
        size.longest_edge = geometry.max_pixels
    if hasattr(ip, "min_pixels"):
        ip.min_pixels = geometry.min_pixels
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = geometry.max_pixels


def load_processor(cfg: ModelConfig) -> tuple[Any, VisualGeometry]:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(cfg.hf_id, revision=cfg.revision)
    geometry = VisualGeometry.from_processor(processor)
    if cfg.image_min_pixels is not None or cfg.image_max_pixels is not None:
        geometry = geometry.with_pixel_budget(min_pixels=cfg.image_min_pixels,
                                              max_pixels=cfg.image_max_pixels)
        _set_pixel_budget(processor, geometry)
    return processor, geometry


def load_model(cfg: ModelConfig) -> LoadedModel:
    import torch
    from transformers import AutoModelForImageTextToText

    t0 = time.time()
    dtype = resolve_dtype(cfg.dtype)
    processor, geometry = load_processor(cfg)
    attn = cfg.attn_implementation
    if attn == "flash_attention_2" and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
        attn = "sdpa"
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.hf_id,
        revision=cfg.revision,
        dtype=dtype,
        attn_implementation=attn,
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    print(f"  loaded {cfg.hf_id} in {time.time() - t0:.0f}s")
    return LoadedModel(model=model, processor=processor, geometry=geometry,
                       dtype=str(dtype).replace("torch.", ""))


def apply_lora(loaded: LoadedModel, cfg: ModelConfig) -> LoadedModel:
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules(cfg),
        modules_to_save=None,
    )
    loaded.model = get_peft_model(loaded.model, lora_cfg)
    return loaded


def load_adapter(loaded: LoadedModel, adapter: str, *, trainable: bool) -> LoadedModel:
    from peft import PeftModel

    loaded.model = PeftModel.from_pretrained(loaded.model, adapter, is_trainable=trainable)
    return loaded


def prepare_for_training(loaded: LoadedModel, cfg: ModelConfig) -> LoadedModel:
    model = loaded.model
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    return loaded


VISION_PATTERNS = ("visual", "vision_tower", "vision_model", "vision_encoder", "image_encoder", "vit")
LANGUAGE_PATTERNS = ("language_model", "model.layers", "llm", "text_model", "text_decoder")


def check_lora_coverage(model: Any) -> dict[str, int]:
    counts = {"vision": 0, "language": 0, "other": 0}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" not in name.lower():
            raise RuntimeError(f"non-LoRA parameter is trainable: {name}")
        low = name.lower()
        side = ("vision" if any(k in low for k in VISION_PATTERNS)
                else "language" if any(k in low for k in LANGUAGE_PATTERNS) else "other")
        counts[side] += p.numel()
    print(f"  trainable LoRA parameters: vision {counts['vision']:,}, language {counts['language']:,}")
    if not counts["vision"] or not counts["language"]:
        raise RuntimeError("LoRA did not reach both the vision tower and the language model")
    return counts
