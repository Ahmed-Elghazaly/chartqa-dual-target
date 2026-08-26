"""Backend: Unsloth.

Unsloth is faster and more memory-frugal than the stock stack, and the compute
budget in `IDEA.md` 14 is anchored to a measurement made with it: Qwen3-VL-**8B**
in 4-bit on one free T4, 30 optimizer steps in 214.98 s, peak 8.213 GiB reserved.

The catch, verified in Phase 0 and still true: Unsloth publishes vision
fine-tuning notebooks for Qwen3-VL **8B**, Qwen2.5-VL **7B**, Qwen2-VL **7B** and
Qwen3.5 **0.8B/2B/4B** — and none for Qwen3-VL-**2B**, the model this project
uses. A 4-bit checkpoint for it exists and has ~50 community finetunes, which is
suggestive but not evidence.

So this backend is implemented and measured, not assumed. If it refuses the
backbone it raises :class:`BackendUnavailable` and Phase 2 records that as the
finding it is, rather than quietly falling back to ``hf_peft`` and leaving the
question unanswered.
"""

from __future__ import annotations

import time

from chartqa_dt.config import ModelConfig
from chartqa_dt.modeling.backends.base import (
    Backend,
    BackendUnavailable,
    LoadedModel,
    peak_reserved_gb,
    register_backend,
    reset_peak_memory,
)
from chartqa_dt.vision.coords import VisualGeometry


@register_backend
class UnslothBackend(Backend):
    name = "unsloth"

    @staticmethod
    def available() -> tuple[bool, str]:
        try:
            import unsloth  # noqa: F401
        except ImportError as exc:
            return False, f"missing dependency: {exc.name} (pip install unsloth)"
        except Exception as exc:  # noqa: BLE001 - unsloth probes CUDA on import
            return False, f"unsloth failed to import: {type(exc).__name__}: {exc}"
        return True, ""

    def load(self, cfg: ModelConfig) -> LoadedModel:
        reset_peak_memory()
        t0 = time.time()
        try:
            from unsloth import FastVisionModel
        except ImportError as exc:
            raise BackendUnavailable(f"unsloth is not installed: {exc}") from exc

        # Unsloth prefers its own pre-quantised checkpoints.
        model_id = cfg.hf_id_4bit if cfg.load_in_4bit and cfg.hf_id_4bit else cfg.hf_id
        try:
            model, processor = FastVisionModel.from_pretrained(
                model_id,
                load_in_4bit=cfg.load_in_4bit,
                use_gradient_checkpointing="unsloth" if cfg.gradient_checkpointing else False,
            )
        except Exception as exc:
            raise BackendUnavailable(
                f"unsloth could not load {model_id!r}: {type(exc).__name__}: {exc}\n"
                "This is the risk IDEA.md 7 records: no Unsloth vision notebook exists for "
                "Qwen3-VL-2B. Record this result and use the hf_peft backend."
            ) from exc

        geometry = VisualGeometry.from_processor(processor)
        if cfg.image_max_pixels is not None:
            geometry = geometry.with_max_pixels(cfg.image_max_pixels)
            from chartqa_dt.modeling.backends.hf_peft_backend import _set_processor_pixel_budget

            _set_processor_pixel_budget(processor, geometry)

        return LoadedModel(
            model=model,
            processor=processor,
            geometry=geometry,
            backend=self.name,
            model_id=model_id,
            load_seconds=time.time() - t0,
            peak_reserved_gb_after_load=peak_reserved_gb(),
            dtype=cfg.dtype,
            quantized_4bit=bool(cfg.load_in_4bit),
            notes={"image_max_pixels": geometry.max_pixels},
        )

    def apply_lora(self, loaded: LoadedModel, cfg: ModelConfig) -> LoadedModel:
        from unsloth import FastVisionModel

        loaded.model = FastVisionModel.get_peft_model(
            loaded.model,
            # These four flags are the whole reason rule 3 exists: the Qwen3-VL
            # trainer has open issues where equivalents are silently ignored.
            # We set them AND verify the result by parameter name and count.
            finetune_vision_layers=cfg.lora_on_vision,
            finetune_language_layers=cfg.lora_on_language,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth" if cfg.gradient_checkpointing else False,
            random_state=0,
        )
        loaded.notes["lora_r"] = cfg.lora_r
        loaded.notes["finetune_vision_layers"] = cfg.lora_on_vision
        return loaded

    def prepare_for_training(self, loaded: LoadedModel, cfg: ModelConfig) -> LoadedModel:
        try:
            from unsloth import FastVisionModel

            FastVisionModel.for_training(loaded.model)
        except Exception:  # noqa: BLE001
            pass
        return loaded
