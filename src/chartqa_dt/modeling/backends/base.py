"""The backend interface: one surface, two implementations.

`PLAN.md` 2.1 requires ``unsloth`` and ``hf_peft`` implementations behind a single
interface, selected by ``--backend``, so that Phase 2 can compare them by
measurement rather than by reputation.

Why an abstraction at all
-------------------------
`IDEA.md` 7 records a verified risk: Unsloth publishes vision fine-tuning
notebooks for Qwen3-VL **8B**, Qwen2.5-VL **7B** and Qwen3.5 **2B/4B**, but none
for Qwen3-VL-**2B** — the model this project is built on. Re-verified in Phase 0
and still true. So we do not know which backend works at this size, and the
project must not be structured such that finding out is expensive.

With this interface, switching backend or backbone is a config change and the
smoke test is the same code either way.

What every backend must guarantee
---------------------------------
* :meth:`Backend.load` returns a model **and** its processor. The processor is
  not optional — :class:`~chartqa_dt.vision.coords.VisualGeometry` is derived
  from it, and hard-coding that geometry is what produced decision 0008.
* :meth:`Backend.apply_lora` must attach adapters to the vision tower *and* the
  language model, and the caller then verifies it with
  :func:`~chartqa_dt.modeling.lora_assert.assert_lora_on_both_sides`. A backend
  is never trusted to have done what it was asked (rule 3).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from chartqa_dt.config import ModelConfig
from chartqa_dt.vision.coords import VisualGeometry


@dataclass
class LoadedModel:
    """A model, its processor, and everything measured about the load."""

    model: Any
    processor: Any
    geometry: VisualGeometry
    backend: str
    model_id: str
    load_seconds: float = 0.0
    peak_reserved_gb_after_load: float = 0.0
    dtype: str = ""
    quantized_4bit: bool = False
    notes: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return (
            f"backend={self.backend}  model={self.model_id}\n"
            f"  4bit={self.quantized_4bit}  dtype={self.dtype}  "
            f"load={self.load_seconds:.1f}s  peak_after_load={self.peak_reserved_gb_after_load:.3f} GiB\n"
            f"  geometry: {self.geometry.describe()}"
        )


class BackendUnavailable(RuntimeError):
    """The backend's dependencies are missing, or it refuses this backbone.

    Raised rather than silently falling back to another backend: Phase 2 exists
    to record *which* backends work at this model size, and a silent substitution
    would erase the measurement it is trying to make.
    """


class Backend(abc.ABC):
    """One way of loading a VLM and attaching LoRA to it."""

    name: str = "base"

    @abc.abstractmethod
    def load(self, cfg: ModelConfig) -> LoadedModel:
        """Load the model in 4-bit (if requested) and return it with its processor."""

    @abc.abstractmethod
    def apply_lora(self, loaded: LoadedModel, cfg: ModelConfig) -> LoadedModel:
        """Attach LoRA adapters to both the vision and the language sides."""

    def prepare_for_training(self, loaded: LoadedModel, cfg: ModelConfig) -> LoadedModel:
        """Optional hook: gradient checkpointing, cache disabling, etc."""
        return loaded

    @staticmethod
    def available() -> tuple[bool, str]:
        """(is_importable, reason_if_not). Never raises."""
        return True, ""


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, type[Backend]] = {}


def register_backend(cls: type[Backend]) -> type[Backend]:
    _REGISTRY[cls.name] = cls
    return cls


def get_backend(name: str) -> Backend:
    """Instantiate a backend by name, failing loudly on an unknown one."""
    # Imported here so that registering a backend does not require its deps.
    from chartqa_dt.modeling.backends import hf_peft_backend, unsloth_backend  # noqa: F401

    if name not in _REGISTRY:
        raise BackendUnavailable(
            f"unknown backend {name!r}; available: {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[name]
    ok, reason = cls.available()
    if not ok:
        raise BackendUnavailable(f"backend {name!r} is not usable here: {reason}")
    return cls()


def list_backends() -> dict[str, tuple[bool, str]]:
    """Every registered backend and whether it can run here."""
    from chartqa_dt.modeling.backends import hf_peft_backend, unsloth_backend  # noqa: F401

    return {name: cls.available() for name, cls in sorted(_REGISTRY.items())}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def peak_reserved_gb() -> float:
    """Peak reserved VRAM in GiB. 0.0 without CUDA.

    *Reserved*, not *allocated*: reserved is what the caching allocator has taken
    from the driver, which is what actually decides whether the next allocation
    OOMs. Reporting allocated would understate how close to the limit a run is,
    and the Phase 2 gate is 13.5 GiB of reserved memory.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_reserved() / 1024**3
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# Verified against the real Qwen3-VL architecture by instantiating it and reading
# `named_modules()` — see tests/test_backends.py. Do NOT edit these from memory.
#
# The vision tower and the language model use entirely different names:
#   vision   : qkv (fused), attn.proj, linear_fc1, linear_fc2
#   language : q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
#
# Targeting only the language names attaches LoRA to the language model alone,
# silently — the exact shape of QwenLM/Qwen3-VL issues #2016 and #2079.
QWEN_VISION_TARGETS: tuple[str, ...] = ("qkv", "attn.proj", "linear_fc1", "linear_fc2")

# `attn.proj` rather than bare `proj` is deliberate. PEFT matches by suffix, and
# the vision tower contains TWO modules whose leaf name is `proj`:
#   model.visual.patch_embed.proj      Conv3d   <- the patch embedding
#   model.visual.blocks.N.attn.proj    Linear   <- the attention output
# A bare `proj` would attach an adapter to the patch embedding as well. Adapting
# how the image is cut into patches at all is a much more invasive change than
# this project intends, and it would have happened without any warning.


def resolve_dtype(requested: str) -> tuple[Any, str]:
    """Pick a compute dtype the GPU supports **natively**, and say so.

    A free Kaggle/Colab T4 is Turing, compute capability 7.5. **bfloat16 needs
    Ampere (8.0).** PyTorch does not refuse bf16 on a T4 — it emulates it. The
    numbers stay correct and everything simply runs far slower, which on a timed
    benchmark reads as "this backbone is too slow" rather than "this setting is
    wrong".

    The test is compute capability, deliberately, **not**
    ``torch.cuda.is_bf16_supported()``. That helper takes
    ``including_emulation: bool = True`` and therefore returns ``True`` on a T4 —
    it answers "can this run?", where the question here is "can this run fast?".
    PyTorch's own implementation checks ``major >= 8`` before falling through to
    the emulation probe, so that is what we check.

    Returns (dtype, note); the note is recorded so a substitution is never silent.
    """
    try:
        import torch
    except ImportError:
        return None, "torch unavailable"

    want = getattr(torch, requested, torch.float16)
    if not torch.cuda.is_available():
        return want, "no CUDA device; dtype not exercised"

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    if want is torch.bfloat16 and major < 8:
        return torch.float16, (
            f"requested bfloat16 but {name} is compute capability {major}.{minor}; "
            "native bf16 needs 8.0+, and below that PyTorch emulates it. Using float16."
        )
    return want, f"{requested} on {name} (sm_{major}{minor})"


def resolve_attn_implementation(requested: str) -> tuple[str, str]:
    """Fall back from flash_attention_2 on hardware that cannot run it."""
    if requested != "flash_attention_2":
        return requested, ""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8:
            return "sdpa", "flash_attention_2 needs Ampere or newer; using sdpa"
    except ImportError:
        pass
    return requested, ""


def resolve_target_modules(cfg: ModelConfig) -> list[str]:
    """LoRA target module names, honouring the vision/language switches.

    Names that do not occur in a given backbone are ignored by PEFT, so this stays
    correct across the fallback ladder. What it must never do is silently target
    nothing on one side — which is why the result is verified by
    :func:`~chartqa_dt.modeling.lora_assert.assert_lora_on_both_sides` afterwards.
    """
    mods: list[str] = list(cfg.lora_target_modules) if cfg.lora_on_language else []
    if cfg.lora_on_vision:
        mods += list(QWEN_VISION_TARGETS)
    if not mods:
        raise ValueError(
            "no LoRA target modules: both lora_on_vision and lora_on_language are false"
        )
    return sorted(set(mods))
