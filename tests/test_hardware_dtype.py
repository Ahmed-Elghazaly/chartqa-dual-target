"""Compute dtype must match what the GPU can actually do.

A free Kaggle/Colab T4 is Turing, compute capability 7.5. **bfloat16 requires
Ampere (8.0).** On a T4 bf16 still runs and still produces correct numbers — it is
simply emulated rather than accelerated. On a timed memory benchmark that looks
like a bad result rather than a wrong setting, which is the failure mode this
whole project keeps meeting: nothing errors, the number is just worse.

The configs request bfloat16 because that is right on Ampere and later. The
substitution to float16 on older hardware happens automatically and is recorded
in the run notes, never silently.
"""

from __future__ import annotations

import pytest

from chartqa_dt.modeling.backends.base import resolve_attn_implementation, resolve_dtype

torch = pytest.importorskip("torch")


class FakeCuda:
    """Stands in for torch.cuda so the T4 path is testable without a T4."""

    def __init__(self, name, capability, bf16):
        self._name, self._cap, self._bf16 = name, capability, bf16

    def is_available(self):
        return True

    def get_device_name(self, i=0):
        return self._name

    def get_device_capability(self, i=0):
        return self._cap

    def is_bf16_supported(self, including_emulation: bool = True):
        # Mirrors torch: True on a T4 when emulation counts, False when it does not.
        return self._bf16 or including_emulation


@pytest.fixture
def as_t4(monkeypatch):
    monkeypatch.setattr(torch, "cuda", FakeCuda("Tesla T4", (7, 5), False))


@pytest.fixture
def as_a100(monkeypatch):
    monkeypatch.setattr(torch, "cuda", FakeCuda("NVIDIA A100-SXM4-40GB", (8, 0), True))


def test_bfloat16_is_downgraded_on_a_t4(as_t4):
    dtype, note = resolve_dtype("bfloat16")
    assert dtype is torch.float16
    assert "7.5" in note and "8.0+" in note
    assert "Tesla T4" in note, "the substitution must name the device that caused it"


def test_capability_is_used_rather_than_is_bf16_supported(as_t4):
    """The trap that broke the first version of this fix.

    torch.cuda.is_bf16_supported() takes `including_emulation: bool = True`, so on
    a T4 it returns True — it answers "can this run?", not "can this run fast?".
    A check built on it would never fire on exactly the hardware it exists for.
    """
    assert torch.cuda.is_bf16_supported() is True, "precondition: the helper says yes on a T4"
    assert torch.cuda.is_bf16_supported(including_emulation=False) is False
    assert resolve_dtype("bfloat16")[0] is torch.float16, (
        "resolve_dtype must key off compute capability, not the emulation-inclusive helper"
    )


def test_bfloat16_is_kept_on_ampere(as_a100):
    dtype, note = resolve_dtype("bfloat16")
    assert dtype is torch.bfloat16
    assert "A100" in note


def test_float16_is_never_upgraded(as_t4):
    assert resolve_dtype("float16")[0] is torch.float16


def test_the_substitution_is_always_reported(as_t4):
    """A silent downgrade would be indistinguishable from the config being honoured."""
    _, note = resolve_dtype("bfloat16")
    assert note, "resolve_dtype must always return a note for the run record"


def test_flash_attention_falls_back_on_pre_ampere(as_t4):
    impl, note = resolve_attn_implementation("flash_attention_2")
    assert impl == "sdpa"
    assert "Ampere" in note


def test_flash_attention_is_kept_on_ampere(as_a100):
    impl, note = resolve_attn_implementation("flash_attention_2")
    assert impl == "flash_attention_2"
    assert note == ""


def test_sdpa_is_left_alone_everywhere(as_t4):
    assert resolve_attn_implementation("sdpa") == ("sdpa", "")


def test_no_cuda_returns_the_request_unchanged(monkeypatch):
    class NoCuda:
        def is_available(self):
            return False

    monkeypatch.setattr(torch, "cuda", NoCuda())
    dtype, note = resolve_dtype("bfloat16")
    assert dtype is torch.bfloat16
    assert "no CUDA" in note
