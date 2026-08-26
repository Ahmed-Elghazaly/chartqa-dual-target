"""LoRA attachment verified against the REAL Qwen3-VL architecture.

`PLAN.md` 2.3 is explicit: *"Verify the pattern lists against the actual parameter
names of the model you load — print them first, then adjust. Do not assume."*

That instruction earned its place immediately. The first draft of
``resolve_target_modules`` guessed the vision MLP was named ``fc1``/``fc2``. It is
actually ``linear_fc1``/``linear_fc2``, so LoRA would have reached the vision
tower's attention but **not** its MLP — while still passing a naive "are there any
trainable vision parameters?" check, because ``qkv`` and ``attn.proj`` matched.

These tests instantiate the genuine ``Qwen3VLForConditionalGeneration`` from the
published config with the layer sizes shrunk, so the module *names and structure*
are exactly the real ones while the weights are ~10 MB of random numbers. That
buys the verification without a 4 GB download, and it runs in CI.
"""

from __future__ import annotations

import pytest

from chartqa_dt.config import ModelConfig
from chartqa_dt.modeling.backends.base import (
    QWEN_VISION_TARGETS,
    BackendUnavailable,
    get_backend,
    list_backends,
    resolve_target_modules,
)
from chartqa_dt.modeling.lora_assert import (
    LoRACoverageError,
    assert_lora_on_both_sides,
    summarise_lora,
)

pytestmark = pytest.mark.official

QWEN3VL_ID = "Qwen/Qwen3-VL-2B-Instruct"


@pytest.fixture(scope="module")
def tiny_config():
    """The published Qwen3-VL config with layer sizes shrunk. Fetched once."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(QWEN3VL_ID)
    t, v = cfg.text_config, cfg.vision_config
    t.hidden_size, t.num_hidden_layers = 64, 2
    t.num_attention_heads, t.num_key_value_heads, t.intermediate_size = 4, 2, 128
    v.hidden_size, v.depth, v.intermediate_size, v.num_heads = 64, 2, 128, 4
    v.out_hidden_size = 64
    if hasattr(v, "deepstack_visual_indexes"):
        v.deepstack_visual_indexes = [0]
    return cfg


@pytest.fixture
def tiny_qwen3vl(tiny_config):
    """A FRESH model per test.

    Module scope would be cheaper, but PEFT mutates the model in place: a test
    that wraps the shared instance leaves its adapters attached for every later
    test. That silently broke the language-only test, which then could not
    observe the vision-side failure it exists to detect — a detector defeated by
    fixture scope is no detector at all.
    """
    import copy

    import torch
    from transformers import AutoModelForImageTextToText

    torch.manual_seed(0)
    return AutoModelForImageTextToText.from_config(copy.deepcopy(tiny_config))


# ------------------------------------------------------------- architecture


def test_the_real_vision_mlp_is_linear_fc_not_fc(tiny_qwen3vl):
    """Pins the naming that the first draft got wrong."""
    names = {n for n, _ in tiny_qwen3vl.named_parameters() if "visual" in n}
    assert any("linear_fc1" in n for n in names)
    assert any("linear_fc2" in n for n in names)
    assert not any(n.endswith(".fc1.weight") for n in names), (
        "if this ever passes, Qwen renamed the vision MLP and QWEN_VISION_TARGETS must be rechecked"
    )


def test_every_declared_vision_target_exists_in_the_real_model(tiny_qwen3vl):
    """A target that matches nothing is a silent no-op, so assert each one hits."""
    module_names = [n for n, _ in tiny_qwen3vl.named_modules()]
    for target in QWEN_VISION_TARGETS:
        hits = [n for n in module_names if n.endswith(target) and "visual" in n]
        assert hits, f"vision LoRA target {target!r} matches no module in the real architecture"


def test_every_declared_language_target_exists_in_the_real_model(tiny_qwen3vl):
    module_names = [n for n, _ in tiny_qwen3vl.named_modules()]
    for target in ModelConfig().lora_target_modules:
        hits = [n for n in module_names if n.endswith(target) and "language_model" in n]
        assert hits, f"language LoRA target {target!r} matches no module in the real architecture"


def test_attn_proj_target_excludes_the_patch_embedding(tiny_qwen3vl):
    """A bare `proj` would also adapt the Conv3d patch embedding. It must not."""
    modules = dict(tiny_qwen3vl.named_modules())
    bare = [n for n in modules if n.endswith("proj") and "visual" in n]
    assert any("patch_embed" in n for n in bare), "precondition: patch_embed.proj exists"
    targeted = [n for n in modules if n.endswith("attn.proj") and "visual" in n]
    assert targeted and not any("patch_embed" in n for n in targeted)


# ------------------------------------------------------------- LoRA wiring


@pytest.fixture
def lora_model(tiny_qwen3vl):
    peft = pytest.importorskip("peft")
    cfg = ModelConfig(lora_r=8, lora_alpha=16, lora_dropout=0.0)
    lcfg = peft.LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=resolve_target_modules(cfg),
    )
    return peft.get_peft_model(tiny_qwen3vl, lcfg)


def test_lora_reaches_both_sides_of_the_real_architecture(lora_model):
    """The end-to-end check for non-negotiable rule 3."""
    info = assert_lora_on_both_sides(lora_model, verbose=False)
    assert info["vision_params"] > 0, "LoRA did not reach the vision tower"
    assert info["language_params"] > 0, "LoRA did not reach the language model"
    assert info["vision_tensors"] >= 8   # 2 blocks x 4 targets x 2 matrices
    assert info["language_tensors"] >= 28  # 2 layers x 7 targets x 2 matrices


def test_lora_reaches_the_vision_mlp_not_only_its_attention(lora_model):
    """The failure the first draft would have shipped: attention adapted, MLP not.

    A bare `vision_params > 0` check passes that. This one does not.
    """
    trainable = [n for n, p in lora_model.named_parameters() if p.requires_grad]
    vision = [n for n in trainable if "visual" in n]
    assert any("linear_fc1" in n for n in vision), "vision MLP fc1 has no adapter"
    assert any("linear_fc2" in n for n in vision), "vision MLP fc2 has no adapter"
    assert any("qkv" in n for n in vision), "vision attention qkv has no adapter"
    assert any("attn.proj" in n for n in vision), "vision attention output has no adapter"


def test_patch_embedding_stays_frozen(lora_model):
    trainable = [n for n, p in lora_model.named_parameters() if p.requires_grad]
    assert not any("patch_embed" in n for n in trainable)


def test_vision_share_is_a_meaningful_fraction(lora_model):
    """Guards against a technically-attached-but-negligible vision adapter."""
    cov = summarise_lora(lora_model)
    assert cov.vision_share > 0.05, f"vision holds only {cov.vision_share:.1%} of trainable params"


def test_language_only_config_fails_the_assertion(tiny_qwen3vl):
    """Prove the detector fires on the real architecture, not just on fakes."""
    peft = pytest.importorskip("peft")
    cfg = ModelConfig(lora_on_vision=False)
    lcfg = peft.LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=resolve_target_modules(cfg),
    )
    model = peft.get_peft_model(tiny_qwen3vl, lcfg)
    with pytest.raises(LoRACoverageError, match="No trainable VISION parameters"):
        assert_lora_on_both_sides(model, verbose=False)


# ------------------------------------------------------------- the registry


def test_disabling_both_sides_is_rejected_rather_than_silently_empty():
    with pytest.raises(ValueError, match="no LoRA target modules"):
        resolve_target_modules(ModelConfig(lora_on_vision=False, lora_on_language=False))


def test_registry_lists_both_backends():
    assert set(list_backends()) == {"hf_peft", "unsloth"}


def test_unknown_backend_raises():
    with pytest.raises(BackendUnavailable, match="unknown backend"):
        get_backend("tensorflow")


def test_unavailable_backend_reports_why_rather_than_falling_back():
    """Phase 2 must record which backends work; a silent substitution erases that."""
    ok, why = list_backends()["unsloth"]
    if not ok:
        assert why, "an unavailable backend must say why"
        with pytest.raises(BackendUnavailable, match="not usable here"):
            get_backend("unsloth")
