"""The LoRA coverage assertion — non-negotiable rule 3.

These tests use a fake model rather than a real one on purpose: the point is to
prove the *detector* works, and a detector for a silent failure has to be tested
against a deliberately staged instance of that failure. Loading a 2B model to
check an assertion would also make this untestable in CI.
"""

from __future__ import annotations

import pytest

from chartqa_dt.modeling.lora_assert import (
    LoRACoverageError,
    assert_lora_on_both_sides,
    classify_parameter,
    print_parameter_names,
    summarise_lora,
)


class FakeParam:
    def __init__(self, n: int, requires_grad: bool):
        self._n = n
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._n


class FakeModel:
    """Minimal stand-in exposing only what the assertion touches."""

    def __init__(self, params: list[tuple[str, int, bool]]):
        self._params = params

    def named_parameters(self):
        for name, n, grad in self._params:
            yield name, FakeParam(n, grad)


# Realistic Qwen3-VL-style names, taken from the published architecture.
def build(*, vision_lora: bool, language_lora: bool, n_vision: int = 24, n_lang: int = 28):
    params: list[tuple[str, int, bool]] = []
    for i in range(n_vision):
        params.append((f"model.visual.blocks.{i}.attn.qkv.weight", 3_145_728, False))
        if vision_lora:
            params.append((f"base_model.model.model.visual.blocks.{i}.attn.qkv.lora_A.default.weight", 16_384, True))
            params.append((f"base_model.model.model.visual.blocks.{i}.attn.qkv.lora_B.default.weight", 49_152, True))
    for i in range(n_lang):
        params.append((f"model.language_model.layers.{i}.self_attn.q_proj.weight", 4_194_304, False))
        if language_lora:
            params.append((f"base_model.model.model.language_model.layers.{i}.self_attn.q_proj.lora_A.default.weight", 32_768, True))
            params.append((f"base_model.model.model.language_model.layers.{i}.self_attn.q_proj.lora_B.default.weight", 32_768, True))
    return FakeModel(params)


# ------------------------------------------------------------ classification


@pytest.mark.parametrize("name", [
    "model.visual.blocks.0.attn.qkv.weight",
    "vision_tower.encoder.layer.3.mlp.fc1.weight",
    "model.vision_model.embeddings.patch_embedding.weight",
    "base_model.model.model.visual.blocks.9.mlp.down_proj.lora_A.default.weight",
])
def test_vision_names_classify_as_vision(name):
    assert classify_parameter(name) == "vision"


@pytest.mark.parametrize("name", [
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.layers.17.mlp.up_proj.weight",
    "llm.decoder.layers.2.o_proj.weight",
])
def test_language_names_classify_as_language(name):
    assert classify_parameter(name) == "language"


def test_vision_is_tested_before_language():
    """A vision block also called 'layers' must not be misfiled as language.

    This ordering is the difference between detecting a vision-side failure and
    silently reporting it as success.
    """
    assert classify_parameter("model.visual.model.layers.3.attn.q_proj.weight") == "vision"


def test_unknown_names_are_other_not_silently_assigned():
    assert classify_parameter("merger.mlp.0.weight") == "other"
    assert classify_parameter("lm_head.weight") == "other"


# ------------------------------------------------------------ the assertion


def test_passes_when_lora_reaches_both_sides(capsys):
    info = assert_lora_on_both_sides(build(vision_lora=True, language_lora=True))
    assert info["vision_params"] > 0 and info["language_params"] > 0
    assert info["vision_tensors"] == 48 and info["language_tensors"] == 56
    assert 0.0 < info["trainable_fraction"] < 0.05
    printed = capsys.readouterr().out
    assert "trainable tensors:" in printed and "vision params:" in printed


def test_fails_loudly_when_lora_misses_the_vision_tower():
    """The exact Qwen3-VL bug (#2016/#2079). Nothing else in the pipeline notices."""
    with pytest.raises(LoRACoverageError, match="No trainable VISION parameters"):
        assert_lora_on_both_sides(build(vision_lora=False, language_lora=True), verbose=False)


def test_the_vision_failure_message_names_the_upstream_issues():
    with pytest.raises(LoRACoverageError) as exc:
        assert_lora_on_both_sides(build(vision_lora=False, language_lora=True), verbose=False)
    msg = str(exc.value)
    assert "#2016" in msg and "#2079" in msg
    assert "never widen the patterns just to make this pass" in msg


def test_fails_when_lora_misses_the_language_model():
    with pytest.raises(LoRACoverageError, match="No trainable LANGUAGE parameters"):
        assert_lora_on_both_sides(build(vision_lora=True, language_lora=False), verbose=False)


def test_fails_when_nothing_is_trainable():
    with pytest.raises(LoRACoverageError, match="No trainable parameters at all"):
        assert_lora_on_both_sides(build(vision_lora=False, language_lora=False), verbose=False)


def test_min_vision_share_catches_a_token_attachment():
    """The subtler failure: LoRA technically reaches vision, but only just.

    A naive `> 0` check passes this. It is still, in substance, a language-only
    fine-tune, and the CV claim of the project would not survive it.
    """
    model = build(vision_lora=True, language_lora=True, n_vision=1, n_lang=28)
    assert_lora_on_both_sides(model, verbose=False)          # passes the bare check
    with pytest.raises(LoRACoverageError, match="too thinly"):
        assert_lora_on_both_sides(model, min_vision_share=0.15, verbose=False)


def test_summary_counts_frozen_parameters_in_the_total():
    cov = summarise_lora(build(vision_lora=True, language_lora=True))
    assert cov.total_params > cov.total_trainable
    assert cov.total_trainable == cov.vision_params + cov.language_params + cov.other_params


def test_unclassified_trainable_tensors_are_reported_not_hidden():
    model = FakeModel([
        ("model.visual.blocks.0.lora_A.weight", 100, True),
        ("model.language_model.layers.0.lora_A.weight", 100, True),
        ("merger.mlp.0.lora_A.weight", 50, True),
    ])
    info = assert_lora_on_both_sides(model, verbose=False)
    assert info["other_params"] == 50
    assert "merger.mlp.0.lora_A.weight" in info["unclassified_names"]


def test_describe_flags_unclassified_tensors():
    model = FakeModel([
        ("model.visual.b.lora_A.weight", 10, True),
        ("model.language_model.l.lora_A.weight", 10, True),
        ("mystery.weight", 10, True),
    ])
    text = summarise_lora(model).describe()
    assert "matched neither side" in text


def test_print_parameter_names_filters(capsys):
    model = build(vision_lora=True, language_lora=True)
    names = print_parameter_names(model, pattern=r"visual.*lora_A", limit=3)
    assert names and all("visual" in n and "lora_A" in n for n in names)
    assert "and " in capsys.readouterr().out    # truncation notice
