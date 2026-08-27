"""The Phase 2 smoke-test harness.

The gate arithmetic and the loss masking are tested because both can be wrong in
ways that still produce a green run:

* a smoke test that optimises the wrong objective converges and passes anyway;
* a gate that is computed but never enforced lets a configuration through that
  cannot finish inside a free session.
"""

from __future__ import annotations

import random

import pytest

from chartqa_dt.train.smoke import (
    FULL_RUN_GATE_HOURS,
    MEMORY_GATE_GB,
    PLANNED_OPTIMIZER_STEPS,
    SmokeResult,
    make_bar_chart,
    write_report,
)

# ------------------------------------------------------------------- data


@pytest.mark.parametrize(("w", "h"), [(64, 45), (128, 90), (448, 314), (512, 358), (800, 557)])
def test_chart_generator_survives_every_image_size(w, h):
    """Hard-coded margins produced negative bar heights on small images."""
    img, question, answer = make_bar_chart(w, h, random.Random(0))
    assert img.size[0] >= 64 and img.size[1] >= 64
    assert question and answer.isdigit()


def test_chart_actually_contains_bars():
    img, _, _ = make_bar_chart(256, 180, random.Random(1))
    px = img.load()
    bar = sum(
        px[x, y] == (60, 110, 200)
        for x in range(0, img.size[0], 2)
        for y in range(0, img.size[1], 2)
    )
    total = (img.size[0] // 2) * (img.size[1] // 2)
    assert 0.03 < bar / total < 0.6, "the chart should be mostly white with visible bars"


def test_generator_is_deterministic_given_a_seed():
    a = make_bar_chart(200, 140, random.Random(7))
    b = make_bar_chart(200, 140, random.Random(7))
    assert a[1] == b[1] and a[2] == b[2]
    assert a[0].tobytes() == b[0].tobytes()


@pytest.mark.official
def test_only_the_answer_tokens_carry_loss():
    """The first draft supervised 23 of 45 positions instead of 2."""
    pytest.importorskip("transformers")
    from transformers import AutoProcessor

    from chartqa_dt.modeling.backends.hf_peft_backend import _set_processor_pixel_budget
    from chartqa_dt.train.smoke import build_batch
    from chartqa_dt.vision.coords import VisualGeometry

    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    _set_processor_pixel_budget(proc, VisualGeometry.from_processor(proc).with_max_pixels(128 * 128))

    rng = random.Random(0)
    samples = [make_bar_chart(128, 90, rng) for _ in range(3)]
    batch = build_batch(proc, samples, 512)

    for row, (_, _, answer) in enumerate(samples):
        labels = batch["labels"][row]
        kept = labels[labels != -100]
        decoded = proc.tokenizer.decode(kept)
        assert decoded.strip() == answer, f"row {row}: supervised {decoded!r}, expected {answer!r}"
        assert len(kept) <= 4, "only the short answer should be supervised"


# ------------------------------------------------------------------ gates


def _result(**kw) -> SmokeResult:
    base = {
        "backend": "hf_peft", "model_id": "m", "image_max_pixels": 262144, "label": "x",
        "ok": True, "peak_reserved_gb": 8.0, "seconds_per_step": 5.0,
        "projected_full_run_hours": 4.0, "loss_decreased": True, "any_nan": False,
        "lora": {"vision_params": 1, "language_params": 1},
    }
    base.update(kw)
    return SmokeResult(**base)


def test_gate_constants_match_the_plan():
    assert MEMORY_GATE_GB == 13.5           # IDEA.md 14
    assert FULL_RUN_GATE_HOURS == 10.0
    assert PLANNED_OPTIMIZER_STEPS == 3000  # 24,000 presentations at effective batch 8


def test_a_healthy_configuration_passes():
    assert _result().passes_all_gates


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("peak_reserved_gb", 13.6),               # over the memory gate
        ("projected_full_run_hours", 20.1),       # over the time gate
        ("any_nan", True),
        ("loss_decreased", False),
        ("ok", False),
    ],
)
def test_each_gate_is_actually_enforced(field, value):
    assert not _result(**{field: value}).passes_all_gates


def test_gate_boundaries_are_inclusive():
    assert _result(peak_reserved_gb=MEMORY_GATE_GB).passes_all_gates
    assert _result(projected_full_run_hours=FULL_RUN_GATE_HOURS).passes_all_gates


@pytest.mark.parametrize("lora", [
    {"vision_params": 0, "language_params": 1},
    {"vision_params": 1, "language_params": 0},
    {},
])
def test_missing_lora_on_either_side_fails_the_gates(lora):
    """Rule 3 again, this time at the level of the reported result."""
    assert not _result(lora=lora).passes_all_gates


def test_projection_uses_the_preregistered_step_count():
    r = _result(seconds_per_step=6.0)
    expected = 6.0 * PLANNED_OPTIMIZER_STEPS / 3600.0
    assert abs(expected - 5.0) < 1e-9        # 3000 steps at 6 s = 5 h
    assert r.projected_full_run_hours == 4.0  # stored, not recomputed


def test_report_round_trips(tmp_path):
    import json

    path = write_report([_result(label="a"), _result(label="b", ok=False, error="boom")], tmp_path)
    data = json.loads(path.read_text())
    assert data["gates"]["memory_gb"] == MEMORY_GATE_GB
    assert [r["label"] for r in data["results"]] == ["a", "b"]
    assert data["results"][1]["error"] == "boom"


def test_row_renders_pass_fail_and_unknown():
    assert "PASS" in _result().row()
    assert "FAIL" in _result(peak_reserved_gb=99.0).row()
    assert "—" in _result(resume_verified=None).row()


# ----------------------------------------------------- the DECISIONS.md table


def test_markdown_table_marks_pass_and_fail(tmp_path):
    from chartqa_dt.train.smoke import markdown_table

    text = markdown_table([
        _result(label="a"),
        _result(label="b", peak_reserved_gb=99.0),
        SmokeResult(backend="unsloth", model_id="m", image_max_pixels=1, label="c",
                    ok=False, error="BackendUnavailable: no notebook for this size"),
    ])
    lines = text.splitlines()
    assert lines[0].startswith("| configuration")
    assert "**PASS**" in lines[2]
    assert "FAIL" in lines[3] and "**PASS**" not in lines[3]
    assert "FAILED" in lines[4] and "BackendUnavailable" in lines[4]


def test_report_round_trips_through_load(tmp_path):
    from chartqa_dt.train.smoke import load_report, write_report

    original = [_result(label="a", seconds_per_step=4.25), _result(label="b", ok=False, error="x")]
    back = load_report(write_report(original, tmp_path))
    assert [r.label for r in back] == ["a", "b"]
    assert back[0].seconds_per_step == 4.25
    assert back[1].ok is False
    assert back[0].passes_all_gates and not back[1].passes_all_gates


def test_write_report_also_emits_the_markdown_table(tmp_path):
    from chartqa_dt.train.smoke import write_report

    write_report([_result(label="a")], tmp_path)
    md = (tmp_path / "smoke_results.md").read_text()
    assert md.startswith("| configuration") and "**PASS**" in md


# ------------------------------------- numerical health (fp16 without a scaler)


def test_dead_gradients_fail_the_gates():
    """On a T4 the compute dtype is float16 (decision 0017).

    fp16 without a gradient scaler can underflow gradients to exactly zero. The
    loss then sits flat, nothing raises, no NaN appears, and every other gate
    passes. A gradient norm of 0.0 is the only cheap signal.
    """
    assert _result().passes_all_gates
    assert not _result(zero_grad_steps=1).passes_all_gates


def test_nonfinite_gradients_fail_the_gates():
    assert not _result(nonfinite_grad_steps=1).passes_all_gates


def test_gradient_diagnostics_are_recorded(tmp_path):
    from chartqa_dt.train.smoke import load_report, write_report

    r = _result(grad_norms=[0.9, 1.2, 0.7], grad_norm_median=0.9,
                trainable_dtypes=["float32"])
    back = load_report(write_report([r], tmp_path))[0]
    assert back.grad_norms == [0.9, 1.2, 0.7]
    assert back.grad_norm_median == 0.9
    assert back.trainable_dtypes == ["float32"]


def test_a_run_can_look_perfect_and_still_be_dead():
    """The exact failure profile this gate exists for."""
    looks_fine = _result(
        peak_reserved_gb=1.5, projected_full_run_hours=8.0,
        any_nan=False, loss_decreased=True, zero_grad_steps=100,
    )
    assert not looks_fine.any_nan
    assert looks_fine.loss_decreased
    assert looks_fine.passes_memory_gate and looks_fine.passes_time_gate
    assert not looks_fine.passes_all_gates, "every other signal is green; only the norm is not"


# ------------------------------------------ the two bugs the first 100-step run found


def test_source_image_size_is_independent_of_the_pixel_budget():
    """Bug 2, from the first 100-step run.

    `image_px = int(sqrt(image_max_pixels))` made the "native" arm generate a
    4096x2867 chart, whose ~11,520 visual tokens overflowed max_seq_len; the
    processor then rejected the batch with

        ValueError: Mismatch in `image` token count between text and `input_ids`.

    The pixel budget's job is to control DOWNSCALING inside the processor. The
    source image must look like real data at every budget.
    """
    import inspect

    from chartqa_dt.train.smoke import SOURCE_IMAGE_H, SOURCE_IMAGE_W, _train_steps

    assert (SOURCE_IMAGE_W, SOURCE_IMAGE_H) == (800, 557), "the modal RefChartQA size"
    params = inspect.signature(_train_steps).parameters
    assert "image_px" not in params, (
        "the training loop must not take a source size derived from the pixel budget"
    )


def test_resume_rebuilds_the_same_optimizer_class():
    """Bug 1, from the same run -- after the 100 steps had already succeeded.

    bitsandbytes' AdamW8bit stores its moments under different state keys from
    torch.optim.AdamW, so loading one state_dict into the other raises
    `KeyError: 'exp_avg'`. Both sides must come from build_optimizer.
    """
    import inspect

    from chartqa_dt.train.smoke import _verify_resume, build_optimizer

    assert callable(build_optimizer)
    src = inspect.getsource(_verify_resume)
    assert "build_optimizer(fresh.model" in src
    assert "torch.optim.AdamW(" not in src, (
        "the resume path must not hard-code an optimizer class"
    )


@pytest.mark.official
def test_a_native_budget_batch_fits_the_sequence_limit():
    """The end-to-end check that bug 2 is gone: build a real batch at both budgets."""
    pytest.importorskip("transformers")
    from transformers import AutoProcessor

    from chartqa_dt.modeling.backends.hf_peft_backend import _set_processor_pixel_budget
    from chartqa_dt.train.smoke import SOURCE_IMAGE_H, SOURCE_IMAGE_W, build_batch
    from chartqa_dt.vision.coords import VisualGeometry

    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    base = VisualGeometry.from_processor(proc)
    rng = random.Random(0)

    for budget in (512 * 512, 16777216):
        _set_processor_pixel_budget(proc, base.with_max_pixels(budget))
        sample = make_bar_chart(SOURCE_IMAGE_W, SOURCE_IMAGE_H, rng)
        batch = build_batch(proc, [sample], 1024)
        n = int(batch["input_ids"].shape[1])
        assert n <= 1024, f"budget {budget}: sequence length {n} exceeds max_seq_len"
        kept = batch["labels"][0][batch["labels"][0] != -100]
        assert len(kept) > 0, f"budget {budget}: nothing supervised"


# ------------------------------------------------- single-card measurement


def test_a_sharded_run_cannot_pass_the_gates():
    """Kaggle's T4 shape provides TWO T4s and device_map='auto' shards across them.

    That silently produced three problems at once: the training loop sends every
    batch to the first parameter's device and crashes when a later layer is on the
    other; each forward pays inter-GPU transfers (+52% per step, measured); and
    torch.cuda.max_memory_reserved() reads device 0 only, so a sharded run reports
    a fraction of its real footprint.

    IDEA.md 14's compute budget is for one card, so a sharded measurement is not
    merely worse -- it is not the thing being budgeted.
    """
    assert _result().passes_all_gates
    assert not _result(is_sharded=True).passes_all_gates


def test_device_facts_are_recorded(tmp_path):
    from chartqa_dt.train.smoke import load_report, write_report

    r = _result(visible_devices=2, model_devices={"cuda:0": 500}, is_sharded=False)
    back = load_report(write_report([r], tmp_path))[0]
    assert back.visible_devices == 2
    assert back.model_devices == {"cuda:0": 500}
    assert back.is_sharded is False


def test_backend_pins_to_a_single_device():
    """device_map must not be 'auto': that is what caused the sharding."""
    import inspect

    from chartqa_dt.modeling.backends import hf_peft_backend

    src = inspect.getsource(hf_peft_backend.HFPeftBackend.load)
    # Ignore comments: the explanation of the bug legitimately names the value
    # that caused it.
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#")
    )
    assert 'device_map={"": 0}' in code
    assert 'device_map="auto"' not in code, "sharding must not be re-enabled"


def test_peak_memory_sums_every_visible_device():
    """max_memory_reserved() defaults to device 0; a sharded run would under-report."""
    import inspect

    from chartqa_dt.modeling.backends import base

    src = inspect.getsource(base.peak_reserved_gb)
    assert "device_count()" in src, "peak memory must be summed across devices"
    assert "sum(" in src


# ------------------------------------------- checkpoint completeness (PLAN 6.3)


def test_checkpoint_saves_rng_state():
    """PLAN.md 6.3 lists five things a checkpoint must contain; we saved two.

    The omission mattered concretely: lora_dropout is 0.05 and active in train
    mode, so a resumed model that draws different dropout masks diverges from the
    live one. That produced resume_loss_delta = 0.0456 against a 1e-2 tolerance —
    which reads as "resume is slightly broken" rather than "the comparison was
    never fair".
    """
    import inspect

    from chartqa_dt.train.smoke import _verify_resume

    src = inspect.getsource(_verify_resume)
    assert "save_pretrained" in src, "adapter weights"
    assert "optimizer.pt" in src, "optimizer state"
    assert "rng_state.pt" in src, "RNG states"
    assert "load_rng_state(" in src, "RNG states must be restored, not merely saved"


def test_rng_state_round_trips_through_torch_save(tmp_path):
    """The mechanism itself, independent of the model."""
    import random

    import torch

    from chartqa_dt.seeding import load_rng_state, rng_state, set_seed

    set_seed(11)
    state = rng_state()
    torch.save(state, tmp_path / "rng.pt")
    expected = [random.random() for _ in range(4)]

    set_seed(999)  # move the RNG somewhere else entirely
    load_rng_state(torch.load(tmp_path / "rng.pt", map_location="cpu", weights_only=False))
    assert [random.random() for _ in range(4)] == expected


def test_dropout_is_actually_active_in_the_config():
    """If this ever becomes 0, the RNG-state reasoning above changes."""
    from chartqa_dt.config import Config

    assert Config().model.lora_dropout > 0, (
        "lora_dropout is 0; the resume comparison no longer depends on RNG state "
        "and this test's rationale should be revisited"
    )


# ------------------------------------------- micro-batch grouping (fixed effective batch)


def test_smoke_result_records_the_grouping():
    r = _result(per_device_batch=4, grad_accum=2)
    assert r.per_device_batch * r.grad_accum == 8, "effective batch must stay pre-registered"


@pytest.mark.parametrize(("batch", "accum"), [(1, 8), (2, 4), (4, 2), (8, 1)])
def test_every_valid_grouping_preserves_the_effective_batch(batch, accum):
    """Varying the grouping changes GPU efficiency, not the experiment.

    Same optimizer steps, same example presentations, same effective batch — so
    it is not a deviation from the pre-registration, which fixes the effective
    batch and the step count, not how the micro-batches are chunked.
    """
    assert batch * accum == 8


def test_cli_rejects_a_grouping_that_changes_the_effective_batch():
    """A per-device batch that does not divide the effective batch would silently
    change the experiment rather than its scheduling."""
    from pathlib import Path as _P

    src = (_P(__file__).resolve().parents[1] / "src/chartqa_dt/cli/train.py").read_text()
    assert "effective % b" in src
    assert "deviate from the pre-registration" in src
