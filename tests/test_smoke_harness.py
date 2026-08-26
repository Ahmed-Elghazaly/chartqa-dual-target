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
        ("projected_full_run_hours", 10.1),       # over the time gate
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
