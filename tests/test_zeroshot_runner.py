"""The Phase 5 runner's decision logic — `PLAN.md` 5.2.

The gate is the part that has to be right *before* the numbers exist, which is the whole
reason `PLAN.md` states it in advance. Once a comparison table is on screen it is very
easy to find a reason why one condition "does not really apply here", so the thresholds
live in `VARIANT_GATE` and the verdict is computed, not judged.

Nothing here touches a GPU: `decide_variant` is a pure function of a measurement table,
which is exactly what makes it testable before the measurement happens.
"""

from __future__ import annotations

import json

import pytest
from scripts.run_zeroshot import (
    VARIANT_GATE,
    VARIANTS,
    decide_variant,
    format_variant_table,
)


def table(*, inst_acc=0.70, think_acc=0.75, inst_lat=1.0, think_lat=1.5,
          think_valid=0.95, inst_valid=0.95):
    """`think_valid` is the SCHEMA-valid fraction — the rate the gate uses.

    JSON validity and schema validity come apart: the first probe produced a record with
    `"args": {"label": "Zara", "value": 99}`, which parses cleanly and the executor
    rejects. Rule 3 makes that a failure, so the gate is on the stricter number.
    """
    return {
        "instruct": {"relaxed_accuracy": inst_acc, "valid_json_fraction": inst_valid,
                     "schema_valid_fraction": inst_valid, "repaired_fraction": 0.02,
                     "median_latency_s": inst_lat, "median_new_tokens": 120,
                     "hit_token_cap_fraction": 0.0},
        "thinking": {"relaxed_accuracy": think_acc, "valid_json_fraction": think_valid,
                     "schema_valid_fraction": think_valid, "repaired_fraction": 0.05,
                     "median_latency_s": think_lat, "median_new_tokens": 300,
                     "hit_token_cap_fraction": 0.1},
    }


def test_the_gate_uses_schema_validity_not_merely_json_validity():
    """A record the executor rejects is a failure, whatever its syntax (rule 3)."""
    t = table()
    t["thinking"]["valid_json_fraction"] = 1.00      # parses every time
    t["thinking"]["schema_valid_fraction"] = 0.55    # ... and half are unusable
    d = decide_variant(t)
    assert d["choice"] == "instruct"
    assert d["checks"]["schema_valid_fraction"]["value"] == pytest.approx(0.55)


def test_the_gate_thresholds_are_the_ones_the_plan_states():
    """`PLAN.md` 5.2: >= 2 accuracy points, >= 90% valid JSON, <= 2x latency."""
    assert VARIANT_GATE == {"min_accuracy_gain_points": 2.0,
                            "min_valid_json_fraction": 0.90,
                            "max_latency_ratio": 2.0}


def test_thinking_wins_only_when_all_three_conditions_hold():
    d = decide_variant(table(think_acc=0.75, think_valid=0.94, think_lat=1.8))
    assert d["choice"] == "thinking"
    assert all(c["pass"] for c in d["checks"].values())


@pytest.mark.parametrize(("name", "kwargs"), [
    ("gain too small", {"think_acc": 0.71}),          # +1 point, needs +2
    ("gain exactly at the boundary minus epsilon", {"think_acc": 0.7199}),
    ("schema-invalid records", {"think_valid": 0.89}),
    ("too slow", {"think_lat": 2.01}),
    ("worse accuracy", {"think_acc": 0.60}),
])
def test_any_single_failure_selects_instruct(name, kwargs):
    """`PLAN.md` 5.2: "Otherwise choose Instruct." One failure is enough."""
    d = decide_variant(table(**kwargs))
    assert d["choice"] == "instruct", name
    assert not all(c["pass"] for c in d["checks"].values())


def test_the_boundaries_are_inclusive_as_written():
    """">= 2 points", ">= 90%", "<= 2x" — a value exactly on the line passes."""
    d = decide_variant(table(think_acc=0.72, think_valid=0.90, think_lat=2.0))
    assert d["choice"] == "thinking"
    for name, check in d["checks"].items():
        assert check["pass"], f"{name} should pass exactly on the boundary"


def test_a_single_measured_variant_defaults_to_instruct():
    t = table()
    del t["thinking"]
    d = decide_variant(t)
    assert d["choice"] == "instruct"
    assert "only one variant" in d["reason"]


def test_zero_latency_does_not_divide_by_zero():
    d = decide_variant(table(inst_lat=0.0))
    assert d["choice"] == "instruct"
    assert d["checks"]["latency_ratio"]["value"] == float("inf")


def test_the_decision_records_every_measured_value_not_just_the_verdict():
    """A verdict without its inputs cannot be checked by a reader later."""
    d = decide_variant(table(think_acc=0.78, think_valid=0.80, think_lat=3.0))
    checks = d["checks"]
    assert set(checks) == {"accuracy_gain_points", "schema_valid_fraction",
                           "latency_ratio"}
    assert checks["accuracy_gain_points"]["value"] == pytest.approx(8.0)
    assert checks["latency_ratio"]["value"] == pytest.approx(3.0)
    for c in checks.values():
        assert "required" in c and isinstance(c["pass"], bool)


def test_the_table_renders_the_verdict_and_its_reasons():
    t = table(think_acc=0.78, think_valid=0.80, think_lat=3.0)
    t["decision"] = decide_variant(t)
    text = format_variant_table(t)
    assert "instruct" in text and "FAIL" in text and "PASS" in text
    assert "schema_valid_fraction" in text
    assert "schema" in text and "capped" in text, \
        "the table must show what actually drove the verdict"


def test_the_decision_is_json_serialisable():
    """It is written to disk and quoted in the pre-registration."""
    d = decide_variant(table())
    assert json.loads(json.dumps(d))["choice"] in ("instruct", "thinking")


def test_both_variants_point_at_real_checkpoints():
    assert VARIANTS["instruct"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert VARIANTS["thinking"] == "Qwen/Qwen3-VL-2B-Thinking"


def test_the_prereg_generator_runs_before_the_numbers_exist():
    """`PREREGISTRATION.md` must be writable at any point, filling gaps honestly.

    If it could only be generated after 5.2, it could not be committed before the test
    split is opened — which is the one thing `PLAN.md` 5.5 requires of it.
    """
    import importlib

    module = importlib.import_module("scripts.write_prereg")
    assert hasattr(module, "main")
    assert module.read_json("verification/does-not-exist.json") == {}


class TestAdapterEvaluation:
    """Phase 7 evaluates the fine-tuned system through the SAME path as Phase 5.

    That is what makes the before/after comparison matched rather than merely adjacent:
    same prompts, same decoding, same slice logic, same scoring. A second runner would
    give both arms a chance to differ for reasons nobody chose.
    """

    def test_the_runner_accepts_an_adapter_and_a_tag(self) -> None:
        import inspect

        from scripts import run_zeroshot

        source = inspect.getsource(run_zeroshot.main)
        assert '"--adapter"' in source and '"--tag"' in source

    def test_every_generating_stage_threads_the_adapter_through(self) -> None:
        """A stage that quietly dropped it would report the base model's numbers as the
        fine-tuned ones, which is the worst possible silent failure here."""
        import inspect

        from scripts import run_zeroshot

        for stage in (run_zeroshot.stage_chartqa, run_zeroshot.stage_refchartqa):
            source = inspect.getsource(stage)
            assert "adapter=args.adapter" in source, f"{stage.__name__} drops --adapter"

    def test_output_names_carry_the_tag_so_a_baseline_is_not_overwritten(self) -> None:
        import inspect
        import types

        from scripts import run_zeroshot

        assert run_zeroshot.run_tag(types.SimpleNamespace(tag="")) == ""
        assert run_zeroshot.run_tag(types.SimpleNamespace(tag="finetuned")) == "_finetuned"
        for stage in (run_zeroshot.stage_chartqa, run_zeroshot.stage_refchartqa):
            assert "run_tag(args)" in inspect.getsource(stage), \
                f"{stage.__name__} would overwrite the zero-shot baseline it is compared to"

    def test_the_adapter_is_recorded_in_the_result_file(self) -> None:
        """A results file that does not say which adapter produced it cannot be trusted
        later to be the fine-tuned arm rather than the baseline."""
        import inspect

        from scripts import run_zeroshot

        for stage in (run_zeroshot.stage_chartqa, run_zeroshot.stage_refchartqa):
            assert '"adapter"' in inspect.getsource(stage)
