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
    return {
        "instruct": {"relaxed_accuracy": inst_acc, "valid_json_fraction": inst_valid,
                     "repaired_fraction": 0.02, "median_latency_s": inst_lat},
        "thinking": {"relaxed_accuracy": think_acc, "valid_json_fraction": think_valid,
                     "repaired_fraction": 0.05, "median_latency_s": think_lat},
    }


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
    ("invalid JSON", {"think_valid": 0.89}),
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
    assert set(checks) == {"accuracy_gain_points", "valid_json_fraction", "latency_ratio"}
    assert checks["accuracy_gain_points"]["value"] == pytest.approx(8.0)
    assert checks["latency_ratio"]["value"] == pytest.approx(3.0)
    for c in checks.values():
        assert "required" in c and isinstance(c["pass"], bool)


def test_the_table_renders_the_verdict_and_its_reasons():
    t = table(think_acc=0.78, think_valid=0.80, think_lat=3.0)
    t["decision"] = decide_variant(t)
    text = format_variant_table(t)
    assert "instruct" in text and "FAIL" in text and "PASS" in text
    assert "valid_json_fraction" in text


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
