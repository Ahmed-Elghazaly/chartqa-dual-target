

class TestTheSuccessBarIsWrittenDownFirst:
    """`PLAN.md` rule 2: never change a decision after seeing test results.

    Section 11 of the pre-registration promises the fine-tuned system will beat *our own
    zero-shot baseline*. That is only a commitment if the baseline is written down before
    the test split opens — otherwise it is a bar that can move. Section 12 records it, and
    the placeholder guard keeps the seal closed while it says TBD.
    """

    def test_the_preregistration_names_the_baselines_it_promises_to_beat(self) -> None:
        from pathlib import Path

        text = (Path(__file__).resolve().parents[1] / "PREREGISTRATION.md").read_text(
            encoding="utf-8")
        assert "zero-shot baselines this project must beat" in text
        for row in ("ChartQA relaxed accuracy", "RefChartQA AP@0.5", "RefChartQA P@F1"):
            assert row in text, f"section 12 does not record {row}"

    def test_an_unfilled_baseline_table_keeps_the_seal_closed(self, tmp_path) -> None:
        from chartqa_dt.splits import PREREGISTRATION_PLACEHOLDERS

        table = "| ChartQA relaxed accuracy | human | **TBD** |"
        assert any(marker in table for marker in PREREGISTRATION_PLACEHOLDERS), (
            "an unfilled baseline row must match a placeholder marker, or committing the "
            "document would open the test split with the bar still blank")
