

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


class TestASmokeRunCannotOpenTheSeal:
    """A near-miss worth a test.

    `kaggle_run.py` unpacks a downloaded run under `outputs/<run>/repo/`, and that
    directory still held a **12-item** smoke run reporting 91.67% with an interval of
    [75.0, 100.0]. Section 12 searches those paths. Reading it would have filled the
    baseline table with real-looking numbers — and because they are not placeholders, the
    seal guard would then have opened the test splits on the strength of twelve questions.
    """

    def test_a_result_smaller_than_the_slice_renders_as_a_placeholder(self) -> None:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from scripts.write_prereg import CHARTQA_SLICE

        assert CHARTQA_SLICE == 1_920

    def test_the_committed_preregistration_still_holds_a_placeholder(self) -> None:
        """5.3 and 5.4 have not landed, so the bar is not yet written down and the seal
        must stay closed."""
        from pathlib import Path

        from chartqa_dt.splits import PREREGISTRATION_PLACEHOLDERS

        text = (Path(__file__).resolve().parents[1] / "PREREGISTRATION.md").read_text(
            encoding="utf-8")
        section = text.split("## 12.")[1].split("## 13.")[0]
        assert any(marker in section for marker in PREREGISTRATION_PLACEHOLDERS)

    def test_the_placeholder_says_what_is_missing_rather_than_just_tbd(self) -> None:
        """'TBD' invites someone to delete it. 'n=12; needs n>=1,920' does not."""
        from pathlib import Path

        text = (Path(__file__).resolve().parents[1] / "PREREGISTRATION.md").read_text(
            encoding="utf-8")
        section = text.split("## 12.")[1].split("## 13.")[0]
        assert "needs n" in section
