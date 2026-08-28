"""The Phase 7 report must refuse to overstate what the project measured."""

from __future__ import annotations

import pytest

from chartqa_dt.eval.results_table import (
    PUBLISHED_REFCHARTQA_AP50,
    PUBLISHED_STATUS,
    Cell,
    Claim,
    Claims,
    Comparison,
    SystemRow,
    assert_claims_honest,
    build_report,
    render_table,
)


def _claims(**overrides: Claim) -> Claims:
    base = {
        "official_training_reproduced": Claim("no", "No RefChartQA training was run."),
        "official_checkpoint_evaluated": Claim("no", "No checkpoint is released."),
        "matched_student_baseline_trained": Claim("yes", "Direct-answer LoRA, same data."),
        "before_after_improvement": Claim("yes", "See comparison table."),
        "published_baseline_exceeded": Claim("not applicable", PUBLISHED_STATUS),
    }
    return Claims(**{**base, **overrides})


class TestCell:
    def test_interval_must_contain_the_point_estimate(self) -> None:
        with pytest.raises(ValueError, match="does not contain"):
            Cell(value=0.5, lo=0.6, hi=0.7)

    def test_renders_as_percentage_with_its_interval(self) -> None:
        assert Cell(0.5, 0.43, 0.57).render() == "50.00% [43.00, 57.00]"

    def test_overlap_is_symmetric(self) -> None:
        a, b = Cell(0.50, 0.45, 0.55), Cell(0.54, 0.52, 0.60)
        assert a.overlaps(b) and b.overlaps(a)

    def test_disjoint_intervals_do_not_overlap(self) -> None:
        assert not Cell(0.50, 0.45, 0.55).overlaps(Cell(0.70, 0.60, 0.80))


class TestTable:
    def test_missing_cells_render_as_dashes_never_as_numbers(self) -> None:
        table = render_table([SystemRow("Zero-shot", chartqa_all=Cell(0.5, 0.4, 0.6))])
        assert "| Zero-shot | — | — | 50.00% [40.00, 60.00] | — | — |" in table

    def test_published_reference_always_carries_its_status(self) -> None:
        table = render_table([SystemRow("Ours", chartqa_all=Cell(0.6, 0.5, 0.7))])
        assert f"{PUBLISHED_REFCHARTQA_AP50:.2f}" in table
        assert "Level C" in table


class TestComparison:
    def _cmp(self, *, matched: bool, system: Cell) -> Comparison:
        return Comparison(label="Grounded vs zero-shot", baseline="Zero-shot",
                          system="Grounded", metric="ChartQA relaxed",
                          baseline_cell=Cell(0.50, 0.45, 0.55), system_cell=system,
                          matched=matched, matched_on="same base model, same prompt")

    def test_unmatched_comparison_says_so_loudly(self) -> None:
        rendered = self._cmp(matched=False, system=Cell(0.70, 0.65, 0.75)).render()
        assert "**NOT matched**" in rendered

    def test_overlapping_intervals_are_not_called_a_gain(self) -> None:
        rendered = self._cmp(matched=True, system=Cell(0.54, 0.49, 0.59)).render()
        assert "not yet a demonstrated gain" in rendered

    def test_disjoint_intervals_are_reported_as_disjoint(self) -> None:
        rendered = self._cmp(matched=True, system=Cell(0.70, 0.65, 0.75)).render()
        assert "intervals are disjoint" in rendered
        assert "+20.00 pts" in rendered


class TestClaims:
    def test_five_claims_render_separately_and_numbered(self) -> None:
        rendered = _claims().render()
        for n in range(1, 6):
            assert f"{n}. **" in rendered
        assert "the mandatory result" in rendered

    def test_cannot_claim_official_training_was_reproduced(self) -> None:
        claims = _claims(official_training_reproduced=Claim("yes", "we retrained it"))
        with pytest.raises(AssertionError, match="retrains RefChartQA"):
            assert_claims_honest(claims)

    def test_cannot_claim_the_published_baseline_bare(self) -> None:
        claims = _claims(published_baseline_exceeded=Claim("yes", "we scored 40 vs 32.83"))
        with pytest.raises(AssertionError, match="genuinely comparable"):
            assert_claims_honest(claims)

    def test_may_claim_it_when_the_comparability_status_is_carried(self) -> None:
        claims = _claims(published_baseline_exceeded=Claim(
            "yes", f"40.1 [38, 42] against 32.83, which is {PUBLISHED_STATUS}."))
        assert_claims_honest(claims)  # does not raise


class TestReport:
    def test_report_lists_unmatched_comparisons_for_the_writer(self) -> None:
        comparison = Comparison(label="vs published", baseline="RefChartQA paper",
                                system="Ours", metric="AP@0.5",
                                baseline_cell=Cell(0.3283, 0.3283, 0.3283),
                                system_cell=Cell(0.40, 0.38, 0.42), matched=False)
        report = build_report([SystemRow("Ours", refchartqa_ap50=Cell(0.4, 0.38, 0.42))],
                              [comparison], _claims())
        assert report["unmatched_comparisons"] == ["vs published"]

    def test_a_dishonest_report_cannot_be_built(self) -> None:
        claims = _claims(official_training_reproduced=Claim("yes", "reproduced"))
        with pytest.raises(AssertionError):
            build_report([], [], claims)
