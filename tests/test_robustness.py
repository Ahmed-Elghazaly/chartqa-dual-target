"""`PLAN.md` 9.5. A counterfactual's answer is recomputed, never assumed unchanged."""

from __future__ import annotations

import pytest
from PIL import Image

from chartqa_dt.synth.robustness import (
    APPEARANCE,
    VacuousCounterfactual,
    counterfactual,
    counterfactual_or_none,
    perturb_image,
    recompute_answer,
)

SERIES = [("2019", 245.0), ("2018", 210.0), ("2017", 180.0)]
DIFFERENCE = {"op": "difference", "args": ["2019", "2018"]}


def _chart(size=(64, 48)) -> Image.Image:
    img = Image.new("RGB", size, (250, 250, 250))
    for x in range(10, 30):
        for y in range(10, 40):
            img.putpixel((x, y), (30, 90, 200))
    return img


class TestAppearancePerturbations:
    @pytest.mark.parametrize("kind", sorted(APPEARANCE))
    def test_every_perturbation_returns_an_image_of_the_same_size(self, kind: str) -> None:
        out = perturb_image(_chart(), kind)
        assert out.size == (64, 48) and out.mode == "RGB"

    @pytest.mark.parametrize("kind", sorted(APPEARANCE))
    def test_every_perturbation_actually_changes_the_pixels(self, kind: str) -> None:
        """A no-op perturbation would report robustness the model has not demonstrated."""
        original = _chart()
        assert list(perturb_image(original, kind).get_flattened_data()) != list(original.get_flattened_data())

    def test_greyscale_removes_colour(self) -> None:
        out = perturb_image(_chart(), "greyscale")
        assert all(r == g == b for r, g, b in out.get_flattened_data())

    def test_an_unknown_perturbation_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="unknown perturbation"):
            perturb_image(_chart(), "rotate")

    def test_noise_is_reproducible_from_its_seed(self) -> None:
        a = perturb_image(_chart(), "noise", seed=7)
        b = perturb_image(_chart(), "noise", seed=7)
        assert list(a.get_flattened_data()) == list(b.get_flattened_data())


class TestRecomputation:
    def test_the_answer_comes_from_the_plan_and_the_data(self) -> None:
        assert recompute_answer(DIFFERENCE, SERIES) == "35"

    def test_changing_the_data_changes_the_answer(self) -> None:
        changed = [("2019", 392.0), ("2018", 210.0), ("2017", 180.0)]
        assert recompute_answer(DIFFERENCE, changed) == "182"


class TestCounterfactual:
    def test_the_new_answer_is_recomputed_not_carried_over(self) -> None:
        """Scoring against the OLD answer would mark a model that read the new chart
        correctly as wrong, and a model ignoring the image as right."""
        pair = counterfactual(SERIES, DIFFERENCE, "35", label="2019", factor=1.6)
        assert pair.answer == "182"
        assert dict(pair.series)["2019"] == pytest.approx(392.0)
        assert dict(pair.series)["2018"] == pytest.approx(210.0), "only one value moves"

    def test_a_pair_whose_answer_did_not_move_is_refused(self) -> None:
        """2017 is not read by this plan, so changing it cannot change the answer."""
        with pytest.raises(VacuousCounterfactual, match="left the answer at"):
            counterfactual(SERIES, DIFFERENCE, "35", label="2017")

    def test_the_refusal_explains_why_the_pair_is_worthless(self) -> None:
        with pytest.raises(VacuousCounterfactual) as exc:
            counterfactual(SERIES, DIFFERENCE, "35", label="2017")
        assert "duplicate of the original" in str(exc.value)

    def test_the_default_label_is_one_the_plan_reads(self) -> None:
        pair = counterfactual(SERIES, DIFFERENCE, "35")
        assert pair.series is not None
        moved = [lab for lab, v in pair.series if v != dict(SERIES)[lab]]
        assert moved and moved[0] in ("2019", "2018")

    def test_a_label_not_in_the_series_is_refused(self) -> None:
        with pytest.raises(VacuousCounterfactual, match="not in the series"):
            counterfactual(SERIES, DIFFERENCE, "35", label="1999")

    def test_a_single_point_series_cannot_be_varied(self) -> None:
        with pytest.raises(VacuousCounterfactual, match="fewer than two"):
            counterfactual([("A", 1.0)], {"op": "lookup", "args": ["A"]}, "1")

    def test_a_change_below_the_relaxed_tolerance_counts_as_vacuous(self) -> None:
        """The metric cannot distinguish them, so the pair adds nothing."""
        with pytest.raises(VacuousCounterfactual):
            counterfactual(SERIES, {"op": "lookup", "args": ["2019"]}, "245",
                           label="2019", factor=1.01)

    def test_the_batch_helper_returns_none_instead_of_raising(self) -> None:
        assert counterfactual_or_none(SERIES, DIFFERENCE, "35", label="2017") is None
        assert counterfactual_or_none(SERIES, DIFFERENCE, "35", label="2019") is not None

    def test_the_batch_helper_survives_an_unexecutable_plan(self) -> None:
        assert counterfactual_or_none(SERIES, {"op": "lookup", "args": ["Nowhere"]},
                                      "1") is None
