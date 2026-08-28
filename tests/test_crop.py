"""`PLAN.md` 8.2. The plan expects this to fail; the accounting must let it."""

from __future__ import annotations

import pytest

from chartqa_dt.eval.crop import (
    CROP_EXPANSION,
    AcceptanceRule,
    CropBudget,
    ThirdPassRefused,
    crop_region,
    expand_box,
    run_crop_pass,
    should_crop,
)


class TestExpansion:
    def test_a_box_grows_by_the_pre_registered_fraction(self) -> None:
        assert CROP_EXPANSION == 0.15
        assert expand_box([400, 400, 600, 600]) == (385.0, 385.0, 615.0, 615.0)

    def test_expansion_is_clamped_to_the_image(self) -> None:
        assert expand_box([0, 0, 100, 100]) == (0.0, 0.0, 107.5, 107.5)

    def test_an_inverted_box_is_normalised_rather_than_producing_a_negative_crop(self) -> None:
        assert expand_box([600, 600, 400, 400]) == expand_box([400, 400, 600, 600])

    def test_the_region_is_in_original_pixels(self) -> None:
        assert crop_region([400, 400, 600, 600], 800, 600) == (308, 231, 492, 369)

    def test_a_tiny_box_yields_no_crop(self) -> None:
        """Cropping to a bad box removes the axis the value is read from."""
        assert crop_region([500, 500, 505, 505], 800, 600) is None

    def test_a_degenerate_box_yields_no_crop(self) -> None:
        assert crop_region([500, 500, 500, 500], 800, 600) is None


class TestBudget:
    def test_one_re_read_per_record(self) -> None:
        budget = CropBudget()
        budget.take("r1")
        with pytest.raises(ThirdPassRefused, match="forbids a third"):
            budget.take("r1")

    def test_different_records_each_get_one(self) -> None:
        budget = CropBudget()
        budget.take("r1")
        budget.take("r2")          # does not raise

    def test_the_refusal_names_why_it_matters(self) -> None:
        budget = CropBudget()
        budget.take("r1")
        with pytest.raises(ThirdPassRefused, match="search over the test set"):
            budget.take("r1")


class TestOffering:
    RULE = AcceptanceRule()

    def test_only_unreliable_records_are_offered_a_crop(self) -> None:
        """Cropping a record the system already got right can only cost."""
        assert not should_crop({"reliability": 0.9, "focus_box": [1, 2, 3, 4]}, self.RULE)
        assert should_crop({"reliability": 0.1, "focus_box": [1, 2, 3, 4]}, self.RULE)

    def test_a_record_with_no_focus_box_is_not_offered_one(self) -> None:
        assert not should_crop({"reliability": 0.1, "focus_box": []}, self.RULE)

    def test_a_record_with_no_reliability_score_is_not_offered_one(self) -> None:
        assert not should_crop({"focus_box": [1, 2, 3, 4]}, self.RULE)


class TestAcceptanceRule:
    def test_the_rule_is_compared_by_value_so_a_tuned_rule_is_a_different_rule(self) -> None:
        """Frozen on validation; a rule adjusted after seeing the help rate is not it."""
        assert AcceptanceRule() == AcceptanceRule()
        assert AcceptanceRule() != AcceptanceRule(max_reliability=0.8)

    def test_a_second_pass_that_fails_the_schema_is_rejected(self) -> None:
        assert not AcceptanceRule().accepts({"schema_valid": False, "agrees": True})

    def test_a_second_pass_whose_executor_disagrees_is_rejected(self) -> None:
        assert not AcceptanceRule().accepts({"schema_valid": True, "agrees": False})

    def test_boxes_outside_the_crop_are_rejected(self) -> None:
        assert not AcceptanceRule().accepts({"schema_valid": True, "agrees": True,
                                             "boxes_in_crop": False})

    def test_a_clean_second_pass_is_accepted(self) -> None:
        assert AcceptanceRule().accepts({"schema_valid": True, "agrees": True})


class TestAccounting:
    @staticmethod
    def _item(rid: str, *, correct: bool, reliability: float = 0.1) -> dict:
        return {"id": rid, "correct": correct, "reliability": reliability,
                "focus_box": [300, 300, 700, 700], "image_size": (800, 600)}

    def test_help_and_harm_are_counted_separately_never_netted(self) -> None:
        """A technique that fixes twenty and breaks twenty is a coin flip applied to
        answers that were already right, not a neutral result."""
        items = [self._item("a", correct=False), self._item("b", correct=True)]

        def reread(item, region):
            return {"schema_valid": True, "agrees": True,
                    "correct": item["id"] == "a"}      # fixes a, breaks b

        _, outcome = run_crop_pass(items, reread)
        report = outcome.to_dict(len(items))
        assert outcome.helped == 1 and outcome.harmed == 1
        assert report["help_rate"] == 0.5 and report["harm_rate"] == 0.5
        assert report["net_records"] == 0

    def test_reliable_records_are_never_re_read(self) -> None:
        calls = []
        items = [self._item("a", correct=True, reliability=0.95)]
        run_crop_pass(items, lambda i, r: calls.append(i) or {})
        assert calls == []

    def test_a_rejected_second_pass_leaves_the_first_answer_alone(self) -> None:
        items = [self._item("a", correct=True)]
        final, outcome = run_crop_pass(
            items, lambda i, r: {"schema_valid": False, "correct": False})
        assert outcome.rejected == 1 and outcome.accepted == 0
        assert final[0]["correct"] is True

    def test_an_unusable_box_is_counted_and_not_re_read(self) -> None:
        items = [{"id": "a", "correct": False, "reliability": 0.1,
                  "focus_box": [500, 500, 502, 502], "image_size": (800, 600)}]
        calls = []
        _, outcome = run_crop_pass(items, lambda i, r: calls.append(1) or {})
        assert outcome.unusable_box == 1 and calls == []

    def test_the_crop_region_is_recorded_on_an_accepted_record(self) -> None:
        items = [self._item("a", correct=False)]
        final, _ = run_crop_pass(
            items, lambda i, r: {"schema_valid": True, "agrees": True, "correct": True})
        assert final[0]["crop_region"] == crop_region([300, 300, 700, 700], 800, 600)

    def test_the_request_rate_is_over_all_records_not_just_offered_ones(self) -> None:
        items = [self._item("a", correct=False),
                 self._item("b", correct=True, reliability=0.99)]
        _, outcome = run_crop_pass(
            items, lambda i, r: {"schema_valid": True, "agrees": True, "correct": True})
        assert outcome.to_dict(len(items))["request_rate"] == 0.5

    def test_a_record_is_never_re_read_twice_within_one_pass(self) -> None:
        items = [self._item("a", correct=False), self._item("a", correct=False)]
        with pytest.raises(ThirdPassRefused):
            run_crop_pass(items, lambda i, r: {"schema_valid": True, "agrees": True,
                                               "correct": True})

    def test_the_description_shows_help_and_harm_side_by_side(self) -> None:
        items = [self._item("a", correct=False)]
        _, outcome = run_crop_pass(
            items, lambda i, r: {"schema_valid": True, "agrees": True, "correct": True})
        text = outcome.describe(len(items))
        assert "helped 1" in text and "harmed 0" in text and "net +1" in text
