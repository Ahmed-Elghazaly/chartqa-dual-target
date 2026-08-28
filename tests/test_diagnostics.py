"""`PLAN.md` 9.3 and 9.4. Tree comparison must respect argument order where it matters."""

from __future__ import annotations

import json

import pytest

from chartqa_dt.eval.diagnostics import (
    diagnose,
    diagnose_one,
    normalise_plan,
    trees_match,
)

EVIDENCE = [{"label": "2019", "value": 245, "unit": "units", "bbox": [1, 2, 3, 4]},
            {"label": "2018", "value": 210, "unit": "units", "bbox": [5, 6, 7, 8]}]


def _raw(plan: dict, answer: str = "35", evidence=None) -> str:
    return json.dumps({"answerable": True, "evidence": evidence or EVIDENCE,
                       "plan": plan, "model_answer": answer}, separators=(",", ":"))


class TestTreeComparison:
    def test_commutative_arguments_may_be_reordered(self) -> None:
        assert trees_match({"op": "sum", "args": ["A", "B"]},
                           {"op": "sum", "args": ["B", "A"]})

    def test_order_matters_for_difference(self) -> None:
        """difference(A,B) and difference(B,A) differ by a sign. Calling them equal would
        report agreement on exactly the error the executor exists to catch."""
        assert not trees_match({"op": "difference", "args": ["A", "B"]},
                               {"op": "difference", "args": ["B", "A"]})

    @pytest.mark.parametrize("op", ["ratio", "percent_change", "compare", "trend"])
    def test_order_matters_for_every_non_commutative_operation(self, op: str) -> None:
        assert not trees_match({"op": op, "args": ["A", "B"]},
                               {"op": op, "args": ["B", "A"]})

    def test_a_different_operation_never_matches(self) -> None:
        assert not trees_match({"op": "sum", "args": ["A", "B"]},
                               {"op": "mean", "args": ["A", "B"]})

    def test_nested_trees_are_compared_recursively(self) -> None:
        a = {"op": "difference", "args": ["X", {"op": "sum", "args": ["A", "B"]}]}
        b = {"op": "difference", "args": ["X", {"op": "sum", "args": ["B", "A"]}]}
        assert trees_match(a, b)

    def test_a_missing_plan_does_not_match_anything(self) -> None:
        assert not trees_match(None, {"op": "sum", "args": []})

    def test_normalisation_is_stable(self) -> None:
        plan = {"op": "sum", "args": ["B", "A"]}
        assert normalise_plan(plan) == normalise_plan(normalise_plan(plan))


class TestDiagnoseOne:
    def test_a_good_record_passes_everything(self) -> None:
        out = diagnose_one(_raw({"op": "difference", "args": ["2019", "2018"]}),
                           gold_plan={"op": "difference", "args": ["2019", "2018"]},
                           gold_evidence=EVIDENCE)
        assert all(out[k] for k in ("parsed", "schema_valid", "has_plan", "executes",
                                    "agrees", "tree_exact", "operands_exact",
                                    "units_exact"))

    def test_unparseable_output_fails_every_measure(self) -> None:
        """Rule 4: an invalid record is a failed record, never an abstention."""
        out = diagnose_one("sorry, I cannot", gold_plan={"op": "sum", "args": []})
        assert not any(v for v in out.values() if isinstance(v, bool))

    def test_the_right_answer_with_the_wrong_tree_is_caught(self) -> None:
        """Executor agreement and tree match are different questions."""
        out = diagnose_one(_raw({"op": "difference", "args": ["2019", "2018"]}),
                           gold_plan={"op": "difference", "args": ["2018", "2019"]})
        assert out["agrees"] and not out["tree_exact"]

    def test_wrong_operands_are_reported_separately_from_the_tree(self) -> None:
        other = [{"label": "2017", "value": 245, "unit": "units", "bbox": [1, 2, 3, 4]},
                 {"label": "2016", "value": 210, "unit": "units", "bbox": [5, 6, 7, 8]}]
        out = diagnose_one(_raw({"op": "difference", "args": ["2017", "2016"]},
                                evidence=other),
                           gold_plan={"op": "difference", "args": ["2017", "2016"]},
                           gold_evidence=EVIDENCE)
        assert out["tree_exact"] and not out["operands_exact"]

    def test_a_wrong_unit_fails_units_but_not_operands(self) -> None:
        wrong = [{**e, "unit": "percent"} for e in EVIDENCE]
        out = diagnose_one(_raw({"op": "difference", "args": ["2019", "2018"]},
                                evidence=wrong), gold_evidence=EVIDENCE)
        assert out["operands_exact"] and not out["units_exact"]

    def test_a_plan_that_cannot_execute_is_not_counted_as_agreeing(self) -> None:
        out = diagnose_one(_raw({"op": "lookup", "args": ["Nowhere"]}))
        assert out["has_plan"] and not out["executes"] and not out["agrees"]


class TestTransfer:
    def _items(self, synth_ok: int, synth_bad: int, real_ok: int, real_bad: int) -> list:
        good = _raw({"op": "difference", "args": ["2019", "2018"]})
        bad = "not a record"
        return ([{"raw": good, "source": "synthetic"}] * synth_ok
                + [{"raw": bad, "source": "synthetic"}] * synth_bad
                + [{"raw": good, "source": "chartqa"}] * real_ok
                + [{"raw": bad, "source": "chartqa"}] * real_bad)

    def test_the_drop_from_synthetic_to_real_is_reported_in_points(self) -> None:
        result = diagnose(self._items(10, 0, 5, 5))
        transfer = result["transfer"]
        assert transfer["measurable"]
        assert transfer["drop_points"]["executor_agreement"] == pytest.approx(50.0)
        assert transfer["n_synthetic"] == 10 and transfer["n_real"] == 10

    def test_every_real_source_is_merged_into_one_real_group(self) -> None:
        items = [*self._items(4, 0, 2, 0),
                 {"raw": _raw({"op": "sum", "args": []}), "source": "refchartqa"}]
        assert diagnose(items)["transfer"]["n_real"] == 3

    def test_transfer_is_not_reported_when_one_side_is_missing(self) -> None:
        result = diagnose([{"raw": "x", "source": "synthetic"}])
        assert result["transfer"]["measurable"] is False

    def test_per_source_rates_are_kept_alongside_the_merged_view(self) -> None:
        result = diagnose(self._items(4, 0, 1, 3))
        assert set(result["by_source"]) == {"synthetic", "chartqa"}
        assert result["by_source"]["synthetic"]["executor_agreement"] == pytest.approx(1.0)
        assert result["by_source"]["chartqa"]["executor_agreement"] == pytest.approx(0.25)

    def test_the_operation_histogram_is_recorded(self) -> None:
        result = diagnose(self._items(3, 0, 0, 0))
        assert result["by_source"]["synthetic"]["ops"] == {"difference": 3}
