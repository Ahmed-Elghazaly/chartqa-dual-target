"""Every source reader must produce records `build_target` can actually use.

`DECISIONS.md` 0071. The synthetic reader wrote its per-element metadata under
`meta["evidence"]` while `build_target` reads `meta["elements"]`. Every field was present
and correctly shaped; only the key differed. The records looked complete, `build_target`
fell through to its placeholder branch, labelled the evidence `item1, item2, ...`, and then
refused each record because its plan referenced the *real* labels — so **all 12,000 stage-1
targets were lost**, silently, since the feed skips a refusal and moves on.

Nothing failed. The mixture files were valid, the tests were green, and the only visible
symptom would have been a training run that took ten GPU hours to learn nothing.

These tests run the real readers over tiny fixtures, with no data cache, and assert that a
target comes out the other end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chartqa_dt.data.records import ELEMENTS_KEY
from chartqa_dt.train.targets import TargetError, build_target

EVIDENCE = [
    {"label": "2019", "value": 245.0, "unit": "units", "bbox": [412, 180, 468, 640],
     "bbox_px": [200, 90, 230, 320]},
    {"label": "2018", "value": 210.0, "unit": "units", "bbox": [330, 240, 386, 640],
     "bbox_px": [160, 120, 190, 320]},
]


def _manifest(tmp_path: Path, **overrides) -> Path:
    example = {
        "example_id": "synth_v_bar_L2_1_0", "holdout": False,
        "image_path": str(tmp_path / "chart.png"),
        "image_sha256": "0" * 64,
        "question": "What is the difference between 2019 and 2018?",
        "answer": "35", "table": {"2019": 245.0, "2018": 210.0},
        "evidence": EVIDENCE,
        "plan": {"op": "difference", "args": ["2019", "2018"]},
        "level": "L2", "chart_type": "v_bar", "style_seed": 1, "data_seed": 2,
    }
    example.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"examples": [example]}), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_records():
    from scripts.build_mixtures import synthetic_records as fn

    return fn


class TestSyntheticReader:
    def test_a_synthetic_record_becomes_a_training_target(self, tmp_path, synthetic_records):
        """The whole point. This assertion was false for every record in the project."""
        [record] = synthetic_records(_manifest(tmp_path))
        target = json.loads(build_target(record))
        assert target["model_answer"] == "35"
        assert [e["label"] for e in target["evidence"]] == ["2019", "2018"]
        assert target["plan"] == {"op": "difference", "args": ["2019", "2018"]}

    def test_element_metadata_lands_under_the_key_the_target_builder_reads(
            self, tmp_path, synthetic_records):
        [record] = synthetic_records(_manifest(tmp_path))
        assert record.meta[ELEMENTS_KEY] == EVIDENCE
        assert "evidence" not in record.meta, "one canonical key, not two"

    def test_the_evidence_carries_the_real_labels_not_placeholders(
            self, tmp_path, synthetic_records):
        """The symptom that made every plan unexecutable: `item1`, `item2`, ..."""
        [record] = synthetic_records(_manifest(tmp_path))
        labels = [e["label"] for e in json.loads(build_target(record))["evidence"]]
        assert not any(label.startswith("item") for label in labels), labels

    def test_holdout_examples_are_not_returned(self, tmp_path, synthetic_records):
        """Sealed for Phase 9.5 robustness; leaking one into training invalidates it."""
        assert synthetic_records(_manifest(tmp_path, holdout=True)) == []

    def test_a_missing_manifest_is_empty_rather_than_an_error(self, tmp_path,
                                                              synthetic_records):
        assert synthetic_records(tmp_path / "absent.json") == []


class TestKeyIsShared:
    def test_no_source_writes_a_hand_spelled_elements_key(self) -> None:
        """Both readers and the target builder must go through the same constant."""
        root = Path(__file__).resolve().parents[1]
        offenders = []
        for path in [root / "src/chartqa_dt/data/chartqa.py",
                     root / "src/chartqa_dt/train/targets.py",
                     root / "scripts/build_mixtures.py"]:
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if '"elements"' in line and not line.lstrip().startswith("#"):
                    offenders.append(f"{path.name}:{i}: {line.strip()}")
        assert not offenders, "use ELEMENTS_KEY:\n" + "\n".join(offenders)


FIVE = [{"label": n, "value": v, "unit": None,
         "bbox": [10 * i, 10, 10 * i + 8, 90]}
        for i, (n, v) in enumerate(
            [("Alpha", 286.0), ("Beta", 242.0), ("Gamma", 198.0),
             ("Delta", 235.0), ("Epsilon", 146.0)], start=1)]


class TestFoldOverEvidence:
    """`DECISIONS.md` 0071 defect 2: a plan that folds over the chart needs the chart."""

    def _record(self, tmp_path, plan, answer, evidence):
        from scripts.build_mixtures import synthetic_records

        [record] = synthetic_records(_manifest(
            tmp_path, plan=plan, answer=answer, evidence=evidence,
            table={e["label"]: e["value"] for e in evidence}))
        return record

    def test_a_nested_fold_gets_every_element_not_just_the_named_label(self, tmp_path):
        """difference("Alpha", mean-of-all) = 286 - 221.4 = 64.6, never 0."""
        record = self._record(
            tmp_path, {"op": "difference", "args": ["Alpha", {"op": "mean", "args": []}]},
            "64.6", FIVE)
        target = json.loads(build_target(record))
        assert [e["label"] for e in target["evidence"]] == \
            ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]

    def test_a_plan_with_no_fold_still_gets_only_what_it_names(self, tmp_path):
        """0067's selection is right for every other shape and must not be widened."""
        record = self._record(
            tmp_path, {"op": "difference", "args": ["Alpha", "Gamma"]}, "88", FIVE)
        target = json.loads(build_target(record))
        assert [e["label"] for e in target["evidence"]] == ["Alpha", "Gamma"]

    def test_a_fold_over_more_elements_than_the_schema_holds_is_refused(self, tmp_path):
        """Truncating would change the aggregate, so the target would not round-trip."""
        many = [{"label": f"E{i}", "value": float(i), "unit": None,
                 "bbox": [i, 10, i + 5, 90]} for i in range(12)]
        record = self._record(
            tmp_path, {"op": "difference", "args": ["E0", {"op": "mean", "args": []}]},
            "-5.5", many)
        with pytest.raises(TargetError, match="folds over all 12 elements"):
            build_target(record)


class TestZeroIsAnAnswer:
    """`DECISIONS.md` 0071 defect 3: scoring keeps the official quirk, round-trip does not."""

    def test_the_official_metric_still_reproduces_the_published_behaviour(self) -> None:
        from chartqa_dt.eval.metrics import relaxed_correctness

        assert relaxed_correctness("0", "0.0") is False

    def test_but_a_plan_that_correctly_computes_zero_agrees_with_itself(self) -> None:
        from chartqa_dt.plans.roundtrip import answers_agree

        assert answers_agree("0", 0.0)
        assert answers_agree("0", 0)

    def test_a_wrong_answer_of_zero_still_disagrees(self) -> None:
        from chartqa_dt.plans.roundtrip import answers_agree

        assert not answers_agree("0", 5.0)
        assert not answers_agree("5", 0.0)

    def test_the_tolerance_is_still_relative_and_still_five_percent(self) -> None:
        from chartqa_dt.plans.roundtrip import answers_agree

        assert answers_agree("100", 104.0)
        assert not answers_agree("100", 106.0)

    def test_a_difference_of_two_equal_operands_round_trips(self, tmp_path) -> None:
        """The 512 records this discarded: `difference(max, X)` where X *is* the max."""
        from scripts.build_mixtures import synthetic_records

        evidence = [{"label": "North", "value": 54.98, "unit": None,
                     "bbox": [10, 10, 20, 90]},
                    {"label": "West", "value": 73.96, "unit": None,
                     "bbox": [30, 10, 40, 90]}]
        [record] = synthetic_records(_manifest(
            tmp_path, plan={"op": "difference", "args": [{"op": "max", "args": []}, "West"]},
            answer="0", evidence=evidence,
            table={e["label"]: e["value"] for e in evidence}))
        assert json.loads(build_target(record))["model_answer"] == "0"
