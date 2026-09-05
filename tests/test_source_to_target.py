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

from chartqa_dt.plans.schema import MAX_EVIDENCE
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

    def test_elements_land_on_the_field_the_target_builder_reads(
            self, tmp_path, synthetic_records):
        """`DECISIONS.md` 0071 was a *spelling* defect: the synthetic reader wrote
        `evidence` where the builder read `elements`, and all 12,000 stage-1 targets went
        silently. A shared string constant was the first fix; a dataclass **field** is the
        real one, because a misspelling is now an error rather than an empty dict (0124).
        """
        [record] = synthetic_records(_manifest(tmp_path))
        assert record.elements, "no elements on the record"
        assert record.evidence is not None, "synthetic knows which marks its plan needs"
        assert all(record.elements[i] in record.elements for i in record.evidence)

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
    def test_every_source_populates_the_elements_field(self) -> None:
        """The contract that replaced the shared-constant one (0124).

        `ELEMENTS_KEY` is still written for readers of already-cached records, but the
        field is what `_evidence_from` reads, so a source that sets only the meta key
        would produce records that look complete and ground nothing.
        """
        root = Path(__file__).resolve().parents[1]
        # ChartQA's record construction lives in `build_mixtures.py` alone since 0132
        # removed the dead duplicate in `data/chartqa.py`.
        sources = {
            "src/chartqa_dt/data/refchartqa.py": "elements=",
            "scripts/build_mixtures.py": "elements=",
        }
        missing = [name for name, needle in sources.items()
                   if needle not in (root / name).read_text(encoding="utf-8")]
        assert not missing, f"these sources never set the elements field: {missing}"

    def test_every_source_says_whether_it_knows_its_evidence(self) -> None:
        """`evidence=None` is a claim ("unknown"), not an omission, and each source has to
        make it deliberately — that is what 0116 got wrong when it was inferred."""
        root = Path(__file__).resolve().parents[1]
        for name in ("src/chartqa_dt/data/refchartqa.py",
                     "scripts/build_mixtures.py"):
            text = (root / name).read_text(encoding="utf-8")
            assert "evidence=" in text, f"{name} never sets evidence"


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
        n = MAX_EVIDENCE + 1                 # one past the cap, whatever the cap is
        many = [{"label": f"E{i}", "value": float(i), "unit": None,
                 "bbox": [i, 10, i + 5, 90]} for i in range(n)]
        answer = str(0.0 - (n - 1) / 2)      # E0 minus the mean of 0..n-1
        record = self._record(
            tmp_path, {"op": "difference", "args": ["E0", {"op": "mean", "args": []}]},
            answer, many)
        with pytest.raises(TargetError, match=f"folds over all {n} elements"):
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


class TestSourceDrawsCannotDrift:
    """`DECISIONS.md` 0072. A mixture holds ids; training rebuilds the records.

    If training rebuilds a smaller pool than the mixture was built from, the ids at the
    tail resolve to nothing. `load_mixture_records` refuses on that — loudly, but on the
    GPU, an hour into a run.
    """

    def test_neither_side_hand_writes_a_source_draw(self) -> None:
        root = Path(__file__).resolve().parents[1]
        offenders = []
        for name in ["src/chartqa_dt/cli/train.py", "scripts/build_mixtures.py"]:
            path = root / name
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                for call, arg in (("chartqa_records", "limit="), ("refchartqa_records", "cap=")):
                    if call in line and arg in line and not any(
                            c.isalpha() for c in line.split(arg)[1][:1]):
                        offenders.append(f"{name}:{i}: {stripped}")
        assert not offenders, ("use CHARTQA_DRAW / REFCHARTQA_CAP from data.mixture:\n"
                               + "\n".join(offenders))

    def test_the_constants_are_the_ones_both_sides_import(self) -> None:
        from chartqa_dt.data.mixture import CHARTQA_DRAW, REFCHARTQA_CAP

        assert CHARTQA_DRAW >= 20_901, "must cover the whole ChartQA machine split"
        assert REFCHARTQA_CAP > 0


class TestOneConstructorPerSource:
    """Two constructors for one source is how the same defect happened twice.

    `DECISIONS.md` 0067 and 0071 are the `elements`/`evidence` spelling defect, once per
    reader. 0119 is an edit that landed in the *dead* ChartQA constructor while the live
    one kept its old behaviour. 0132 removed the dead path; this keeps it removed.
    """

    def test_only_one_place_builds_a_chartqa_chart_record(self) -> None:
        root = Path(__file__).resolve().parents[1]
        builders = []
        for path in list((root / "src").rglob("*.py")) + list((root / "scripts").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "ChartRecord(" not in text:
                continue
            if 'source="chartqa"' in text or '"chartqa", split' in text:
                builders.append(str(path.relative_to(root)))
        assert len(builders) <= 1, (
            f"{len(builders)} places build a ChartQA ChartRecord: {builders}. An edit to "
            f"one of them will silently miss the other, which is DECISIONS.md 0119.")

    def test_the_dead_record_path_stays_removed(self) -> None:
        from chartqa_dt.data import chartqa

        for gone in ("row_to_record", "iter_records", "iter_records_from_archive"):
            assert not hasattr(chartqa, gone), (
                f"chartqa.{gone} is back. Nothing outside that module ever called it, and "
                f"its presence is what let an edit land in the wrong constructor (0132).")

    def test_refchartqa_still_has_exactly_one(self) -> None:
        """RefChartQA's `row_to_record` is live — `scripts/cache_refchartqa.py` uses it."""
        from chartqa_dt.data import refchartqa

        assert hasattr(refchartqa, "row_to_record")


class TestDroppedBoxesAreCounted:
    """`feed.py` records four defects of one shape — something the pipeline cannot use,
    caught by an `except`, and skipped — and notes that from outside, such a handler is
    indistinguishable from there being no failures. `_norm_or_none` was the fifth
    (`DECISIONS.md` 0135)."""

    def test_the_counter_exists_and_is_exported(self) -> None:
        from chartqa_dt.data.chartqa import DROPPED_BOXES, __all__

        assert "DROPPED_BOXES" in __all__
        assert hasattr(DROPPED_BOXES, "most_common")

    def test_a_degenerate_box_is_counted_not_merely_dropped(self) -> None:
        from chartqa_dt.data import chartqa

        before = sum(chartqa.DROPPED_BOXES.values())
        assert chartqa._norm_or_none({"x": 0, "y": 0, "w": 0, "h": 0}, 400, 400) is None
        assert sum(chartqa.DROPPED_BOXES.values()) == before + 1

    def test_a_usable_box_is_not_counted(self) -> None:
        from chartqa_dt.data import chartqa

        before = sum(chartqa.DROPPED_BOXES.values())
        assert chartqa._norm_or_none({"x": 10, "y": 10, "w": 50, "h": 50}, 400, 400)
        assert sum(chartqa.DROPPED_BOXES.values()) == before

    def test_the_two_reasons_are_distinguished(self) -> None:
        """"not normalisable" and "degenerate after normalising" have different causes;
        measured on real data, all 722 were the second — ChartQA ships literal
        `{x:0, y:0, w:0, h:0}` placeholders."""
        from chartqa_dt.data import chartqa

        chartqa._norm_or_none({"x": 0, "y": 0, "w": 0, "h": 0}, 400, 400)
        chartqa._norm_or_none("not a box", 400, 400)
        assert set(chartqa.DROPPED_BOXES) == {"degenerate after normalising",
                                              "not normalisable"}
