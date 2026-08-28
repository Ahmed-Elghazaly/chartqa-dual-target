"""Generated LaTeX must be safe to drop into the document, and honest when data is missing."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from chartqa_dt.reporting.build import build_tables, load_results, summarise
from chartqa_dt.reporting.latex import ci, escape, has_todo, num, row, tabular
from chartqa_dt.reporting.tables import BUILDERS

ROOT = Path(__file__).resolve().parents[1]


class TestEscaping:
    @pytest.mark.parametrize(("raw", "expected"), [
        ("percent_change", r"percent\_change"),
        ("14.07%", r"14.07\%"),
        ("a & b", r"a \& b"),
        ("#1", r"\#1"),
        ("$5", r"\$5"),
        ("x^2", r"x\textasciicircum{}2"),
    ])
    def test_special_characters_are_escaped(self, raw: str, expected: str) -> None:
        assert escape(raw) == expected

    def test_a_backslash_does_not_escape_the_escapes(self) -> None:
        """Rewriting `&` before `\\` would turn `\\&` into `\\\\&` and break the row."""
        assert escape(r"a\&b") == r"a\textbackslash{}\&b"

    def test_a_percent_would_otherwise_swallow_the_rest_of_the_line(self) -> None:
        """The dangerous one: an unescaped % comments out the \\label after it."""
        assert "%" not in escape("yield 14%").replace(r"\%", "")


class TestNumbers:
    def test_a_missing_number_is_a_visible_todo_not_a_zero(self) -> None:
        assert num(None) == r"\TODO{}"
        assert ci(None) == r"\TODO{}"

    def test_zero_is_still_rendered_as_zero(self) -> None:
        """`None` means unmeasured; 0.0 is a measurement and must survive."""
        assert num(0.0) == "0.00"

    def test_an_interval_is_rendered_beside_its_estimate(self) -> None:
        assert ci(32.83, 28.1, 37.2) == r"32.8\,\tiny{[28.1, 37.2]}"


class TestTabular:
    def test_every_row_has_the_same_column_count_as_the_header(self) -> None:
        body = tabular("lcc", ["a", "b", "c"], [row([1, 2, 3]), row([4, 5, 6])])
        counts = {line.count("&") for line in body.splitlines() if line.endswith(r"\\")}
        assert counts == {2}


class TestBuilders:
    @pytest.fixture
    def results(self) -> dict:
        return load_results(ROOT)

    @pytest.mark.parametrize("name", sorted(BUILDERS))
    def test_every_builder_produces_balanced_latex(self, name: str, results: dict) -> None:
        text = BUILDERS[name](results)
        assert text.count("{") == text.count("}"), f"{name}: unbalanced braces"
        opens = re.findall(r"\\begin\{(\w+\*?)\}", text)
        closes = re.findall(r"\\end\{(\w+\*?)\}", text)
        assert opens == closes[::-1] or sorted(opens) == sorted(closes), \
            f"{name}: {opens} vs {closes}"

    @pytest.mark.parametrize("name", sorted(BUILDERS))
    def test_every_tabular_row_matches_its_column_count(self, name: str,
                                                        results: dict) -> None:
        """A row with one `&` too many is a compile error, and it is easy to write."""
        text = BUILDERS[name](results)
        spec = re.search(r"\\begin\{tabular\}\{([lcr|@{}\\.\s]+)\}", text)
        if spec is None:
            return                                   # tab_headline builds its own header
        columns = sum(spec.group(1).count(c) for c in "lcr")
        for line in text.splitlines():
            if line.endswith(r"\\") and not line.startswith("%") and "multicolumn" not in line:
                assert line.count("&") == columns - 1, f"{name}: {line!r}"

    @pytest.mark.parametrize("name", sorted(BUILDERS))
    def test_a_builder_with_no_data_still_produces_a_table(self, name: str) -> None:
        """The skeleton must compile from Phase 1, with the gaps red rather than absent."""
        text = BUILDERS[name]({})
        assert r"\begin{table" in text and r"\end{table" in text
        assert has_todo(text), f"{name} claims completeness with no data at all"

    def test_the_tables_that_have_data_carry_it(self, results: dict) -> None:
        text = BUILDERS["plan_yield"](results)
        assert "3,983" in text and r"14.07\%" in text
        assert not has_todo(text)


class TestBuild:
    def test_an_unknown_table_is_refused_by_name(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown table"):
            build_tables({}, tmp_path, "not_a_table")

    def test_all_writes_every_registered_table(self, tmp_path: Path) -> None:
        written = build_tables(load_results(ROOT), tmp_path, "all")
        assert set(written) == set(BUILDERS)
        assert {p.name for p in tmp_path.glob("*.tex")} == {f"tab_{n}.tex" for n in BUILDERS}

    def test_the_summary_reports_incompleteness_rather_than_hiding_it(self,
                                                                     tmp_path: Path) -> None:
        text = summarise(build_tables({}, tmp_path, "all"))
        # Every registered table, none of them complete. Asserted against the registry
        # rather than a literal, so adding a table does not silently make this pass.
        assert f"0/{len(BUILDERS)} tables complete" in text
        assert "not final" in text

    def test_extra_results_are_keyed_by_filename_stem(self, tmp_path: Path) -> None:
        extra = tmp_path / "results"
        extra.mkdir()
        (extra / "oracle.json").write_text(json.dumps(
            {"n_eligible": 120, "n_excluded_no_gold_plan": 8,
             "cells": {"gold_gold": {"accuracy": 0.991, "executor_refused": 0}}}),
            encoding="utf-8")
        results = load_results(ROOT, extra)
        rendered = BUILDERS["oracle"](results)
        assert "99.10" in rendered and "120" in rendered


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX toolchain")
def test_the_report_skeleton_compiles(tmp_path: Path) -> None:
    """The direct measurement. The skeleton once contained `\\TODO{caption for ap_by_area}`,
    whose raw underscore is a LaTeX error — it had never been compiled."""
    build_tables(load_results(ROOT), ROOT / "report" / "tables", "all")
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-output-directory={tmp_path}", "main.tex"],
        cwd=ROOT / "report", capture_output=True, text=True, timeout=300, check=False)
    assert proc.returncode == 0, proc.stdout[-2500:]
    assert (tmp_path / "main.pdf").is_file()


ZEROSHOT_ARMS = {
    "arms": {
        "structured": {"n": 1920, "relaxed_accuracy": 0.412, "ci": [0.390, 0.434],
                       "median_new_tokens": 198.0, "median_latency_s": 11.44,
                       "valid_json_fraction": 0.665},
        "plain": {"n": 1920, "relaxed_accuracy": 0.762, "ci": [0.743, 0.781],
                  "median_new_tokens": 4.0, "median_latency_s": 1.20,
                  "valid_json_fraction": 1.0},
    }
}


class TestStructuredCostFillsFromTheRun:
    """`PLAN.md` 5.3 and 8.1. The table must read what run_zeroshot writes."""

    def test_both_arms_render_with_their_intervals(self) -> None:
        text = BUILDERS["structured_cost"]({"chartqa_zeroshot": ZEROSHOT_ARMS})
        assert "41.20" in text and "76.20" in text
        assert not has_todo(text)

    def test_the_gap_is_stated_as_a_cost_in_points(self) -> None:
        text = BUILDERS["structured_cost"]({"chartqa_zeroshot": ZEROSHOT_ARMS})
        assert "-35.00 points" in text

    def test_the_caption_carries_the_sample_size(self) -> None:
        text = BUILDERS["structured_cost"]({"chartqa_zeroshot": ZEROSHOT_ARMS})
        assert "1,920 ChartQA validation questions" in text

    def test_a_missing_run_leaves_the_table_visibly_unfilled(self) -> None:
        text = BUILDERS["structured_cost"]({})
        assert has_todo(text) and "points" not in text.split("Published")[0].split("\\\\")[-1]
