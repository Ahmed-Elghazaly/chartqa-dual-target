"""Two parsers, two kinds of text, and the rule for telling them apart.

The project has one parser for **gold answers** and one for **chart values**, and confusing
them has now caused four separate defects (`DECISIONS.md` 0082 and this file's own history):

* `eval.metrics.to_float` reads a gold ANSWER. It is byte-faithful to the official ChartQA
  evaluator, which divides a trailing `%` by 100 and cannot read a spaced thousand. It must
  not be "improved" — a more generous parser makes our numbers incomparable with the
  literature while looking better (`DECISIONS.md` 0045).
* `plans.executor.parse_numeric` reads a chart VALUE — a table cell or an annotated bar. It
  keeps the scale the chart is drawn in and understands the separators Statista uses.

The rule: **answers use `to_float`, everything drawn on a chart uses `parse_numeric`.**
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from chartqa_dt.eval.metrics import to_float
from chartqa_dt.plans.executor import parse_numeric

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "chartqa_dt"


def test_the_two_parsers_deliberately_disagree():
    """If these ever agree, one of them has been quietly changed."""
    assert to_float("43.6%") == pytest.approx(0.436), "the official parser divides percents"
    assert parse_numeric("43.6%") == pytest.approx(43.6), "a bar keeps the chart's scale"
    assert to_float("1 234") is None, "the official parser cannot read a spaced thousand"
    assert parse_numeric("1 234") == pytest.approx(1234.0)
    assert to_float("1,234") is None, "nor a comma — it is `float()` and nothing more"
    assert parse_numeric("1,234") == pytest.approx(1234.0)


def test_they_agree_where_agreement_is_required():
    """A plain number must parse identically, or a target could never round-trip."""
    for text in ("82.1", "-4", "0", "3.5", "0.436"):
        assert to_float(text) == pytest.approx(parse_numeric(text)), text


#: Call sites outside `eval/` that pass a chart value to the answer parser. Each entry is a
#: real defect that was found by running the pipeline, not by reading it.
_ALLOWED_TO_FLOAT = {
    # `resolve.candidates` compares a gold ANSWER against chart values; the answer side is
    # correct here and the value side uses `parse_numeric`.
    ("plans/resolve.py", "answer"),
    # `roundtrip.values_match` compares a stated ANSWER with an executed number.
    ("plans/roundtrip.py", "stated"),
    ("plans/roundtrip.py", "str(got)"),
    # `build_record` reads the record's own ANSWER.
    ("train/targets.py", "record.answer"),
}


def _to_float_arguments() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in _SRC.rglob("*.py"):
        if path.parts[-2] == "eval":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "to_float" and node.args):
                rel = "/".join(path.parts[path.parts.index("chartqa_dt") + 1:])
                found.add((rel, ast.unparse(node.args[0])))
    return found


def test_the_answer_parser_is_only_ever_given_an_answer():
    """The guard against a fifth occurrence.

    Four defects came from calling `to_float` on something drawn on a chart: a table cell in
    `_table_values`, both sides of `values_agree`, and an element value in
    `resolve.candidates`. Each looked right in isolation and each produced supervision that
    was silently 100x off. A new call site has to be justified here, deliberately.
    """
    unexpected = _to_float_arguments() - _ALLOWED_TO_FLOAT
    assert not unexpected, (
        "`to_float` is the GOLD ANSWER parser. These call sites pass something else:\n  "
        + "\n  ".join(f"{where}: to_float({what})" for where, what in sorted(unexpected))
        + "\nUse `plans.executor.parse_numeric` for anything drawn on a chart, or add the "
          "call to _ALLOWED_TO_FLOAT with a reason if it really is an answer.")
