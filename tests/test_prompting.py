"""Prompts and output parsing — `PLAN.md` 5.1, and non-negotiable rule 3.

The rule under test is that **an invalid output is a failure**. A parser that always
returns something would turn every model failure into a plausible number and the score
would measure the parser. So the tests below draw one line repeatedly: recovering from
*transport* noise is allowed and counted; supplying *content* the model did not produce
never is.
"""

from __future__ import annotations

import json

import pytest

from chartqa_dt.prompting.parsing import (
    REQUIRED_FIELDS,
    ParseStats,
    answer_of,
    coerce_boxes,
    parse_record,
)
from chartqa_dt.prompting.prompts import (
    ALLOWED_OPS,
    PLAIN_PROMPT,
    build_plain_prompt,
    build_structured_prompt,
    example_record,
    example_record_json,
    prompt_fingerprint,
)

# ------------------------------------------------------------------------- prompts


def test_the_plain_prompt_is_the_published_one_verbatim():
    """It anchors the 79.1 comparison; paraphrasing it would break the comparison.

    From the Qwen3-VL report's evaluation-prompt appendix (`verification/phase0.md` F9).
    """
    assert PLAIN_PROMPT == "{question}\nAnswer the question using a single word or phrase."
    assert build_plain_prompt("How many?") == \
        "How many?\nAnswer the question using a single word or phrase."


def test_the_structured_prompt_names_every_allowed_operation():
    """A model cannot be blamed for inventing an op it was never shown."""
    prompt = build_structured_prompt("q")
    for op in ALLOWED_OPS:
        assert op in prompt, f"{op} is accepted by the executor but absent from the prompt"


def test_the_structured_prompt_states_the_coordinate_convention():
    prompt = build_structured_prompt("q")
    assert "0 to 999" in prompt
    assert "top-left" in prompt and "bottom-right" in prompt


def test_the_prompt_carries_the_question_exactly_once():
    prompt = build_structured_prompt("What is the value for 2019?")
    assert prompt.count("What is the value for 2019?") == 1
    assert "{question}" not in prompt


def test_prompts_are_fingerprinted_so_a_silent_edit_is_detectable():
    """`PLAN.md` 5.5 seals the prompt; a hash makes "unchanged" checkable."""
    a = prompt_fingerprint()
    assert set(a) == {"structured", "plain"}
    assert all(len(v) == 64 for v in a.values())
    assert a == prompt_fingerprint()


def test_the_example_record_satisfies_the_output_schema():
    from chartqa_dt.plans.schema import validate_record

    result = validate_record(example_record())
    assert result.ok, result.errors


# ------------------------------------------------------------------------- parsing


def test_a_clean_record_parses_without_repair():
    result = parse_record(example_record_json())
    assert result.ok and result.repairs == []
    assert answer_of(result.record) == "35"


@pytest.mark.parametrize(("name", "wrapper"), [
    ("code fence", "```json\n{body}\n```"),
    ("bare fence", "```\n{body}\n```"),
    ("chatty prefix", "Sure! Here is the record:\n{body}"),
    ("chatty suffix", "{body}\nHope that helps."),
    ("both", "Here you go:\n```json\n{body}\n```\nLet me know."),
])
def test_transport_noise_is_recovered_and_counted(name, wrapper):
    """Recovery is legitimate — the model produced the record, the wrapper is noise."""
    result = parse_record(wrapper.format(body=example_record_json()))
    assert result.ok, f"{name}: {result.reason}"
    assert result.repairs, f"{name}: a recovery must be recorded, not silent"
    assert answer_of(result.record) == "35"


def test_a_trailing_comma_is_repaired():
    body = '{"answerable": true, "evidence": [], "plan": {"op": "lookup", "args": ["a"]},' \
           ' "model_answer": "5",}'
    result = parse_record(body)
    assert result.ok and "removed trailing comma" in result.repairs


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_a_missing_field_is_a_failure_not_a_default(field):
    """The line this module exists to hold: never supply what the model did not produce."""
    record = example_record()
    del record[field]
    result = parse_record(json.dumps(record))
    assert not result.ok
    assert field in result.reason
    assert result.record is None


@pytest.mark.parametrize(("name", "text"), [
    ("empty", ""),
    ("whitespace", "   \n  "),
    ("prose", "The answer is 35."),
    ("truncated", '{"answerable": true, "evidence": [{"label"'),
    ("a list", "[1, 2, 3]"),
    ("a bare number", "35"),
])
def test_unparseable_output_is_a_failure_with_a_reason(name, text):
    result = parse_record(text)
    assert not result.ok, name
    assert result.reason, f"{name}: a failure must say why"
    assert result.raw == text, "the raw output is kept for inspection"


def test_boxes_that_are_not_boxes_are_dropped_not_guessed():
    record = {"answerable": True, "model_answer": "1",
              "plan": {"op": "lookup", "args": ["a"]},
              "evidence": [
                  {"label": "a", "bbox": [1, 2, 3, 4]},
                  {"label": "b", "bbox": [1, 2, 3]},          # too short
                  {"label": "c", "bbox": "1,2,3,4"},          # a string
                  {"label": "d"},                             # no box at all
                  {"label": "e", "bbox": [1, 2, "x", 4]},     # unparseable
                  "not an object",
              ]}
    assert coerce_boxes(record) == [[1.0, 2.0, 3.0, 4.0]]


def test_coerce_boxes_does_not_clamp():
    """Clamping is the emitter's job (`DECISIONS.md` 0004); this reports what was said."""
    record = {"evidence": [{"label": "a", "bbox": [-5, 0, 2000, 999]}]}
    assert coerce_boxes(record) == [[-5.0, 0.0, 2000.0, 999.0]]


def test_a_missing_answer_becomes_an_empty_string_not_a_guess():
    assert answer_of({"model_answer": None}) == ""
    assert answer_of({}) == ""
    assert answer_of({"model_answer": 35}) == "35"


# --------------------------------------------------------------------------- stats


def test_parse_stats_track_validity_and_repair_rate():
    """`PLAN.md` 5.2 gates the variant choice on >= 90% valid JSON."""
    stats = ParseStats()
    for _ in range(9):
        stats.add(parse_record(example_record_json()))
    stats.add(parse_record("not json at all"))
    assert stats.total == 10 and stats.valid == 9
    assert stats.valid_fraction == pytest.approx(0.9)
    assert stats.repaired == 0
    assert sum(stats.reasons.values()) == 1


def test_the_repair_rate_is_reported_separately_from_validity():
    stats = ParseStats()
    stats.add(parse_record(example_record_json()))
    stats.add(parse_record("```json\n" + example_record_json() + "\n```"))
    assert stats.valid == 2
    assert stats.repaired == 1
    assert stats.repaired_fraction == pytest.approx(0.5)
    assert "needed a repair" in stats.describe()


def test_empty_stats_do_not_divide_by_zero():
    stats = ParseStats()
    assert stats.valid_fraction == 0.0 and stats.repaired_fraction == 0.0


# ------------------------------------------------------------- generation plumbing


def test_decoding_is_greedy_and_sealed():
    """`PLAN.md` 5.5 seals decoding. Sampling would make the baseline a distribution.

    The "before" number has to be exactly reproducible from the committed config, or the
    before/after comparison carries sampling noise it cannot separate from a real effect.
    """
    from chartqa_dt.eval.generate import DECODING

    assert DECODING["do_sample"] is False
    assert DECODING["num_beams"] == 1
    assert DECODING["temperature"] is None and DECODING["top_p"] is None


def test_the_message_structure_carries_image_and_the_right_prompt():
    from chartqa_dt.eval.generate import build_messages

    sentinel = object()
    for mode, marker in (("structured", "JSON object"),
                         ("plain", "single word or phrase")):
        messages = build_messages("How many?", sentinel, mode)
        assert len(messages) == 1 and messages[0]["role"] == "user"
        kinds = [c["type"] for c in messages[0]["content"]]
        assert kinds == ["image", "text"]
        assert messages[0]["content"][0]["image"] is sentinel
        assert marker in messages[0]["content"][1]["text"]
        assert "How many?" in messages[0]["content"][1]["text"]


def test_generations_round_trip_through_disk(tmp_path):
    """A prediction set must be re-scorable without a GPU — three metrics needed
    correcting after they were first produced (`DECISIONS.md` 0053)."""
    from chartqa_dt.eval.generate import Generation, read_generations, write_generations

    gens = [
        Generation(record_id="a", raw=example_record_json(), seconds=1.5,
                   prompt_mode="structured", parsed_ok=True, answer="35",
                   boxes=[[1.0, 2.0, 3.0, 4.0]], plan={"op": "lookup", "args": ["x"]},
                   image_size=(800, 557)),
        Generation(record_id="b", raw="nope", seconds=0.5, prompt_mode="structured",
                   parsed_ok=False, reason="invalid JSON"),
    ]
    path = write_generations(gens, tmp_path / "g.jsonl")
    back = read_generations(path)
    assert [g.record_id for g in back] == ["a", "b"]
    assert back[0].image_size == (800, 557)
    assert back[0].boxes == [[1.0, 2.0, 3.0, 4.0]]
    assert back[1].parsed_ok is False and back[1].reason == "invalid JSON"


def test_the_frozen_slices_are_recorded_by_id_and_hash():
    """`PLAN.md` 5.2 needs a *frozen* slice; a re-sampled one defeats the point."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = root / "data/slices/chartqa_variant_200.json"
    if not path.exists():
        pytest.skip("slices are built from the archive")
    data = json.loads(path.read_text())
    assert data["n"] == 200
    assert len(data["record_ids"]) == 200 == len(set(data["record_ids"]))
    assert len(data["slice_sha256"]) == 64
    assert data["split"] == "val", "rule 1: never the test split"
    for forbidden in ("question", "answer", "questions", "answers"):
        assert forbidden not in data, "rule 7: no dataset content in the repository"
