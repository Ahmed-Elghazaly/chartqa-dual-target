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
    assert "0-999" in prompt or "0 to 999" in prompt
    assert "top-left" in prompt and "bottom-right" in prompt


def test_the_structured_prompt_asks_for_compact_output():
    """Iterated on measurement: the model imitates the example's formatting.

    A pretty-printed example produced 308-token records, a third of which hit the
    512-token cap mid-JSON. Four of five parse failures were pure truncation.
    """
    prompt = build_structured_prompt("q")
    assert "compact" in prompt and "single line" in prompt
    assert "no indentation" in prompt or "no newlines" in prompt
    # Every worked example must itself be compact, or the instruction is contradicted
    # by the demonstration and the demonstration wins.
    for line in prompt.splitlines():
        if line.startswith('{"answerable"'):
            assert ", " not in line and line.strip() == line
            assert len(line) > 60, "an example that short is not a real record"


def test_the_prompt_shows_plan_args_as_a_list():
    """The probe produced `"args": {"label": "Zara", "value": 99}` — an object.

    That parses as JSON and the executor rejects it, so the prompt states the rule and
    every example demonstrates it.
    """
    prompt = build_structured_prompt("q")
    assert '"args":["Zara"]' in prompt
    assert '"args":["2019","2018"]' in prompt
    assert '"args" is always a LIST' in prompt
    assert '"args":[]' in prompt, "the unanswerable and aggregate forms need an example"


def test_the_prompt_gives_the_unanswerable_case_a_complete_example():
    """One failure emitted only `answerable` and `evidence`, dropping two required keys."""
    prompt = build_structured_prompt("q")
    assert '{"answerable":false,"evidence":[],"plan":{"op":"unanswerable","args":[]},' \
        '"model_answer":""}' in prompt
    assert "All four keys are required" in prompt


def test_the_prompt_carries_the_question_exactly_once():
    prompt = build_structured_prompt("What is the value for 2019?")
    assert prompt.count("What is the value for 2019?") == 1
    assert "{question}" not in prompt


def test_prompts_are_fingerprinted_so_a_silent_edit_is_detectable():
    """`PLAN.md` 5.5 seals the prompt; a hash makes "unchanged" checkable."""
    a = prompt_fingerprint()
    assert set(a) == {"structured", "plain", "training"}
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


def test_schema_validity_and_json_validity_are_measured_separately():
    """They come apart, and the gap is what `PLAN.md` 5.2 should gate on.

    The first probe's "successful" record used `"args": {"label": ..., "value": ...}`.
    It parses. The executor rejects it. Counting it as a success reports a rate the
    pipeline cannot act on, and rule 3 makes it a failure.
    """
    from chartqa_dt.prompting.parsing import ParseStats, schema_ok

    args_as_object = {
        "answerable": True,
        "evidence": [{"label": "Zara", "value": 99, "unit": "stores",
                      "bbox": [340, 180, 650, 200]}],
        "plan": {"op": "lookup", "args": {"label": "Zara", "value": 99}},
        "model_answer": "99",
    }
    ok, why = schema_ok(args_as_object)
    assert ok is False and "args" in why

    fixed = {**args_as_object, "plan": {"op": "lookup", "args": ["Zara"]}}
    assert schema_ok(fixed)[0] is True

    stats = ParseStats()
    stats.add(parse_record(json.dumps(args_as_object)))
    stats.add(parse_record(json.dumps(fixed)))
    assert stats.valid == 2, "both parse"
    assert stats.schema_valid == 1, "only one is usable"
    assert stats.valid_fraction == pytest.approx(1.0)
    assert stats.schema_valid_fraction == pytest.approx(0.5)
    assert stats.schema_reasons, "a schema failure must record why"
    assert "the gate that matters" in stats.describe()


def test_the_prompt_states_the_schema_limits_it_must_satisfy():
    """Every limit the first compact probe violated is now named in the prompt.

    Measured schema failures, 5 in 20 parsed records: `args` of 5 and 8 elements (cap 4),
    a 35-character unit (cap 32), duplicate evidence labels, and `op: "average"` which is
    not in the enum. None of those limits appeared in the prompt, so the model had no way
    to respect them.
    """
    from chartqa_dt.prompting.prompts import MAX_ARGS, MAX_EVIDENCE, MAX_UNIT_CHARS

    prompt = build_structured_prompt("q")
    assert f"NEVER more than {MAX_EVIDENCE} items" in prompt
    assert f"at most {MAX_ARGS} elements" in prompt
    assert f"at most {MAX_UNIT_CHARS} characters" in prompt
    assert "appears at most ONCE" in prompt
    assert 'Use "mean" (not "average")' in prompt
    assert "never list the labels" in prompt


def test_the_prompt_limits_are_read_from_the_schema_not_restated():
    """A prompt that hard-codes a limit drifts from the schema the moment one changes."""
    from chartqa_dt.plans.schema import OUTPUT_SCHEMA
    from chartqa_dt.prompting.prompts import MAX_ARGS, MAX_EVIDENCE, MAX_UNIT_CHARS

    evidence = OUTPUT_SCHEMA["properties"]["evidence"]
    assert evidence["maxItems"] == MAX_EVIDENCE
    assert evidence["items"]["properties"]["unit"]["maxLength"] == MAX_UNIT_CHARS
    assert OUTPUT_SCHEMA["$defs"]["node"]["properties"]["args"]["maxItems"] == MAX_ARGS


def test_every_operation_the_prompt_offers_is_one_the_executor_accepts():
    """The reverse of the earlier check: nothing offered that would be rejected."""
    from chartqa_dt.plans.schema import OUTPUT_SCHEMA
    from chartqa_dt.prompting.prompts import ALLOWED_OPS

    assert set(ALLOWED_OPS) == set(OUTPUT_SCHEMA["$defs"]["node"]["properties"]["op"]["enum"])


def test_the_prompt_discourages_over_emitting_boxes():
    """`DECISIONS.md` 0014, and it is also why records were truncating.

    Records that hit the token cap were listing every bar in the chart. Extra boxes cost
    AP (1.00 -> 0.68 for one spurious box per image), cost tokens, and were the direct
    cause of 4 of 5 parse failures. One instruction addresses all three.
    """
    prompt = build_structured_prompt("q")
    assert "Fewer is better" in prompt
    assert "most important first" in prompt
    assert "Do NOT keep listing" in prompt


def test_a_stray_quote_after_an_array_is_repaired():
    """Measured 11-21 times per affected record in the zero-shot probe.

    `"bbox":[10,20,30,40]"` — a quotation mark where no valid JSON can have one. There is
    exactly one reading, so removing it invents nothing; it is transport noise by the same
    standard as a code fence, and it is counted like one.
    """
    body = ('{"answerable":true,"evidence":[{"label":"UBS","value":682,"unit":"GBP",'
            '"bbox":[100,250,250,270]"},{"label":"X","value":1,"unit":null,'
            '"bbox":[1,2,3,4]}],"plan":{"op":"lookup","args":["UBS"]},'
            '"model_answer":"682"}')
    result = parse_record(body)
    assert result.ok
    assert "removed stray quote after an array" in result.repairs
    assert [e["bbox"] for e in result.record["evidence"]] == \
        [[100, 250, 250, 270], [1, 2, 3, 4]], "no coordinate may be altered"


def test_the_stray_quote_repair_does_not_touch_legitimate_quotes():
    """A quoted string that merely follows an array must survive untouched."""
    body = ('{"answerable":true,"evidence":[{"label":"a","value":1,"unit":null,'
            '"bbox":[1,2,3,4]}],"plan":{"op":"lookup","args":["a"]},'
            '"model_answer":"ok"}')
    result = parse_record(body)
    assert result.ok and result.repairs == []
    assert result.record["model_answer"] == "ok"
    assert result.record["plan"]["args"] == ["a"]


def test_the_prompt_avoids_negative_instructions_about_syntax():
    """`DECISIONS.md` 0062: telling the model not to emit a token did not help.

    v4 added "never write bbox":[...]"" and a worked closing example. The stray quote
    still appeared in every failure, the prompt grew by 1,266 characters, and median
    output nearly doubled — 118 tokens to 229. Negative instructions can raise the
    probability of the token they name, and here they measurably cost length for no
    measurable gain. The parser repairs the artefact instead; that is what a repair is
    for.
    """
    prompt = build_structured_prompt("q")
    assert 'never "bbox"' not in prompt
    assert "Close every object" not in prompt
    assert "bbox is four integers 0-999" in prompt


def test_the_prompt_tells_the_model_what_to_do_when_a_chart_is_too_long():
    """58.5% of ChartQA tables have more than 8 rows; the schema caps evidence at 8.

    Without an instruction the model kept enumerating and ran off the token budget, and
    an unfinished record scores zero. Stopping at the cap and still answering correctly is
    strictly better.
    """
    from chartqa_dt.prompting.prompts import MAX_EVIDENCE

    prompt = build_structured_prompt("q")
    assert f"NEVER more than {MAX_EVIDENCE} items" in prompt
    assert "an unfinished record scores zero" in prompt
    assert "whole-chart total or average" in prompt


# ------------------------------------------------------- the training sequence budget


#: Measured with the real `Qwen/Qwen3-VL-2B-Instruct` tokenizer, not estimated.
MEASURED_TOKENS = {
    "structured_prompt": 980,
    "training_prompt": 117,
    "plain_prompt": 27,
    "target_2_evidence": 106,
    "target_8_evidence": 241,
    "visual_tokens_512px": 247,     # DECISIONS.md 0027
    "chat_template_overhead": 30,
}


def test_a_training_example_fits_the_sequence_budget():
    """The defect this constant table exists to prevent — `DECISIONS.md` 0064.

    Training an example costs visual tokens + prompt + target + chat overhead. With the
    980-token zero-shot prompt that is 1,363–1,498 tokens against a `max_seq_len` of
    1,024, so **every example would have been silently truncated** — and a truncated
    target teaches incomplete records while the loss curve looks entirely normal.

    Raising the limit was measured and rejected: 1,536 tokens implies at least 14.9 h for
    3,000 steps against a 10 h gate, and that is a lower bound because attention is
    quadratic. The training prompt is short instead.
    """
    from chartqa_dt.config import ModelConfig

    limit = ModelConfig().max_seq_len
    m = MEASURED_TOKENS
    fixed = m["visual_tokens_512px"] + m["chat_template_overhead"]

    worst_case = fixed + m["training_prompt"] + m["target_8_evidence"]
    assert worst_case < limit, (
        f"a worst-case training example is {worst_case} tokens against a {limit} limit"
    )
    assert limit - worst_case > 300, "keep real headroom; targets vary"

    would_have_been = fixed + m["structured_prompt"] + m["target_8_evidence"]
    assert would_have_been > limit, (
        "the zero-shot prompt must NOT fit — if it does, this test has stopped guarding "
        "anything and the constants need re-measuring"
    )


def test_the_training_prompt_is_far_shorter_than_the_zero_shot_one():
    """After fine-tuning the format lives in the weights; the long prompt is for eliciting
    it from a model that has never seen it."""
    from chartqa_dt.prompting.prompts import build_training_prompt

    short = build_training_prompt("q")
    long_ = build_structured_prompt("q")
    assert len(short) < len(long_) / 4
    assert "{question}" not in short and short.rstrip().endswith("q")
    # It must still pin the one convention a target cannot express by example alone.
    assert "0-999" in short
    for key in ("answerable", "evidence", "plan", "model_answer", "bbox"):
        assert key in short, f"the training prompt must still name {key}"


def test_all_three_prompts_are_fingerprinted():
    """`PLAN.md` 5.5 seals the prompt text; there are three of them and all are sealed."""
    fp = prompt_fingerprint()
    assert set(fp) == {"structured", "plain", "training"}
    assert len(set(fp.values())) == 3, "three distinct prompts, three distinct hashes"


def test_evidence_the_schema_cannot_hold_is_dropped_and_counted():
    """The third repair category: drop, never add — `DECISIONS.md` 0068.

    Measured on 200 real zero-shot generations: 17 records carried an evidence item with
    no `bbox`, and 24 of 133 exceeded the eight-item cap by enumerating a whole chart.
    Either invalidates the record outright, so the choice is between dropping the items
    and dropping the record. Dropping items took schema validity from 35.5% to 46.5% and
    usable records from 49 to 61.
    """
    from chartqa_dt.plans.schema import MAX_EVIDENCE

    record = {
        "answerable": True,
        "evidence": [{"label": "a", "value": 1, "unit": None, "bbox": [1, 2, 3, 4]},
                     {"label": "no-box", "value": 2},
                     *[{"label": f"c{i}", "value": i, "unit": None, "bbox": [1, 2, 3, 4]}
                       for i in range(12)]],
        "plan": {"op": "lookup", "args": ["a"]},
        "model_answer": "1",
    }
    result = parse_record(json.dumps(record))
    assert result.ok
    kept = result.record["evidence"]
    assert len(kept) == MAX_EVIDENCE
    assert all("bbox" in e for e in kept)
    assert kept[0]["label"] == "a", "the model's own ordering is preserved"
    assert any("no bbox" in r for r in result.repairs)
    assert any("first 8" in r for r in result.repairs)


def test_a_record_within_the_limits_is_left_untouched():
    """The repair must not fire on healthy records, or the repair rate is meaningless."""
    result = parse_record(example_record_json())
    assert result.ok and result.repairs == []
    assert len(result.record["evidence"]) == 2


def test_dropping_never_adds_a_field():
    """The invariant across all three repair categories: drop, unwrap, never add."""
    record = {"answerable": True, "evidence": [{"label": "a"}],
              "plan": {"op": "unanswerable", "args": []}, "model_answer": ""}
    result = parse_record(json.dumps(record))
    assert result.ok
    assert result.record["evidence"] == [], "the box-less item is removed, not invented"
    assert set(result.record) == set(record), "no key is added"
