"""The proposing half of LLM mining: what gets asked, and how the reply is read.

Nothing here touches the network. The one function that would (`call_anthropic`) is tested
only for refusing to run without a key -- a run that cannot reach a model must fail loudly
rather than return nothing and look like a measurement.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from chartqa_dt.plans.executor import NEEDS_TABLE, OPS
from chartqa_dt.plans.teacher import (
    OFFERED,
    PROVENANCE_KEY,
    SIGNATURES,
    TeacherError,
    TeacherRequest,
    build_prompt,
    build_system,
    call_anthropic,
    estimate_cost,
    parse_proposal,
    provenance,
)

TABLE = {"columns": ["Characteristic", "Value"], "rows": [["Nigeria", "154.3"],
                                                          ["Egypt", "54.74"]]}
EVIDENCE = [{"label": "Nigeria", "value": 154.3}, {"label": "Egypt", "value": 54.74}]


def a_prompt(**kw):
    args = {"question": "How many users did Nigeria have?", "answer": "154.3",
            "table": TABLE, "evidence": EVIDENCE}
    return build_prompt(**{**args, **kw})


def test_only_implemented_operations_are_offered():
    """`filter`, `rank` and `multiple_choice` are in OPS but raise in the executor.

    Offering them would earn guaranteed rejections and make the teacher's failure profile
    look like bad judgement rather than an unimplemented operation.
    """
    assert set(OFFERED) == set(OPS) - set(NEEDS_TABLE)
    assert not set(OFFERED) & set(NEEDS_TABLE)


def test_every_offered_operation_has_a_signature():
    assert set(OFFERED) <= set(SIGNATURES), f"undocumented: {set(OFFERED) - set(SIGNATURES)}"


def test_the_prompt_is_deterministic():
    assert a_prompt() == a_prompt()
    assert TeacherRequest("r", a_prompt()).sha256 == TeacherRequest("r", a_prompt()).sha256


def test_a_different_question_gives_a_different_prompt_hash():
    """The hash is the cache key, so two records must not share one."""
    one = TeacherRequest("r", a_prompt()).sha256
    two = TeacherRequest("r", a_prompt(question="Which country had the most?")).sha256
    assert one != two


def test_the_prompt_lists_the_evidence_labels_verbatim():
    """A label the teacher invents is rejected downstream, so it must be given them."""
    text = a_prompt()
    assert "'Nigeria'" in text
    assert "'Egypt'" in text


def test_marked_regions_are_flagged_only_when_present():
    assert "[MARKED]" not in a_prompt()
    marked = a_prompt(marked_labels={"Nigeria"})
    assert "[MARKED]" in marked
    assert marked.index("[MARKED]") > marked.index("'Nigeria'")


def test_the_prompt_warns_against_the_right_number_by_the_wrong_route():
    """The single-element blind spot (0080) is not checkable by the gates, so it is asked
    for in words: this sentence is the only defence against it."""
    assert "wrong route" in build_system()
    assert "lookup even when the item happens to be the largest" in build_system()


@pytest.mark.parametrize("reply,plan,refused", [
    ('```json\n{"op": "lookup", "args": ["Nigeria"]}\n```',
     {"op": "lookup", "args": ["Nigeria"]}, False),
    ('sure, here:\n```json\n{"refused": "no yes/no operator"}\n```', None, True),
    ('{"op": "max", "args": []}', {"op": "max", "args": []}, False),
    ("I could not work this one out.", None, False),
    ('```json\n{"nope": 1}\n```', None, False),
    ("```json\n[1, 2, 3]\n```", None, False),
])
def test_parse_proposal(reply, plan, refused):
    got = parse_proposal(reply)
    assert (got.plan, got.refused) == (plan, refused)
    assert got.usable is (plan is not None)


def test_the_last_json_block_wins():
    """Models often restate the format before answering; the answer is the final block."""
    got = parse_proposal('```json\n{"op": "sum", "args": []}\n```\n'
                         'on reflection:\n```json\n{"op": "max", "args": []}\n```')
    assert got.plan == {"op": "max", "args": []}


# ------------------------------------------------- asking for a missing operation


def test_a_missing_operation_is_a_third_answer_not_a_refusal():
    """A refusal says *this question cannot be answered*. A suggestion says *the question is
    fine and our DSL is not*. Counting them together would hide the only signal that says
    what to build next."""
    got = parse_proposal('```json\n{"needs_operator": {"name": "less_than", '
                         '"signature": "less_than(a, b) -> Yes | No", '
                         '"why": "Is the value in 2019 less than 2018?"}}\n```')
    assert got.needs_operator is not None
    assert got.needs_operator["name"] == "less_than"
    assert got.refused is False, "a missing operator is not a refusal"
    assert got.plan is None and got.usable is False


def test_a_suggestion_without_a_name_is_not_usable():
    got = parse_proposal('```json\n{"needs_operator": {"why": "something"}}\n```')
    assert got.needs_operator is None
    assert "name" in got.note


@pytest.mark.parametrize("field,limit", [("name", 64), ("signature", 200), ("why", 300)])
def test_a_suggestion_cannot_flood_the_report(field, limit):
    """The teacher writes these; they end up in a report and a JSON file, so they are
    bounded like any other untrusted text."""
    import json as _json
    got = parse_proposal("```json\n"
                         + _json.dumps({"needs_operator": {"name": "x", field: "z" * 900}})
                         + "\n```")
    assert len(got.needs_operator[field]) <= limit


def test_the_prompt_invites_a_missing_operation():
    text = build_system()
    assert "needs_operator" in text
    assert "do NOT force a fit" in text


# ------------------------------------------------- the split that makes caching possible


def test_the_stable_half_carries_no_per_record_text():
    """Prompt caching is a PREFIX match, so anything that varies per record must come after
    everything that does not. The first version of this prompt opened with the question and
    its table and closed with the operation catalogue -- exactly backwards -- which would
    have meant paying full input price for the catalogue on every one of tens of thousands
    of calls."""
    system = build_system()
    for volatile in ("Nigeria", "Egypt", "154.3", "How many users"):
        assert volatile not in system, f"{volatile!r} varies per record; it cannot be cached"


def test_the_stable_half_is_byte_identical_across_records():
    assert build_system() == build_system()
    one = a_prompt()
    two = a_prompt(question="Which country had the most?", answer="Nigeria")
    assert one != two, "the per-record half must actually vary"


def test_the_operation_catalogue_lives_in_the_cached_half():
    """It is the largest fixed block in the request and the whole reason to cache."""
    system = build_system()
    for op in ("lookup", "argmax", "percent_change"):
        assert op in system
        assert op not in a_prompt()


def test_a_cost_estimate_prices_the_cached_half_once():
    """The system half is billed in full once and at the cache-read rate thereafter; an
    estimate that charged it every time would misprice a large run by most of its cost."""
    small = estimate_cost(n_records=1, system_chars=2500, record_chars=200,
                          model="claude-opus-5", batch=False)
    big = estimate_cost(n_records=1000, system_chars=2500, record_chars=200,
                        model="claude-opus-5", batch=False)
    assert small["input_system_cached"] == 0.0
    naive = small["input_system_first"] * 1000
    assert big["input_system_first"] + big["input_system_cached"] < naive * 0.2


def test_batch_pricing_is_half():
    kw = {"n_records": 500, "system_chars": 2500, "record_chars": 200,
          "model": "claude-opus-5"}
    assert estimate_cost(**kw, batch=True)["total"] == pytest.approx(
        estimate_cost(**kw, batch=False)["total"] / 2)


def test_no_api_key_raises_rather_than_returning_nothing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(TeacherError, match="ANTHROPIC_API_KEY"):
        call_anthropic("hi", model="claude-opus-5")


def test_provenance_records_what_is_needed_to_reproduce():
    p = provenance(model="claude-opus-5", prompt_sha256="abc", verifier_gates=["shape"])
    assert p["model"] == "claude-opus-5"
    assert p["prompt_sha256"] == "abc"
    assert p["method"] == "llm_teacher"
    assert "prompt_version" in p, "without it, two prompt revisions are indistinguishable"
    assert PROVENANCE_KEY


def test_record_ids_are_stable_across_processes():
    """`hash()` on a string is salted per process. An id built from it would give every run
    fresh ids, so a plan mined in one session could never be joined back in the next.

    `make_record_id` is what the mixture builder and `attach_mined_plans` key on, so its
    stability is the property that actually matters.
    """
    code = ("import sys; sys.path.insert(0, 'src'); "
            "from chartqa_dt.data.records import make_record_id; "
            "print(make_record_id('chartqa', 'train', 'deadbeef', 'how many?'))")
    runs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           check=True).stdout.strip() for _ in range(3)}
    assert len(runs) == 1, f"unstable record ids across processes: {runs}"
