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
    call_anthropic,
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
    assert "wrong route" in a_prompt()
    assert "lookup even when the item happens to be the largest" in a_prompt()


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
    got_plan, got_refused, _ = parse_proposal(reply)
    assert (got_plan, got_refused) == (plan, refused)


def test_the_last_json_block_wins():
    """Models often restate the format before answering; the answer is the final block."""
    plan, _, _ = parse_proposal('```json\n{"op": "sum", "args": []}\n```\n'
                                'on reflection:\n```json\n{"op": "max", "args": []}\n```')
    assert plan == {"op": "max", "args": []}


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
    fresh ids, so every cache lookup would miss and a resumed run would pay twice."""
    code = ("import sys; sys.path.insert(0, 'scripts'); "
            "from mine_with_llm import stable_id; print(stable_id('chartqa', 'a.png', 'q?'))")
    runs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           check=True).stdout.strip() for _ in range(3)}
    assert len(runs) == 1, f"unstable record ids across processes: {runs}"
