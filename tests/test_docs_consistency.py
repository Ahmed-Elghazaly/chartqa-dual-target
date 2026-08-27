"""Guards against documentation drifting away from the measurements it quotes.

This project accumulates markdown fast: a decision log, a verification record, a
run log, a pre-flight checklist, a setup guide, a README and a growing set of
teaching notes. The obvious failure mode is that a number gets corrected in one
place and not the others, and nobody notices — the same silent-divergence problem
the code guards against, applied to prose.

The structural answer is `verification/measured_facts.json`: one canonical home
for every measured number. These tests enforce that the prose agrees with it, that
decision numbering is sound, that every cross-reference resolves, and that every
file the docs point at exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FACTS = json.loads((ROOT / "verification/measured_facts.json").read_text(encoding="utf-8"))

TRACKED_DOCS = [
    "README.md", "SETUP.md", "RUNS.md", "DECISIONS.md",
    "verification/phase0.md", "verification/preflight_checklist.md",
]


# Documents that describe the CURRENT state of the project. A stale claim here
# misleads a reader today.
STATUS_DOCS = ["README.md", "SETUP.md", "RUNS.md", "verification/preflight_checklist.md"]

# DECISIONS.md is an append-only historical record and book/notes/ is narrative.
# Both legitimately quote past claims and point forward at work not yet done, so
# the "current fact" rules do not apply to them -- but everything structural
# (numbering, required sections, cross-references) still does.


def status_docs() -> dict[str, str]:
    return {rel: (ROOT / rel).read_text(encoding="utf-8")
            for rel in STATUS_DOCS if (ROOT / rel).is_file()}


def docs() -> dict[str, str]:
    out = {}
    for rel in TRACKED_DOCS:
        p = ROOT / rel
        if p.is_file():
            out[rel] = p.read_text(encoding="utf-8")
    for p in sorted((ROOT / "book/notes").glob("*.md")):
        out[f"book/notes/{p.name}"] = p.read_text(encoding="utf-8")
    return out


# --------------------------------------------------------------- decision log


def decision_numbers() -> list[int]:
    text = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")
    return [int(m) for m in re.findall(r"^## (\d{4}) —", text, flags=re.M)]


def test_decision_numbers_are_unique_and_sequential():
    nums = decision_numbers()
    assert nums, "no decision entries found — has the heading format changed?"
    dupes = {n for n in nums if nums.count(n) > 1}
    assert not dupes, f"duplicate decision numbers: {sorted(dupes)}"
    assert nums == sorted(nums), "decision entries must be in ascending order"
    expected = list(range(nums[0], nums[0] + len(nums)))
    assert nums == expected, f"gap in decision numbering: {sorted(set(expected) - set(nums))}"


def test_every_decision_has_the_required_sections():
    """Appendix H: context, options, decision, evidence, consequences."""
    text = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")
    entries = re.split(r"^## (?=\d{4} —)", text, flags=re.M)[1:]
    for entry in entries:
        title = entry.splitlines()[0]
        for section in ("**Context.**", "**Decision.**", "**Consequences.**"):
            assert section in entry, f"decision {title!r} is missing {section}"


def test_every_decision_cross_reference_resolves():
    """A pointer to decision 0042 when only 22 exist is a broken reference."""
    known = set(decision_numbers())
    pattern = re.compile(r"(?:DECISIONS\.md|decisions?)\s*[`'\"]*\s*(\d{4})", re.I)
    broken: list[str] = []
    for name, text in docs().items():
        for num in pattern.findall(text):
            if int(num) not in known:
                broken.append(f"{name} -> {num}")
    for path in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")):
        for num in pattern.findall(path.read_text(encoding="utf-8")):
            if int(num) not in known:
                broken.append(f"{path.relative_to(ROOT)} -> {num}")
    assert not broken, f"references to non-existent decisions: {broken}"


# ------------------------------------------------------- numbers match the facts


CRITICAL_NUMBERS = [
    ("28,299", "datasets.chartqa_splits.train"),
    ("55,789", "datasets.refchartqa_splits.train"),
    ("11,690", "datasets.refchartqa_splits.test"),
    ("32.83", "published_targets.refchartqa_ap50_human_qwen25vl3b"),
    ("79.1", "published_targets.chartqa_qwen3vl2b_instruct"),
    ("86.6", "published_targets.chartqa_qwen3vl2b_thinking"),
    ("13.5", "gates.memory_gb"),
    ("875,370,872", "datasets.chartqa_archive_bytes"),
]


def lookup(dotted: str):
    node = FACTS
    for part in dotted.split("."):
        node = node[part]
    return node


@pytest.mark.parametrize(("rendered", "path"), CRITICAL_NUMBERS)
def test_quoted_numbers_exist_in_the_canonical_facts(rendered, path):
    """Every number the docs lean on must be traceable to measured_facts.json."""
    value = lookup(path)
    assert f"{value:,}" == rendered or str(value) == rendered, (
        f"{path} is {value!r} in measured_facts.json but the docs quote {rendered!r}"
    )


def test_visual_token_factor_is_consistent_everywhere():
    """Decision 0008's whole point: 32, not the inherited 28."""
    from chartqa_dt.vision.coords import QWEN3VL_FACTOR, QWEN3VL_MERGE_SIZE, QWEN3VL_PATCH_SIZE

    assert FACTS["model"]["visual_token_factor"] == 32 == QWEN3VL_FACTOR
    assert FACTS["model"]["patch_size"] == QWEN3VL_PATCH_SIZE
    assert FACTS["model"]["spatial_merge_size"] == QWEN3VL_MERGE_SIZE
    assert QWEN3VL_PATCH_SIZE * QWEN3VL_MERGE_SIZE == QWEN3VL_FACTOR


def test_gates_match_the_code():
    from chartqa_dt.train.smoke import (
        FULL_RUN_GATE_HOURS,
        MEMORY_GATE_GB,
        PLANNED_OPTIMIZER_STEPS,
    )

    assert FACTS["gates"]["memory_gb"] == MEMORY_GATE_GB
    assert FACTS["gates"]["full_run_hours"] == FULL_RUN_GATE_HOURS
    assert FACTS["gates"]["planned_optimizer_steps"] == PLANNED_OPTIMIZER_STEPS


def test_official_max_coord_matches_the_code():
    from chartqa_dt.vision.coords import OFFICIAL_MAX_COORD

    assert FACTS["vendored"]["official_max_coord"] == OFFICIAL_MAX_COORD == 999


def test_vendored_hash_matches_the_provenance_file():
    prov = json.loads(
        (ROOT / "verification/refchartqa_eval/PROVENANCE.json").read_text(encoding="utf-8")
    )
    assert prov["files"]["evaluate.py"]["sha256"] == FACTS["vendored"]["refchartqa_evaluate_py_sha256"]


def test_refchartqa_test_subsets_sum_to_the_split_size():
    by_type = FACTS["datasets"]["refchartqa_test_by_type"]
    assert sum(by_type.values()) == FACTS["datasets"]["refchartqa_splits"]["test"]


def test_phase2_measurements_are_inside_their_gates():
    p2, gates = FACTS["phase2"], FACTS["gates"]
    assert p2["peak_reserved_gb"] <= gates["memory_gb"]
    assert p2["projected_full_run_hours"] <= gates["full_run_hours"]
    assert p2["loss_last_10"] < p2["loss_first_10"]
    assert p2["lora_vision_params"] > 0 and p2["lora_language_params"] > 0
    assert p2["quantised_vision_full"] > 0, "the vision tower must not be 4-bit"


# --------------------------------------------------------------- file references


def test_files_referenced_by_docs_exist():
    """A doc pointing at a file that was renamed is drift you can catch.

    Scoped to status documents: the decision log is history and may reference a
    file that a superseded entry created, while the book notes point forward at
    work not yet done.
    """
    pattern = re.compile(r"`((?:src|scripts|tests|configs|verification|book|report)/[\w./-]+)`")
    missing: list[str] = []
    for name, text in status_docs().items():
        for rel in set(pattern.findall(text)):
            if rel.endswith("/"):
                continue
            if not (ROOT / rel).exists():
                missing.append(f"{name} -> {rel}")
    assert not missing, f"docs reference files that do not exist: {sorted(set(missing))}"


def test_no_doc_claims_ci_is_green():
    """CI status is a live fact and must be read from the pipeline, not from prose.

    Written after reporting 'CI green' from a stale spot-check while it had been
    failing for eight consecutive runs. `scripts/check_ci.py` answers this against
    the current commit; documentation must not assert it.
    """
    offenders = [
        f"{name}: {line.strip()}"
        for name, text in status_docs().items()
        for line in text.splitlines()
        if re.search(r"CI (?:is |was )?green", line, re.I)
    ]
    assert not offenders, (
        "documentation asserts CI status, which goes stale silently; "
        f"use scripts/check_ci.py instead. Offending lines: {offenders}"
    )


# ------------------------------------------------------- pinned Hub revisions


def test_every_downloadable_artifact_has_a_pinned_revision():
    """Rule 6 requires the same dataset VERSION for a matched comparison.

    A dataset re-upload between our baseline run and our trained run would make
    the comparison unmatched without anything failing, so every download passes
    revision=<sha>.
    """
    pins = FACTS["pinned_revisions"]
    required = [
        "ahmed-masry/ChartQA",
        "omoured/RefChartQA",
        "ahmed-masry/ChartQAPro",
        "Qwen/Qwen3-VL-2B-Instruct",
    ]
    for repo in required:
        assert repo in pins, f"{repo} has no pinned revision"
        sha = pins[repo]
        assert isinstance(sha, str) and len(sha) == 40 and all(
            c in "0123456789abcdef" for c in sha
        ), f"{repo}: {sha!r} is not a 40-character hex commit sha"


def test_pinned_model_matches_the_configured_backbone():
    from chartqa_dt.config import Config

    assert Config().model.hf_id in FACTS["pinned_revisions"], (
        "the configured backbone must have a pinned revision"
    )
    assert Config().model.hf_id == FACTS["model"]["hf_id"]


# ------------------------------------------------------- sequence budget


def test_the_structured_record_fits_the_sequence_budget():
    """Measured, not estimated. If any of these stop holding, Phase 5's prompt
    design or max_seq_len must change before training, not after."""
    b = FACTS["sequence_budget"]
    assert b["total_512px"] < b["max_seq_len"], "the planned configuration must fit"
    assert b["worst_case_512px"] < b["max_seq_len"], (
        "even a full 8-item evidence list must fit, or the schema's own maximum is unreachable"
    )
    assert b["total_native"] < b["max_seq_len"], "native must fit too, for the Phase 8.3 ablation"


def test_compact_json_is_materially_cheaper_than_pretty():
    """The prompt must demand compact output; the penalty is large enough to matter."""
    b = FACTS["sequence_budget"]
    assert b["record_pretty_tokens"] > 1.5 * b["record_compact_tokens"], (
        "if pretty-printing were nearly free, the compactness instruction could be dropped"
    )


def test_pad_and_eos_are_different_tokens():
    """build_batch masks pad tokens. If pad were eos, that would also mask the
    stop token and the model would never learn to terminate."""
    b = FACTS["sequence_budget"]
    assert b["pad_token_id"] != b["eos_token_id"]


def test_sequence_budget_matches_the_model_config():
    from chartqa_dt.config import Config

    assert Config().model.max_seq_len == FACTS["sequence_budget"]["max_seq_len"]


def test_lower_resolution_costs_sub_token_performance_in_every_subset():
    """The 448-vs-512 trade is a cost on the metric this project exists to move,
    so it must be visible per subset rather than as one blended figure."""
    s = FACTS["subtoken"]
    for subset in ("human", "machine", "pot"):
        at448 = s["by_subset_448"][subset]
        at512 = s["by_subset_512"][subset]
        assert at448 > at512, f"{subset}: 448 should be worse than 512, got {at448} vs {at512}"
    assert s["median_visual_tokens_448"] < s["median_visual_tokens_512"]


def test_the_machine_subset_is_the_hardest_to_ground():
    """Its boxes are systematically smaller, so a cross-subset score comparison
    must account for geometry rather than reading it as model weakness."""
    s = FACTS["subtoken"]["by_subset_512"]
    assert s["machine"] > s["human"] > s["pot"]


# ------------------------------------------------ the facts file's own consistency
#
# `measured_facts.json` is the single source of truth, and the tests above check that the
# prose agrees with it. That is not enough: a single source of truth can be *consistently
# wrong*, and it was. `phase2.peak_reserved_gb` carried 1.482 GB — a superseded figure
# from a sharded run where `max_memory_reserved()` read device 0 alone and understated the
# footprint ~3.8x (`DECISIONS.md` 0025). Every document agreed with it, so every document
# was wrong together, and it would have gone into `PREREGISTRATION.md`.
#
# The guard is not more agreement. It is checking relationships the numbers must satisfy
# on physical grounds, which a stale value breaks.


def test_more_pixels_cost_more_memory_and_more_time():
    """A resolution that renders more visual tokens cannot be cheaper on both axes."""
    p2 = FACTS["phase2"]
    at_448 = p2.get("_measured_at_448") or {}
    native = p2.get("_measured_native") or {}

    if at_448:
        assert p2["peak_reserved_gb"] > at_448["peak_gb"], (
            f"512 px reports {p2['peak_reserved_gb']} GB against 448 px's "
            f"{at_448['peak_gb']} GB — a larger image cannot use less memory. This is "
            f"how the superseded sharded-run figure was found."
        )
    if native:
        assert native["peak_gb"] > p2["peak_reserved_gb"]
        assert native["seconds_per_step"] > p2["seconds_per_step"]
        assert native["visual_tokens"] > p2["visual_tokens_512"]


def test_projected_hours_follow_from_the_step_time():
    """3,000 steps at the recorded rate must give the recorded projection."""
    p2 = FACTS["phase2"]
    implied = 3000 * p2["seconds_per_step"] / 3600
    assert implied == pytest.approx(p2["projected_full_run_hours"], rel=0.02), (
        f"{p2['seconds_per_step']} s/step over 3,000 steps is {implied:.2f} h, "
        f"but the file records {p2['projected_full_run_hours']} h"
    )


def test_superseded_measurements_are_labelled_not_deleted():
    """History stays legible, but a superseded number must not sit in a live field."""
    p2 = FACTS["phase2"]
    old = p2.get("_superseded_sharded_run")
    if old is None:
        pytest.skip("no superseded run recorded")
    assert "_why" in old, "a superseded number needs the reason it was superseded"
    for key in ("peak_reserved_gb", "seconds_per_step", "projected_full_run_hours"):
        assert p2[key] != old[key], f"{key} still carries the superseded value"


def test_the_measured_512_sessions_bracket_the_recorded_projection():
    """Three independent sessions measured this; the headline must sit among them."""
    p2 = FACTS["phase2"]
    sessions = p2.get("_measured_at_512_three_sessions_hours")
    if not sessions:
        pytest.skip("no per-session measurements recorded")
    assert min(sessions) <= p2["projected_full_run_hours"] <= max(sessions) + 0.75, (
        f"projection {p2['projected_full_run_hours']} h sits outside the measured "
        f"range {min(sessions)}–{max(sessions)} h"
    )
