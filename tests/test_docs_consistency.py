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
