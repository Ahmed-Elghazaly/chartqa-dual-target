"""Non-negotiable rule 1, enforced mechanically.

Rule 1 appears in the README, the pre-flight checklist and the non-negotiable
list, and was still violated (`DECISIONS.md` 0031) — because prose is not a
control. Every invariant in this project that actually holds is an assertion.
This gives sealed-split access the same treatment.
"""

from __future__ import annotations

import subprocess

import pytest

from chartqa_dt.splits import (
    PREREGISTRATION,
    SEALED_SPLITS,
    SealedSplitError,
    assert_split_allowed,
    is_sealed,
    seal_status,
)

# ------------------------------------------------------------- what is sealed


@pytest.mark.parametrize(
    ("dataset", "split", "sealed"),
    [
        ("chartqa", "test", True),
        ("chartqa", "val", False),
        ("chartqa", "train", False),
        ("refchartqa", "test", True),
        ("refchartqa", "validation", False),
        ("refchartqa", "train", False),
        # ChartQAPro is test-only and sealed in its entirety (IDEA.md 6.4).
        ("chartqapro", "test", True),
        ("chartqapro", "train", True),
        ("chartqapro", "validation", True),
    ],
)
def test_sealing_matches_rule_1(dataset, split, sealed):
    assert is_sealed(dataset, split) is sealed


def test_case_and_whitespace_do_not_defeat_the_seal():
    for variant in ("TEST", " test ", "Test"):
        assert is_sealed("chartqa", variant), f"{variant!r} slipped past the seal"
    for variant in ("ChartQA", " chartqa"):
        assert is_sealed(variant, "test"), f"dataset {variant!r} slipped past the seal"


def test_every_dataset_in_the_project_is_covered():
    assert set(SEALED_SPLITS) == {"chartqa", "refchartqa", "chartqapro"}


# -------------------------------------------------------------- the refusal


def test_unsealed_splits_pass_without_ceremony():
    assert_split_allowed("chartqa", "val")
    assert_split_allowed("refchartqa", "train")


def test_a_sealed_split_is_refused_by_default():
    with pytest.raises(SealedSplitError, match="SEALED split"):
        assert_split_allowed("chartqa", "test")


def test_the_refusal_names_the_rule_and_the_alternative():
    with pytest.raises(SealedSplitError) as exc:
        assert_split_allowed("refchartqa", "test")
    msg = str(exc.value)
    assert "rule 1" in msg
    assert "validation split" in msg, "the message must say what to do instead"


def test_authorisation_alone_is_not_enough(tmp_path):
    """A flag can be passed by habit; a committed pre-registration cannot."""
    with pytest.raises(SealedSplitError, match="may not be opened yet"):
        assert_split_allowed("chartqa", "test", authorised=True, reason="Phase 7",
                             repo_root=tmp_path)


def test_a_reason_is_required_even_when_everything_else_holds(tmp_path):
    _make_committed_prereg(tmp_path)
    with pytest.raises(SealedSplitError, match="requires a stated reason"):
        assert_split_allowed("chartqa", "test", authorised=True, reason="  ",
                             repo_root=tmp_path)


# --------------------------------------------------- the pre-registration gate


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)


def _make_committed_prereg(root):
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / PREREGISTRATION).write_text("frozen decisions\n", encoding="utf-8")
    _git(root, "add", PREREGISTRATION)
    _git(root, "commit", "-q", "-m", "prereg")


def test_seal_is_closed_when_the_preregistration_is_absent(tmp_path):
    status = seal_status(tmp_path)
    assert not status.may_open
    assert "does not exist" in status.reason


def test_seal_is_closed_when_the_preregistration_is_uncommitted(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / PREREGISTRATION).write_text("draft\n", encoding="utf-8")
    status = seal_status(tmp_path)
    assert not status.may_open
    assert "not committed" in status.reason


def test_seal_is_closed_when_the_preregistration_has_uncommitted_edits(tmp_path):
    """'Frozen before the test split was opened' must be literally true."""
    _make_committed_prereg(tmp_path)
    (tmp_path / PREREGISTRATION).write_text("edited after committing\n", encoding="utf-8")
    status = seal_status(tmp_path)
    assert not status.may_open
    assert "uncommitted changes" in status.reason


def test_seal_opens_only_when_committed_and_clean(tmp_path):
    _make_committed_prereg(tmp_path)
    status = seal_status(tmp_path)
    assert status.may_open, status.reason
    assert_split_allowed("chartqa", "test", authorised=True,
                         reason="Phase 7 headline evaluation", repo_root=tmp_path)


def test_opening_the_seal_is_logged(tmp_path, capsys):
    _make_committed_prereg(tmp_path)

    class Recorder:
        def __init__(self):
            self.events = []

        def event(self, name, **fields):
            self.events.append((name, fields))

    rec = Recorder()
    assert_split_allowed("chartqa", "test", authorised=True, reason="Phase 7",
                         repo_root=tmp_path, logger=rec)
    assert "SEALED SPLIT OPENED" in capsys.readouterr().out
    assert rec.events and rec.events[0][0] == "sealed_split_opened"
    assert rec.events[0][1]["reason"] == "Phase 7"


def test_the_repository_seal_is_currently_closed():
    """Until Phase 5.5 commits the pre-registration, this must stay closed."""
    status = seal_status()
    assert not status.may_open, (
        "the seal reports open, but PREREGISTRATION.md should not exist yet — "
        "if Phase 5 has completed, update this test deliberately"
    )
