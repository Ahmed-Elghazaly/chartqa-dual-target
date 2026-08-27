"""Mechanical enforcement of non-negotiable rule 1: test splits are sealed.

Rule 1 says: *"Never train on, tune on, or even inspect ChartQA test, RefChartQA
test, or ChartQAPro."* That sentence appears in the README, the pre-flight
checklist and the non-negotiable list — and it was still violated (`DECISIONS.md`
0031), because a sentence is not a control.

Every invariant in this project that actually holds is enforced by an assertion:
LoRA coverage, the quantisation skip, code freshness, device pinning,
documentation consistency. Sealed-split access had only prose. This module gives
it the same treatment.

The gate is deliberately hard to pass by accident and easy to pass on purpose,
once, at the point the plan intends:

* the default is **refusal**;
* opening a seal requires `PREREGISTRATION.md` to exist, to be **committed**, and
  to be **clean** in the working tree — which is exactly `PLAN.md` 5.5's condition
  (*"After this file is committed, test splits may be opened. Not before."*);
* every opening is logged with a reason, so the audit trail is automatic rather
  than remembered.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Split names that are sealed, per dataset. ChartQAPro is test-only and sealed
# in its entirety (IDEA.md 6.4).
SEALED_SPLITS: dict[str, frozenset[str]] = {
    "chartqa": frozenset({"test"}),
    "refchartqa": frozenset({"test"}),
    "chartqapro": frozenset({"test", "train", "validation", "val"}),
}

PREREGISTRATION = "PREREGISTRATION.md"


class SealedSplitError(RuntimeError):
    """Raised when sealed data is touched without authorisation. Never caught and logged away."""


@dataclass(frozen=True)
class SealStatus:
    preregistration_exists: bool
    preregistration_committed: bool
    preregistration_clean: bool
    reason: str

    @property
    def may_open(self) -> bool:
        return (
            self.preregistration_exists
            and self.preregistration_committed
            and self.preregistration_clean
        )

    def describe(self) -> str:
        if self.may_open:
            return "seal MAY be opened: PREREGISTRATION.md exists, is committed and is clean"
        return f"seal is CLOSED: {self.reason}"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def is_sealed(dataset: str, split: str) -> bool:
    """True if this (dataset, split) is sealed by rule 1."""
    return split.lower().strip() in SEALED_SPLITS.get(dataset.lower().strip(), frozenset())


def seal_status(repo_root: Path | None = None) -> SealStatus:
    """Whether the pre-registration condition in PLAN.md 5.5 is satisfied."""
    root = repo_root or _repo_root()
    path = root / PREREGISTRATION

    if not path.is_file():
        return SealStatus(False, False, False, f"{PREREGISTRATION} does not exist")

    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True, text=True, timeout=15, check=True,
            ).stdout
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return None

    tracked = git("ls-files", "--error-unmatch", PREREGISTRATION)
    if not tracked:
        return SealStatus(True, False, False, f"{PREREGISTRATION} is not committed to git")

    # Uncommitted edits mean the committed pre-registration is not the one on disk,
    # so "frozen before the test split was opened" would not be a true statement.
    dirty = git("status", "--porcelain", "--", PREREGISTRATION)
    if dirty is None or dirty.strip():
        return SealStatus(True, True, False, f"{PREREGISTRATION} has uncommitted changes")

    return SealStatus(True, True, True, "")


def assert_split_allowed(
    dataset: str,
    split: str,
    *,
    authorised: bool = False,
    reason: str = "",
    repo_root: Path | None = None,
    logger: object | None = None,
) -> None:
    """Refuse to touch a sealed split unless explicitly and legitimately authorised.

    ``authorised=True`` alone is not enough — the pre-registration condition must
    also hold. That combination is deliberate: a flag can be passed by habit, but
    a committed, clean ``PREREGISTRATION.md`` is a deliberate act with a git
    history behind it.
    """
    if not is_sealed(dataset, split):
        return

    where = f"{dataset}/{split}"

    if not authorised:
        raise SealedSplitError(
            f"{where} is a SEALED split (non-negotiable rule 1: never train on, tune on, "
            f"or even inspect it).\n"
            f"Opening it requires explicit authorisation AND a committed, clean "
            f"{PREREGISTRATION}. If you are at Phase 7, pass authorised=True with a reason.\n"
            f"If you are not at Phase 7, use the validation split instead."
        )

    status = seal_status(repo_root)
    if not status.may_open:
        raise SealedSplitError(
            f"{where} may not be opened yet — {status.describe()}.\n"
            f"PLAN.md 5.5: \"After this file is committed, test splits may be opened. Not before.\""
        )

    if not reason.strip():
        raise SealedSplitError(
            f"opening {where} requires a stated reason, so the audit trail is automatic."
        )

    message = f"SEALED SPLIT OPENED: {where} — {reason}"
    print(message, flush=True)
    if logger is not None and hasattr(logger, "event"):
        logger.event("sealed_split_opened", dataset=dataset, split=split, reason=reason)


def sealed_split_env_override() -> bool:
    """Escape hatch for tests only, never for real runs.

    Returns True only when ``CDT_ALLOW_SEALED_FOR_TESTS`` is set, which the CLI
    never sets and which is documented as test-only so its presence in a real run
    is visibly wrong.
    """
    return bool(os.environ.get("CDT_ALLOW_SEALED_FOR_TESTS"))
