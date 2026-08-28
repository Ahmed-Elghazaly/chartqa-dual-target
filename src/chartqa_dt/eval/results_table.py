"""The master results table and the five claims — `PLAN.md` 7.4 and 7.5.

The plan's requirements here are about honesty rather than computation, so they are
enforced in code rather than left to the writer:

* *"Every cell carries a confidence interval."* A `Cell` without one cannot be rendered.
* *"Every comparison states whether it is matched."* A `Comparison` must declare it.
* *"Write the five separate claims explicitly, never merged."* `Claims` has exactly five
  fields, each with its own verdict and evidence, and the renderer refuses to collapse
  them.

The fifth claim is the one that needs a guard. "Published baseline exceeded" is only
allowed *"if genuinely comparable"*, and `DECISIONS.md` 0052 established that RefChartQA's
32.83 cannot be reproduced by anyone — no released predictions, no checkpoints. So that
claim is constrained to say so, and `assert_claims_honest` fails a report that quietly
upgrades it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: `DECISIONS.md` 0052. Kept here so a report cannot cite it without its status.
PUBLISHED_REFCHARTQA_AP50 = 32.83
PUBLISHED_STATUS = ("Level C — published, not independently reproducible: RefChartQA "
                    "releases no per-model predictions and no checkpoints")


@dataclass(frozen=True)
class Cell:
    """One measured number. A point estimate without an interval is not reportable."""

    value: float
    lo: float
    hi: float
    n: int = 0

    def __post_init__(self) -> None:
        if not (self.lo <= self.value <= self.hi):
            raise ValueError(
                f"interval [{self.lo}, {self.hi}] does not contain {self.value}")

    def render(self, *, percent: bool = True) -> str:
        scale = 100.0 if percent else 1.0
        suffix = "%" if percent else ""
        return (f"{scale * self.value:.2f}{suffix} "
                f"[{scale * self.lo:.2f}, {scale * self.hi:.2f}]")

    def overlaps(self, other: Cell) -> bool:
        """Whether two intervals overlap — the honest test for 'is this a real gain'."""
        return self.lo <= other.hi and other.lo <= self.hi


@dataclass
class SystemRow:
    """One row of `PLAN.md` 7.4's table."""

    name: str
    chartqa_human: Cell | None = None
    chartqa_machine: Cell | None = None
    chartqa_all: Cell | None = None
    refchartqa_ap50: Cell | None = None
    refchartqa_p_at_f1: Cell | None = None
    note: str = ""

    def cells(self) -> dict[str, Cell | None]:
        return {"ChartQA H": self.chartqa_human, "ChartQA M": self.chartqa_machine,
                "ChartQA all": self.chartqa_all, "RefChartQA AP@0.5": self.refchartqa_ap50,
                "RefChartQA P@F1": self.refchartqa_p_at_f1}


@dataclass
class Comparison:
    """A stated difference between two systems, with its matching status declared."""

    label: str
    baseline: str
    system: str
    metric: str
    baseline_cell: Cell
    system_cell: Cell
    matched: bool
    matched_on: str = ""

    @property
    def delta(self) -> float:
        return self.system_cell.value - self.baseline_cell.value

    @property
    def intervals_overlap(self) -> bool:
        return self.system_cell.overlaps(self.baseline_cell)

    def render(self) -> str:
        status = (f"matched on {self.matched_on}" if self.matched
                  else "**NOT matched** — the systems differ in more than the change")
        strength = ("intervals overlap, so this is not yet a demonstrated gain"
                    if self.intervals_overlap else "intervals are disjoint")
        return (f"- **{self.label}** ({self.metric}): {self.baseline} "
                f"{self.baseline_cell.render()} → {self.system} "
                f"{self.system_cell.render()}, "
                f"**{100 * self.delta:+.2f} pts**. {status}; {strength}.")


@dataclass
class Claim:
    verdict: str          # "yes" | "no" | "not applicable"
    evidence: str

    def render(self, number: int, text: str) -> str:
        return f"{number}. **{text}** — *{self.verdict}*. {self.evidence}"


@dataclass
class Claims:
    """`PLAN.md` 7.5's five claims. Five fields, never merged into a summary sentence."""

    official_training_reproduced: Claim
    official_checkpoint_evaluated: Claim
    matched_student_baseline_trained: Claim
    before_after_improvement: Claim
    published_baseline_exceeded: Claim

    TEXT = (
        "official training reproduced",
        "official checkpoint evaluated",
        "matched student baseline trained",
        "before/after improvement achieved (the mandatory result)",
        "published baseline exceeded",
    )

    def ordered(self) -> list[Claim]:
        return [self.official_training_reproduced, self.official_checkpoint_evaluated,
                self.matched_student_baseline_trained, self.before_after_improvement,
                self.published_baseline_exceeded]

    def render(self) -> str:
        return "\n".join(claim.render(i + 1, text)
                         for i, (claim, text) in enumerate(zip(self.ordered(), self.TEXT)))


def assert_claims_honest(claims: Claims) -> None:
    """Refuse a report that overstates what this project can support.

    Two constraints follow from measurements already recorded, not from caution:

    * **Official training was not reproduced.** Nothing in this project retrains
      RefChartQA's model, so claim 1 cannot be "yes".
    * **The published RefChartQA baseline is not comparable.** `DECISIONS.md` 0052 —
      re-scoring their own released file gives 28.33 against a published 32.83 on human,
      and +11.97 and +20.34 on machine and PoT. Deltas in both directions rule out a
      scoring error; the file is a format example. So claim 5 may not be a bare "yes":
      it must carry the reason.
    """
    if claims.official_training_reproduced.verdict == "yes":
        raise AssertionError(
            "claim 1 says official training was reproduced. Nothing in this project "
            "retrains RefChartQA's model; the claim is 'no' (PLAN.md 7.5).")

    fifth = claims.published_baseline_exceeded
    if fifth.verdict == "yes" and "Level C" not in fifth.evidence:
        raise AssertionError(
            "claim 5 asserts the published baseline was exceeded without recording that "
            f"32.83 is {PUBLISHED_STATUS}. `PLAN.md` 7.5 allows this claim only 'if "
            "genuinely comparable' (DECISIONS.md 0052).")


def render_table(rows: list[SystemRow]) -> str:
    """`PLAN.md` 7.4's table. A missing cell renders as a dash, never as a bare number."""
    headers = ["System", *next(iter(rows)).cells().keys()] if rows else ["System"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = [c.render() if c is not None else "—" for c in row.cells().values()]
        lines.append("| " + " | ".join([row.name, *cells]) + " |")
    lines.append(f"| RefChartQA published reference | — | — | — | "
                 f"{PUBLISHED_REFCHARTQA_AP50:.2f} | — |")
    lines.append("")
    lines.append(f"_The published reference is {PUBLISHED_STATUS}._")
    return "\n".join(lines)


def build_report(rows: list[SystemRow], comparisons: list[Comparison],
                 claims: Claims) -> dict[str, Any]:
    """Assemble the Phase 7 report, refusing to emit a dishonest one."""
    assert_claims_honest(claims)
    unmatched = [c.label for c in comparisons if not c.matched]
    return {
        "table_markdown": render_table(rows),
        "comparisons_markdown": "\n".join(c.render() for c in comparisons),
        "claims_markdown": claims.render(),
        "unmatched_comparisons": unmatched,
        "published_reference": {"ap50": PUBLISHED_REFCHARTQA_AP50,
                                "status": PUBLISHED_STATUS},
    }


__all__ = ["PUBLISHED_REFCHARTQA_AP50", "PUBLISHED_STATUS", "Cell", "Claim", "Claims",
           "Comparison", "SystemRow", "assert_claims_honest", "build_report",
           "render_table"]
