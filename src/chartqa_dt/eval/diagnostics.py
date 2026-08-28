"""Plan diagnostics and synthetic-to-real transfer — `PLAN.md` 9.3 and 9.4.

9.3 asks what the emitted plans are actually like: do they parse, do they execute, do they
name the right operands and units, and — on synthetic charts, where the true tree is known
by construction — is the tree itself right?

9.4 asks the question the whole training set rests on. `DECISIONS.md` 0072 measured the
supply: **4,483 real records against 23,966 synthetic ones**. If plan quality collapses
when the chart is real, the synthetic data is teaching a format rather than a skill, and
the project should say so. So every 9.3 metric is reported per source, and the difference
between them is the transfer result.

**Tree comparison is order-sensitive for some operations and not others.** `sum(A, B)` and
`sum(B, A)` are the same plan; `difference(A, B)` and `difference(B, A)` are not, and
scoring them as equal would report agreement on the exact error the executor exists to
catch. `COMMUTATIVE` names the operations whose arguments are compared as a multiset.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from chartqa_dt.plans.executor import EvidenceItem, execute
from chartqa_dt.plans.roundtrip import answers_agree
from chartqa_dt.prompting.parsing import parse_record, schema_ok

#: Operations whose arguments may be reordered without changing the result. `difference`,
#: `ratio`, `percent_change`, `compare` and `trend` are deliberately absent: for those,
#: argument order *is* the plan.
COMMUTATIVE = frozenset({"sum", "mean", "median", "min", "max", "count",
                         "argmin", "argmax"})


def normalise_plan(plan: Any) -> Any:
    """A comparable form: commutative arguments sorted, everything else left alone."""
    if not isinstance(plan, dict):
        return plan
    op = plan.get("op")
    args = [normalise_plan(a) for a in (plan.get("args") or [])]
    if op in COMMUTATIVE:
        args = sorted(args, key=repr)
    return {"op": op, "args": args}


def trees_match(predicted: Any, gold: Any) -> bool:
    """Exact operation-tree match, up to commutative reordering."""
    if not isinstance(predicted, dict) or not isinstance(gold, dict):
        return False
    return normalise_plan(predicted) == normalise_plan(gold)


@dataclass
class PlanDiagnostics:
    """`PLAN.md` 9.3, for one group of records."""

    source: str = ""
    n: int = 0
    parsed: int = 0
    schema_valid: int = 0
    has_plan: int = 0
    executes: int = 0
    agrees: int = 0
    #: Only counted where a gold plan exists — synthetic charts always, real ones sometimes.
    n_with_gold_plan: int = 0
    tree_exact: int = 0
    operands_exact: int = 0
    units_exact: int = 0
    ops: dict[str, int] = field(default_factory=dict)

    def _rate(self, numerator: int, denominator: int | None = None) -> float:
        d = self.n if denominator is None else denominator
        return numerator / d if d else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source, "n": self.n,
            "valid_json": self._rate(self.parsed),
            "schema_valid": self._rate(self.schema_valid),
            "plan_coverage": self._rate(self.has_plan),
            "executor_success": self._rate(self.executes),
            "executor_agreement": self._rate(self.agrees),
            "n_with_gold_plan": self.n_with_gold_plan,
            "tree_exact": self._rate(self.tree_exact, self.n_with_gold_plan),
            "operands_exact": self._rate(self.operands_exact, self.n_with_gold_plan),
            "units_exact": self._rate(self.units_exact, self.n_with_gold_plan),
            "ops": dict(sorted(self.ops.items(), key=lambda kv: -kv[1])),
        }

    def describe(self) -> str:
        d = self.to_dict()
        rows = [("valid JSON", "valid_json"), ("schema-valid", "schema_valid"),
                ("has a plan", "plan_coverage"), ("executes", "executor_success"),
                ("agrees with its answer", "executor_agreement")]
        lines = [f"  {self.source or 'all':<12} n={self.n:,}"]
        lines += [f"    {label:<24}{100 * d[key]:6.2f}%" for label, key in rows]
        if self.n_with_gold_plan:
            lines.append(f"    against {self.n_with_gold_plan:,} known gold plans:")
            for label, key in [("exact tree", "tree_exact"),
                               ("exact operands", "operands_exact"),
                               ("exact units", "units_exact")]:
                lines.append(f"      {label:<22}{100 * d[key]:6.2f}%")
        return "\n".join(lines)


def _labels(evidence: Sequence[dict[str, Any]]) -> list[str]:
    return sorted(str(e.get("label")) for e in evidence if isinstance(e, dict))


def _units(evidence: Sequence[dict[str, Any]]) -> list[Any]:
    return sorted((str(e.get("label")), e.get("unit")) for e in evidence
                  if isinstance(e, dict))


def diagnose_one(raw: str, *, gold_plan: Any = None,
                 gold_evidence: Sequence[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Every 9.3 measure for one generation.

    An unparseable output fails every measure rather than abstaining — `PLAN.md` rule 4.
    """
    out = {"parsed": False, "schema_valid": False, "has_plan": False, "executes": False,
           "agrees": False, "tree_exact": False, "operands_exact": False,
           "units_exact": False, "op": ""}
    result = parse_record(raw)
    if not result.ok or result.record is None:
        return out
    record = result.record
    out["parsed"] = True
    out["schema_valid"] = schema_ok(record)[0]

    plan = record.get("plan")
    if isinstance(plan, dict) and plan.get("op"):
        out["has_plan"] = True
        out["op"] = str(plan.get("op"))
    evidence = [e for e in (record.get("evidence") or []) if isinstance(e, dict)]

    if out["has_plan"]:
        try:
            got = execute(plan, [EvidenceItem(str(e.get("label")), e.get("value"),
                                              e.get("unit")) for e in evidence])
        except Exception:                   # noqa: BLE001 — every refusal is one outcome
            got = None
            out["executes"] = False
        else:
            out["executes"] = True
            out["agrees"] = answers_agree(str(record.get("model_answer", "")), got)

    if gold_plan is not None:
        out["tree_exact"] = trees_match(plan, gold_plan)
    if gold_evidence is not None:
        out["operands_exact"] = _labels(evidence) == _labels(gold_evidence)
        out["units_exact"] = _units(evidence) == _units(gold_evidence)
    return out


def diagnose(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Diagnostics overall and per source. Each item supplies `raw` and `source`.

    `gold_plan` and `gold_evidence` are optional per item; the tree, operand and unit rates
    are reported over the records that have them, with that count printed beside the rate.
    """
    groups: dict[str, PlanDiagnostics] = {}
    overall = PlanDiagnostics(source="all")
    for item in items:
        source = str(item.get("source") or "unknown")
        gold_plan = item.get("gold_plan")
        gold_evidence = item.get("gold_evidence")
        one = diagnose_one(str(item.get("raw", "")), gold_plan=gold_plan,
                           gold_evidence=gold_evidence)
        for group in (overall, groups.setdefault(source, PlanDiagnostics(source=source))):
            group.n += 1
            for key in ("parsed", "schema_valid", "executes", "agrees"):
                setattr(group, key, getattr(group, key) + one[key])
            group.has_plan += one["has_plan"]
            if one["op"]:
                group.ops[one["op"]] = group.ops.get(one["op"], 0) + 1
            if gold_plan is not None or gold_evidence is not None:
                group.n_with_gold_plan += 1
                group.tree_exact += one["tree_exact"]
                group.operands_exact += one["operands_exact"]
                group.units_exact += one["units_exact"]

    return {"overall": overall.to_dict(),
            "by_source": {k: v.to_dict() for k, v in sorted(groups.items())},
            "transfer": _transfer(groups)}


def _transfer(groups: dict[str, PlanDiagnostics]) -> dict[str, Any]:
    """`PLAN.md` 9.4 — how much of what synthetic charts teach survives on real ones.

    Reported as a difference in percentage points, with both sides' `n`, because the real
    groups are small: 4,483 usable real records exist in total (`DECISIONS.md` 0072), so a
    per-source rate can carry a wide interval and a gap must be read against it.
    """
    synth = groups.get("synthetic")
    real = [g for name, g in groups.items() if name != "synthetic"]
    if synth is None or not real:
        return {"measurable": False,
                "reason": "needs a synthetic group and at least one real group"}
    merged = PlanDiagnostics(source="real")
    for g in real:
        for key in ("n", "parsed", "schema_valid", "has_plan", "executes", "agrees",
                    "n_with_gold_plan", "tree_exact", "operands_exact", "units_exact"):
            setattr(merged, key, getattr(merged, key) + getattr(g, key))
    s, r = synth.to_dict(), merged.to_dict()
    return {
        "measurable": True,
        "n_synthetic": synth.n, "n_real": merged.n,
        "drop_points": {key: 100 * (s[key] - r[key])
                        for key in ("schema_valid", "plan_coverage", "executor_success",
                                    "executor_agreement")},
        "synthetic": s, "real": r,
    }


__all__ = ["COMMUTATIVE", "PlanDiagnostics", "diagnose", "diagnose_one", "normalise_plan",
           "trees_match"]
