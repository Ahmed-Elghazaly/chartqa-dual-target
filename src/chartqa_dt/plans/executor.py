from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

MAX_DEPTH = 4
MAX_FACTS = 12
SERIES_SEPARATOR = " · "
FILTER_FIELDS = ("compare", "threshold")
FILTER_COMPARISONS = ("greater", "greater_or_equal", "less", "less_or_equal", "equal")


@dataclass(frozen=True)
class OperatorSpec:
    input_type: str
    output_type: str
    min_arity: int
    max_arity: int | None
    unit_rule: str
    formatting_rule: str
    reference_only: bool = False
    extra_fields: tuple[str, ...] = ()


OPERATOR_SPECS: dict[str, OperatorSpec] = {
    "lookup": OperatorSpec("fact", "scalar", 1, 1, "preserve", "fact value", True),
    "label_of": OperatorSpec("fact", "string", 1, 1, "ignore", "fact label", True),
    "difference": OperatorSpec("numeric", "number", 2, 2, "same", "finite number"),
    "absolute_difference": OperatorSpec(
        "numeric", "number", 2, 2, "same", "non-negative finite number"
    ),
    "ratio": OperatorSpec("numeric", "number", 2, 2, "same", "finite number"),
    "percentage": OperatorSpec("numeric", "number", 2, 2, "same", "percent"),
    "percent_change": OperatorSpec("numeric", "number", 2, 2, "same", "percent"),
    "sum": OperatorSpec("numeric", "number", 2, None, "same", "finite number"),
    "mean": OperatorSpec("numeric", "number", 2, None, "same", "finite number"),
    "median": OperatorSpec("facts", "number", 2, None, "same", "finite number", True),
    "min": OperatorSpec("facts", "number", 2, None, "same", "finite number", True),
    "max": OperatorSpec("facts", "number", 2, None, "same", "finite number", True),
    "argmin": OperatorSpec("facts", "string", 2, None, "same", "fact label", True),
    "argmax": OperatorSpec("facts", "string", 2, None, "same", "fact label", True),
    "count": OperatorSpec("facts", "number", 2, None, "ignore", "integer", True),
    "trend": OperatorSpec("facts", "string", 2, None, "same", "trend word", True),
    "compare": OperatorSpec("numeric", "string", 2, 2, "same", "greater/less/equal"),
    "greater_than": OperatorSpec("numeric", "yes_no", 2, 2, "same", "Yes/No"),
    "less_than": OperatorSpec("numeric", "yes_no", 2, 2, "same", "Yes/No"),
    "equal_to": OperatorSpec("numeric", "yes_no", 2, 2, "same", "Yes/No"),
    "rank": OperatorSpec(
        "facts", "string", 2, None, "same", "fact label", True, ("position", "order"),
    ),
    "rank_value": OperatorSpec(
        "facts", "scalar", 2, None, "same", "fact value", True, ("position", "order"),
    ),
    "count_where": OperatorSpec(
        "facts", "number", 2, None, "ignore", "integer", True, FILTER_FIELDS,
    ),
    "sum_where": OperatorSpec(
        "facts", "number", 2, None, "same", "finite number", True, FILTER_FIELDS,
    ),
    "mean_where": OperatorSpec(
        "facts", "number", 2, None, "same", "finite number", True, FILTER_FIELDS,
    ),
    "label_where": OperatorSpec(
        "facts", "string", 2, None, "same", "fact label", True, FILTER_FIELDS,
    ),
    "category_of": OperatorSpec("label", "string", 1, 1, "ignore", "category label"),
    "meets_threshold": OperatorSpec(
        "numeric", "yes_no", 1, 1, "ignore", "Yes/No", False, FILTER_FIELDS,
    ),
    "unanswerable": OperatorSpec("none", "null", 0, 0, "ignore", "empty answer"),
}
OPS = frozenset(OPERATOR_SPECS)
FOLD_OPS = frozenset({
    "sum", "mean", "median", "min", "max", "argmin", "argmax", "count", "trend", "rank",
    "rank_value", "count_where", "sum_where", "mean_where", "label_where",
})
FILTER_OPS = frozenset({"count_where", "sum_where", "mean_where", "label_where"})
THRESHOLD_OPS = FILTER_OPS | {"meets_threshold"}
LABEL_OPS = frozenset({"label_of", "argmax", "argmin", "rank", "label_where"})


class ExecutorError(ValueError):
    pass


@dataclass(frozen=True)
class EvidenceItem:
    label: str
    value: float | str | None
    unit: str | None = None
    fact_id: str | None = None


def with_fact_ids(items: list[EvidenceItem]) -> list[EvidenceItem]:
    return [item if item.fact_id else replace(item, fact_id=f"f{index}")
            for index, item in enumerate(items, start=1)]


def is_fact_ref(value: Any) -> bool:
    return (isinstance(value, dict) and set(value) == {"ref"}
            and isinstance(value.get("ref"), str) and bool(value["ref"]))


def plan_depth(node: Any) -> int:
    if not isinstance(node, dict) or is_fact_ref(node):
        return 0
    args = node.get("args")
    if not isinstance(args, list):
        return 1
    return 1 + max((plan_depth(arg) for arg in args), default=0)


def plan_fact_ids(plan: Any) -> list[str]:
    if is_fact_ref(plan):
        return [plan["ref"]]
    if not isinstance(plan, dict):
        return []
    out: list[str] = []
    for arg in plan.get("args") or []:
        out.extend(plan_fact_ids(arg))
    return out


_SEPARATORS = str.maketrans({
    ",": "", "$": "", " ": "", "\xa0": "", " ": "", " ": "",
})


def parse_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if not isinstance(value, str):
        return None
    try:
        number = float(value.strip().translate(_SEPARATORS).rstrip("%"))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def to_number(value: Any) -> float:
    number = parse_numeric(value)
    if number is None:
        raise ExecutorError(f"finite number required, got {value!r}")
    return number


def _finite(value: Any, where: str) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ExecutorError(f"{where} produced a non-finite result")
    return value


def validate_plan_shape(node: Any, *, max_facts: int = MAX_FACTS, _root: bool = True) -> list[str]:
    errors: list[str] = []
    if not isinstance(node, dict) or is_fact_ref(node):
        return ["plan root must be an operation object"] if _root else []
    op = node.get("op")
    if not isinstance(op, str):
        return ["operation name must be a string"]
    spec = OPERATOR_SPECS.get(op)
    if spec is None:
        return [f"unknown operation {op!r}"]
    allowed = {"op", "args", *spec.extra_fields}
    unknown = sorted(set(node) - allowed)
    missing = sorted({"op", "args", *spec.extra_fields} - set(node))
    if unknown:
        errors.append(f"{op}: unknown fields {unknown}")
    if missing:
        errors.append(f"{op}: missing fields {missing}")
    args = node.get("args")
    if not isinstance(args, list):
        return [*errors, f"{op}: args must be a list"]
    max_arity = max_facts if spec.max_arity is None else spec.max_arity
    if not spec.min_arity <= len(args) <= max_arity:
        errors.append(
            f"{op}: requires {spec.min_arity}"
            + ("" if spec.min_arity == max_arity else f"–{max_arity}")
            + f" operands, got {len(args)}"
        )
    for index, arg in enumerate(args):
        if is_fact_ref(arg):
            continue
        if isinstance(arg, dict) and not spec.reference_only:
            nested = validate_plan_shape(arg, max_facts=max_facts, _root=False)
            if not nested and is_fact_ref(arg):
                continue
            errors.extend(f"{op}.args[{index}].{error}" for error in nested)
            nested_op = arg.get("op")
            nested_spec = OPERATOR_SPECS.get(nested_op) if isinstance(nested_op, str) else None
            if (spec.input_type == "numeric" and nested_spec is not None
                    and nested_spec.output_type not in {"number", "scalar"}):
                errors.append(
                    f"{op}.args[{index}]: {arg.get('op')} returns "
                    f"{nested_spec.output_type}, not a numeric value"
                )
            if spec.input_type == "label" and nested_op not in LABEL_OPS:
                errors.append(
                    f"{op}.args[{index}]: {nested_op} does not return one fact's label"
                )
            continue
        errors.append(
            f"{op}.args[{index}]: operand must be an explicit fact reference"
            + ("" if spec.reference_only else " or nested expression")
        )
    if op in {"rank", "rank_value"}:
        position = node.get("position")
        if isinstance(position, bool) or not isinstance(position, int) or position < 1:
            errors.append(f"{op}.position must be a positive integer")
        elif isinstance(args, list) and position > len(args):
            errors.append(f"{op}.position exceeds the number of candidate facts")
        if node.get("order") not in {"ascending", "descending"}:
            errors.append(f"{op}.order must be 'ascending' or 'descending'")
    if op in THRESHOLD_OPS:
        if node.get("compare") not in FILTER_COMPARISONS:
            errors.append(f"{op}.compare must be one of {list(FILTER_COMPARISONS)}")
        threshold = node.get("threshold")
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold))):
            errors.append(f"{op}.threshold must be a finite number")
    if _root and plan_depth(node) > MAX_DEPTH:
        errors.append(f"plan depth {plan_depth(node)} exceeds {MAX_DEPTH}")
    return errors


def execute(node: Any, evidence: list[EvidenceItem], *, max_facts: int = MAX_FACTS,
            _depth_checked: bool = False) -> Any:
    if not _depth_checked:
        errors = validate_plan_shape(node, max_facts=max_facts)
        if errors:
            raise ExecutorError("; ".join(errors))
    facts = with_fact_ids(evidence)
    by_id = {fact.fact_id: fact for fact in facts}
    if len(by_id) != len(facts):
        raise ExecutorError("fact ids must be unique")

    def fact(ref: Any) -> EvidenceItem:
        if not is_fact_ref(ref):
            raise ExecutorError(f"explicit fact reference required, got {ref!r}")
        fact_id = ref["ref"]
        if fact_id not in by_id:
            raise ExecutorError(f"unknown fact reference {fact_id!r}")
        return by_id[fact_id]

    def resolve(arg: Any) -> Any:
        if is_fact_ref(arg):
            return fact(arg).value
        if isinstance(arg, dict):
            return execute(arg, facts, max_facts=max_facts, _depth_checked=True)
        raise ExecutorError("raw constants are forbidden as computation operands")

    def numbers(args: list[Any]) -> list[float]:
        return [to_number(resolve(arg)) for arg in args]

    def same_unit(values: list[Any], where: str) -> str:
        units = {unit_of(value) for value in values}
        if len(units) > 1:
            raise ExecutorError(f"{where} unit mismatch: {sorted(units)}")
        return next(iter(units), "")

    def unit_of(arg: Any) -> str:
        if is_fact_ref(arg):
            return str(fact(arg).unit or "")
        if not isinstance(arg, dict):
            raise ExecutorError("raw constants have no declared unit")
        nested_op = arg["op"]
        nested_args = arg["args"]
        if nested_op == "lookup":
            return unit_of(nested_args[0])
        operand_unit = (same_unit(nested_args, nested_op)
                        if OPERATOR_SPECS[nested_op].unit_rule == "same" else "")
        if nested_op in {
            "absolute_difference", "difference", "sum", "mean", "median", "min", "max",
            "rank_value", "sum_where", "mean_where",
        }:
            return operand_unit
        if nested_op in {"percent_change", "percentage"}:
            return "%"
        return ""

    op = node["op"]
    args = node["args"]
    spec = OPERATOR_SPECS[op]
    if spec.unit_rule == "same":
        same_unit(args, op)

    if op == "unanswerable":
        return None
    if op == "lookup":
        return fact(args[0]).value
    if op == "label_of":
        return fact(args[0]).label
    if op == "count":
        return float(len(args))
    if op in {"sum", "mean", "median", "min", "max"}:
        values = numbers(args)
        if op == "sum":
            result = sum(values)
        elif op == "mean":
            result = statistics.fmean(values)
        elif op == "median":
            result = statistics.median(values)
        elif op == "min":
            result = min(values)
        else:
            result = max(values)
        return _finite(result, op)
    if op in {"absolute_difference", "difference", "ratio", "percentage", "percent_change"}:
        left, right = numbers(args)
        if op == "difference":
            return _finite(left - right, op)
        if op == "absolute_difference":
            return _finite(abs(left - right), op)
        if right == 0:
            raise ExecutorError("division by zero")
        if op == "ratio":
            value = left / right
        elif op == "percentage":
            value = 100 * left / right
        else:
            value = 100 * (left - right) / right
        return _finite(value, op)
    if op in {"argmin", "argmax"}:
        pairs = [(fact(arg), to_number(fact(arg).value)) for arg in args]
        selected = (min if op == "argmin" else max)(pairs, key=lambda pair: pair[1])
        winners = [item for item, value in pairs if value == selected[1]]
        if len(winners) != 1:
            raise ExecutorError(f"{op} is ambiguous because the extreme value is tied")
        return selected[0].label
    if op == "trend":
        values = numbers(args)
        deltas = [right - left for left, right in pairwise(values)]
        if all(delta > 0 for delta in deltas):
            return "increasing"
        if all(delta < 0 for delta in deltas):
            return "decreasing"
        if all(delta == 0 for delta in deltas):
            return "flat"
        return "mixed"
    if op in {"compare", "greater_than", "less_than", "equal_to"}:
        left, right = numbers(args)
        if op == "compare":
            return "greater" if left > right else ("less" if left < right else "equal")
        predicate = {
            "greater_than": left > right,
            "less_than": left < right,
            "equal_to": left == right,
        }[op]
        return "Yes" if predicate else "No"
    if op == "rank":
        pairs = [(fact(arg), to_number(fact(arg).value)) for arg in args]
        reverse = node["order"] == "descending"
        pairs.sort(key=lambda pair: pair[1], reverse=reverse)
        position = node["position"] - 1
        selected_value = pairs[position][1]
        if sum(value == selected_value for _, value in pairs) > 1:
            raise ExecutorError("rank selection is ambiguous because the selected value is tied")
        return pairs[position][0].label
    if op == "rank_value":
        pairs = [(fact(arg), to_number(fact(arg).value)) for arg in args]
        pairs.sort(key=lambda pair: pair[1], reverse=node["order"] == "descending")
        return pairs[node["position"] - 1][0].value
    if op in THRESHOLD_OPS:
        threshold = float(node["threshold"])
        compare = node["compare"]

        def passes(value: float) -> bool:
            if compare == "greater":
                return value > threshold
            if compare == "greater_or_equal":
                return value >= threshold
            if compare == "less":
                return value < threshold
            if compare == "less_or_equal":
                return value <= threshold
            return value == threshold

        if op == "meets_threshold":
            return "Yes" if passes(to_number(resolve(args[0]))) else "No"
        candidates = [(fact(arg), to_number(fact(arg).value)) for arg in args]
        kept = [(item, value) for item, value in candidates if passes(value)]
        if op == "count_where":
            return float(len(kept))
        if op == "label_where":
            if len(kept) != 1:
                raise ExecutorError(f"label_where needs exactly one qualifying fact, got {len(kept)}")
            return kept[0][0].label
        if not kept:
            raise ExecutorError(f"{op} has no fact satisfying the comparison")
        values = [value for _, value in kept]
        return _finite(sum(values) if op == "sum_where" else statistics.fmean(values), op)
    if op == "category_of":
        label = fact(args[0]).label if is_fact_ref(args[0]) else resolve(args[0])
        if not isinstance(label, str):
            raise ExecutorError("category_of needs a fact label")
        return label.split(SERIES_SEPARATOR, 1)[1] if SERIES_SEPARATOR in label else label
    raise ExecutorError(f"unhandled operation {op!r}")


def resolve_label_plan(plan: Any, evidence: list[EvidenceItem]) -> dict[str, Any]:
    facts = with_fact_ids(evidence)
    if isinstance(plan, dict) and not validate_plan_shape(plan):
        missing = sorted(set(plan_fact_ids(plan)) - {str(fact.fact_id) for fact in facts})
        if missing:
            raise ExecutorError(f"typed plan references unknown facts {missing}")
        return plan
    by_label: dict[str, list[str]] = {}
    for item in facts:
        by_label.setdefault(item.label, []).append(str(item.fact_id))

    def convert(node: Any) -> Any:
        if isinstance(node, str):
            matches = by_label.get(node, [])
            if len(matches) != 1:
                raise ExecutorError(
                    f"label operand {node!r} resolves to {len(matches)} facts; expected exactly one"
                )
            return {"ref": matches[0]}
        if isinstance(node, (int, float, bool)) or node is None:
            raise ExecutorError("raw constants cannot be converted into computation operands")
        if not isinstance(node, dict) or not isinstance(node.get("op"), str):
            raise ExecutorError(f"malformed plan node {node!r}")
        op = node["op"]
        if op == "boolean":
            raise ExecutorError("boolean is ambiguous; use a typed comparison predicate")
        if op == "within":
            raise ExecutorError("within is unsupported; use explicit series-qualified facts")
        if op not in OPERATOR_SPECS:
            raise ExecutorError(f"operation {op!r} is not executable")
        raw_args = node.get("args")
        if not isinstance(raw_args, list):
            raise ExecutorError(f"{op}.args must be a list")
        if op in FOLD_OPS and not raw_args:
            raw_args = [item.label for item in facts]
        converted = {"op": op, "args": [convert(arg) for arg in raw_args]}
        if op == "rank":
            converted["position"] = node.get("position")
            converted["order"] = node.get("order")
        errors = validate_plan_shape(converted)
        if errors:
            raise ExecutorError("; ".join(errors))
        return converted

    result = convert(plan)
    if not isinstance(result, dict):
        raise ExecutorError("label plan did not produce an operation")
    return result


def execute_label_plan(plan: Any, evidence: list[EvidenceItem]) -> Any:
    facts = with_fact_ids(evidence)
    return execute(resolve_label_plan(plan, facts), facts)
