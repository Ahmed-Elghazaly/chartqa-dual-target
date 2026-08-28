"""Table builders — one per pre-named stub in `report/tables/` (`PLAN.md` 10.1).

Each builder takes the recorded results and returns a LaTeX fragment. A builder whose data
does not exist yet returns a fragment full of `\\TODO{}` rather than raising: the skeleton
is meant to compile from Phase 1 onwards, with the unmeasured cells red on the page.

The registry at the bottom is what `cdt-report --what` dispatches on, so adding a table
means adding a function and one entry.
"""

from __future__ import annotations

from typing import Any

from chartqa_dt.reporting.latex import (
    TODO,
    ci,
    escape,
    num,
    row,
    table,
    tabular,
)

#: Operations in the order the report discusses them, not the order they were mined.
OP_ORDER = ["lookup", "difference", "sum", "mean", "ratio", "count", "percent_change"]


def _get(facts: dict[str, Any], *path: str) -> Any:
    """Walk a path into the facts, returning None rather than raising on a gap."""
    node: Any = facts
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def build_plan_yield(results: dict[str, Any]) -> str:
    """How much of ChartQA's training set carries a recoverable plan (`PLAN.md` 3.4)."""
    mining = results.get("mining_yield") or {}
    by_kind = mining.get("by_kind") or {}
    ops = mining.get("operations") or {}

    rows = []
    for kind in ("human", "machine"):
        d = by_kind.get(kind) or {}
        questions, mined = d.get("questions"), d.get("mined")
        pct = 100 * mined / questions if questions and mined is not None else None
        rows.append(row([escape(kind),
                         f"{questions:,}" if questions else TODO,
                         f"{mined:,}" if mined is not None else TODO,
                         num(pct, 1, percent=True),
                         f"{d.get('rejected:ambiguous', 0):,}" if d else TODO,
                         f"{d.get('rejected:non_numeric', 0):,}" if d else TODO]))
    total_q, total_m = mining.get("questions"), mining.get("mined")
    rows.append(row([r"\textbf{all}",
                     f"\\textbf{{{total_q:,}}}" if total_q else TODO,
                     f"\\textbf{{{total_m:,}}}" if total_m is not None else TODO,
                     f"\\textbf{{{num(mining.get('yield_pct'), 2, percent=True)}}}",
                     "", ""]))

    present = [op for op in OP_ORDER if op in ops] or OP_ORDER
    op_line = ", ".join(f"{escape(op)} {ops[op]:,}" for op in present if op in ops) or TODO
    note = (f"Mined operations: {op_line}. "
            "A question is mined only when exactly one operation over the gold table "
            "reproduces its answer; ambiguity is a rejection, not a guess "
            "(\\textsc{decisions} 0045).")
    return table("tab_plan_yield",
                 "Plan supervision recovered from ChartQA's training split by mining the "
                 "gold data tables. Yield is low by construction: the uniqueness rule "
                 "discards every question whose answer more than one operation reproduces.",
                 "tab:plan_yield",
                 tabular("lrrrrr",
                         ["Question kind", "Questions", "Mined", "Yield",
                          "Rej.\\ ambiguous", "Rej.\\ non-numeric"],
                         rows, midrules_before=[len(rows) - 1]),
                 note=note)


def build_variant_selection(results: dict[str, Any]) -> str:
    """Which checkpoint variant was selected, and on what evidence (`PLAN.md` 5.2)."""
    facts = results.get("measured_facts") or {}
    sel = _get(facts, "phase5", "variant_selection_5_2") or {}
    probe = _get(facts, "phase5", "prompt_iteration") or {}
    thinking = _get(facts, "published_targets", "chartqa_qwen3vl2b_thinking")

    n = sel.get("n")
    rows = [
        row(["Relaxed accuracy", num(sel.get("relaxed_accuracy_pct"), 1, percent=True),
             "answers scored by the official evaluator"]),
        row(["Valid JSON", num(sel.get("valid_json_pct"), 1, percent=True),
             "parses at all"]),
        row(["Schema-valid (raw)",
             num(sel.get("schema_valid_pct_before_repair"), 1, percent=True),
             "satisfies the output schema as emitted"]),
        row(["Schema-valid (repaired)",
             num(sel.get("schema_valid_pct_after_repair"), 1, percent=True),
             "after drop-and-unwrap repair (\\textsc{decisions} 0068)"]),
        row(["Executor agreement",
             num(sel.get("roundtrip_agreement_pct"), 1, percent=True),
             "plan recomputes the model's own answer"]),
        row(["Executable plans",
             num(sel.get("roundtrip_executable_pct"), 1, percent=True),
             "plan runs without raising"]),
        row(["Median latency",
             f"{num(sel.get('median_latency_s'), 2)}\\,s", "per question, 4-bit, T4"]),
        row(["Median new tokens", num(sel.get("median_new_tokens"), 1),
             "generated per record"]),
    ]
    note = (f"Variant selected: \\textbf{{{escape(sel.get('variant_chosen', TODO))}}}. "
            f"The Thinking variant was measured on a {probe.get('v3_schema_limits_stated', {}).get('n', TODO)}-question "
            "probe and failed two of three pre-registered gates — latency and valid-JSON "
            "rate — so it was not carried to the full slice"
            + (f"; its published ChartQA figure is {thinking}." if thinking else "."))
    return table("tab_variant_selection",
                 "Zero-shot behaviour of the selected checkpoint on a frozen "
                 f"{n if n else TODO}-question ChartQA validation slice. These numbers set "
                 "the before side of the mandatory before/after comparison.",
                 "tab:variant_selection",
                 tabular("lrl", ["Measure", "Value", "Meaning"], rows),
                 note=note)


def build_compute(results: dict[str, Any]) -> str:
    """The compute budget actually spent (`PLAN.md` 10.1, and the USD 20 constraint)."""
    facts = results.get("measured_facts") or {}
    spent = results.get("compute") or {}
    gates = _get(facts, "gates") or {}
    p2 = _get(facts, "phase2") or {}

    rows = [
        row(["Peak reserved memory", f"{num(p2.get('peak_reserved_gb'))}\\,GiB",
             f"gate {num(gates.get('memory_gb'))}\\,GiB"]),
        row(["Seconds per optimizer step", num(p2.get("seconds_per_step"), 3),
             "batch 1, grad-accum 8, 512\\,px"]),
        row(["Projected full run",
             f"{num(p2.get('projected_full_run_hours'))}\\,h",
             f"gate {num(gates.get('full_run_hours'))}\\,h"]),
        row(["Planned optimizer steps",
             f"{gates['planned_optimizer_steps']:,}" if gates.get("planned_optimizer_steps")
             else TODO, "pre-registered budget"]),
        row(["Weekly GPU quota",
             f"{num(gates.get('weekly_gpu_quota_hours'), 0)}\\,h", "free tier, per account"]),
        row(["GPU hours spent", num(spent.get("gpu_hours_used"), 1),
             "across all phases"]),
        row(["Cash cost", f"USD\\,{num(spent.get('usd'), 2)}",
             "hard ceiling USD 20"]),
    ]
    return table("tab_compute",
                 "Compute. The project runs entirely on free-tier GPUs; the memory and "
                 "wall-clock gates were fixed before training and are reported against "
                 "what was measured.",
                 "tab:compute",
                 tabular("lrl", ["Quantity", "Measured", "Constraint"], rows))


def build_headline(results: dict[str, Any]) -> str:
    """`PLAN.md` 7.4. Grounding is per subset: the official evaluator has no aggregate."""
    facts = results.get("measured_facts") or {}
    published = _get(facts, "published_targets") or {}
    systems = results.get("headline") or {}

    def cell(system: str, key: str) -> str:
        d = (systems.get(system) or {}).get(key)
        if not d:
            return TODO
        return ci(d.get("value"), d.get("lo"), d.get("hi"))

    names = [("zeroshot", "Untouched model (zero-shot)"),
             ("control", "Direct-answer LoRA (control)"),
             ("grounded", r"\textbf{Grounded plan + executor}")]
    keys = ["chartqa_human", "chartqa_machine", "ap50_human", "ap50_machine", "ap50_pot"]
    rows = [row([label, *(cell(key, k) for k in keys)]) for key, label in names]
    rows.append(row(["Published reference (3B, 1 epoch)", "---", "---",
                     num(published.get("refchartqa_ap50_human_qwen25vl3b")),
                     num(published.get("refchartqa_ap50_machine_qwen25vl3b")),
                     num(published.get("refchartqa_ap50_pot_qwen25vl3b"))]))

    body = "\n".join([
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"& \multicolumn{2}{c}{ChartQA relaxed accuracy} & \multicolumn{3}{c}{RefChartQA AP@0.5} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        row(["System", "Human", "Machine", "Human", "Machine", "PoT"]), r"\midrule",
        *rows[:-1], r"\midrule", rows[-1], r"\bottomrule", r"\end{tabular}"])
    return table("tab_headline",
                 "Headline results. Every cell carries a 95\\% bootstrap confidence "
                 "interval. Grounding is reported per question subset because the official "
                 "evaluator produces no aggregate; the published reference applies to the "
                 "human subset alone, where $n=500$ and differences below roughly four "
                 "points are not distinguishable from sampling noise.",
                 "tab:headline", body, wide=True,
                 note="The published reference is cited, not reproduced: RefChartQA "
                      "releases no per-model predictions and no checkpoints, and "
                      "re-scoring its own released file does not return its printed "
                      "figures (\\textsc{decisions} 0052). It is not a matched comparison.")


def build_oracle(results: dict[str, Any]) -> str:
    """`PLAN.md` 9.1 — where the error actually comes from.

    Reads the shape `eval/oracle.decompose` writes, so the table cannot drift from the
    computation. All four cells share their record set by construction there; `n_eligible`
    is printed once, in the caption, rather than per row, because it is the same number.
    """
    o = results.get("oracle") or {}
    cells = o.get("cells") or {}
    labels = {"pred_pred": ("predicted", "predicted", "the real system"),
              "gold_pred": (r"\textbf{gold}", "predicted", r"error from \emph{seeing}"),
              "pred_gold": ("predicted", r"\textbf{gold}", r"error from \emph{reasoning}"),
              "gold_gold": (r"\textbf{gold}", r"\textbf{gold}",
                            "the executor's own ceiling")}
    rows = []
    for key, (ev, plan, tells) in labels.items():
        cell = cells.get(key) or {}
        acc = cell.get("accuracy")
        rows.append(row([ev, plan,
                         num(100 * acc, 2, percent=True) if acc is not None else TODO,
                         num(cell.get("executor_refused"), 0)
                         if cell.get("executor_refused") is not None else TODO,
                         tells]))
    n = o.get("n_eligible")
    excluded = o.get("n_excluded_no_gold_plan")
    note = ""
    if excluded:
        note = (f"{excluded:,} records are excluded because they carry no gold plan; "
                "ChartQA supplies one only where a single operation over its gold table "
                "uniquely reproduces the answer (\\textsc{decisions} 0045). A predicted "
                "plan that does not fit the gold evidence is counted as a failure in the "
                "gold-evidence rows, not skipped \\textemdash{} skipping it would flatter "
                "exactly the records the model got most wrong.")
    return table("tab_oracle",
                 "Oracle decomposition"
                 + (f" over the {n:,} records eligible for all four cells" if n else "")
                 + ". Substituting gold evidence isolates visual error; substituting the "
                 "gold plan isolates reasoning error; substituting both leaves only what "
                 "the executor itself cannot do. Every cell is computed on the same "
                 "records, so the differences are like-for-like.",
                 "tab:oracle",
                 tabular("llrrl", ["Evidence", "Plan", "Relaxed accuracy",
                                   "Executor refused", "Tells you"], rows),
                 note=note)


def build_stratified(results: dict[str, Any]) -> str:
    """`PLAN.md` 9.2 — AP by target-box area, the axis the model is expected to struggle on.

    Reads the shape `eval/stratified` writes so the table cannot drift from the
    computation. `by_area` comes from `stratify`, whose buckets follow COCO's area-range
    semantics; `by_chart_type` and `by_question_kind` come from `stratify_by`, where each
    group is an independent evaluation.
    """
    strata = results.get("stratified") or {}
    facts = results.get("measured_facts") or {}
    sub = _get(facts, "phase4", "stratification") or {}

    def block(groups: dict[str, Any], heading: str) -> list[str]:
        if not groups:
            return [row([f"\\emph{{{heading}}}", TODO, TODO, TODO])]
        out = [row([f"\\emph{{{heading}}}", "", "", ""])]
        for name, g in groups.items():
            out.append(row([f"\\quad {escape(name)}",
                            f"{g.get('n', 0):,}",
                            num(100 * g["ap50"], 2, percent=True)
                            if g.get("ap50") is not None else TODO,
                            num(100 * g["p_at_f1"], 2, percent=True)
                            if g.get("p_at_f1") is not None else TODO]))
        return out

    rows = (block(strata.get("by_area") or {}, "by target-box area")
            + block(strata.get("by_chart_type") or {}, "by chart type")
            + block(strata.get("by_question_kind") or {}, "by question kind"))
    note = ("Sub-token boxes are those narrower than one visual token on at least one "
            f"axis: {num(sub.get('subtoken_fraction_by_axis_pct'), 1, percent=True)} of "
            "RefChartQA training boxes at 512\\,px. A box the encoder cannot resolve is a "
            "box the decoder cannot point at precisely, so that stratum bounds what any "
            "amount of training achieves at this resolution. AP is recomputed within each "
            "group rather than averaged across groups, because AP is not a mean of "
            "per-item scores.")
    return table("tab_stratified",
                 "Grounding stratified three ways.", "tab:stratified",
                 tabular("lrrr", ["Stratum", "$n$", "AP@0.5", "P@F1"], rows), note=note)


def build_plan_diagnostics(results: dict[str, Any]) -> str:
    """`PLAN.md` 9.3 and 9.4 in one float — the per-source view *is* the transfer result."""
    d = results.get("diagnostics") or {}
    by_source = d.get("by_source") or {}
    transfer = d.get("transfer") or {}

    measures = [("valid JSON", "valid_json"), ("schema-valid", "schema_valid"),
                ("has a plan", "plan_coverage"), ("executor succeeds", "executor_success"),
                ("executor agrees", "executor_agreement"),
                ("exact operation tree", "tree_exact"),
                ("exact operands", "operands_exact")]
    names = list(by_source) or ["synthetic", "chartqa", "refchartqa"]
    rows = [row([label, *[num(100 * (by_source.get(n, {}).get(key) or 0), 2, percent=True)
                          if by_source.get(n, {}).get(key) is not None else TODO
                          for n in names]])
            for label, key in measures]
    header = ["Measure", *[escape(n) for n in names]]

    note = ""
    if transfer.get("measurable"):
        drop = transfer.get("drop_points", {})
        note = ("Transfer (9.4): moving from synthetic charts to real ones costs "
                + ", ".join(f"{v:+.2f} pts {escape(k.replace('_', ' '))}"
                            for k, v in drop.items())
                + f", over {transfer.get('n_synthetic', 0):,} synthetic and "
                  f"{transfer.get('n_real', 0):,} real records. The real supply is the "
                  "binding constraint — 4,483 records in total — so these gaps carry wide "
                  "intervals and are read as direction rather than magnitude.")
    return table("tab_plan_diagnostics",
                 "What the emitted plans are actually like, by chart source. Tree and "
                 "operand exactness are measured only where the true plan is known. "
                 "Operation-tree comparison allows commutative arguments to be reordered "
                 "and nothing else: \\texttt{difference} is not symmetric, and scoring it "
                 "as though it were would report agreement on the error the executor "
                 "exists to catch.",
                 "tab:plan_diagnostics",
                 tabular("l" + "r" * len(names), header, rows), note=note)


def _todo_table(name: str, caption: str, columns: list[str]) -> str:
    return table(f"tab_{name}", caption, f"tab:{name}",
                 tabular("l" + "c" * (len(columns) - 1), columns,
                         [row([TODO] * len(columns))]))


def build_structured_cost(results: dict[str, Any]) -> str:
    """`PLAN.md` 5.3 — what the structured output costs against a plain-prompt baseline."""
    d = results.get("structured_cost") or {}
    rows = []
    for key, label in [("plain", "Plain prompt (published)"),
                       ("structured", "Structured record")]:
        r = d.get(key) or {}
        rows.append(row([label, ci(r.get("accuracy"), r.get("lo"), r.get("hi")),
                         num(r.get("median_new_tokens"), 0),
                         f"{num(r.get('median_latency_s'), 2)}\\,s"
                         if r.get("median_latency_s") is not None else TODO,
                         num(r.get("valid_pct"), 1, percent=True)]))
    return table("tab_structured_cost",
                 "The cost of asking for a structured record instead of a bare answer, "
                 "measured zero-shot on the same frozen validation slice with the same "
                 "checkpoint and decoding. This is the price the fine-tune has to earn back.",
                 "tab:structured_cost",
                 tabular("lcccc",
                         ["Prompt", "Relaxed accuracy", "Median tokens", "Median latency",
                          "Valid output"], rows))


def build_crop(results: dict[str, Any]) -> str:
    return _todo_table("crop", "Predicted-box crop re-read (\\textsc{plan} 8.1).",
                       ["Configuration", "Relaxed accuracy", "$\\Delta$", "Extra tokens"])


def build_resolution(results: dict[str, Any]) -> str:
    return _todo_table("resolution", "Input resolution ablation (\\textsc{plan} 8.2).",
                       ["Long side", "AP@0.5", "Relaxed accuracy", "s/step", "Peak GiB"])


BUILDERS = {
    "headline": build_headline,
    "oracle": build_oracle,
    "stratified": build_stratified,
    "plan_diagnostics": build_plan_diagnostics,
    "structured_cost": build_structured_cost,
    "crop": build_crop,
    "resolution": build_resolution,
    "variant_selection": build_variant_selection,
    "plan_yield": build_plan_yield,
    "compute": build_compute,
}

__all__ = ["BUILDERS", "OP_ORDER"]
