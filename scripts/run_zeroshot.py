"""Zero-shot baselines on GPU — `PLAN.md` 5.2, 5.3, 5.4.

Stages, each runnable alone so a session that runs out of quota loses only its own stage:

* `probe`      — 20 items, both variants. Measures latency so the later stages are sized
                 from a measurement rather than a guess (`PLAN.md` Appendix F: cheap gates
                 before expensive runs).
* `variant`    — `PLAN.md` 5.2 on the frozen 200-question slice: Instruct against Thinking,
                 with the three-part gate decided in advance.
* `chartqa`    — 5.3 zero-shot ChartQA on validation, structured and plain prompts.
* `refchartqa` — 5.4 zero-shot grounding on validation, stratified by box area.

**Nothing is redistributed.** Datasets are fetched from their pinned upstream revisions
here, exactly as they are locally, so rule 7 holds and the run is reproducible from public
sources.

**Validation only.** Every slice is a validation slice; `chartqa_dt.splits` refuses test
splits and nothing here asks for one.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

VARIANTS = {"instruct": "Qwen/Qwen3-VL-2B-Instruct",
            "thinking": "Qwen/Qwen3-VL-2B-Thinking"}

#: `PLAN.md` 5.2 — Thinking is chosen only if ALL THREE hold. Written here, before the
#: numbers exist, so the decision cannot be reshaped around them.
VARIANT_GATE = {"min_accuracy_gain_points": 2.0,
                "min_valid_json_fraction": 0.90,
                "max_latency_ratio": 2.0}


def out_dir() -> Path:
    d = Path(os.environ.get("CDT_OUT", "outputs/phase5"))
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------- data


def chartqa_reader():
    from chartqa_dt.data.chartqa import ArchiveReader
    from chartqa_dt.data.download import fetch_archive
    from chartqa_dt.data.sources import CHARTQA_ARCHIVE
    from chartqa_dt.env import get_env

    result = fetch_archive(CHARTQA_ARCHIVE, data_root=Path(get_env().data_root))
    print(f"ChartQA archive: {result.size_bytes:,} B  sha256 {result.sha256[:16]}…",
          flush=True)
    return ArchiveReader(result.path)


def load_slice(name: str) -> list[dict[str, Any]]:
    """The frozen slice, from the local working copy or rebuilt from the archive."""
    from chartqa_dt.env import get_env

    local = Path(get_env().data_root) / "slices" / f"{name}.jsonl"
    if local.exists():
        return [json.loads(x) for x in local.read_text().splitlines() if x.strip()]

    import scripts.build_val_slices as builder

    reader = builder.archive()
    rows = builder.chartqa_val_rows(reader)
    size = 200 if name.endswith("_200") else len(rows)
    built = builder.build_chartqa_slice(reader, rows, size=size, seed=20260827)
    builder.write_slice(name, built, {"purpose": "rebuilt on the worker",
                                      "seed": 20260827, "split": "val",
                                      "source": "chartqa"})
    return built


def attach_images(reader, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import io

    from PIL import Image

    out = []
    for row in rows:
        raw = reader.read(row["image_member"])
        out.append({**row, "image": Image.open(io.BytesIO(raw)).convert("RGB")})
    return out


def refchartqa_val(limit: int, seed: int = 0) -> list[dict[str, Any]]:
    """Stratified validation sample, streamed from the pinned revision."""
    from datasets import load_dataset

    from chartqa_dt.data.refchartqa import boxes_to_norm1000
    from chartqa_dt.data.sources import REFCHARTQA_PARQUET as spec

    per_kind = max(1, limit // 3)
    taken = {"human": 0, "machine": 0, "pot": 0}
    out: list[dict[str, Any]] = []
    stream = load_dataset(spec.repo_id, split="validation", streaming=True,
                          revision=spec.revision).shuffle(seed=seed, buffer_size=3000)
    for row in stream:
        kind = str(row.get("type", "")).lower()
        if kind not in taken or taken[kind] >= per_kind:
            if sum(taken.values()) >= limit:
                break
            continue
        image = row["image"].convert("RGB")
        w, h = image.size
        out.append({"record_id": row["id"], "question": row["query"],
                    "answer": str(row["label"]), "question_kind": kind,
                    "image": image, "image_size": (w, h),
                    # The official evaluator quantises the raw {x,y,w,h} boxes itself, so
                    # they are kept unconverted alongside our normalised copy.
                    "raw_boxes": list(row.get("grounding_bboxes") or []),
                    "gt_boxes": boxes_to_norm1000(row.get("grounding_bboxes"), w, h)})
        taken[kind] += 1
        if sum(taken.values()) >= limit:
            break
    print(f"RefChartQA validation sample: {taken}", flush=True)
    return out


# ---------------------------------------------------------------------- model


def load_model(variant: str, *, load_in_4bit: bool = True):
    from chartqa_dt.config import ModelConfig
    from chartqa_dt.modeling.backends.base import get_backend

    cfg = ModelConfig(hf_id=VARIANTS[variant], load_in_4bit=load_in_4bit,
                      image_max_pixels=512 * 512)
    loaded = get_backend("hf_peft").load(cfg)
    print(loaded.describe(), flush=True)
    return loaded


# --------------------------------------------------------------------- stages


def stage_probe(args) -> dict[str, Any]:
    """20 items per variant, to size everything that follows."""
    from chartqa_dt.eval.generate import generate_over, write_generations

    reader = chartqa_reader()
    rows = attach_images(reader, load_slice("chartqa_variant_200")[:args.probe_n])
    report: dict[str, Any] = {}
    for variant in args.variants.split(","):
        variant = variant.strip()
        loaded = load_model(variant)
        for mode in ("structured", "plain"):
            gens, rep = generate_over(loaded, rows, mode=mode, progress_every=5,
                                      max_new_tokens=args.max_new_tokens or None)
            # Keep the generations. A probe that reports a failure rate without keeping
            # the failures gives a number nobody can act on — which is what the first
            # probe did.
            write_generations(gens, out_dir() / f"probe_{variant}_{mode}.jsonl")
            report[f"{variant}/{mode}"] = {
                "n": rep.n, "median_latency_s": rep.median_latency,
                "median_new_tokens": rep.median_new_tokens,
                "hit_token_cap_fraction": rep.capped_fraction,
                "valid_json_fraction": rep.parse.valid_fraction if mode == "structured"
                else 1.0,
                "schema_valid_fraction": rep.parse.schema_valid_fraction
                if mode == "structured" else 1.0,
                "failure_reasons": dict(rep.parse.reasons),
                "schema_failure_reasons": dict(rep.parse.schema_reasons),
            }
            print(f"  {variant}/{mode}: median {rep.median_latency:.2f}s/item, "
                  f"{rep.median_new_tokens:.0f} tokens, capped "
                  f"{100 * rep.capped_fraction:.0f}%, "
                  f"json {rep.parse.valid}/{rep.parse.total}, "
                  f"schema {rep.parse.schema_valid}/{rep.parse.total}", flush=True)
            if mode == "structured":
                # The project's central claim, measured from the first baseline onward.
                from chartqa_dt.plans.roundtrip import check_many
                from chartqa_dt.prompting.parsing import parse_record, schema_ok

                recs = []
                for g in gens:
                    res = parse_record(g.raw)
                    if res.ok and schema_ok(res.record)[0]:
                        recs.append(res.record)
                _, rt = check_many(recs)
                report[f"{variant}/{mode}"]["roundtrip_agreement"] = rt.agreement
                report[f"{variant}/{mode}"]["roundtrip_executable"] = rt.executable
                report[f"{variant}/{mode}"]["roundtrip_counts"] = dict(rt.counts)
                print(rt.describe(), flush=True)
                for g in gens:
                    if not g.parsed_ok:
                        print(f"      FAIL [{g.reason[:44]}] tokens={g.new_tokens} "
                              f"capped={g.hit_token_cap} tail={g.raw[-90:]!r}", flush=True)
        del loaded
        _free()
    (out_dir() / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    _print_budget(report)
    return report


def _free() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _print_budget(probe: dict[str, Any]) -> None:
    print("\nprojected cost at these latencies:")
    for key, m in probe.items():
        lat = m["median_latency_s"]
        for name, n in (("variant slice (200)", 200), ("ChartQA val (1,920)", 1920),
                        ("RefChartQA val (1,200)", 1200)):
            print(f"  {key:<22} {name:<24} {n * lat / 3600:6.2f} h")
        break


def stage_variant(args) -> dict[str, Any]:
    """`PLAN.md` 5.2, on the frozen 200-question slice."""
    from chartqa_dt.eval.generate import generate_over, write_generations
    from chartqa_dt.eval.metrics import relaxed_correctness

    reader = chartqa_reader()
    rows = attach_images(reader, load_slice("chartqa_variant_200"))
    if args.limit:
        rows = rows[:args.limit]

    table: dict[str, Any] = {}
    for variant in args.variants.split(","):
        variant = variant.strip()
        loaded = load_model(variant)
        gens, rep = generate_over(loaded, rows, mode="structured")
        write_generations(gens, out_dir() / f"variant_{variant}.jsonl")
        correct = sum(relaxed_correctness(r["answer"], g.answer)
                      for r, g in zip(rows, gens))
        from chartqa_dt.plans.roundtrip import check_many
        from chartqa_dt.prompting.parsing import parse_record, schema_ok

        recs = [res.record for g in gens
                if (res := parse_record(g.raw)).ok and schema_ok(res.record)[0]]
        _, rt = check_many(recs)
        table[variant] = {
            "n": len(gens),
            "relaxed_accuracy": correct / max(1, len(gens)),
            "valid_json_fraction": rep.parse.valid_fraction,
            # The gate uses this one: a record the executor rejects is a failure
            # (non-negotiable rule 3), whatever its JSON syntax.
            "schema_valid_fraction": rep.parse.schema_valid_fraction,
            "repaired_fraction": rep.parse.repaired_fraction,
            "median_latency_s": rep.median_latency,
            "median_new_tokens": rep.median_new_tokens,
            "hit_token_cap_fraction": rep.capped_fraction,
            "failure_reasons": dict(rep.parse.reasons),
            "schema_failure_reasons": dict(rep.parse.schema_reasons),
            "roundtrip_agreement": rt.agreement,
            "roundtrip_executable": rt.executable,
        }
        print(f"\n{variant}: accuracy {100 * table[variant]['relaxed_accuracy']:.2f}%  "
              f"schema-valid {100 * rep.parse.schema_valid_fraction:.1f}%  "
              f"round-trip {100 * rt.agreement:.1f}%  "
              f"median {rep.median_latency:.2f}s", flush=True)
        del loaded
        _free()

    table["gate"] = VARIANT_GATE
    table["decision"] = decide_variant(table)
    (out_dir() / "variant_selection.json").write_text(json.dumps(table, indent=2) + "\n")
    print("\n" + format_variant_table(table))
    return table


def decide_variant(table: dict[str, Any]) -> dict[str, Any]:
    """Apply `PLAN.md` 5.2's three-part gate. All three must hold, or Instruct wins."""
    inst, think = table.get("instruct"), table.get("thinking")
    if not (inst and think):
        return {"choice": "instruct", "reason": "only one variant was measured"}
    gain = 100 * (think["relaxed_accuracy"] - inst["relaxed_accuracy"])
    ratio = (think["median_latency_s"] / inst["median_latency_s"]
             if inst["median_latency_s"] else float("inf"))
    checks = {
        "accuracy_gain_points": {"value": gain,
                                 "required": f">= {VARIANT_GATE['min_accuracy_gain_points']}",
                                 "pass": gain >= VARIANT_GATE["min_accuracy_gain_points"]},
        "schema_valid_fraction": {"value": think.get("schema_valid_fraction",
                                                     think["valid_json_fraction"]),
                                  "required": f">= {VARIANT_GATE['min_valid_json_fraction']}",
                                  "pass": think.get("schema_valid_fraction",
                                                    think["valid_json_fraction"])
                                  >= VARIANT_GATE["min_valid_json_fraction"]},
        "latency_ratio": {"value": ratio,
                          "required": f"<= {VARIANT_GATE['max_latency_ratio']}",
                          "pass": ratio <= VARIANT_GATE["max_latency_ratio"]},
    }
    passed = all(c["pass"] for c in checks.values())
    return {"choice": "thinking" if passed else "instruct", "checks": checks,
            "reason": "all three gate conditions hold" if passed
            else "at least one gate condition fails; PLAN 5.2 defaults to Instruct"}


def format_variant_table(table: dict[str, Any]) -> str:
    lines = [f"{'variant':<12}{'accuracy':>10}{'json':>8}{'schema':>9}{'capped':>9}"
             f"{'tokens':>8}{'median s':>10}"]
    for name in ("instruct", "thinking"):
        m = table.get(name)
        if not m:
            continue
        lines.append(f"  {name:<10}{100 * m['relaxed_accuracy']:>9.2f}%"
                     f"{100 * m['valid_json_fraction']:>7.1f}%"
                     f"{100 * m.get('schema_valid_fraction', 0):>8.1f}%"
                     f"{100 * m.get('hit_token_cap_fraction', 0):>8.1f}%"
                     f"{m.get('median_new_tokens', 0):>8.0f}"
                     f"{m['median_latency_s']:>10.2f}")
    d = table.get("decision", {})
    lines.append(f"\n  decision: {d.get('choice')}  ({d.get('reason')})")
    for name, c in (d.get("checks") or {}).items():
        lines.append(f"    {'PASS' if c['pass'] else 'FAIL'}  {name}: "
                     f"{c['value']:.3f}  needs {c['required']}")
    return "\n".join(lines)


#: `PLAN.md` 5.3 says "validation split". All 1,920 questions would resolve ~3 points and
#: cost up to 7 h of quota; 800 resolves ~5 points for 1.5–2.9 h, which is ample for a
#: baseline and leaves the budget for Phases 6 and 7. Sized from the power calculation
#: rather than by taking the whole split because it was there (design pass, step 5).
STRUCTURED_N = 800


def stage_chartqa(args) -> dict[str, Any]:
    """`PLAN.md` 5.3 — zero-shot ChartQA on validation, structured and plain.

    Both prompts over the same items, so the gap between them is what structured output
    costs with nothing else varying.

    **The plain arm is not a reproduction of the published 79.1** (`DECISIONS.md` 0063).
    That figure is the ChartQA **test** split; this runs on validation, because rule 1
    seals test until the pre-registration is committed. The comparison is a sanity
    indication here and becomes a real reproduction at Phase 7.
    """
    from chartqa_dt.eval.generate import generate_over, write_generations
    from chartqa_dt.eval.runner import evaluate_predictions, score_item, write_results
    from chartqa_dt.plans.roundtrip import check_many
    from chartqa_dt.prompting.parsing import parse_record, schema_ok

    reader = chartqa_reader()
    all_rows = attach_images(reader, load_slice("chartqa_val"))
    if args.limit:
        all_rows = all_rows[:args.limit]
    loaded = load_model(args.variant)

    report: dict[str, Any] = {
        "variant": args.variant, "split": "validation",
        "published_reference": {"value": 79.1, "split": "test",
                                "note": "NOT comparable to this run — different split "
                                        "(DECISIONS.md 0063); reproduced at Phase 7"},
        "arms": {},
    }
    for mode in ("structured", "plain"):
        # The plain arm is cheap (0.28 s/item), so it runs on everything; the structured
        # arm is sized to what it can resolve.
        rows = all_rows if mode == "plain" else all_rows[:args.structured_n]
        gens, rep = generate_over(loaded, rows, mode=mode)
        write_generations(gens, out_dir() / f"chartqa_val_{mode}.jsonl")
        items = [score_item(r["record_id"], r["answer"], g.answer,
                            subset=r["question_kind"])
                 for r, g in zip(rows, gens)]
        result = evaluate_predictions(items, seeds=[0, 1, 2], ap_resamples=1)

        arm: dict[str, Any] = {
            "n": len(rows),
            "relaxed_accuracy": result.relaxed_accuracy.mean,
            "ci": [result.relaxed_accuracy.lo, result.relaxed_accuracy.hi],
            "by_subset": {k: v["relaxed_accuracy"] for k, v in result.by_subset.items()},
            "median_latency_s": rep.median_latency,
            "median_new_tokens": rep.median_new_tokens,
        }
        if mode == "structured":
            recs = [res.record for g in gens
                    if (res := parse_record(g.raw)).ok and schema_ok(res.record)[0]]
            _, rt = check_many(recs)
            arm.update(valid_json_fraction=rep.parse.valid_fraction,
                       schema_valid_fraction=rep.parse.schema_valid_fraction,
                       hit_token_cap_fraction=rep.capped_fraction,
                       roundtrip_agreement=rt.agreement,
                       roundtrip_executable=rt.executable,
                       roundtrip_counts=dict(rt.counts))
            print(rt.describe(), flush=True)
        write_results(result, out_dir() / f"chartqa_val_{mode}.json",
                      meta={"variant": args.variant, "mode": mode, "split": "validation",
                            **{k: v for k, v in arm.items() if k != "by_subset"}})
        report["arms"][mode] = arm
        print(f"\n{mode} (n={len(rows)}): {result.relaxed_accuracy.as_percent}", flush=True)

    a, b = report["arms"]["structured"], report["arms"]["plain"]
    report["structured_output_cost_points"] = 100 * (b["relaxed_accuracy"]
                                                     - a["relaxed_accuracy"])
    (out_dir() / "chartqa_zeroshot.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nstructured costs {report['structured_output_cost_points']:+.2f} points "
          f"against the plain prompt (the elicitation behind the published figure)")
    print(f"plain arm {100 * b['relaxed_accuracy']:.2f}% on VALIDATION; the published "
          f"79.1 is TEST and is reproduced at Phase 7, not here (DECISIONS.md 0063)")
    return report


def stage_refchartqa(args) -> dict[str, Any]:
    """`PLAN.md` 5.4 — zero-shot grounding. The measurement that decides the framing."""
    from chartqa_dt.eval.generate import generate_over, write_generations
    from chartqa_dt.eval.metrics import average_precision_coco, p_at_f1
    from chartqa_dt.eval.official_format import build_rows, score_with_official
    from chartqa_dt.eval.runner import evaluate_predictions, score_item, write_results
    from chartqa_dt.eval.stratified import stratify
    from chartqa_dt.vision.coords import clamp_for_official_evaluator

    rows = refchartqa_val(args.refchartqa_n)
    if args.limit:
        rows = rows[:args.limit]
    loaded = load_model(args.variant)
    gens, rep = generate_over(loaded, rows, mode="structured")
    write_generations(gens, out_dir() / "refchartqa_val.jsonl")

    preds: list[tuple[str, float, list[float]]] = []
    gts: dict[str, list[list[float]]] = {}
    pairs = []
    items = []
    strat = []
    for row, gen in zip(rows, gens):
        key = row["record_id"]
        boxes = [list(map(float, clamp_for_official_evaluator(b))) for b in gen.boxes]
        gt = row["gt_boxes"]
        if gt:
            gts[key] = gt
        preds.extend((key, 1.0, b) for b in boxes)
        pairs.append((boxes, gt))
        items.append(score_item(key, row["answer"], gen.answer,
                                pred_boxes=boxes, gt_boxes=gt,
                                subset=row["question_kind"]))
        w, h = row["image_size"]
        scale = 512 / max(w, h)
        strat.append({"pred_boxes": boxes, "gt_boxes": gt,
                      "resized_size": (w * scale, h * scale)})

    result = evaluate_predictions(items, seeds=[0, 1, 2], ap_resamples=200)
    report_strat = stratify(strat)

    # `PLAN.md` 5.4 requires the RELEASED evaluator, and `DECISIONS.md` 0003 makes it the
    # scorer of record. Ours agrees to within 0.07 pp on 11,690 real predictions, and
    # "agrees with" is still not "is" — so the reported number comes from the vendored
    # code, while ours supplies the strata and intervals it cannot produce.
    official_rows = build_rows([
        {"pred_boxes": [list(map(float, clamp_for_official_evaluator(b)))
                        for b in gen.boxes],
         "answer": gen.answer, "label": row["answer"], "image_size": row["image_size"],
         "grounding_bboxes": row["raw_boxes"], "question_kind": row["question_kind"]}
        for row, gen in zip(rows, gens)])
    official = score_with_official(official_rows)
    by_subset = {}
    for kind in ("human", "machine", "pot"):
        subset = [r for r in official_rows if r["type"] == kind]
        if subset:
            by_subset[kind] = score_with_official(subset)

    write_results(result, out_dir() / "refchartqa_zeroshot.json",
                  meta={"variant": args.variant, "n": len(rows),
                        "valid_json_fraction": rep.parse.valid_fraction,
                        "schema_valid_fraction": rep.parse.schema_valid_fraction,
                        "official": official, "official_by_subset": by_subset,
                        "published_reference_ap50": 32.83,
                        "published_reference_note":
                            "Level C — not independently reproducible, DECISIONS.md 0052"},
                  stratified=report_strat.to_dict())

    ours_ap = average_precision_coco(preds, gts, 0.5)
    print("\n" + result.describe())
    print("\n" + report_strat.describe())
    print("\nOFFICIAL evaluator — this is the number of record:")
    print(f"  accuracy {100 * official['accuracy']:.2f}%   "
          f"AP@0.5 {100 * official['AP_50']:.2f}%   "
          f"P@F1 {100 * official['P_at_FI']:.2f}%")
    for kind, m in by_subset.items():
        print(f"    {kind:<8} AP@0.5 {100 * m['AP_50']:>6.2f}%   "
              f"P@F1 {100 * m['P_at_FI']:>6.2f}%   acc {100 * m['accuracy']:>6.2f}%")
    print(f"\n  ours, cross-check: AP@0.5 {100 * ours_ap:.2f}%   "
          f"P@F1 {100 * p_at_f1(pairs):.2f}%   "
          f"(delta {abs(100 * ours_ap - 100 * official['AP_50']):.3f} pp)")
    print("  published reference 32.83 is the human TEST subset — Level C, not "
          "independently reproducible (DECISIONS.md 0052)")
    return {"official": official, "ours_ap50": ours_ap,
            "stratified": report_strat.to_dict()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["probe", "variant", "chartqa", "refchartqa"])
    ap.add_argument("--variants", default="instruct,thinking")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--probe-n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="override the structured budget; 0 keeps the default")
    ap.add_argument("--refchartqa-n", type=int, default=1200)
    ap.add_argument("--structured-n", type=int, default=STRUCTURED_N,
                    help="items for the 5.3 structured arm; sized from what "
                         "it can resolve, not from the split size")
    ap.add_argument("--variant", default="instruct",
                    help="the variant 5.2 selected; used by the 5.3 and 5.4 stages")
    args = ap.parse_args()

    started = time.time()
    stages = {"probe": stage_probe, "variant": stage_variant,
              "chartqa": stage_chartqa, "refchartqa": stage_refchartqa}
    stages[args.stage](args)
    print(f"\nstage {args.stage} finished in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
