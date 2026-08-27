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
    from chartqa_dt.eval.generate import generate_over

    reader = chartqa_reader()
    rows = attach_images(reader, load_slice("chartqa_variant_200")[:args.probe_n])
    report: dict[str, Any] = {}
    for variant in args.variants.split(","):
        loaded = load_model(variant.strip())
        for mode in ("structured", "plain"):
            _gens, rep = generate_over(loaded, rows, mode=mode, progress_every=5)
            report[f"{variant}/{mode}"] = {
                "n": rep.n, "median_latency_s": rep.median_latency,
                "valid_json_fraction": rep.parse.valid_fraction if mode == "structured"
                else 1.0,
            }
            print(f"  {variant}/{mode}: median {rep.median_latency:.2f}s/item, "
                  f"valid {rep.parse.valid}/{rep.parse.total}", flush=True)
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
        table[variant] = {
            "n": len(gens),
            "relaxed_accuracy": correct / max(1, len(gens)),
            "valid_json_fraction": rep.parse.valid_fraction,
            "repaired_fraction": rep.parse.repaired_fraction,
            "median_latency_s": rep.median_latency,
            "failure_reasons": dict(rep.parse.reasons),
        }
        print(f"\n{variant}: accuracy {100 * table[variant]['relaxed_accuracy']:.2f}%  "
              f"valid JSON {100 * rep.parse.valid_fraction:.1f}%  "
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
        "valid_json_fraction": {"value": think["valid_json_fraction"],
                                "required": f">= {VARIANT_GATE['min_valid_json_fraction']}",
                                "pass": think["valid_json_fraction"]
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
    lines = [f"{'variant':<12}{'accuracy':>10}{'valid JSON':>12}{'repaired':>10}"
             f"{'median s':>10}"]
    for name in ("instruct", "thinking"):
        m = table.get(name)
        if not m:
            continue
        lines.append(f"  {name:<10}{100 * m['relaxed_accuracy']:>9.2f}%"
                     f"{100 * m['valid_json_fraction']:>11.1f}%"
                     f"{100 * m['repaired_fraction']:>9.1f}%"
                     f"{m['median_latency_s']:>10.2f}")
    d = table.get("decision", {})
    lines.append(f"\n  decision: {d.get('choice')}  ({d.get('reason')})")
    for name, c in (d.get("checks") or {}).items():
        lines.append(f"    {'PASS' if c['pass'] else 'FAIL'}  {name}: "
                     f"{c['value']:.3f}  needs {c['required']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["probe", "variant", "chartqa", "refchartqa"])
    ap.add_argument("--variants", default="instruct,thinking")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--probe-n", type=int, default=20)
    ap.add_argument("--refchartqa-n", type=int, default=1200)
    args = ap.parse_args()

    started = time.time()
    {"probe": stage_probe, "variant": stage_variant}[args.stage](args) \
        if args.stage in ("probe", "variant") else _not_yet(args)
    print(f"\nstage {args.stage} finished in {(time.time() - started) / 60:.1f} min")
    return 0


def _not_yet(args):
    raise SystemExit(f"stage {args.stage} lands after the probe sizes it")


if __name__ == "__main__":
    raise SystemExit(main())
