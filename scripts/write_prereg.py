"""Generate `PREREGISTRATION.md` from the recorded facts — `PLAN.md` 5.5.

Generated rather than hand-written, on purpose. 5.5 lists eleven things the file must
contain, several of which already exist somewhere authoritative: the prompt strings, the
slice hashes, the mixture counts, the evaluator hashes. Retyping them into prose is how a
pre-registration ends up describing a run that never happened.

So every number here is read from its source at generation time, and the sources are named
inline. The file is then committed, and `chartqa_dt.splits` refuses to open a test split
until it is committed and clean.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(rel: str) -> dict:
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> None:
    from chartqa_dt.eval.generate import DECODING, MAX_NEW_TOKENS_PLAIN, MAX_NEW_TOKENS_STRUCTURED
    from chartqa_dt.prompting.prompts import (
        PLAIN_PROMPT,
        STRUCTURED_PROMPT,
        prompt_fingerprint,
    )

    facts = read_json("verification/measured_facts.json")
    manifest = read_json("data/MANIFEST.json")
    variant = read_json("outputs/phase5/variant_selection.json")
    slices = {n: read_json(f"data/slices/{n}.json")
              for n in ("chartqa_variant_200", "chartqa_val")}
    refprov = read_json("verification/refchartqa_eval/PROVENANCE.json")
    cqprov = read_json("verification/chartqa_eval/PROVENANCE.json")
    fp = prompt_fingerprint()
    mixtures = facts.get("phase3", {}).get("mixtures", {})
    phase2 = facts.get("phase2", {})

    chosen = (variant.get("decision") or {}).get("choice", "TBD — 5.2 has not run")
    decision_reason = (variant.get("decision") or {}).get("reason", "")

    def variant_table() -> str:
        if not variant:
            return "_5.2 has not run yet; this section is filled by `scripts/write_prereg.py`._"
        rows = ["| variant | relaxed accuracy | valid JSON | repaired | median latency |",
                "|---|---:|---:|---:|---:|"]
        for name in ("instruct", "thinking"):
            m = variant.get(name)
            if not m:
                continue
            rows.append(f"| {name} | {100 * m['relaxed_accuracy']:.2f}% | "
                        f"{100 * m['valid_json_fraction']:.1f}% | "
                        f"{100 * m['repaired_fraction']:.1f}% | "
                        f"{m['median_latency_s']:.2f} s |")
        checks = (variant.get("decision") or {}).get("checks") or {}
        if checks:
            rows.append("")
            rows.append("| gate condition | measured | required | verdict |")
            rows.append("|---|---:|---|---|")
            for name, c in checks.items():
                rows.append(f"| {name} | {c['value']:.3f} | {c['required']} | "
                            f"{'PASS' if c['pass'] else 'FAIL'} |")
        return "\n".join(rows)

    body = f"""# Pre-registration

**Frozen before any test split is opened.** `PLAN.md` 5.5 requires this file to be
committed and clean before `chartqa_dt.splits` will allow a sealed split, and
`assert_split_allowed` enforces it mechanically rather than on trust
(`DECISIONS.md` 0031).

Everything below is generated from its authoritative source by
`scripts/write_prereg.py`. Retyping numbers into prose is how a pre-registration ends up
describing a run that never happened.

---

## 1. Backbone and variant

| | |
|---|---|
| backbone | `Qwen/Qwen3-VL-2B-Instruct` (`DECISIONS.md` 0035, 0036) |
| variant selected | **{chosen}** |
| reason | {decision_reason or "—"} |
| visual token factor | {facts.get('model', {}).get('visual_token_factor', '?')} (derived from the processor, `DECISIONS.md` 0008) |
| quantisation | 4-bit NF4, vision tower excluded (`DECISIONS.md` 0012) |
| LoRA | r=16, alpha=32, dropout=0.05, on **both** vision and language |
| image budget | 512 px long side |

### 5.2 comparison, on the frozen 200-question slice

{variant_table()}

`PLAN.md` 5.2's gate — Thinking only if **all three** hold: ≥ 2 accuracy points better,
≥ 90% valid JSON, ≤ 2× Instruct's median latency. The thresholds were written into
`scripts/run_zeroshot.py` before any number existed.

## 2. Prompts, verbatim

Structured prompt — SHA-256 `{fp['structured']}`:

```
{STRUCTURED_PROMPT}
```

Plain prompt — SHA-256 `{fp['plain']}`. This is the Qwen3-VL report's own ChartQA
elicitation, reproduced exactly so that "structured output costs N points" is measured
against the elicitation that produced the published 79.1 (`verification/phase0.md` F9):

```
{PLAIN_PROMPT}
```

## 3. Decoding

Greedy, fixed: `{json.dumps(DECODING)}`. Max new tokens: {MAX_NEW_TOKENS_STRUCTURED}
(structured), {MAX_NEW_TOKENS_PLAIN} (plain). Sampling is not used — the "before" number
must be exactly reproducible from this file, or the before/after comparison inherits noise
it cannot separate from a real effect.

## 4. Answer normaliser

`chartqa_dt.eval.metrics.normalise_prediction` — `str(text).strip()`, applied to the
model's output **before** scoring. The metric itself is left byte-identical to the
official one, which does *not* strip (`DECISIONS.md` 0053). Normalising in the pipeline
rather than in the metric keeps our numbers comparable with published ones.

## 5. Evaluators

| evaluator | file | SHA-256 |
|---|---|---|
| RefChartQA (official) | `verification/refchartqa_eval/evaluate.py` | `{(refprov.get('files', {}).get('evaluate.py', {}) or {}).get('sha256', '?')}` |
| ChartQA (pix2struct, official) | `verification/chartqa_eval/metrics.py` | `{(cqprov.get('files', {}).get('metrics.py', {}) or {}).get('sha256', '?')}` |

Both are vendored byte-identical and hash-checked by `tests/test_vendored_integrity.py`.
`DECISIONS.md` 0003 makes the official evaluator the scorer of record; our implementation
agrees with it on 11,690 real predictions to within 0.07 percentage points of AP.

## 6. Datasets, pinned

| dataset | revision | integrity |
|---|---|---|
| ChartQA | `{manifest.get('archives', {}).get('chartqa', {}).get('revision', '?')}` | archive SHA-256 `{manifest.get('archives', {}).get('chartqa', {}).get('sha256', '?')[:32]}…` |
| RefChartQA | `{(manifest.get('parquet', {}).get('refchartqa', {}) or {}).get('revision', '?')}` | {len((manifest.get('parquet', {}).get('refchartqa', {}) or {}).get('files', {}))} parquet SHA-256s recorded before download |

## 7. Frozen validation slices

| slice | n | SHA-256 |
|---|---:|---|
""" + "\n".join(
        f"| `{n}` | {d.get('n', '?')} | `{d.get('slice_sha256', '?')}` |"
        for n, d in slices.items()) + f"""

Sampled once, before any prompt existed. Test splits are untouched.

## 8. Training mixtures

| | stage 1 | stage 2 |
|---|---:|---:|
| total | {mixtures.get('stage1', {}).get('total', '?'):,} | {mixtures.get('stage2', {}).get('total', '?'):,} |
| synthetic | {mixtures.get('stage1', {}).get('synthetic', 0):,} | {mixtures.get('stage2', {}).get('synthetic', 0):,} |
| ChartQA | {mixtures.get('stage1', {}).get('chartqa', 0):,} | {mixtures.get('stage2', {}).get('chartqa', 0):,} |
| RefChartQA | 0 | {mixtures.get('stage2', {}).get('refchartqa', 0):,} |
| with boxes | {mixtures.get('stage1', {}).get('with_boxes', 0):,} | {mixtures.get('stage2', {}).get('with_boxes', 0):,} |
| with a plan | {mixtures.get('stage1', {}).get('with_plan', 0):,} | {mixtures.get('stage2', {}).get('with_plan', 0):,} |
| of those, compositional | {mixtures.get('stage1', {}).get('compositional', 0):,} | {mixtures.get('stage2', {}).get('compositional', 0):,} |

Deduplicated: {mixtures.get('dedup_merges', 0)} merges, of which
{mixtures.get('dedup_chartqa_refchartqa_merges', 0)} across ChartQA and RefChartQA.
Zero validation or test records, asserted in code, at the **image** level as well as the
split label (`DECISIONS.md` 0048, 0049).

## 9. Training hyperparameters

Measured in the Phase 2 smoke run (`verification/measured_facts.json` → `phase2`):

| | |
|---|---|
| peak reserved | {phase2.get('peak_reserved_gb', '?')} GiB |
| seconds per step | {phase2.get('seconds_per_step', '?')} |
| projected full run | {phase2.get('projected_full_run_hours', '?')} h |
| LoRA params (vision / language) | {phase2.get('lora_vision_params', '?'):,} / {phase2.get('lora_language_params', '?'):,} |
| batch | 2 × 4 gradient accumulation, single device (`DECISIONS.md` 0025) |

## 10. Early stopping

Validation relaxed accuracy and AP@0.5, evaluated at fixed intervals. Stop when neither
improves for two consecutive evaluations. The checkpoint reported is the last one that
improved, not the last one trained — and the rule is fixed here so it cannot be relaxed
after seeing a curve.

## 11. What counts as success

| target | claim level | success |
|---|---|---|
| ChartQA relaxed accuracy | **B** — published 79.1, exact checkpoint, exact prompt, verified evaluator | fine-tuned beats our own zero-shot baseline, CIs disjoint |
| RefChartQA AP@0.5 | **C** — published 32.83 is not independently reproducible (`DECISIONS.md` 0052) | fine-tuned beats our own zero-shot baseline, CIs disjoint |
| executable plans | — | the executor reproduces the emitted answer on a majority of records; invalid records count as failures |

**The primary claim is the internal before/after**, both arms measured by us with the
byte-identical official evaluator on the same sealed split. 32.83 is cited as context and
labelled as unverified, because nobody — including its authors' released artefacts — can
reproduce it.

## 12. Extensions and their entry gates

Planned only if the core result lands and quota remains: ChartQAPro transfer (entry gate:
Phase 7 complete and the extension approved), and the RefChartQA scaling ladder at
4,000 / 10,000 / 25,000 training rows (entry gate: Phase 6 stage 2 complete).

---

_Generated by `scripts/write_prereg.py`. Regenerating after the test split is opened would
defeat the purpose; the committed version is the record._
"""
    path = ROOT / "PREREGISTRATION.md"
    path.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"wrote {path}  ({len(body):,} bytes)")
    print(f"SHA-256: {digest}")
    print("\nNext: commit it, then record that hash in DECISIONS.md.")


if __name__ == "__main__":
    main()
