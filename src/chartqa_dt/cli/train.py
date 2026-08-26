"""``cdt-train`` — LoRA fine-tuning, stages 1 and 2 and the control (PLAN Phase 6)."""

from __future__ import annotations

from chartqa_dt.cli._common import NotYetBuilt, base_parser, setup


def main() -> None:
    p = base_parser("cdt-train", "Train the grounding curriculum, the joint stage, or the direct-answer control.")
    p.add_argument("--stage", type=str, default=None,
                   choices=["smoke", "stage1", "stage2", "control"],
                   help="smoke: the Phase 2 100-step backbone test; stage1: grounding; "
                        "stage2: joint box+plan+answer; control: direct-answer baseline")
    p.add_argument("--backend", type=str, default=None, choices=["hf_peft", "unsloth"])
    p.add_argument("--resume", type=str, default=None, help="checkpoint directory or hub path to resume from")
    p.add_argument("--steps", type=int, default=None, help="hard cap on optimizer steps")
    setup(p)  # validates config, dumps provenance
    raise NotYetBuilt("cdt-train", "Phase 2 (smoke) / Phase 6 (training)")
