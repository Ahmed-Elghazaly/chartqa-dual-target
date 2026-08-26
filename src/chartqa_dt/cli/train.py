"""``cdt-train`` — LoRA fine-tuning, and the Phase 2 backbone smoke test."""

from __future__ import annotations

import sys

from chartqa_dt.cli._common import NotYetBuilt, base_parser, setup


def main() -> None:
    p = base_parser(
        "cdt-train",
        "Train the grounding curriculum, the joint stage, or the direct-answer control.\n"
        "--stage smoke runs the Phase 2 backbone test and answers whether this model\n"
        "trains at all on this hardware, inside the 13.5 GiB / 10 h gates.",
    )
    p.add_argument("--stage", type=str, default=None,
                   choices=["smoke", "stage1", "stage2", "control"])
    p.add_argument("--backend", type=str, default=None, nargs="*",
                   choices=["hf_peft", "unsloth"],
                   help="backends to try; default tries every available one")
    p.add_argument("--resume", type=str, default=None, help="checkpoint dir or hub path")
    p.add_argument("--steps", type=int, default=100, help="optimizer steps for the smoke test")
    p.add_argument("--resolutions", type=str, default="512,native",
                   help="comma-separated input budgets to measure (decision 0010). "
                        "'native' means the model's own max_pixels; a number R means R^2 pixels.")
    p.add_argument("--no-resume-test", action="store_true",
                   help="skip the kill-and-resume verification (it doubles model loads)")
    ctx = setup(p)

    if ctx.args.stage != "smoke":
        raise NotYetBuilt(f"cdt-train --stage {ctx.args.stage}", "Phase 6 — Training")

    _run_smoke(ctx)


def _run_smoke(ctx) -> None:
    from chartqa_dt.modeling.backends.base import list_backends
    from chartqa_dt.train.smoke import HEADER, run_smoke, write_report

    available = list_backends()
    print("\nbackends:")
    for name, (ok, why) in available.items():
        print(f"  {name:<10} available={ok}  {why}")

    wanted = ctx.args.backend or [n for n, (ok, _) in available.items() if ok]
    if not wanted:
        print("\nNo backend is available. Install one:  pip install -e '.[gpu]'")
        raise SystemExit(2)

    # Decision 0010: the base input resolution is chosen on measured evidence,
    # not inherited from an analysis that used the wrong visual-token factor.
    budgets: list[tuple[str, int]] = []
    for token in (t.strip() for t in ctx.args.resolutions.split(",") if t.strip()):
        if token == "native":
            budgets.append(("native", 16777216))   # the model's own max_pixels
        else:
            r = int(token)
            budgets.append((f"{r}px", r * r))

    logger = ctx.logger()
    results = []
    try:
        for backend_name in wanted:
            for res_label, max_px in budgets:
                label = f"{backend_name}/{res_label}"
                print(f"\n{'=' * 78}\n  {label}\n{'=' * 78}")
                r = run_smoke(
                    ctx.cfg,
                    backend_name=backend_name,
                    image_max_pixels=max_px,
                    label=label,
                    steps=ctx.args.steps,
                    out_dir=ctx.out_dir,
                    logger=logger,
                    test_resume=not ctx.args.no_resume_test,
                )
                results.append(r)
                if not r.ok:
                    print(f"  FAILED: {r.error}")
                else:
                    print(f"  peak {r.peak_reserved_gb:.2f} GiB   {r.seconds_per_step:.2f} s/step   "
                          f"projected {r.projected_full_run_hours:.2f} h   "
                          f"loss {r.loss_first_10:.3f} -> {r.loss_last_10:.3f}")
                logger.event("smoke_result", **{k: v for k, v in vars(r).items() if k != "losses"})

        path = write_report(results, ctx.out_dir)
        print(f"\n{HEADER}\n{'-' * len(HEADER)}")
        for r in results:
            print(r.row())
        print(f"\nwritten: {path}")

        passing = [r for r in results if r.passes_all_gates]
        if not passing:
            print("\nNO CONFIGURATION PASSED ALL GATES.")
            print("Apply the Phase 2 fallback ladder in order: batch 2->1; image 512->448;")
            print("attention-only modules while KEEPING LoRA on both sides; then move down the")
            print("model ladder in IDEA.md 7. If nothing passes, stop and report (rule 9).")
            sys.exit(1)

        best = min(passing, key=lambda r: r.projected_full_run_hours)
        print(f"\nFastest passing configuration: {best.label}")
        print("Record the choice in DECISIONS.md with this table before proceeding (PLAN 2.4).")
    finally:
        logger.close()
