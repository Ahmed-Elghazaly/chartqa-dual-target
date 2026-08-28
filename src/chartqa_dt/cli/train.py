"""``cdt-train`` — LoRA fine-tuning, and the Phase 2 backbone smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

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
    p.add_argument("--batches", type=str, default=None,
                   help="comma-separated per-device batch sizes to compare, e.g. 2,4,8. "
                        "grad_accum is set so the EFFECTIVE batch stays at the "
                        "pre-registered value, so only micro-batch grouping changes.")
    p.add_argument("--resolutions", type=str, default="512,native",
                   help="comma-separated input budgets to measure (decision 0010). "
                        "'native' means the model's own max_pixels; a number R means R^2 pixels.")
    p.add_argument("--no-resume-test", action="store_true",
                   help="skip the kill-and-resume verification (it doubles model loads)")
    p.add_argument("--mixture", type=str, default="data/mixture_stage1.json",
                   help="mixture file for stage1/stage2/control")
    p.add_argument("--lr", type=float, default=None,
                   help="override the stage learning rate; the 6.x fallback is 2e-5")
    p.add_argument("--summarise", type=str, default=None,
                   help="print the DECISIONS.md table for an existing smoke_results.json and exit")
    ctx = setup(p)

    if ctx.args.summarise:
        from chartqa_dt.train.smoke import load_report, markdown_table
        print(markdown_table(load_report(Path(ctx.args.summarise))))
        return

    if ctx.args.stage in ("stage1", "stage2", "control"):
        _run_stage(ctx)
        return
    if ctx.args.stage != "smoke":
        raise NotYetBuilt(f"cdt-train --stage {ctx.args.stage}", "Phase 6 — Training")

    _run_smoke(ctx)


def _run_stage(ctx) -> None:
    """`PLAN.md` 6.1, 6.2 and 6.4 — the real training stages."""
    import json

    from chartqa_dt.config import ModelConfig
    from chartqa_dt.modeling.backends.base import get_backend
    from chartqa_dt.train.checkpoint import load_checkpoint
    from chartqa_dt.train.feed import MixtureFeed
    from chartqa_dt.train.loop import TrainConfig, train
    from chartqa_dt.train.smoke import build_optimizer
    from chartqa_dt.train.validate import make_evaluator

    args = ctx.args
    stage = args.stage
    records = _records_for(ctx)
    print(f"\n{stage}: {len(records):,} records from {args.mixture}")

    # `PLAN.md` 6.1 orders stage 1 easy->hard; 6.2 shuffles stage 2. The control (6.4)
    # trains on the SAME records as the arm it is a control for, so it inherits its
    # ordering — only the target differs.
    feed = MixtureFeed(records, shuffle=(stage != "stage1"), seed=ctx.cfg.seed,
                       answer_only=(stage == "control"),
                       image_root=Path(ctx.env.data_root))

    cfg = TrainConfig(stage=stage, steps=args.steps or 3000,
                      batch_size=ctx.cfg.train.per_device_batch,
                      grad_accum=ctx.cfg.train.grad_accum,
                      max_len=ctx.cfg.model.max_seq_len,
                      lr=args.lr, seed=ctx.cfg.seed,
                      out_dir=ctx.out_dir / "checkpoints",
                      answer_only=(stage == "control"))
    print(f"  lr {cfg.learning_rate}  batch {cfg.batch_size}x{cfg.grad_accum} "
          f"= {cfg.batch_size * cfg.grad_accum}  max_len {cfg.max_len}  "
          f"steps {cfg.steps}")

    model_cfg = ModelConfig(image_max_pixels=512 * 512)
    loaded = get_backend(args.backend[0] if args.backend else "hf_peft").load(model_cfg)
    print(loaded.describe())

    state = optimizer = None
    if args.resume:
        optimizer, state = load_checkpoint(
            args.resume, model=loaded.model,
            optimizer_factory=lambda m: build_optimizer(m, cfg.learning_rate))
        feed.load_state_dict(state.feed)
        print(f"  resumed at step {state.step}, feed position {state.feed.get('position')}")

    # `PLAN.md` 6.5/6.6. The stopping signal is validation loss, not AP — an AP measured
    # on an affordable slice has a +/-8.7 point interval and cannot detect "has not
    # improved" (`DECISIONS.md` 0069).
    evaluate = None
    holdout = _holdout_examples(ctx, records, feed)
    if holdout:
        evaluate = make_evaluator(loaded, holdout, max_len=cfg.max_len)
        print(f"  validation: {len(holdout)} held-out examples, "
              f"early stopping on loss, patience {cfg.patience}")

    result = train(loaded, feed, cfg, state=state, optimizer=optimizer,
                   evaluate=evaluate, on_log=_progress,
                   on_checkpoint=_hub_pusher(ctx, stage))
    print("\n" + result.summary())
    print(feed.stats.describe())

    if evaluate is not None:
        report_path = ctx.out_dir / f"{stage}_validation.json"
        report_path.write_text(
            json.dumps([r.to_dict() for r in evaluate.reports], indent=2) + "\n",
            encoding="utf-8")
        print(f"  validation curve -> {report_path}")

    report = {"stage": stage, "steps": result.state.step,
              "stopped_early": result.stopped_early,
              "losses": result.state.losses, "grad_norms": result.state.grad_norms,
              "feed": feed.state_dict(), "usable": feed.stats.usable,
              "offered": feed.stats.offered, "refused": feed.stats.refused,
              "logs": [x.to_dict() for x in result.logs]}
    out = ctx.out_dir / f"{stage}_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwritten to {out}")


def _hub_pusher(ctx, stage: str):
    """`PLAN.md` 6.3: push to the Hub on every save.

    Non-strict on purpose. A checkpoint is already safe on local disk by the time this
    runs, so a transient upload failure must not end a run that is otherwise healthy —
    losing one periodic push costs nothing, losing ten hours costs the phase.
    """
    from chartqa_dt.hub import HubStore

    store = HubStore(repo_id=ctx.cfg.hub.repo_id) if getattr(ctx.cfg, "hub", None) \
        else None
    if store is None or not store.enabled:
        print("  hub: disabled (no token or no repo configured); checkpoints stay local")
        return None

    def push(path, state) -> None:
        ok = store.push_dir(path, f"{stage}/{path.name}",
                            commit_message=f"{stage} step {state.step}", strict=False)
        print(f"  hub: {'pushed' if ok else 'push failed (kept locally)'} {path.name}",
              flush=True)

    return push


def _holdout_examples(ctx, records, feed) -> list:
    """A fixed slice held out of training, for the loss signal.

    Taken from the **end** of the mixture and excluded from the feed, because validating
    on examples the model is also training on measures memorisation rather than
    generalisation — and the curve looks better for it, which is the dangerous direction.
    """
    from chartqa_dt.train.validate import LOSS_SLICE

    if len(records) <= LOSS_SLICE * 2:
        return []
    holdout = records[-LOSS_SLICE:]
    feed.records = list(records[:-LOSS_SLICE])
    feed.load_state_dict({**feed.state_dict(), "n": len(feed.records)})

    examples = []
    for record in holdout:
        example = feed._example(record)
        if example is not None:
            examples.append(example)
    return examples


def _progress(log) -> None:
    if log.step % 25 == 0 or log.step <= 3:
        print(f"    step {log.step:>5}  loss {log.loss:.4f}  |grad| {log.grad_norm:.2f}  "
              f"{log.seconds:.2f}s  peak {log.peak_gb:.2f} GiB", flush=True)


def _records_for(ctx):
    """Rehydrate the mixture. Ids only live in the file (rule 7), so records are rebuilt."""
    import json

    from chartqa_dt.train.feed import load_mixture_records

    by_id = {}
    for record in _all_source_records(ctx):
        by_id[record.record_id] = record
    path = Path(ctx.args.mixture)
    ids = json.loads(path.read_text(encoding="utf-8"))["record_ids"]
    print(f"  mixture lists {len(ids):,} ids; rebuilt {len(by_id):,} source records")
    return load_mixture_records(path, by_id)


def _all_source_records(ctx):
    """Every record the mixtures can draw from, rebuilt from the pinned sources."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.build_mixtures import (
        archive_path,
        chartqa_records,
        refchartqa_records,
        synthetic_records,
    )

    from chartqa_dt.data.chartqa import ArchiveReader

    root = Path(ctx.env.data_root)
    yield from synthetic_records(root / "synthetic/train/manifest.json")
    yield from chartqa_records(ArchiveReader(archive_path()), limit=8000,
                               seed=ctx.cfg.seed)
    yield from refchartqa_records(cap=4000, cache=root / "refchartqa_train.jsonl")


def _run_smoke(ctx) -> None:
    from chartqa_dt.modeling.backends.base import list_backends
    from chartqa_dt.train.smoke import HEADER, markdown_table, run_smoke, write_report

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

    # Effective batch is fixed by the pre-registration; only its grouping varies.
    effective = ctx.cfg.train.per_device_batch * ctx.cfg.train.grad_accum
    if ctx.args.batches:
        groupings = []
        for token in (t.strip() for t in ctx.args.batches.split(",") if t.strip()):
            b = int(token)
            if effective % b:
                raise SystemExit(
                    f"per-device batch {b} does not divide the effective batch {effective}; "
                    "changing the effective batch would deviate from the pre-registration"
                )
            groupings.append((b, effective // b))
    else:
        groupings = [(ctx.cfg.train.per_device_batch, ctx.cfg.train.grad_accum)]
    print(f"effective batch fixed at {effective}; groupings: "
          + ", ".join(f"{b}x{a}" for b, a in groupings))

    logger = ctx.logger()
    results = []
    try:
        for backend_name in wanted:
          for batch, accum in groupings:
            for res_label, max_px in budgets:
                label = (f"{backend_name}/{res_label}" if len(groupings) == 1
                         else f"{backend_name}/{res_label}/b{batch}x{accum}")
                print(f"\n{'=' * 78}\n  {label}\n{'=' * 78}")
                r = run_smoke(
                    ctx.cfg,
                    backend_name=backend_name,
                    image_max_pixels=max_px,
                    label=label,
                    steps=ctx.args.steps,
                    per_device_batch=batch,
                    grad_accum=accum,
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
        print("\n--- DECISIONS.md table ---")
        print(markdown_table(results))

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


if __name__ == "__main__":
    main()
