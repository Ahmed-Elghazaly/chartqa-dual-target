from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from chartqa_dt.config import Config, dump_config
from chartqa_dt.data.chartqa import ArchiveReader
from chartqa_dt.data.mixture import read_mixture_ids
from chartqa_dt.data.pool import load_real_pool, synthetic_records
from chartqa_dt.data.records import ChartRecord
from chartqa_dt.data.sources import chartqa_archive_path
from chartqa_dt.model.prompts import build_grounding_prompt, build_training_prompt
from chartqa_dt.plans.cache import attach_plans
from chartqa_dt.train.collate import Example, encoded_lengths
from chartqa_dt.train.feed import MixtureFeed
from chartqa_dt.train.validate import LOSS_SLICE, METRIC_SLICE


def source_records(cfg: Config, data_root: Path) -> dict[str, ChartRecord]:
    records = synthetic_records(data_root / "synthetic" / "train" / "manifest.json")
    real = load_real_pool(
        chartqa_archive_path(data_root),
        chartqa_limit=cfg.data.chartqa_limit_per_kind,
        seed=cfg.data.mixture_seed,
        refchartqa_cache=data_root / "refchartqa_train.jsonl",
        aligned_cache=data_root / "refchartqa_aligned.jsonl",
    )
    real = attach_plans(real, data_root / "real_plans.jsonl")
    return {record.record_id: record for record in [*records, *real]}


def stratified_holdout(records: Sequence[ChartRecord], *, seed: int, size: int = LOSS_SLICE
                       ) -> tuple[list[ChartRecord], list[ChartRecord]]:
    rows = list(records)
    groups: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(rows):
        plan = record.plan if isinstance(record.plan, dict) else {}
        key = (str(record.source), str(record.question_kind), str(record.meta.get("level", "n/a")),
               str(plan.get("op", "no-plan")))
        groups[key].append(index)
    rng = random.Random(seed)
    for key in sorted(groups):
        rng.shuffle(groups[key])
    exact = {key: size * len(indices) / len(rows) for key, indices in groups.items()}
    allocation = {key: min(len(groups[key]), math.floor(value)) for key, value in exact.items()}
    remaining = size - sum(allocation.values())
    candidates = sorted(groups, key=lambda key: (-(exact[key] - allocation[key]), key))
    while remaining:
        for key in candidates:
            if allocation[key] < len(groups[key]):
                allocation[key] += 1
                remaining -= 1
                if not remaining:
                    break
    held = {index for key, indices in groups.items() for index in indices[: allocation[key]]}
    return ([record for i, record in enumerate(rows) if i not in held],
            [record for i, record in enumerate(rows) if i in held])


def prefilter(records: Sequence[ChartRecord], feed: MixtureFeed, processor: Any, *, max_len: int,
              batch_size: int, prompt_builder: Any) -> list[ChartRecord]:
    accepted: list[ChartRecord] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start:start + batch_size]
        prepared: list[tuple[ChartRecord, Example]] = []
        for record in chunk:
            example = feed.example(record)
            if example is not None:
                prepared.append((record, example))
        if not prepared:
            continue
        try:
            lengths = encoded_lengths(processor, [ex for _, ex in prepared], prompt_builder=prompt_builder)
        except Exception:
            lengths = []
            for _, ex in prepared:
                try:
                    lengths.append(encoded_lengths(processor, [ex], prompt_builder=prompt_builder)[0])
                except Exception:
                    lengths.append(max_len + 1)
        for (record, example), length in zip(prepared, lengths):
            example.image.close()
            if length <= max_len:
                accepted.append(record)
    print(f"  prefilter: {len(accepted):,}/{len(records):,} accepted")
    return accepted


def grounding_truth(record: ChartRecord) -> list[list[float]]:
    evidence = record.evidence_elements
    return [list(element["bbox"]) for element in evidence] if evidence is not None else []


def run_stage(cfg: Config, *, mixture: Path, data_root: Path, out_dir: Path,
              init_adapter: str | None = None, resume: Path | None = None) -> None:
    from chartqa_dt.model.loading import (
        apply_lora,
        check_lora_coverage,
        load_adapter,
        load_model,
        prepare_for_training,
    )
    from chartqa_dt.seeding import set_seed
    from chartqa_dt.train.checkpoint import TrainState, restore_checkpoint
    from chartqa_dt.train.loop import LoopConfig, build_lr_scheduler, build_optimizer, make_grad_scaler, train
    from chartqa_dt.train.validate import make_evaluator

    stage = cfg.train.stage
    set_seed(cfg.seed)
    dump_config(cfg, out_dir)
    by_id = source_records(cfg, data_root)
    records = [by_id[record_id] for record_id in read_mixture_ids(mixture)]
    print(f"\n{stage}: {len(records):,} records from {mixture}")
    train_records, validation_records = stratified_holdout(records, seed=cfg.seed)

    loaded = load_model(cfg.model)
    if resume is not None:
        loaded = load_adapter(loaded, str(resume), trainable=True)
    elif init_adapter is not None:
        loaded = load_adapter(loaded, init_adapter, trainable=True)
    else:
        loaded = apply_lora(loaded, cfg.model)
    loaded = prepare_for_training(loaded, cfg.model)
    check_lora_coverage(loaded.model)

    grounding_only = stage == "stage1"
    prompt_builder = build_grounding_prompt if grounding_only else build_training_prompt
    archive = ArchiveReader(chartqa_archive_path(data_root))
    options = {"seed": cfg.seed, "grounding_only": grounding_only, "image_root": data_root, "archive": archive}
    probe = MixtureFeed([], shuffle=False, **options)
    train_records = prefilter(train_records, probe, loaded.processor, max_len=cfg.model.max_seq_len,
                              batch_size=cfg.train.per_device_batch, prompt_builder=prompt_builder)
    validation_records = prefilter(validation_records, probe, loaded.processor, max_len=cfg.model.max_seq_len,
                                   batch_size=cfg.train.per_device_batch, prompt_builder=prompt_builder)

    feed = MixtureFeed(train_records, shuffle=stage != "stage1", **options)
    steps = cfg.train.max_steps or max(1, math.ceil(
        len(train_records) * cfg.train.epochs / (cfg.train.per_device_batch * cfg.train.grad_accum)))
    loop_cfg = LoopConfig(
        steps=steps, stage=stage, batch_size=cfg.train.per_device_batch, grad_accum=cfg.train.grad_accum,
        max_len=cfg.model.max_seq_len, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay,
        warmup_ratio=cfg.train.warmup_ratio, lr_scheduler=cfg.train.lr_scheduler,
        optimizer_name=cfg.train.optim, max_grad_norm=cfg.train.max_grad_norm,
        save_every=cfg.train.save_every_steps, eval_every=cfg.train.eval_every_steps,
        out_dir=out_dir / "checkpoints",
    )
    print(f"  lr {loop_cfg.lr}  batch {loop_cfg.batch_size}x{loop_cfg.grad_accum}  "
          f"max_len {loop_cfg.max_len}  steps {steps}")

    loss_examples: list[Example] = []
    metric_items: list[dict[str, Any]] = []
    for record in validation_records:
        example = probe.example(record)
        assert example is not None
        loss_examples.append(example)
        metric_items.append({"record_id": record.record_id, "question": record.question,
                             "image": example.image, "answer": str(record.answer or ""),
                             "boxes": grounding_truth(record)})
    order = list(range(len(metric_items)))
    random.Random(cfg.seed).shuffle(order)
    metric_items = [metric_items[i] for i in order[:METRIC_SLICE]]

    log_path = out_dir / "metrics.jsonl"
    validation_reports: list[dict[str, Any]] = []

    def log(entry: dict[str, Any]) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        if entry.get("step", 0) % 25 == 0 or entry.get("step", 0) <= 3:
            print(f"    step {entry['step']:>5}  loss {entry['loss']:.4f}  |grad| {entry['grad_norm']:.2f}  "
                  f"{entry['seconds']:.2f}s  peak {entry['peak_gb']:.2f} GiB", flush=True)

    def on_report(report: dict[str, Any]) -> None:
        validation_reports.append(report)
        (out_dir / f"{stage}_validation.json").write_text(json.dumps(validation_reports, indent=2) + "\n")

    evaluate = make_evaluator(
        loaded, loss_examples, metric_items, max_len=loop_cfg.max_len, loss_batch_size=loop_cfg.batch_size,
        metric_batch_size=cfg.train.metric_batch_size, metric_every=cfg.train.metric_every_steps,
        final_step=steps, mode="grounding" if grounding_only else "training",
        prompt_builder=prompt_builder, on_report=on_report,
    )

    optimizer = build_optimizer(loaded.model, loop_cfg.lr, weight_decay=loop_cfg.weight_decay,
                                optim=loop_cfg.optimizer_name)
    scheduler = build_lr_scheduler(optimizer, loop_cfg)
    scaler = make_grad_scaler(loaded)
    state = TrainState(stage=stage)
    if resume is not None:
        state = restore_checkpoint(resume, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        feed.load_state_dict(state.feed)
        print(f"  resumed at step {state.step}")

    state = train(loaded, feed, loop_cfg, prompt_builder=prompt_builder, evaluate=evaluate, state=state,
                  optimizer=optimizer, scheduler=scheduler, scaler=scaler, on_log=log)
    (out_dir / f"{stage}_report.json").write_text(json.dumps({
        "stage": stage, "steps": state.step, "best_checkpoint": state.best_checkpoint,
        "best_metric": state.best_metric, "train_records": len(train_records),
        "validation_records": len(validation_records), "refused_by_feed": feed.refused,
    }, indent=2) + "\n")
    print(f"\ndone: {state.step} steps; best checkpoint {state.best_checkpoint}")
