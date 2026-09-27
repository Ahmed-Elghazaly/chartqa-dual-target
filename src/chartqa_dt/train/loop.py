from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chartqa_dt.train.checkpoint import TrainState, load_adapter_weights, save_checkpoint
from chartqa_dt.train.collate import build_batch
from chartqa_dt.train.feed import MixtureFeed


@dataclass
class LoopConfig:
    steps: int
    stage: str
    batch_size: int
    grad_accum: int
    max_len: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler: str
    optimizer_name: str
    max_grad_norm: float
    save_every: int
    eval_every: int
    out_dir: Path


def build_optimizer(model: Any, lr: float, *, weight_decay: float, optim: str) -> Any:
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    if optim == "adamw_fused":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
    if optim == "adamw_8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def build_lr_scheduler(optimizer: Any, cfg: LoopConfig) -> Any:
    import torch

    warmup = round(cfg.steps * cfg.warmup_ratio)

    def scale(step: int) -> float:
        if warmup and step < warmup:
            return max(step, 1) / warmup
        if cfg.lr_scheduler == "constant":
            return 1.0
        progress = min(max((step - warmup) / max(cfg.steps - warmup, 1), 0.0), 1.0)
        if cfg.lr_scheduler == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def make_grad_scaler(loaded: Any) -> Any:
    import torch

    enabled = bool(torch.cuda.is_available() and loaded.dtype == "float16")
    return torch.amp.GradScaler("cuda", enabled=enabled)


def _peak_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return float(max(torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())) / 1024**3)


def train(loaded: Any, feed: MixtureFeed, cfg: LoopConfig, *, prompt_builder: Callable[[str], str],
          evaluate: Callable[[Any, int], float | None], state: TrainState, optimizer: Any, scheduler: Any,
          scaler: Any, on_log: Callable[[dict[str, Any]], None]) -> TrainState:
    import torch

    model = loaded.model
    model.train()
    device = next(model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    stream = feed.batches(cfg.batch_size)

    def checkpoint(name: str) -> Path:
        state.feed = feed.state_dict()
        return save_checkpoint(cfg.out_dir / name, model=model, optimizer=optimizer, scheduler=scheduler,
                               scaler=scaler, state=state, processor=loaded.processor)

    while state.step < cfg.steps:
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total, supervised = 0.0, 0
        for _ in range(cfg.grad_accum):
            examples = next(stream)
            try:
                batch, n_supervised = build_batch(loaded.processor, examples, cfg.max_len,
                                                  prompt_builder=prompt_builder)
            finally:
                for example in examples:
                    example.image.close()
            supervised += n_supervised
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            loss = model(**batch).loss / cfg.grad_accum
            loss_value = float(loss.detach())
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"non-finite training loss at step {state.step + 1}")
            scaler.scale(loss).backward()
            total += loss_value

        scaler.unscale_(optimizer)
        norm = float(torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm))
        if not math.isfinite(norm) or norm <= 0.0:
            raise FloatingPointError(f"invalid gradient norm {norm} at step {state.step + 1}")
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        state.step += 1
        state.epoch = feed.epoch
        state.losses.append(total)
        state.grad_norms.append(norm)
        on_log({"step": state.step, "loss": total, "grad_norm": norm,
                "seconds": time.perf_counter() - started, "peak_gb": _peak_gb(), "supervised": supervised})

        if cfg.eval_every and (state.step % cfg.eval_every == 0 or state.step == cfg.steps):
            metric = evaluate(model, state.step)
            if metric is not None and (state.best_metric is None or metric > state.best_metric):
                state.best_metric, state.best_step = metric, state.step
                state.best_checkpoint = f"{cfg.stage}-best-step{state.step}"
                checkpoint(state.best_checkpoint)

        if cfg.save_every and state.step % cfg.save_every == 0:
            checkpoint(f"{cfg.stage}-step{state.step}")

    checkpoint(f"{cfg.stage}-final")
    if state.best_checkpoint is not None:
        load_adapter_weights(cfg.out_dir / state.best_checkpoint, model)
    return state
