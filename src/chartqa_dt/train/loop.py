"""The Phase 6 training loop — `PLAN.md` 6.1, 6.2, 6.5, 6.6.

Assembled from parts that were each verified separately: `train/collate.py` for masking,
`train/feed.py` for ordering and resume, `train/checkpoint.py` for state, and
`train/smoke.py`'s `build_optimizer`, which is the only place that decides between
`AdamW8bit` and `torch.optim.AdamW`.

Two things the loop does that are easy to leave out and expensive to discover:

* **Gradient norms are recorded every step.** On a T4 the compute dtype is float16
  (`DECISIONS.md` 0017), and float16 without a gradient scaler can underflow gradients to
  exactly zero. The loss then sits flat while nothing else looks wrong, so a norm of 0.0
  or a non-finite norm — not a NaN loss — is the signal.
* **`assert_lora_on_both_sides` runs before the first step**, per the Phase 6 acceptance
  criteria. A run that trains only the language side looks entirely healthy and answers
  the wrong research question.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chartqa_dt.train.checkpoint import (
    SAVE_EVERY_STEPS,
    EarlyStopping,
    TrainState,
    save_checkpoint,
)
from chartqa_dt.train.collate import build_batch
from chartqa_dt.train.feed import MixtureFeed

#: `PLAN.md` 6.1 and 6.2.
STAGE_LR = {"stage1": 1e-4, "stage2": 5e-5}
#: The Phase 6 fallback: "if Stage 2 destabilises, reduce learning rate to 2e-5 before
#: touching anything else."
STAGE2_FALLBACK_LR = 2e-5
#: Gradient-norm clipping. 1.0 is the standard value for transformer fine-tuning and is not a
#: tuned choice; the measured gradient-norm medians on this model were 13.3 and 14.1 with zero
#: dead or non-finite steps, so clipping is active and the run is stable under it.
GRAD_CLIP = 1.0


@dataclass
class TrainConfig:
    stage: str = "stage1"
    steps: int = 3000                    # `PLAN.md` 6.6: ~24,000 presentations
    batch_size: int = 2
    grad_accum: int = 4                  # effective batch 8
    max_len: int = 1024
    lr: float | None = None              # None -> STAGE_LR[stage]
    save_every: int = SAVE_EVERY_STEPS
    eval_every: int = 250
    patience: int = 2
    seed: int = 0
    out_dir: Path = Path("outputs/phase6")
    answer_only: bool = False            # `PLAN.md` 6.4 control

    @property
    def learning_rate(self) -> float:
        return self.lr if self.lr is not None else STAGE_LR[self.stage]


@dataclass
class StepLog:
    """`PLAN.md` 6.5: loss, grad norm, memory and step time, at every step."""

    step: int
    loss: float
    grad_norm: float
    seconds: float
    peak_gb: float
    supervised: int

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "loss": self.loss, "grad_norm": self.grad_norm,
                "seconds": self.seconds, "peak_gb": self.peak_gb,
                "supervised": self.supervised}


@dataclass
class TrainResult:
    state: TrainState
    logs: list[StepLog] = field(default_factory=list)
    stopped_early: bool = False

    def summary(self) -> str:
        if not self.logs:
            return "  no steps run"
        first = sum(x.loss for x in self.logs[:10]) / min(10, len(self.logs))
        last = sum(x.loss for x in self.logs[-10:]) / min(10, len(self.logs))
        dead = sum(1 for x in self.logs if x.grad_norm == 0.0 or x.grad_norm != x.grad_norm)
        return (f"  steps {len(self.logs)}   loss {first:.3f} -> {last:.3f}\n"
                f"  peak {max(x.peak_gb for x in self.logs):.3f} GiB   "
                f"{sum(x.seconds for x in self.logs) / len(self.logs):.2f} s/step\n"
                f"  dead or non-finite gradient steps: {dead}"
                + ("  <- float16 underflow, the loss curve will not show it" if dead else ""))


def train(loaded: Any, feed: MixtureFeed, cfg: TrainConfig, *,
          evaluate: Any = None, state: TrainState | None = None,
          optimizer: Any = None, on_log: Any = None,
          on_checkpoint: Any = None) -> TrainResult:
    """Run `cfg.steps` optimizer steps, checkpointing and evaluating as configured."""
    import torch

    from chartqa_dt.modeling.lora_assert import assert_lora_on_both_sides
    from chartqa_dt.train.smoke import build_optimizer

    model = loaded.model
    # Phase 6 acceptance criterion. A run that trains one side looks healthy and answers
    # the wrong question, so this is checked before any compute is spent.
    assert_lora_on_both_sides(model)

    model.train()
    device = next(model.parameters()).device
    if optimizer is None:
        optimizer = build_optimizer(model, cfg.learning_rate)
    state = state or TrainState(stage=cfg.stage)
    stopper = EarlyStopping(patience=cfg.patience)
    result = TrainResult(state=state)
    trainable = [p for p in model.parameters() if p.requires_grad]
    stream = feed.batches(cfg.batch_size)

    while state.step < cfg.steps:
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total, supervised = 0.0, 0
        for _ in range(cfg.grad_accum):
            examples = next(stream)
            batch = build_batch(loaded.processor, examples, cfg.max_len, strict=False)
            supervised += int(batch.pop("_supervised_positions", 0))
            batch = {k: (v.to(device) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            loss = model(**batch).loss / cfg.grad_accum
            loss.backward()
            total += float(loss.detach())

        norm = float(torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP))
        optimizer.step()
        state.step += 1
        state.epoch = feed.epoch
        state.losses.append(total)
        state.grad_norms.append(norm)

        log = StepLog(step=state.step, loss=total, grad_norm=norm,
                      seconds=time.perf_counter() - started,
                      peak_gb=_peak_gb(), supervised=supervised)
        result.logs.append(log)
        if on_log is not None:
            on_log(log)

        if cfg.save_every and state.step % cfg.save_every == 0:
            state.feed = feed.state_dict()
            path = save_checkpoint(cfg.out_dir / f"{cfg.stage}-step{state.step}",
                                   model=model, optimizer=optimizer, state=state)
            # `PLAN.md` 6.3 pushes on every save. The callback owns the failure policy:
            # losing a periodic upload must never end a run that is otherwise healthy.
            if on_checkpoint is not None:
                on_checkpoint(path, state)

        if evaluate is not None and cfg.eval_every and state.step % cfg.eval_every == 0:
            metric = float(evaluate(model, state.step))
            if stopper.update(state, metric, state.step):
                result.stopped_early = True
                break

    state.feed = feed.state_dict()
    final = save_checkpoint(cfg.out_dir / f"{cfg.stage}-final", model=model,
                            optimizer=optimizer, state=state)
    if on_checkpoint is not None:
        on_checkpoint(final, state)
    return result


def _peak_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return max(torch.cuda.max_memory_reserved(i) for i in
                   range(torch.cuda.device_count())) / 1024 ** 3
    except Exception:  # noqa: BLE001 - reporting must never break a run
        return 0.0


__all__ = ["GRAD_CLIP", "STAGE2_FALLBACK_LR", "STAGE_LR", "StepLog", "TrainConfig",
           "TrainResult", "train"]
