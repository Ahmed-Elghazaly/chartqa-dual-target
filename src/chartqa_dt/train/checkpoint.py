"""Checkpointing that a resume can actually use — `PLAN.md` 6.3.

    Save every 100 optimizer steps: adapter weights, optimizer state, scheduler state,
    RNG states, and the dataloader position. **Test the resume path by deliberately
    killing a run** — a resume that has never been tested does not work.

Phase 2 proved that instruction correct twice over, and both lessons are encoded here
rather than described:

* **The optimizer must be rebuilt by the same factory.** `bitsandbytes`' `AdamW8bit`
  stores its moments under different state keys, so loading its `state_dict` into a
  `torch.optim.AdamW` raises `KeyError: 'exp_avg'`. That is how the first 100-step run
  failed — *after* the 100 steps had already succeeded (`train/smoke.py`).
* **RNG state belongs in the checkpoint.** Without it, resume reproduced a loss delta of
  0.0438; with it, 0.0014–0.0053 (`DECISIONS.md` 0026). A missing RNG state does not
  crash — it just makes a resumed run quietly different from an uninterrupted one.

Everything needed to continue is written to one directory, and `verify_resume` compares a
resumed step against the uninterrupted one so "resume works" is a measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: `PLAN.md` 6.3. Also the resume granularity: a crash costs at most this many steps.
SAVE_EVERY_STEPS = 100

#: A resumed step whose loss differs from the uninterrupted one by more than this is a
#: broken resume, not noise. Phase 2 measured 0.0014–0.0053 with the RNG state restored
#: and 0.0438 without it, so the threshold sits well inside that gap.
RESUME_TOLERANCE = 0.02


@dataclass
class TrainState:
    """Everything about *where* a run is, as opposed to what its weights are."""

    step: int = 0
    epoch: int = 0
    stage: str = ""
    feed: dict[str, Any] = field(default_factory=dict)
    best_metric: float | None = None
    best_step: int | None = None
    evals_without_improvement: int = 0
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_checkpoint(directory: str | Path, *, model: Any, optimizer: Any,
                    state: TrainState, scheduler: Any = None,
                    processor: Any = None) -> Path:
    """Write a checkpoint that `load_checkpoint` can fully restore."""
    import torch

    from chartqa_dt.seeding import rng_state

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(path))                      # adapter weights only
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), path / "scheduler.pt")
    # `weights_only=False` on load, so this must never contain untrusted data. It does
    # not: it is our own RNG state.
    torch.save(rng_state(), path / "rng_state.pt")
    (path / "train_state.json").write_text(
        json.dumps(state.to_dict(), indent=2) + "\n", encoding="utf-8")
    if processor is not None:
        processor.save_pretrained(str(path))
    return path


def load_checkpoint(directory: str | Path, *, model: Any, optimizer_factory: Any,
                    scheduler: Any = None) -> tuple[Any, TrainState]:
    """Restore a run. Returns the rebuilt optimizer and the training state.

    `optimizer_factory` rather than an optimizer, deliberately: the factory is the single
    place that decides between `AdamW8bit` and `torch.optim.AdamW`, and a checkpoint
    written by one cannot be loaded into the other.
    """
    import torch

    from chartqa_dt.seeding import load_rng_state

    path = Path(directory)
    missing = [n for n in ("optimizer.pt", "rng_state.pt", "train_state.json")
               if not (path / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"{path} is not a resumable checkpoint; missing {missing}. Resuming from a "
            f"partial checkpoint silently changes the run.")

    model.load_adapter(str(path), adapter_name="default", is_trainable=True) \
        if hasattr(model, "load_adapter") else None

    optimizer = optimizer_factory(model)
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu"))
    if scheduler is not None and (path / "scheduler.pt").exists():
        scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location="cpu"))
    load_rng_state(torch.load(path / "rng_state.pt", map_location="cpu",
                              weights_only=False))

    raw = json.loads((path / "train_state.json").read_text(encoding="utf-8"))
    return optimizer, TrainState(**raw)


@dataclass
class EarlyStopping:
    """`PLAN.md` 6.6, with N fixed in the pre-registration rather than after a curve.

    The checkpoint reported is the last one that *improved*, not the last one trained —
    which is why `best_step` is carried in the state rather than inferred later.
    """

    patience: int = 2
    min_delta: float = 0.0

    def update(self, state: TrainState, metric: float, step: int) -> bool:
        """Record an evaluation. Returns True when training should stop."""
        if state.best_metric is None or metric > state.best_metric + self.min_delta:
            state.best_metric = metric
            state.best_step = step
            state.evals_without_improvement = 0
            return False
        state.evals_without_improvement += 1
        return state.evals_without_improvement >= self.patience


def resume_delta(uninterrupted: float, resumed: float) -> float:
    """How far a resumed step's loss sits from the uninterrupted one."""
    return abs(uninterrupted - resumed)


def assert_resume_matched(uninterrupted: float, resumed: float,
                          tolerance: float = RESUME_TOLERANCE) -> float:
    """Fail loudly when a resume is not equivalent. `PLAN.md` 6.3 wants this measured."""
    delta = resume_delta(uninterrupted, resumed)
    if delta > tolerance:
        raise AssertionError(
            f"resumed loss {resumed:.4f} differs from uninterrupted {uninterrupted:.4f} "
            f"by {delta:.4f}, over the {tolerance} tolerance. Phase 2 saw exactly this "
            f"when the RNG state was missing from the checkpoint (DECISIONS.md 0026)."
        )
    return delta


__all__ = ["RESUME_TOLERANCE", "SAVE_EVERY_STEPS", "EarlyStopping", "TrainState",
           "assert_resume_matched", "load_checkpoint", "resume_delta", "save_checkpoint"]
