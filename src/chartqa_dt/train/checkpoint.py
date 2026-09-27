from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    stage: str = ""
    feed: dict[str, Any] = field(default_factory=dict)
    best_metric: float | None = None
    best_step: int | None = None
    best_checkpoint: str | None = None
    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)


def save_checkpoint(directory: Path, *, model: Any, optimizer: Any, scheduler: Any, scaler: Any,
                    state: TrainState, processor: Any) -> Path:
    import torch

    from chartqa_dt.seeding import rng_state

    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        model.save_pretrained(str(temporary))
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        torch.save(scaler.state_dict(), temporary / "scaler.pt")
        torch.save(rng_state(), temporary / "rng_state.pt")
        (temporary / "train_state.json").write_text(
            json.dumps(asdict(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        processor.save_pretrained(str(temporary))
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(temporary, directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return directory


def load_adapter_weights(directory: Path, model: Any) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(model, load_file(str(directory / "adapter_model.safetensors")))


def restore_checkpoint(directory: Path, *, optimizer: Any, scheduler: Any, scaler: Any) -> TrainState:
    import torch

    from chartqa_dt.seeding import load_rng_state

    optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True))
    scheduler.load_state_dict(torch.load(directory / "scheduler.pt", map_location="cpu", weights_only=True))
    scaler.load_state_dict(torch.load(directory / "scaler.pt", map_location="cpu", weights_only=True))
    load_rng_state(torch.load(directory / "rng_state.pt", map_location="cpu", weights_only=False))
    return TrainState(**json.loads((directory / "train_state.json").read_text(encoding="utf-8")))
